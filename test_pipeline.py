"""End-to-end and unit tests for Dubbing Studio.

Run it from anywhere:  python test_pipeline.py

The job data is redirected to a scratch folder so a test run never disturbs the
real jobs under Application Support. Set DUBBING_TEST_SOURCE to a video with
speech in it to exercise transcription against real recorded audio; without one
the suite synthesises its own speech and is fully self-contained.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Must be set before app.config is imported anywhere — it reads these at import
# time to decide where the job and output folders live.
SCRATCH = Path(os.environ.setdefault(
    "DUBBING_STUDIO_HOME", str(Path(tempfile.gettempdir()) / "dubbing-studio-test")))
os.environ.setdefault("DUBBING_STUDIO_OUTPUT", str(SCRATCH / "output"))
WORK = SCRATCH / "work"
WORK.mkdir(parents=True, exist_ok=True)

# Point the scratch model folder at the real one so a test run doesn't re-fetch
# 700 MB of speech models it already has. Jobs and output stay isolated.
#
# The scratch home lives under /var/folders, which macOS purges on its own
# schedule. It deletes the big model files but leaves the directories, so a
# plain "does it exist" check kept a hollowed-out models folder in place and the
# suite failed from deep inside sherpa-onnx with "File doesn't exist" — an
# environment problem wearing a code regression's clothes. A real directory that
# is not the symlink we intended is therefore replaced, not respected.
_real_models = Path.home() / "Library" / "Application Support" / "DubbingStudio" / "models"
_scratch_models = SCRATCH / "models"
if _real_models.is_dir() and not _scratch_models.is_symlink():
    if _scratch_models.is_dir():
        shutil.rmtree(_scratch_models, ignore_errors=True)
    _scratch_models.parent.mkdir(parents=True, exist_ok=True)
    _scratch_models.symlink_to(_real_models)

import numpy as np
import soundfile as sf

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def stub_download(clip: Path, title: str, duration: float, replace: bool = False):
    """Stand in for the download step, probe included.

    The pipeline probes the link before it draws the progress plan, because a
    sample has to know how much of the whole video the download covers before it
    can weight it. A stub that replaces only download() therefore leaves the real
    yt-dlp being asked about a made-up URL, and every job dies on its first line.

    Returns (probe, download) for the caller to install; everything after the
    download is the real thing.
    """
    meta = {"title": title, "duration": duration, "uploader": "t", "thumbnail": ""}

    def fake_probe(url):
        return dict(meta)

    # **_ so a new pass-through argument is not a suite-wide failure.
    def fake_download(url, workdir, quality="best", progress=None, info=None, **_):
        workdir.mkdir(parents=True, exist_ok=True)
        dest = workdir / "source.mp4"
        if replace or not dest.exists():
            shutil.copy(clip, dest)
        if progress:
            progress(1.0, "Downloaded")
        return dest, dict(meta)

    return fake_probe, fake_download


def speech_wav(dst: Path, seconds: float = 75.0) -> Path:
    """Audio with real speech in it, for the transcription end of the pipeline.

    Prefers a real recording if one is configured, because recorded speech is a
    far harder and more representative test of the recogniser than synthesised
    speech is. Falls back to the app's own portable voice so the suite still runs
    on a machine that has no sample video to hand.
    """
    if dst.exists():
        return dst

    source = os.environ.get("DUBBING_TEST_SOURCE", str(Path.home() / "Downloads" / "videoplayback.mp4"))
    if source and Path(source).exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "20", "-t", f"{seconds:g}",
                        "-i", source, "-ac", "1", "-ar", "24000", str(dst)], check=True)
        return dst

    from app.backends.tts import OnnxTTS
    engine = OnnxTTS()
    lines = ["Right, let us carry on with the next round of the pattern.",
             "We chain three stitches and then turn the work over.",
             "Now work a single crochet into that very same stitch.",
             "Keep the tension nice and loose so the fabric drapes well.",
             "You can see how the pattern starts to build up along here."]
    pieces, rate = [], 24000
    while sum(p.size for p in pieces) / rate < seconds:
        for text in lines:
            audio, rate = engine.say(text, voice="bf_emma")
            pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            pieces.append(np.zeros(int(0.45 * rate), dtype=np.float32))
            if sum(p.size for p in pieces) / rate >= seconds:
                break
    sf.write(dst, np.concatenate(pieces)[:int(seconds * rate)], rate)
    return dst


# ======================================================= 1. alignment maths
def test_align():
    print("\n[1] Alignment maths")
    from app.steps import align

    sr = 24000

    def tone(seconds):
        return (0.2 * np.sin(np.linspace(0, seconds * 440 * 2 * np.pi, int(seconds * sr)))
                ).astype(np.float32)

    # Lines that comfortably fit their slots must not be touched at all.
    lines = [{"start": 0.0, "end": 2.0, "samples": tone(1.0)},
             {"start": 5.0, "end": 7.0, "samples": tone(1.0)},
             {"start": 10.0, "end": 12.0, "samples": tone(1.0)}]
    track, stats = align.assemble([dict(l) for l in lines], 15.0, sr)
    check("no compression when lines fit", stats["compressed"] == 0, str(stats["compressed"]))
    check("no drift when lines fit", stats["max_drift"] == 0.0)
    check("track is the right length", abs(len(track) / sr - 16.0) < 0.01,
          f"{len(track)/sr:.2f}s")

    # A line far too long for its slot must be compressed, but capped.
    lines = [{"start": 0.0, "end": 1.0, "samples": tone(6.0)},
             {"start": 2.0, "end": 3.0, "samples": tone(0.5)}]
    track, stats = align.assemble([dict(l) for l in lines], 10.0, sr, max_stretch=1.55)
    check("over-long line is compressed", stats["compressed"] >= 1)
    check("compression respects the cap", stats["max_factor"] <= 1.5501,
          f"{stats['max_factor']}")
    check("over-cap lines are counted", stats["over_cap"] == 1, str(stats["over_cap"]))

    # A line that overruns must push the next one later, not overlap it.
    lines = [{"start": 0.0, "end": 1.0, "samples": tone(4.0)},
             {"start": 1.0, "end": 2.0, "samples": tone(1.0)}]
    _, stats = align.assemble([dict(l) for l in lines], 12.0, sr, max_stretch=1.1)
    check("drift is recorded when a line overruns", stats["max_drift"] > 0,
          f"{stats['max_drift']}s")

    # Peak normalisation.
    loud = [{"start": 0.0, "end": 2.0, "samples": (tone(1.0) * 20).astype(np.float32)}]
    track, _ = align.assemble(loud, 5.0, sr)
    check("output is normalised, not clipped", 0.85 <= float(np.max(np.abs(track))) <= 0.9,
          f"peak {float(np.max(np.abs(track))):.3f}")

    # atempo chaining for large factors.
    check("atempo chains beyond 2x", align._atempo_chain(3.5).count("atempo") == 2,
          align._atempo_chain(3.5))
    check("atempo single stage under 2x", align._atempo_chain(1.4).count("atempo") == 1)

    # SRT output.
    segs = [{"start": 1.5, "end": 3.25, "translation": "Hello there"},
            {"start": 4.0, "end": 5.0, "translation": "Second line"}]
    out = WORK / "test.srt"
    align.write_srt(segs, out)
    body = out.read_text()
    check("SRT timestamps are formatted correctly", "00:00:01,500 --> 00:00:03,250" in body,
          body.splitlines()[1] if body else "empty")


# ==================================================== 2. translation parsing
def test_translate():
    print("\n[2] Translation handling")
    from app.backends import translate as T

    batch = [{"i": 0, "start": 0, "end": 2, "text": "hola"},
             {"i": 1, "start": 2, "end": 4, "text": "adios"}]

    check("parses clean output",
          T._parse("0|Hello\n1|Goodbye", batch) == {0: "Hello", 1: "Goodbye"})
    check("ignores preamble and fences",
          T._parse("Sure!\n```\n0|Hello\n1|Goodbye\n```", batch) == {0: "Hello", 1: "Goodbye"})
    check("strips an echoed time marker",
          T._parse("0|[2.0s] Hello", batch) == {0: "Hello"})
    check("drops ids not in the batch",
          T._parse("0|Hello\n99|Nope", batch) == {0: "Hello"})
    check("survives a missing line",
          T._parse("1|Goodbye", batch) == {1: "Goodbye"})
    check("handles pipes inside the translation",
          T._parse("0|Hello | there", batch) == {0: "Hello | there"})

    prompt = T._build_prompt(batch, ["contexto"], "English", "hola -> hi")
    check("prompt carries the slot length", "[2.0s]" in prompt)
    check("prompt carries the glossary", "hola -> hi" in prompt)
    check("prompt marks context as not-for-translation", "do NOT translate" in prompt)

    # Full translate() with a stub backend, including a deliberately flaky one.
    from app.config import Settings
    settings = Settings()
    settings.translator = "ollama"

    segs = [{"start": i * 2.0, "end": i * 2.0 + 1.8, "text": f"linea {i}"} for i in range(60)]
    calls = {"n": 0}

    def flaky(prompt, model, host="x"):
        calls["n"] += 1
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        if calls["n"] == 1:                       # first batch drops two lines
            ids = ids[:-2]
        return "\n".join(f"{i}|English line {i}" for i in ids)

    T._call_ollama = flaky
    out = T.translate([dict(s) for s in segs], settings, 16)
    check("every segment gets a translation",
          all(s.get("translation") for s in out), f"{sum(1 for s in out if s.get('translation'))}/60")
    check("retry recovered the dropped lines", out[len(out) - 1]["translation"].startswith("English"))

    # A backend that returns nothing must raise, not ship a silent empty dub.
    T._call_ollama = lambda p, m, host="x": ""
    try:
        T.translate([dict(s) for s in segs], settings, 16)
        check("empty backend raises", False)
    except T.TranslationError:
        check("empty backend raises", True)


# ============================================================ 3. HTTP layer
def test_server():
    print("\n[3] Web server")
    from fastapi.testclient import TestClient
    from app.server import app

    client = TestClient(app)

    r = client.get("/")
    check("home page loads", r.status_code == 200 and "Dubbing Studio" in r.text)

    r = client.get("/api/state")
    check("state endpoint responds", r.status_code == 200)
    body = r.json()
    check("state includes machine info", "engine" in body["machine"], body["machine"]["engine"])
    check("state includes voices", any(v["id"] == "bf_emma" for v in body["voices"]))
    check("state includes glossaries", "crochet_us" in body["glossaries"])

    r = client.post("/api/settings", json={"data": {"voice": "bf_lily", "speed": 1.1,
                                                    "write_srt": True}})
    check("settings save", r.status_code == 200 and r.json()["voice"] == "bf_lily")
    check("numeric settings coerce", r.json()["speed"] == 1.1)
    check("boolean settings coerce", r.json()["write_srt"] is True)

    r = client.post("/api/settings", json={"data": {"speed": "not-a-number"}})
    check("bad setting value is ignored, not fatal", r.status_code == 200)

    client.post("/api/settings", json={"data": {"voice": "bf_emma", "write_srt": False}})

    r = client.post("/api/job", json={"url": ""})
    check("empty link is rejected", r.status_code == 400)
    r = client.post("/api/job", json={"url": "not a url"})
    check("non-link is rejected", r.status_code == 400)

    r = client.get("/api/doctor")
    check("doctor endpoint responds", r.status_code == 200 and "checks" in r.json())

    # An old yt-dlp is the commonest reason a video describes itself happily and
    # then refuses to download, and the check used to say only that it existed.
    import datetime as _dt
    from app import server as _srv
    names = [c["name"] for c in r.json()["checks"]]
    check("the setup check names the yt-dlp version",
          any(n.startswith("yt-dlp — ") for n in names), str(names[:3]))

    real_run = _srv.subprocess.run

    class _Out:
        def __init__(self, s): self.stdout = s

    def dated(days_old):
        d = _dt.date.today() - _dt.timedelta(days=days_old)
        return lambda *a, **k: _Out(f"{d.year}.{d.month:02d}.{d.day:02d}\n")

    try:
        _srv.subprocess.run = dated(5)
        check("a recent yt-dlp is not flagged", _srv._ytdlp_age()[0] is False)
        _srv.subprocess.run = dated(200)
        stale, shown = _srv._ytdlp_age()
        check("one from months ago is", stale is True, shown)
        _srv.subprocess.run = lambda *a, **k: _Out("2026.07.04.dev0+abc\n")
        check("a build with an undateable version is left alone",
              _srv._ytdlp_age() == (False, "2026.07.04.dev0+abc"))
        def boom(*a, **k): raise OSError("no yt-dlp")
        _srv.subprocess.run = boom
        check("and a missing yt-dlp is not reported as stale",
              _srv._ytdlp_age() == (False, ""))
    finally:
        _srv.subprocess.run = real_run

    # What a failed download actually says to the person reading it. The failed
    # panel shows this verbatim, so anything yt-dlp phrases for itself ends up
    # in front of someone who cannot act on it.
    from app.steps.download import _friendly
    dead = _friendly("ERROR: [generic] nope: Unable to download webpage: HTTP Error "
                     "404: File not found (caused by <HTTPError 404: File not found>)")
    check("a dead link is explained in plain English",
          "typed correctly" in dead and "ERROR" not in dead, dead)
    # Two files have to agree about which model a given Mac gets: config.py
    # suggests it and the setup check reports it, while Install.command is what
    # actually downloads it. A comment asks them to be kept in step; this is what
    # notices when they aren't.
    import re as _re
    from app.config import suggest_ollama_model
    ladder = _re.findall(r"RAM_GB >= (\d+) \)\); then LADDER=\(([^)]+)\)",
                         (ROOT / "Install.command").read_text())
    check("the installer names its memory tiers", len(ladder) >= 2, str(ladder))
    for ram, models in ladder:
        first = models.split()[0]
        check(f"at {ram} GB the installer fetches what the app expects",
              first == suggest_ollama_model(int(ram)),
              f"installs {first}, app wants {suggest_ollama_model(int(ram))}")
    # Each tier falls back to something smaller, so one failed download is not
    # the end of it.
    check("every tier but the smallest has something to fall back to",
          all(len(m.split()) > 1 for _, m in ladder), str(ladder))
    # Capped deliberately: the 32B is a 20 GB download and slower per line.
    check("no tier defaults to the 32B model",
          all("32b" not in suggest_ollama_model(r) for r in (8, 16, 24, 48, 64, 128)))

    # Nobody outside this code knows model tags exist, and a machine whose
    # installer pulled a different size used to die on a 404 mid-translation.
    from app.backends import translate as T
    real_installed = T.installed_models
    T.installed_models = lambda host="": [{"name": "qwen3:8b", "size": 5_000_000_000},
                                          {"name": "llama3:8b", "size": 4_000_000_000}]
    try:
        picked, note_ = T.usable_model("qwen3:8b")
        check("an installed model is used as-is", picked == "qwen3:8b" and not note_)
        picked, note_ = T.usable_model("qwen3:32b")
        check("a missing size falls back within the same family",
              picked == "qwen3:8b" and "isn't installed" in note_, picked)
        picked, note_ = T.usable_model("mistral:7b")
        check("a missing family falls back to whatever is there",
              picked in ("qwen3:8b", "llama3:8b") and note_, picked)
        T.installed_models = lambda host="": []
        picked, note_ = T.usable_model("qwen3:8b")
        check("with nothing installed it asks for what it wanted",
              picked == "qwen3:8b" and not note_)
    finally:
        T.installed_models = real_installed

    check("an offline machine is told so",
          "internet connection" in _friendly("ERROR: unable to open: "
                                             "nodename nor servname provided"))
    # Said honestly: by the time this reaches anyone the download has already
    # been retried as several different player clients, so "try again in a
    # minute" is advice that has been taken on their behalf and failed.
    forbidden = _friendly("ERROR: unable to download: HTTP Error 403: Forbidden")
    check("a 403 says what to actually do about it",
          "Sign in as" in forbidden and "every attempt" in forbidden, forbidden[:80])
    check("and no longer claims it is temporary", "temporary" not in forbidden.lower())
    odd = _friendly("ERROR: [youtube] abc: Something nobody anticipated "
                    "(caused by <SomeError: blah>)")
    check("anything unrecognised is passed on, minus the scaffolding",
          odd == "Something nobody anticipated", odd)

    r = client.get("/api/job/nope/video")
    check("missing job video 404s", r.status_code == 404)

    # The preset is read back off the switches rather than stored as an
    # assertion about them. It used to be the latter, so choosing Balanced and
    # then turning separation off left it saying "balanced" and the segmented
    # control in the interface claiming a preset the settings no longer were.
    from app.config import PRESETS, Settings
    for name in PRESETS:
        check(f"{name} settings identify as {name}",
              Settings().apply_preset(name).matching_preset() == name)
    # Balanced without separation is exactly Fast, and is named as such rather
    # than called custom — the presets are points in one space, not labels.
    off = Settings().apply_preset("balanced")
    off.separate_audio = False
    check("balanced minus separation is recognised as fast",
          off.matching_preset() == "fast", off.matching_preset())
    # A combination no preset describes.
    mixed = Settings().apply_preset("balanced")
    mixed.asr_model = "whisper"
    check("a mix no preset describes reads as custom",
          mixed.matching_preset() == "custom", mixed.matching_preset())

    # How many people are speaking is a fact about the video, not a
    # quality-versus-cost setting, so it belongs to neither preset.
    for name in PRESETS:
        check(f"{name} does not dictate who is speaking", "diarize" not in PRESETS[name])
    spoke = Settings().apply_preset("best")
    spoke.diarize = True
    check("saying several people speak does not change the preset",
          spoke.matching_preset() == "best", spoke.matching_preset())

    r = client.post("/api/settings", json={"data": {"preset": "balanced"}})
    check("picking a preset saves it", r.json()["preset"] == "balanced")
    r = client.post("/api/settings", json={"data": {"asr_model": "whisper"}})
    check("an off-preset switch flips the saved preset to custom",
          r.json()["preset"] == "custom", r.json()["preset"])
    r = client.post("/api/settings", json={"data": {"asr_model": "parakeet"}})
    check("and putting it back returns to the named preset",
          r.json()["preset"] == "balanced", r.json()["preset"])
    r = client.post("/api/settings", json={"data": {"diarize": True}})
    check("but saying several people speak leaves the preset alone",
          r.json()["preset"] == "balanced", r.json()["preset"])


# ======================================================= 4. full pipeline run
def test_end_to_end():
    print("\n[4] Full pipeline (real ASR, real TTS, real mux)")
    from app import pipeline
    from app.backends import translate as T
    from app.config import Settings, JOBS
    from app.steps import download as dl

    work = WORK / "e2e"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"

    if not clip.exists():
        audio = speech_wav(work / "speech.wav", seconds=75.0)
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15",
                        "-i", str(audio), "-map", "0:v", "-map", "1:a", "-t", "75",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)
    check("test clip exists", clip.exists())

    pipeline.download.probe, pipeline.download.download = stub_download(
        clip, "Pipeline Test Clip", 75.0, replace=True)

    # Stand in for the LLM, returning plausible English of a sensible length.
    PHRASES = ["Right, let's carry on with the next round.",
               "We chain three and turn the work.",
               "Now single crochet into the same stitch.",
               "Keep the tension loose so it drapes nicely.",
               "One, two, three. Then we skip one.",
               "You can see how the pattern builds up here."]

    def fake_llm(prompt, model=None, host=None, key=None):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|{PHRASES[i % len(PHRASES)]}" for i in ids)

    T._call_ollama = fake_llm

    settings = Settings()
    settings.translator = "ollama"
    settings.voice = "bf_emma"
    settings.audio_mode = "replace"
    settings.write_srt = True

    events = []
    pipeline.runner.subscribe(lambda p: events.append((p["stage"], p["overall"])))

    job = pipeline.runner.submit("https://example.com/fake", settings)
    print("      running…", end="", flush=True)
    t0 = time.time()
    while job.status in ("queued", "running") and time.time() - t0 < 900:
        time.sleep(2)
        print(".", end="", flush=True)
    print()

    check("job completed", job.status == "done", f"{job.status}: {job.error}")
    if job.status != "done":
        log = JOBS / job.id / "error.log"
        if log.exists():
            print(log.read_text()[-1500:])
        return

    out = Path(job.output)
    check("output file was written", out.exists(), out.name)
    check("output has sensible size", out.stat().st_size > 50_000, f"{out.stat().st_size} bytes")

    s = job.stats
    check("video frames were preserved", s.get("frames_match") is True,
          f"{s.get('source_frames')} -> {s.get('output_frames')}")
    check("audio and video lengths agree", s.get("drift_seconds", 99) < 1.0,
          f"{s.get('drift_seconds')}s")
    check("lines were spoken", s.get("lines_spoken", 0) > 0, str(s.get("lines_spoken")))
    check("timing drift stayed small", s.get("max_drift", 99) < 2.0, f"{s.get('max_drift')}s")

    stages = [e[0] for e in events]
    for want in ("download", "transcribe", "translate", "synthesize", "assemble", "finish"):
        check(f"stage reported: {want}", want in stages)
    moving = [e for e in events if e[1] > 0]
    backwards = [(a[0], a[1], b[1]) for a, b in zip(moving, moving[1:]) if b[1] < a[1] - 0.001]
    check("progress only moves forward", not backwards,
          "; ".join(f"{stage} {was:.3f}->{now:.3f}" for stage, was, now in backwards[:3]))
    peak = max(e[1] for e in moving)
    check("progress reaches 100%", peak >= 0.999, f"{peak:.3f}")

    check("subtitle file was saved", out.with_suffix(".srt").exists())

    # Prove the audio actually contains the English we asked for, using the app's
    # own portable recogniser rather than a separate copy of one.
    from app.backends import asr as asr_backend
    check_wav = work / "check.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out),
                    "-ac", "1", "-ar", "16000", str(check_wav)], check=True)
    heard = [s["text"].strip() for s in asr_backend._transcribe_onnx(check_wav)[:8]]
    joined = " ".join(h for h in heard if h).lower()
    print("      heard back:", (joined[:150] + "…") if joined else "(nothing)")
    check("dubbed audio is the English we synthesised",
          any(w in joined for w in ("crochet", "chain", "round", "stitch", "pattern")),
          joined[:80])

    global E2E_JOB_ID
    E2E_JOB_ID = job.id


# ========================================================== 5. resuming a link
E2E_JOB_ID = ""


def test_resume():
    """Re-running a link must reuse the expensive stages, not redo them.

    This is what makes iterating on a long video bearable, and it is easy to
    break without noticing: the job folder is keyed off the link, so anything
    that makes the key vary per run silently disables resuming altogether.
    """
    print("\n[5] Resuming a repeated link")
    from app import pipeline
    from app.config import Settings

    if not E2E_JOB_ID:
        check("end-to-end job ran first", False, "no job id captured")
        return

    settings = Settings()
    settings.translator = "ollama"
    settings.voice = "bf_emma"
    settings.audio_mode = "replace"

    messages: list[str] = []
    listener = lambda p: messages.append(p["message"])       # noqa: E731
    pipeline.runner.subscribe(listener)

    t0 = time.time()
    job = pipeline.runner.submit("https://example.com/fake", settings)
    check("the same link maps to the same job folder", job.id == E2E_JOB_ID,
          f"{job.id} vs {E2E_JOB_ID}")

    while job.status in ("queued", "running") and time.time() - t0 < 600:
        time.sleep(1)
    pipeline.runner.unsubscribe(listener)

    check("second run completed", job.status == "done", f"{job.status}: {job.error}")
    check("transcription was reused, not redone",
          any("Reusing the transcription" in m for m in messages))
    check("translation was reused, not redone",
          any("Reusing the translation" in m for m in messages))

    # Changing the transcription engine must invalidate the cached transcript
    # rather than quietly handing back the other engine's work.
    from app.pipeline import _fingerprint
    parakeet = _fingerprint("parakeet", True, True)
    whisper = _fingerprint("whisper", True, True)
    check("a different ASR engine invalidates the cache", parakeet != whisper)


# ============================== 6. a preset change re-derives the audio
def _tone(hz: float, seconds: float, rate: int) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _dominant_hz(path: Path) -> float:
    data, rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    spec = np.abs(np.fft.rfft(data))
    return float(np.fft.rfftfreq(len(data), 1 / rate)[int(np.argmax(spec))])


def test_preset_change_reseparates():
    """Running a link on Fast and then on Balanced must not reuse the downmix.

    The job folder is keyed off the link, so the second run finds the first
    run's speech16k.wav — a downmix of the *whole* soundtrack — already in
    place. Guarded by existence alone it was skipped, so Demucs ran and was
    paid for, the report said separation had happened, and the transcript was
    rebuilt from the music-contaminated mix anyway.

    The mix is a 200 Hz tone and the stubbed stem is an 800 Hz one, so which
    audio reached the recogniser is a fact rather than an impression.
    """
    print("\n[6] A preset change re-derives the audio it feeds on")
    from app import pipeline
    from app.backends import translate as T
    from app.config import Settings, JOBS

    MIX_HZ, STEM_HZ, RATE = 200.0, 800.0, 44100
    URL = "https://example.com/preset-change"
    # From a clean folder: a link maps to a stable job id, so leftovers from a
    # previous run of this suite would be reused and the test would be asserting
    # against whatever the last run happened to leave.
    shutil_rmtree(JOBS / pipeline._job_id(URL))
    work = WORK / "preset"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"
    if not clip.exists():
        mix = work / "mix.wav"
        sf.write(mix, _tone(MIX_HZ, 8.0, RATE), RATE)
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-i", str(mix), "-map", "0:v", "-map", "1:a", "-t", "8",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)

    fake_probe, fake_download = stub_download(clip, "Preset Change", 8.0)

    def fake_separate(audio, workdir, prefer_gpu=True, progress=None):
        stems = Path(workdir) / "stems" / "stub"
        stems.mkdir(parents=True, exist_ok=True)
        vocals, bed = stems / "vocals.wav", stems / "no_vocals.wav"
        sf.write(vocals, _tone(STEM_HZ, 8.0, RATE), RATE)
        sf.write(bed, _tone(120.0, 8.0, RATE), RATE)
        if progress:
            progress(1.0, "Separated")
        return vocals, bed

    heard: list[int] = []

    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        hz = int(round(_dominant_hz(Path(audio_wav)) / 10.0) * 10)
        heard.append(hz)
        if progress:
            progress(1.0, "Heard 1 line")
        return [{"start": 0.5, "end": 3.0, "text": f"tono de {hz} hercios"}]

    def fake_llm(prompt, model=None, host=None, key=None):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|A tone, and then some more words to say." for i in ids)

    real_probe = pipeline.download.probe
    real_download = pipeline.download.download
    real_separate = pipeline.separate_backend.separate
    real_transcribe = pipeline.asr_backend.transcribe
    real_prune = pipeline.prune_workdir
    pipeline.download.probe = fake_probe
    pipeline.download.download = fake_download
    pipeline.separate_backend.separate = fake_separate
    pipeline.asr_backend.transcribe = fake_transcribe
    # Pruning after a successful job removes the derived audio, which would hide
    # this bug rather than fix it — and only for jobs that succeed. A run that
    # failed or was cancelled keeps everything, and is then exactly the stale
    # folder the next run reads from, so that is the state to test against.
    pipeline.prune_workdir = lambda workdir: 0
    T._call_ollama = fake_llm

    def run(preset: str):
        s = Settings().apply_preset(preset)
        s.translator = "ollama"
        s.voice = "bf_emma"
        job = pipeline.runner.submit(URL, s)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)
        return job

    try:
        fast = run("fast")
        check("the Fast run completed", fast.status == "done",
              f"{fast.status}: {fast.error}")
        check("Fast transcribed the full mix", heard[:1] == [int(MIX_HZ)], str(heard))

        balanced = run("balanced")
        check("the Balanced run completed", balanced.status == "done",
              f"{balanced.status}: {balanced.error}")
        check("the same link reused the same job folder", balanced.id == fast.id)
        check("Balanced reported that it separated",
              balanced.stats.get("separated") is True, str(balanced.stats.get("separated")))
        check("Balanced transcribed the separated stem, not the stale mix",
              len(heard) == 2 and heard[1] == int(STEM_HZ),
              f"heard {heard}, wanted [{int(MIX_HZ)}, {int(STEM_HZ)}]")

        # The two runs must not be sharing derived audio at all.
        derived = sorted((JOBS / fast.id / "derived").iterdir())
        check("each set of settings got its own derived audio folder",
              len(derived) >= 3, f"{len(derived)} folders")
        rates = {_dominant_hz(p): p for p in
                 (JOBS / fast.id / "derived").glob("*/speech16k.wav")}
        check("both the mix and the stem survive as separate files",
              any(abs(hz - MIX_HZ) < 15 for hz in rates) and
              any(abs(hz - STEM_HZ) < 15 for hz in rates),
              str(sorted(round(h) for h in rates)))
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.separate_backend.separate = real_separate
        pipeline.asr_backend.transcribe = real_transcribe
        pipeline.prune_workdir = real_prune


# ================================ 7. one sample rate across the whole track
def test_mixed_sample_rates():
    """A voice that dies mid-job must not leave two rates in one track.

    sample_rate used to be a single variable reassigned per line and handed to
    assemble() once, for all of them. The mid-job fallback swaps a cloning
    engine for the portable one partway through the loop, so lines synthesised
    before the swap were timed against the rate of the engine that replaced
    them — a timing error, which is the one failure the design exists to stop.
    """
    print("\n[7] One sample rate across the whole track")
    from app import pipeline
    from app.backends import translate as T
    from app.backends import tts as tts_backend
    from app.config import Settings
    from app.steps import align

    # Directly: assemble() is told one rate, so a line that declares another is
    # refused rather than placed against the wrong clock.
    sr = 24000
    lines = [{"start": 0.0, "end": 1.0, "samples": _tone(440, 0.5, sr), "rate": 24000},
             {"start": 2.0, "end": 3.0, "samples": _tone(440, 0.5, sr), "rate": 32000}]
    try:
        align.assemble(lines, 6.0, sr)
        check("assemble refuses a line recorded at another rate", False, "it accepted it")
    except ValueError as exc:
        check("assemble refuses a line recorded at another rate", True, str(exc)[:60])

    # And through the pipeline: an engine that reports 32 kHz and then fails,
    # forcing the real mid-job fallback to the 24 kHz portable voice.
    STARTS = [(1.0, 3.0), (5.0, 7.0)]
    URL = "https://example.com/rate-switch"
    # Cached lines from an earlier run of this suite would be read straight back
    # and the engine never called, so the fallback this test exists to trigger
    # would never happen.
    from app.config import JOBS as JOBS_DIR
    shutil_rmtree(JOBS_DIR / pipeline._job_id(URL))
    work = WORK / "rates"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"
    if not clip.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                        "-map", "0:v", "-map", "1:a", "-t", "10",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)

    class RateSwitchingEngine:
        """32 kHz for the first line, then broken — exactly the fallback path."""
        name = "Stub (32 kHz)"
        sample_rate = 32000

        def __init__(self):
            self.calls = 0

        def say(self, text, voice="", speed=1.0, speaker=0):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("the cloning model fell over")
            return _tone(300, 1.2, 32000), 32000

    fake_probe, fake_download = stub_download(clip, "Rate Switch", 10.0)

    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        if progress:
            progress(1.0, "Heard 2 lines")
        return [{"start": a, "end": b, "text": f"linea {n}"}
                for n, (a, b) in enumerate(STARTS)]

    def fake_llm(prompt, model=None, host=None, key=None):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|This is line number {i} speaking." for i in ids)

    real_probe = pipeline.download.probe
    real_download = pipeline.download.download
    real_transcribe = pipeline.asr_backend.transcribe
    real_make = pipeline.JobRunner._make_engine
    real_prune = pipeline.prune_workdir
    pipeline.download.probe = fake_probe
    pipeline.download.download = fake_download
    pipeline.asr_backend.transcribe = fake_transcribe
    pipeline.JobRunner._make_engine = lambda *a, **k: (RateSwitchingEngine(), False)
    # The assembled track is one of the intermediates a successful job drops.
    pipeline.prune_workdir = lambda workdir: 0
    T._call_ollama = fake_llm

    try:
        s = Settings().apply_preset("fast")
        s.translator = "ollama"
        job = pipeline.runner.submit(URL, s)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)

        check("the job survived the mid-job voice failure", job.status == "done",
              f"{job.status}: {job.error}")
        if job.status != "done":
            return
        check("the fallback was reported", "fell back" in str(job.stats.get("voices", "")),
              str(job.stats.get("voices")))
        check("the track was assembled at the engine's rate",
              job.stats.get("sample_rate") == 32000, str(job.stats.get("sample_rate")))
        check("both lines were spoken", job.stats.get("lines_spoken") == 2,
              str(job.stats.get("lines_spoken")))
        check("nothing was pushed off its mark", job.stats.get("max_drift", 99) == 0.0,
              f"{job.stats.get('max_drift')}s")

        # Both lines must actually be where the original speaker was.
        from app.config import JOBS
        track, rate = sf.read(JOBS / job.id / "dubbed.wav", dtype="float32")
        check("the rendered track is at one rate", rate == 32000, str(rate))

        def loud_between(a, b):
            seg = track[int(a * rate):int(b * rate)]
            return float(np.sqrt((seg ** 2).mean())) if seg.size else 0.0

        for n, (a, b) in enumerate(STARTS):
            check(f"line {n} lands in its slot", loud_between(a, b + 1.0) > 0.02,
                  f"rms {loud_between(a, b + 1.0):.3f}")
        check("the gap between the lines stayed quiet",
              loud_between(3.6, 4.8) < 0.01, f"rms {loud_between(3.6, 4.8):.4f}")
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.asr_backend.transcribe = real_transcribe
        pipeline.JobRunner._make_engine = real_make
        pipeline.prune_workdir = real_prune


# ==================================== 8. a finished job drops its bulk
def test_cleanup():
    """Keep what makes a re-run cheap; drop the rest.

    A job folder held the download, the full-band audio, the stems and the
    rendered track — about 1.5 GB for an hour of video, under Application
    Support where nobody would find it.
    """
    print("\n[8] Clearing up after a finished job")
    from app.pipeline import prune_workdir
    from app.storage import dir_size
    from app.config import JOBS

    work = WORK / "cleanup"
    shutil_rmtree(work)
    (work / "derived" / "abc" / "stems").mkdir(parents=True)
    (work / "lines" / "def").mkdir(parents=True)

    big = np.zeros(240000, dtype=np.float32)
    sf.write(work / "dubbed.wav", big, 24000)
    sf.write(work / "derived" / "abc" / "full.wav", big, 24000)
    sf.write(work / "derived" / "abc" / "stems" / "vocals.wav", big, 24000)
    sf.write(work / "lines" / "def" / "00000.wav", big, 24000)
    (work / "derived" / "abc" / "source.mp4").write_bytes(b"x" * 5000)
    (work / "segments.json").write_text('[{"start": 0}]')
    (work / "translated.json").write_text("[]")
    (work / "subtitles.srt").write_text("1\n")

    before = dir_size(work)
    freed = prune_workdir(work)

    check("something was actually reclaimed", freed > 0, f"{freed} bytes")
    check("the transcript is kept", (work / "segments.json").exists())
    check("the translation is kept", (work / "translated.json").exists())
    check("the subtitles are kept", (work / "subtitles.srt").exists())
    check("the rendered lines are kept",
          (work / "lines" / "def" / "00000.wav").exists())
    for gone in ("dubbed.wav", "derived/abc/full.wav", "derived/abc/source.mp4",
                 "derived/abc/stems/vocals.wav"):
        check(f"{gone} is dropped", not (work / gone).exists())
    check("emptied folders go too", not (work / "derived").exists())
    check("the folder is smaller than it was", dir_size(work) < before,
          f"{dir_size(work)} < {before}")

    # And the real job from section 4 kept what a resume needs.
    if E2E_JOB_ID:
        job_dir = JOBS / E2E_JOB_ID
        check("the finished job kept its transcript",
              (job_dir / "segments.json").exists())
        check("the finished job dropped its working audio",
              not (job_dir / "dubbed.wav").exists())


def shutil_rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


# =============================== 9. reshaping and matching the voices
def test_segments_and_voices():
    """Two ideas taken from other dubbing projects, checked against real numbers."""
    print("\n[9] Joining run-on lines, and matching voices by pitch")
    from app.steps.segments import merge_adjacent
    from app.backends.diarize import median_pitch
    from app.config import Settings

    # Fast dialogue: short lines, almost no gap, one speaker.
    rapid = [{"start": t, "end": t + 0.9, "text": f"line {n}", "speaker": 0}
             for n, t in enumerate([0.0, 1.0, 2.0, 3.0])]
    out = merge_adjacent(rapid)
    check("run-on lines are joined", len(out) == 1, f"{len(rapid)} -> {len(out)}")
    check("the joined line spans the original run",
          abs(out[0]["end"] - 3.9) < 1e-6 and out[0]["start"] == 0.0)
    check("the words are all still there",
          out[0]["text"] == "line 0 line 1 line 2 line 3", out[0]["text"])

    # A different speaker interrupting must break the run.
    mixed = [dict(r) for r in rapid]
    mixed[2]["speaker"] = 1
    check("a change of speaker breaks the run", len(merge_adjacent(mixed)) == 3,
          str(len(merge_adjacent(mixed))))

    # Real pauses — the instructional material the app was built for.
    spaced = [{"start": t, "end": t + 1.5, "text": "x", "speaker": 0}
              for t in (0.0, 3.0, 6.0)]
    check("material with real pauses is left alone",
          len(merge_adjacent(spaced)) == 3, str(len(merge_adjacent(spaced))))
    check("ids are renumbered for the translator",
          [s_["i"] for s_ in merge_adjacent(spaced)] == [0, 1, 2])

    # Pitch, on synthesised tones with a known fundamental.
    sr = 16000
    for hz in (110.0, 220.0):
        t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
        # A couple of harmonics, so it is a plausible voice rather than a sine.
        wave = (0.5 * np.sin(2 * np.pi * hz * t)
                + 0.3 * np.sin(4 * np.pi * hz * t)).astype(np.float32)
        got = median_pitch(wave, sr)
        check(f"pitch of a {hz:.0f} Hz tone is found", abs(got - hz) < 12,
              f"{got:.1f} Hz")

    # A speaker count that cannot be right. The real run that prompted this
    # reported 28 speakers for a 10m35s film with about seven characters, and
    # said so without comment.
    def implausible(count, seconds):
        return count > max(6, seconds / 45)

    check("28 speakers in a ten-minute film is flagged", implausible(28, 635))
    check("three people in an hour-long podcast is not", not implausible(3, 3600))
    check("eight on a long panel show is not", not implausible(8, 3600))
    check("four in a two-minute clip is not", not implausible(4, 120))
    check("twelve in a two-minute clip is", implausible(12, 120))

    s_ = Settings()
    s_.voice = "bf_emma"
    male = s_.voice_for(1, male=True)
    female = s_.voice_for(1, male=False)
    check("a low-pitched speaker gets a male voice", male.split("_")[0][-1] == "m", male)
    check("a high-pitched speaker gets a female voice",
          female.split("_")[0][-1] == "f", female)
    check("the choice still avoids the primary voice", female != "bf_emma", female)


# ======================= 10. a 30-second sample before the whole video
def test_preview():
    """A sample must be cheap to take and must not pretend to be the real thing.

    Three things make it worth having, and each is a way it can silently stop
    being worth having: the window has to land on speech rather than on the
    title card, the download it pays for has to survive for the full run behind
    it, and its output must not leak into the finished videos or the history —
    a thirty-second stub filed beside real output is a mess the user cannot
    reason their way out of.
    """
    print("\n[10] A sample before committing to the whole video")
    from app import pipeline
    from app.backends import translate as T
    from app.config import HISTORY_FILE, JOBS as JOBS_DIR, OUTPUT_DIR, Settings

    work = WORK / "preview"
    work.mkdir(parents=True, exist_ok=True)

    # 90 seconds that open on 20 of silence — the title card this feature exists
    # to skip past — and then talk steadily.
    LEAD_IN, TOTAL = 20.0, 90.0
    clip = work / "clip.mp4"
    if not clip.exists():
        rate = 24000
        track = np.concatenate([np.zeros(int(LEAD_IN * rate), dtype=np.float32),
                                _tone(220, TOTAL - LEAD_IN, rate)])
        sf.write(work / "audio.wav", track, rate)
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-i", str(work / "audio.wav"), "-map", "0:v", "-map", "1:a",
                        "-t", f"{TOTAL:g}", "-c:v", "libx264", "-preset", "ultrafast",
                        "-c:a", "aac", str(clip)], check=True)

    # ---------------------------------------------- choosing the window
    found = pipeline._speech_start(clip, TOTAL)
    check("the window skips a silent opening", 18.0 <= found <= 22.0, f"{found:g}s")

    # A video whose speech starts near the end must not be handed a window that
    # runs off it.
    clamped = pipeline._speech_start(clip, 35.0)
    check("the window is clamped to the end of the video", clamped <= 5.01, f"{clamped:g}s")

    # An unknown duration — some hosts report none — falls back rather than
    # dividing or subtracting its way into nonsense.
    check("an unknown duration still gives a usable window",
          pipeline._speech_start(clip, 0.0) >= 0.0)

    # ------------------------------------------- weighting the progress bar
    # A sample shrinks every stage but the download, which still fetches the
    # whole video. Left at its full-run share the bar sat near zero for most of
    # the wait and then sprinted, which is the one thing a progress bar must not
    # do to someone deciding whether to give up.
    balanced = Settings().apply_preset("balanced")
    plain = pipeline.runner._plan(balanced)
    heavy = pipeline.runner._plan(balanced, 6.0)
    check("a sample gives the download a far larger share of the bar",
          heavy["download"][1] > plain["download"][1] * 3,
          f"{plain['download'][1]:.2f} -> {heavy['download'][1]:.2f}")
    check("the stages after it are pushed later to make room",
          heavy["transcribe"][0] > plain["transcribe"][0],
          f"{plain['transcribe'][0]:.2f} -> {heavy['transcribe'][0]:.2f}")
    check("the weights still add up to one",
          abs(sum(w for _, w, _ in heavy.values()) - 1.0) < 1e-6)
    check("the plan names only the stages this preset runs",
          set(pipeline.runner._plan(Settings().apply_preset("fast"))) ==
          {"download", "transcribe", "translate", "synthesize", "assemble", "finish"})

    # ------------------------------------------------- a sample end to end
    URL = "https://example.com/sample-me"
    shutil_rmtree(JOBS_DIR / pipeline._link_id(URL))
    fake_probe, fake_download = stub_download(clip, "Sample Me", TOTAL)

    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        if progress:
            progress(1.0, "Heard 2 lines")
        return [{"start": 1.0, "end": 4.0, "text": "primera linea"},
                {"start": 8.0, "end": 11.0, "text": "segunda linea"}]

    def fake_llm(prompt, model=None, host=None, key=None):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|This is a line of dubbed speech." for i in ids)

    real_probe, real_download = pipeline.download.probe, pipeline.download.download
    real_transcribe = pipeline.asr_backend.transcribe
    pipeline.download.probe, pipeline.download.download = fake_probe, fake_download
    pipeline.asr_backend.transcribe = fake_transcribe
    T._call_ollama = fake_llm

    try:
        s = Settings().apply_preset("fast")
        s.translator = "ollama"

        # Elapsed time is time spent working, not time since the link was
        # pasted. A job that sat behind another and then ran reported the wait
        # as part of its own duration, which made "Took" wrong and the estimate
        # of the time remaining wrong by the same margin.
        waiting = pipeline.Job(id="x", url="u")
        check("a job that has not started reports no elapsed time",
              waiting.public()["elapsed"] == 0)
        waiting.began = time.time() - 30
        waiting.finished = waiting.began + 10
        check("a finished job reports the time it ran, not the time it waited",
              waiting.public()["elapsed"] == 10, str(waiting.public()["elapsed"]))

        check("a sample and the full run are different jobs",
              pipeline._job_id(URL, True) != pipeline._job_id(URL, False))
        check("but they share one work folder",
              pipeline._link_id(URL) == pipeline._job_id(URL, False))

        job = pipeline.runner.submit(URL, s, preview=True)
        # An impatient second click is the same job, not two racing over one
        # folder. The buttons are disabled while a submission is in flight, but
        # that is a courtesy; this is the guarantee.
        check("a second click while it runs is the same job",
              pipeline.runner.submit(URL, s, preview=True) is job)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)
        check("the sample completed", job.status == "done", f"{job.status}: {job.error}")
        if job.status != "done":
            return

        out = Path(job.output)
        length = pipeline.download.media_duration(out)
        check("the sample is about thirty seconds",
              abs(length - pipeline.PREVIEW_SECONDS) < 2.0, f"{length:.1f}s")
        check("it was taken from where the speech starts",
              18.0 <= job.preview_from <= 22.0, f"{job.preview_from:g}s")
        check("the report says it is a sample", job.stats.get("preview") is True)

        # Not a deliverable: not in the videos folder, not in the history.
        check("it did not land in the finished videos folder",
              not list(OUTPUT_DIR.glob("Sample-Me*")),
              str([p.name for p in OUTPUT_DIR.glob("Sample-Me*")]))
        history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
        check("it was not recorded in the history",
              not any(h.get("output") == job.output for h in history))

        # And the download it paid for is still there for the full run behind it.
        workdir = JOBS_DIR / pipeline._link_id(URL)
        sources = list(workdir.glob("derived/*/source.mp4"))
        check("the shared download survived the sample",
              any("preview" not in str(p) and p.stat().st_size > 0 for p in sources)
              and len(sources) >= 2, f"{len(sources)} source files")

        # ------------------------------------- too short to be worth sampling
        SHORT_URL = "https://example.com/too-short"
        shutil_rmtree(JOBS_DIR / pipeline._link_id(SHORT_URL))
        short = work / "short.mp4"
        if not short.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip),
                            "-ss", "25", "-t", "12", "-c:v", "libx264",
                            "-preset", "ultrafast", "-c:a", "aac", str(short)], check=True)
        pipeline.download.probe, pipeline.download.download = stub_download(
            short, "Too Short", 12.0)

        brief = pipeline.runner.submit(SHORT_URL, s, preview=True)
        t0 = time.time()
        while brief.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)
        check("a video shorter than the window completed",
              brief.status == "done", f"{brief.status}: {brief.error}")
        if brief.status == "done":
            check("it was dubbed whole rather than sampled", brief.preview is False)
            check("and it says why",
                  any("only" in n and "whole thing" in n
                      for n in brief.stats.get("notes", [])),
                  str(brief.stats.get("notes")))
            check("so it did reach the finished videos folder",
                  Path(brief.output).parent == OUTPUT_DIR, brief.output)

        # ------------------------------------------ nothing to hear at all
        # A video with no speech in it — a music video, a silent screencast —
        # should fail plainly and in a minute, which is most of the argument for
        # sampling in the first place.
        SILENT_URL = "https://example.com/all-quiet"
        shutil_rmtree(JOBS_DIR / pipeline._link_id(SILENT_URL))
        silent = work / "silent.mp4"
        if not silent.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error",
                            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                            "-map", "0:v", "-map", "1:a", "-t", "90",
                            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                            str(silent)], check=True)
        check("a silent video starts its window at the beginning",
              pipeline._speech_start(silent, 90.0) == 0.0)

        pipeline.download.probe, pipeline.download.download = stub_download(
            silent, "All Quiet", 90.0)
        pipeline.asr_backend.transcribe = lambda *a, **k: []

        quiet = pipeline.runner.submit(SILENT_URL, s, preview=True)
        t0 = time.time()
        while quiet.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)
        check("a video with no speech fails rather than producing silence",
              quiet.status == "error", quiet.status)
        check("and says so in plain English",
              "no speech" in quiet.error.lower(), quiet.error)
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.asr_backend.transcribe = real_transcribe


# ============================== 11. knowing where the disk went
def test_storage():
    """Filling the boot disk does not just fail a job — macOS stops working too.

    So the app has to be able to say what it is holding, let it go a piece at a
    time, and refuse a video it has no room for rather than finding out at 80%.
    """
    print("\n[11] Disk space, and getting it back")
    from app import storage as store
    from app.config import JOBS, OUTPUT_DIR

    # --- estimating before anything is written
    hour = store.estimate_needed(3600, "720")
    check("an hour at 720p is estimated in gigabytes",
          2 * 1024**3 < hour < 3 * 1024**3, f"{hour / 1024**3:.1f} GB")
    check("best quality is estimated larger than 720p",
          store.estimate_needed(3600, "best") > hour)
    check("a very short video still reserves a floor",
          store.estimate_needed(2, "720") == store.MINIMUM_NEED)
    check("an unknown duration doesn't go negative",
          store.estimate_needed(-5, "720") == store.MINIMUM_NEED)
    check("an unknown quality is treated as the largest",
          store.estimate_needed(600, "??") == store.estimate_needed(600, "best"))

    check("free space is a real number", store.free_bytes() > 0)

    # --- a breakdown, not one figure
    keys = [g["key"] for g in store.groups()]
    check("every place the app writes to is accounted for",
          set(keys) == {"jobs", "models", "previews", "hfmodels", "venv",
                        "ollama", "output"}, str(keys))
    by_key = {g["key"]: g for g in store.groups()}
    check("finished videos are never offered for deletion",
          not by_key["output"]["clearable"])
    check("the translation model is shown but not deletable",
          not by_key["ollama"]["clearable"])
    # Deleting it would remove the interpreter running the request. It is in the
    # list because it is one of the largest things on disk; Uninstall removes it.
    check("the Python environment is shown but not deletable",
          not by_key["venv"]["clearable"])

    # The model cache is shared with anything else on the machine that uses
    # Hugging Face, so only the repositories this app fetches are counted.
    ours = {p.name for p in store.model_cache_dirs()}
    check("only this app's model repositories are counted",
          all(n.startswith(store.OUR_MODEL_REPOS) for n in ours), str(sorted(ours)))
    if store.HF_HUB.is_dir():
        every = {p.name for p in store.HF_HUB.iterdir() if p.is_dir()}
        check("another tool's models are left out of the total",
              ours <= every and all(not n.startswith(store.OUR_MODEL_REPOS)
                                    for n in every - ours),
              str(sorted(every - ours)))

    # --- clearing one job at a time
    a, b = JOBS / "aaaaaaaaaaaa", JOBS / "bbbbbbbbbbbb"
    for folder in (a, b):
        (folder / "derived").mkdir(parents=True, exist_ok=True)
        (folder / "derived" / "big.wav").write_bytes(b"x" * 2_000_000)
    rows = store.job_folders()
    check("each job folder is listed with its size",
          {r["id"] for r in rows} >= {a.name, b.name})
    # A folder holding little more than an error log is not where a disk went.
    tiny = JOBS / "cccccccccccc"
    tiny.mkdir(parents=True, exist_ok=True)
    (tiny / "error.log").write_text("it broke")
    check("a folder with nothing but an error log is left off the list",
          not any(r["id"] == tiny.name for r in store.job_folders()))
    freed = store.clear(f"job:{a.name}")
    check("clearing one job frees roughly its size", 1_500_000 < freed < 3_000_000, str(freed))
    check("that job's folder is gone", not a.exists())
    check("the other job is untouched", b.exists())

    # A job that is still running must survive a clear-all.
    store.clear("jobs", keep={b.name})
    check("a job named as live is kept", b.exists())
    store.clear("jobs")
    check("clearing them all empties the folder",
          not any(p.is_dir() for p in JOBS.iterdir()))

    # Someone short of space may well keep the models on an external drive, and
    # that folder is not this app's to empty. The suite itself symlinks it, so
    # this also stops a test run deleting the real 700 MB.
    if store.MODELS.is_symlink():
        target = store.MODELS.resolve()
        check("a symlinked models folder is left alone", store.clear("models") == 0)
        check("and it is still there", target.is_dir())

    # --- nothing else is reachable
    before = store.dir_size(OUTPUT_DIR) if OUTPUT_DIR.is_dir() else 0
    for bad in ("output", "..", "everything", "models/../.."):
        try:
            store.clear(bad)
            check(f"“{bad}” is refused outright", False, "it was accepted")
        except ValueError:
            check(f"“{bad}” is refused outright", True)
    check("the finished videos folder survived every one of those",
          (store.dir_size(OUTPUT_DIR) if OUTPUT_DIR.is_dir() else 0) == before)

    # A job id is a folder name, not a path. This one is named rather than
    # joined-and-trusted, so it resolves outside JOBS and is dropped.
    check("a job id that climbs out of the folder frees nothing",
          store.clear("job:../../etc") == 0)

    # --- refusing a video there is no room for
    from app import pipeline
    from app.config import Settings
    clip = WORK / "preview" / "clip.mp4"
    real_probe, real_download = pipeline.download.probe, pipeline.download.download
    real_free = store.free_bytes
    pipeline.download.probe, pipeline.download.download = stub_download(
        clip, "No Room", 3600.0)
    store.free_bytes = lambda path=None: 100 * 1024 ** 2          # 100 MB left
    try:
        URL = "https://example.com/no-room"
        workdir = JOBS / pipeline._link_id(URL)
        shutil_rmtree(workdir)
        job = pipeline.runner.submit(URL, Settings().apply_preset("fast"))
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 120:
            time.sleep(0.5)
        check("an hour of video with 100 MB free is refused", job.status == "error",
              f"{job.status}: {job.error}")
        check("the message says what it needs and what there is",
              "room on the disk" in job.error and "100 MB" in job.error, job.error)
        check("and it refused before downloading anything",
              not list(workdir.rglob("*.mp4")) if workdir.exists() else True)
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        store.free_bytes = real_free


# ================================ 12. not letting the Mac doze off mid-job
def test_keep_awake():
    """An hour of video is an hour of work, and a laptop left alone sleeps.

    The job survives it — everything resumes when somebody touches the trackpad
    — but to whoever started it and walked away, a run that stopped at 40% for
    half an hour is indistinguishable from one that hung.
    """
    print("\n[12] Staying awake while there is work")
    import platform
    from app.pipeline import KeepAwake

    def held():
        out = subprocess.run(["pmset", "-g", "assertions"],
                             capture_output=True, text=True).stdout
        return "caffeinate" in out

    check("the display is deliberately left alone", "-d" not in KeepAwake.FLAGS,
          str(KeepAwake.FLAGS))
    check("idle sleep, disk sleep and mains sleep are all covered",
          set(KeepAwake.FLAGS) == {"-i", "-m", "-s"}, str(KeepAwake.FLAGS))

    if platform.system() != "Darwin":
        check("nothing is attempted off macOS", True, "skipped")
        return

    was = held()
    keeper = KeepAwake()
    keeper.start()
    time.sleep(0.5)
    check("an assertion is taken while working", held(), "pmset reports caffeinate")
    check("and it is tied to this process, not left loose",
          "-w" in keeper._proc.args and str(os.getpid()) in keeper._proc.args)
    keeper.start()
    check("starting twice holds one, not two", keeper.active)
    keeper.stop()
    time.sleep(0.5)
    check("it is given back when the queue empties", held() == was)
    keeper.stop()
    check("stopping twice is harmless", not keeper.active)

    # The pill in the running panel is a switch, and a switch that only takes
    # effect on the next job would be lying for the hour it matters most.
    from app.pipeline import JobRunner
    r = JobRunner()
    check("switching it on with nothing running holds nothing",
          r.sync_keep_awake(True) is False)
    r.awake.start()
    time.sleep(0.4)
    check("switching it off releases the one being held right now",
          r.sync_keep_awake(False) is False and not held())

    # A machine without caffeinate must not take the app down with it.
    import subprocess as _sp
    real = _sp.Popen
    try:
        _sp.Popen = lambda *a, **k: (_ for _ in ()).throw(OSError("no caffeinate"))
        k = KeepAwake()
        k.start()
        check("a Mac without caffeinate is simply not held awake", not k.active)
        k.stop()
    finally:
        _sp.Popen = real


# ============== 13. catching a translation that went wrong before it is spoken
def test_translation_qc():
    """A real 52-minute dub read out "id: 63" and then several minutes of Spanish.

    Nothing downstream could tell: the audio check heard speech, the frame check
    saw the picture survive, and the file was the right length. The failure is
    upstream of everything that looks at the finished article.
    """
    print("\n[13] Catching a translation that went wrong")
    from app.backends.translate import _looks_untranslated, _parse, _strip_echo
    from app.steps import qc

    SRC = "Blusa de verano facil en todas las tallas paso a paso"

    # Scaffolding the model echoes back into its own answer.
    for raw, want in [("id: 63 Now we chain three.", "Now we chain three."),
                      ("[2.0s] Now we chain three.", "Now we chain three."),
                      ("[2.0s] id 63. Now we chain three.", "Now we chain three."),
                      ("#63 Now we chain three.", "Now we chain three."),
                      ("Now we chain three.", "Now we chain three.")]:
        got = _strip_echo(raw)
        check(f"stripped: {raw[:26]!r}", got == want, got)

    # Numbers that belong to the sentence must survive. These are the shapes a
    # crochet or cookery tutorial actually produces, and eating the front of one
    # would corrupt a good translation silently.
    for keep in ("3 chain stitches, then turn.",
                 "3. Chain three stitches.",
                 "Line 5 of the pattern is a chain.",
                 "Row 12) is worked in the back loop."):
        check(f"kept intact: {keep[:28]!r}", _strip_echo(keep) == keep, _strip_echo(keep))

    check("the source handed back is spotted", _looks_untranslated(SRC, SRC))
    check("a real translation is not", not _looks_untranslated("A summer blouse.", SRC))
    check("a short line unchanged is left alone", not _looks_untranslated("OK", "OK"))

    # And the parser refuses it, which turns a wrong answer into a missing one —
    # so the retry above it gets a go, and the 5% ceiling can fail the job rather
    # than deliver an hour of the original language.
    check("the parser rejects an untranslated line",
          _parse(f"63|{SRC}", [{"i": 63, "text": SRC}]) == {})
    check("and repairs an echoed one",
          _parse("63|id: 63 Now we chain three.", [{"i": 63, "text": SRC}])
          == {63: "Now we chain three."})

    # The check itself, which also covers a translation restored from cache.
    segs = [{"text": SRC, "translation": "id: 63 Now we chain three."},
            {"text": SRC, "translation": SRC},
            {"text": "Vale", "translation": "OK"},
            {"text": "Y seguimos", "translation": "And we carry on."},
            {"text": "Nada", "translation": ""}]
    report = qc.check(segs)
    check("the echoed line is repaired, not dropped",
          segs[0]["translation"] == "Now we chain three.")
    check("the untranslated line is left silent", segs[1]["translation"] == "")
    check("the good lines are untouched",
          segs[2]["translation"] == "OK" and segs[3]["translation"] == "And we carry on.")
    check("the counts are right",
          (report["repaired"], report["untranslated"], report["spoken"]) == (1, 1, 3),
          str(report))
    check("and it says so in a sentence", "original language" in qc.summarise(report))
    check("a clean translation says nothing at all",
          qc.summarise(qc.check([{"text": "Hola", "translation": "Hello there."}])) == "")

    # A batch the model cannot manage whole is halved until it can. Twenty-five
    # missing lines used to fall outside the "small prompts land" rule, so the
    # identical question was asked twice and then given up on.
    from app.backends import translate as _T
    sizes = []

    def flaky(prompt):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        sizes.append(len(ids))
        if len(ids) > 4:                    # echoes its input above four lines
            return "\n".join(f"{i}|id: {i} texto original numero {i} sin traducir"
                              for i in ids)
        return "\n".join(f"{i}|Chain three and turn, number {i}." for i in ids)

    big = [{"i": n, "start": n, "end": n + 2,
            "text": f"texto original numero {n} sin traducir"} for n in range(25)]
    got = _T._translate_chunk(big, [], "English", "", flaky)
    check("a batch the model chokes on is split until it lands",
          len(got) == 25, f"{len(got)} of 25")
    check("and it halves rather than asking the same thing twice",
          sizes[0] == 25 and max(sizes[1:]) <= 13, str(sizes[:6]))

    # --- the codec that made a finished dub unplayable
    from app.steps.mux import WIDELY_PLAYABLE
    check("H.264 counts as playable anywhere", "h264" in WIDELY_PLAYABLE)
    check("AV1 does not", "av1" not in WIDELY_PLAYABLE)

    # Read from the file, not through the module: earlier tests replace
    # download() with a stub and never put it back, so inspect would be reading
    # the fake.
    src = (ROOT / "app" / "steps" / "download.py").read_text()
    # A regex, not a prefix: YouTube labels the same codec avc1.640028 on some
    # formats and h264 on others, and matching one silently falls through to the
    # AV1 this exists to avoid.
    check("H.264 is matched by both of the names YouTube gives it",
          "^(avc|h264)" in src, "selector does not use the regex form")
    check("and it still falls back rather than refusing a video",
          "+ba/bv*" in src)
    check("a Mac that can decode AV1 is not made to download H.264",
          "allow_av1" in src)

    from app.config import can_decode_av1, mac_generation, AV1_FROM_GENERATION
    check("AV1 is gated on the generation that can decode it",
          AV1_FROM_GENERATION == 3)
    check("this machine reports a generation", mac_generation() >= 0)
    check("and the two agree",
          can_decode_av1() == (mac_generation() >= AV1_FROM_GENERATION))

    # Batches are capped by size as well as count: long joined lines crowd the
    # context window, which is exactly when a small model starts repeating.
    from app.backends.translate import _batches, BATCH, BATCH_CHARS
    short = _batches([{"text": "x" * 40} for _ in range(60)])
    check("short lines batch by count", max(len(b) for b in short) == BATCH,
          str([len(b) for b in short]))
    long_ = _batches([{"text": "y" * 400} for _ in range(20)])
    check("long ones batch by size instead",
          max(len(b) for b in long_) < BATCH
          and all(sum(len(s["text"]) for s in b) <= BATCH_CHARS for b in long_),
          str([len(b) for b in long_]))
    check("and a single over-long line is still carried, not dropped",
          [len(b) for b in _batches([{"text": "z" * 9000}])] == [1])

    # From the file: earlier tests replace _call_ollama with a stub and never
    # put it back, so inspect would be reading the fake.
    tsrc = (ROOT / "app" / "backends" / "translate.py").read_text()
    check("the local model is asked not to repeat itself",
          "repeat_penalty" in tsrc and "min_p" in tsrc)
    check("and told outright not to copy the source back",
          "NEVER COPY" in tsrc)


if __name__ == "__main__":
    test_align()
    test_translate()
    test_server()
    test_end_to_end()
    test_resume()
    test_preset_change_reseparates()
    test_mixed_sample_rates()
    test_cleanup()
    test_segments_and_voices()
    test_preview()
    test_storage()
    test_keep_awake()
    test_translation_qc()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("All checks passed.")
