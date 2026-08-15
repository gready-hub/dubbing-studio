"""End-to-end and unit tests for Dubbing Studio.

Run it from anywhere:  python test_pipeline.py

The job data is redirected to a scratch folder so a test run never disturbs the
real jobs under Application Support. Set DUBBING_TEST_SOURCE to a video with
speech in it to exercise transcription against real recorded audio; without one
the suite synthesises its own speech and is fully self-contained.
"""
import json
import os
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
_real_models = Path.home() / "Library" / "Application Support" / "DubbingStudio" / "models"
_scratch_models = SCRATCH / "models"
if _real_models.is_dir() and not _scratch_models.exists():
    _scratch_models.parent.mkdir(parents=True, exist_ok=True)
    _scratch_models.symlink_to(_real_models)

import numpy as np
import soundfile as sf

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


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

    r = client.get("/api/job/nope/video")
    check("missing job video 404s", r.status_code == 404)


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

    # Stand in for the download step — everything after it is the real thing.
    def fake_download(url, workdir, quality="best", progress=None):
        import shutil
        workdir.mkdir(parents=True, exist_ok=True)
        dest = workdir / "source.mp4"
        shutil.copy(clip, dest)
        if progress:
            progress(1.0, "Downloaded")
        return dest, {"title": "Pipeline Test Clip", "duration": 75.0,
                      "uploader": "test", "thumbnail": ""}

    pipeline.download.download = fake_download

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


if __name__ == "__main__":
    test_align()
    test_translate()
    test_server()
    test_end_to_end()
    test_resume()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("All checks passed.")
