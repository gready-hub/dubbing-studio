"""Tests for the multi-speaker / music-preservation upgrade."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Set before app.config is imported — see the note in test_pipeline.py.
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


# ===================================================== 1. preset stage planning
def test_plan():
    print("\n[1] Preset stage planning")
    from app.config import PRESETS, Settings
    from app.pipeline import JobRunner

    r = JobRunner()
    for name in PRESETS:
        s = Settings().apply_preset(name)
        plan = r._plan(s)
        total = sum(w for _, w, _ in plan.values())
        check(f"{name}: weights sum to 1", abs(total - 1.0) < 1e-6, f"{total:.6f}")
        starts = [plan[k][0] for k, _, _ in
                  [(x, 0, 0) for x in plan]]
        check(f"{name}: stages are contiguous",
              all(abs((plan[k][0] + plan[k][1]) - nxt) < 1e-6
                  for k, nxt in zip(list(plan), [plan[x][0] for x in list(plan)[1:]] + [1.0])))

    fast = r._plan(Settings().apply_preset("fast"))
    check("fast skips separation", "separate" not in fast)

    balanced = r._plan(Settings().apply_preset("balanced"))
    check("balanced includes separation", "separate" in balanced)

    # Who is speaking is a fact about the video, not a quality preset, so no
    # preset turns it on or off — the front panel asks instead. Off by default,
    # because this app is used mostly on single-presenter instruction and
    # over-segmentation dubs one person in several voices.
    check("no preset dictates diarization", "diarize" not in balanced)
    several = Settings().apply_preset("balanced")
    several.diarize = True
    check("saying several people speak adds the stage", "diarize" in r._plan(several))

    best = r._plan(Settings().apply_preset("best"))
    check("cloning gets a bigger share of the bar",
          best["synthesize"][1] > balanced["synthesize"][1],
          f"{best['synthesize'][1]:.3f} vs {balanced['synthesize'][1]:.3f}")
    check("whisper gets a bigger share than parakeet",
          best["transcribe"][1] > balanced["transcribe"][1])

    s = Settings().apply_preset("best")
    check("best preset turns cloning on", s.voice_mode == "clone")
    check("best preset selects whisper", s.asr_model == "whisper")
    s.apply_preset("fast")
    check("switching preset resets the switches",
          s.voice_mode == "fixed" and s.asr_model == "parakeet" and not s.separate_audio)


# ====================================================== 2. per-speaker voices
def test_voice_assignment():
    print("\n[2] Per-speaker voice assignment")
    from app.config import Settings

    s = Settings()
    s.voice = "bf_emma"
    check("first speaker keeps the chosen voice", s.voice_for(0) == "bf_emma")
    others = [s.voice_for(i) for i in range(1, 6)]
    check("other speakers get different voices", "bf_emma" not in others, str(others))
    check("assignment is stable", s.voice_for(1) == s.voice_for(1))
    check("adjacent speakers differ", others[0] != others[1], f"{others[0]} vs {others[1]}")


# ========================================================== 3. diarization
def make_two_speaker_clip(path: Path) -> list[tuple[float, float, int]]:
    """Build audio with two clearly different voices alternating."""
    if path.exists():
        return json.loads(path.with_suffix(".json").read_text())

    # Two built-in voices with clearly different timbre — a British female and a
    # British male — so the diarizer has a genuine pair of speakers to separate.
    from app.backends.tts import OnnxTTS

    tts = OnnxTTS()
    lines = [
        ("bf_emma", "Hello and welcome along to the programme this evening."),
        ("bm_george", "Thank you very much for having me here today, it is a pleasure."),
        ("bf_emma", "Now tell me, how did the whole project actually begin?"),
        ("bm_george", "Well it started several years ago in a very small workshop."),
        ("bf_emma", "That is fascinating, and what happened after that?"),
        ("bm_george", "We grew steadily and eventually moved to a much larger building."),
    ]
    pieces, truth, t = [], [], 0.0
    sr = 24000
    for voice, text in lines:
        arr, sr = tts.say(text, voice=voice, speed=1.0)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        truth.append((round(t, 2), round(t + len(arr) / sr, 2), 0 if voice == "bf_emma" else 1))
        pieces.append(arr)
        gap = np.zeros(int(0.4 * sr), dtype=np.float32)
        pieces.append(gap)
        t += len(arr) / sr + 0.4
    sf.write(path, np.concatenate(pieces), sr)
    path.with_suffix(".json").write_text(json.dumps(truth))
    return truth


def test_diarization():
    print("\n[3] Speaker diarization (real models)")
    from app.backends import diarize as D

    clip = WORK / "two_speakers.wav"
    truth = make_two_speaker_clip(clip)
    clip16 = WORK / "two_speakers_16k.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip),
                    "-ar", "16000", "-ac", "1", str(clip16)], check=True)

    turns = D.diarize(clip16, expected_speakers=-1)
    check("diarization returned turns", len(turns) > 0, f"{len(turns)} turns")
    if not turns:
        return
    found = len({t["speaker"] for t in turns})
    check("found exactly two speakers", found == 2, f"found {found}")

    # Label fake transcript segments matching the ground truth and see if the
    # speaker labels alternate the way the audio does.
    segs = [{"start": a, "end": b, "text": f"line {i}"} for i, (a, b, _) in enumerate(truth)]
    labelled = D.label_segments(segs, turns)
    pattern = [s["speaker"] for s in labelled]
    truth_pattern = [t[2] for t in truth]
    # Speaker numbering is arbitrary, so compare the grouping, not the ids.
    same = all((pattern[i] == pattern[j]) == (truth_pattern[i] == truth_pattern[j])
               for i in range(len(pattern)) for j in range(len(pattern)))
    check("segments grouped by the right speaker", same, f"{pattern} vs {truth_pattern}")

    # Reference extraction for cloning.
    audio, rate = sf.read(clip, dtype="float32")
    for spk in sorted({s["speaker"] for s in labelled}):
        ref = D.pick_reference(labelled, spk, audio, rate, min_s=4.0, max_s=10.0)
        check(f"got a reference clip for speaker {spk}",
              ref is not None and ref.size > rate,
              f"{(ref.size/rate):.1f}s" if ref is not None else "none")
        check(f"reference for speaker {spk} respects the cap",
              ref is None or ref.size <= 10.0 * rate + 10)

    check("no speakers means everything is speaker zero",
          all(s["speaker"] == 0 for s in D.label_segments(
              [{"start": 0, "end": 1, "text": "x"}], [])))


# ======================================================== 4. separation path
def test_separation():
    print("\n[4] Separation")
    from app.backends import separate as S

    work = WORK / "septest"
    work.mkdir(parents=True, exist_ok=True)
    src = work / "in.wav"
    if not src.exists():
        # A voice over a bass line, so there is genuinely something to pull apart.
        speech = WORK / "two_speakers.wav"
        make_two_speaker_clip(speech)
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-i", str(speech), "-f", "lavfi",
                        "-i", "sine=frequency=110:duration=8",
                        "-filter_complex",
                        "[0:a]aformat=channel_layouts=stereo:sample_rates=44100[v];"
                        "[1:a]aformat=channel_layouts=stereo:sample_rates=44100,volume=-8dB[m];"
                        "[v][m]amix=inputs=2:duration=first:normalize=0[out]",
                        "-map", "[out]", "-ac", "2", "-ar", "44100", str(src)], check=True)

    # The fallback has to stay graceful whatever this machine has installed, so
    # force it rather than relying on the dependency happening to be absent.
    real_available = S.available
    S.available = lambda: False
    try:
        messages = []
        result = S.separate(src, work, prefer_gpu=False,
                            progress=lambda f, m: messages.append(m))
        check("returns None instead of raising when it can't run", result is None, str(result))
        check("explains itself in the progress message",
              any("carrying on" in m or "Skipping" in m for m in messages),
              messages[-1] if messages else "no messages")
    finally:
        S.available = real_available

    if not S.available():
        check("demucs is installed", False, "quality extras missing — real run skipped")
        return
    check("demucs is installed", True)

    messages = []
    result = S.separate(src, work, prefer_gpu=True,
                        progress=lambda f, m: messages.append(m))
    check("separation produced both stems", result is not None,
          messages[-1] if messages else "no messages")
    if result is None:
        return

    speech_path, bed_path = result
    check("speech stem was written", speech_path.exists())
    check("background stem was written", bed_path.exists())

    for label, path in (("speech", speech_path), ("background", bed_path)):
        data, _ = sf.read(path, dtype="float32")
        finite = bool(np.all(np.isfinite(data)))
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        # Demucs on MPS has historically produced silent or NaN output. Both are
        # silent failures that would ship a broken dub, so assert against them.
        check(f"{label} stem contains no NaN or infinity", finite)
        check(f"{label} stem is not silent", peak > 1e-4, f"peak {peak:.4f}")


# ============================================================ 5. music mixing
def test_music_mix():
    print("\n[5] Mixing the dub over a preserved music bed")
    from app.steps import mux

    work = WORK / "mixtest"
    work.mkdir(parents=True, exist_ok=True)
    dub, bed, out = work / "dub.wav", work / "bed.wav", work / "out.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=5", "-ac", "1",
                    "-ar", "24000", str(dub)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=120:duration=5", "-ac", "2",
                    "-ar", "44100", str(bed)], check=True)

    mux.mix_with_background(dub, bed, out, bed_gain_db=-6.0)
    check("mixed file was written", out.exists())
    data, rate = sf.read(out, dtype="float32")
    check("output is stereo", data.ndim == 2 and data.shape[1] == 2, str(data.shape))
    check("output is 48k", rate == 48000, str(rate))
    check("output holds both signals", float(np.max(np.abs(data))) > 0.1,
          f"peak {float(np.max(np.abs(data))):.2f}")

    # The bed must be quieter than the speech, not louder.
    spec = np.abs(np.fft.rfft(data[:, 0][:48000]))
    freqs = np.fft.rfftfreq(48000, 1 / 48000)
    e440 = spec[(freqs > 420) & (freqs < 460)].max()
    e120 = spec[(freqs > 110) & (freqs < 130)].max()
    check("speech sits above the music bed", e440 > e120, f"{e440:.0f} vs {e120:.0f}")


# ==================================================== 6. cloning falls back
def test_clone_fallback():
    print("\n[6] Cloning unavailable — must fall back, not fail")
    from app import pipeline as P
    from app.backends import clone as C
    from app.config import Settings, detect_machine
    from app.pipeline import JobRunner

    r = JobRunner()
    s = Settings().apply_preset("best")

    # Force the missing-dependency path. Previously this test relied on
    # chatterbox genuinely being absent, so it inverted the moment the quality
    # extras were installed and stopped testing the fallback at all.
    real_available = C.available
    P.clone_backend.available = lambda: False
    try:
        msgs = []
        engine, cloning = r._make_engine(s, detect_machine(), [], WORK / "speech.wav",
                                         WORK / "clonetest", [0],
                                         lambda f, m: msgs.append(m))
        check("fell back to a built-in voice", cloning is False)
        check("engine is still usable", hasattr(engine, "say"))
        check("told the user why",
              any("built-in voice" in m for m in msgs), msgs[-1] if msgs else "none")
    finally:
        P.clone_backend.available = real_available

    # A model that imports but blows up on load must degrade the same way, since
    # that is what a half-downloaded checkpoint actually looks like.
    real_ctor = C.CloneTTS
    P.clone_backend.available = lambda: True

    def exploding(*a, **k):
        raise RuntimeError("checkpoint is corrupt")

    P.clone_backend.CloneTTS = exploding
    try:
        msgs = []
        engine, cloning = r._make_engine(s, detect_machine(), [], WORK / "speech.wav",
                                         WORK / "clonetest", [0],
                                         lambda f, m: msgs.append(m))
        check("a failing clone model falls back rather than raising", cloning is False)
        check("the reason reaches the progress callback",
              any("checkpoint is corrupt" in m for m in msgs), msgs[-1] if msgs else "none")
    finally:
        P.clone_backend.CloneTTS = real_ctor
        P.clone_backend.available = real_available


# ============================================ 7. full balanced-preset run
def test_balanced_end_to_end():
    print("\n[7] Full run on the Balanced preset (2 speakers, real diarization)")
    from app import pipeline
    from app.backends import translate as T
    from app.config import Settings, JOBS

    work = WORK / "e2e2"
    work.mkdir(parents=True, exist_ok=True)
    clip_audio = WORK / "two_speakers.wav"
    make_two_speaker_clip(clip_audio)
    clip = work / "clip.mp4"
    if not clip.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc=size=320x240:rate=15", "-i", str(clip_audio),
                        "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                        "-c:a", "aac", str(clip)], check=True)

    from app.steps import download as dl
    # Probe as well as download: the pipeline asks for the duration before it
    # draws the progress plan, so stubbing only the download leaves the real
    # yt-dlp being asked about a made-up URL.
    meta = {"title": "Two Speaker Test", "duration": dl.media_duration(clip),
            "uploader": "t", "thumbnail": ""}

    def fake_download(url, workdir, quality="best", progress=None, info=None, **_):
        import shutil
        workdir.mkdir(parents=True, exist_ok=True)
        dest = workdir / "source.mp4"
        shutil.copy(clip, dest)
        if progress:
            progress(1.0, "Downloaded")
        return dest, dict(meta)

    pipeline.download.probe = lambda url, *_a, **_kw: dict(meta)
    pipeline.download.download = fake_download

    LINES = ["Welcome along to the programme this evening.",
             "Thank you for having me, it is a real pleasure.",
             "Tell me, how did the project begin?",
             "It started in a very small workshop some years ago.",
             "Fascinating. And what happened next?",
             "We grew, and moved to a far larger building."]

    def fake_llm(prompt, model=None, host=None, key=None):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|{LINES[i % len(LINES)]}" for i in ids)

    T._call_ollama = fake_llm

    s = Settings().apply_preset("balanced")
    s.diarize = True                  # this clip genuinely has two people in it
    s.translator = "ollama"
    s.voice = "bf_emma"

    job = pipeline.runner.submit("https://example.com/two", s)
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
            print(log.read_text()[-1800:])
        return

    st = job.stats
    check("two speakers were detected", st.get("speakers") == 2, str(st.get("speakers")))
    check("separation was attempted and reported", "separated" in st, str(st.get("separated")))
    check("frames preserved", st.get("frames_match") is True)
    check("audio matches video length", st.get("drift_seconds", 99) < 1.0,
          f"{st.get('drift_seconds')}s")
    check("preset recorded in the report", st.get("preset") == "balanced")
    check("output exists", Path(job.output).exists())

    # "Who's speaking?" is on the front panel now, so changing it on a link that
    # has already been dubbed is an ordinary thing to do — and this pipeline has
    # a history of reusing an artefact that no longer matches its settings. The
    # transcript and the translation are rightly kept; the labels and the
    # rendered voices are not, and they are what the answer changes.
    def rerun(diarize: bool):
        again = Settings().apply_preset("balanced")
        again.diarize, again.translator, again.voice = diarize, "ollama", "bf_emma"
        j = pipeline.runner.submit("https://example.com/two", again)
        t = time.time()
        while j.status in ("queued", "running") and time.time() - t < 900:
            time.sleep(2)
        return j

    one = rerun(False)
    check("saying one person speaks re-runs to a single voice",
          one.status == "done" and one.stats.get("speakers") == 1,
          f"{one.status}: {one.stats.get('speakers')}")

    # Rendered lines are cached in a voice-keyed folder, and the one-voice run
    # just above leaves its own behind — in which both speakers legitimately
    # sound the same, because that is what it was asked for. The pitch check
    # below then picked whichever folder name sorted first and, when that was
    # the single-voice one, failed with the two speakers under a hertz apart on
    # a pipeline that was working correctly. Clearing them makes the folder it
    # measures unambiguously the one this run renders.
    import shutil
    shutil.rmtree(JOBS / job.id / "lines", ignore_errors=True)

    two = rerun(True)
    check("and saying several restores them",
          two.status == "done" and two.stats.get("speakers") == 2,
          f"{two.status}: {two.stats.get('speakers')}")

    # Both assigned voices should actually appear in the rendered audio: check
    # the per-line files aren't all identical in pitch.
    segs = json.loads((JOBS / job.id / "translated.json").read_text())
    speakers = {s_["speaker"] for s_ in segs}
    check("segments carry speaker labels", len(speakers) == 2, str(speakers))

    # Pitch by autocorrelation, taken per frame over the voiced parts and then
    # averaged. One autocorrelation across a whole line also spans its silence
    # and onsets, which made the estimate wobble by 20 Hz between runs and
    # understated the real gap between voices badly enough to flip this check.
    # An FFT peak is no good either: it finds harmonics, not the fundamental.
    def fundamental(path):
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        win, hop = int(0.04 * sr), int(0.02 * sr)
        lo, hi = sr // 350, sr // 70           # search 70-350 Hz
        f0s = []
        for i in range(0, max(0, len(a) - win), hop):
            frame = a[i:i + win]
            if np.sqrt((frame ** 2).mean()) < 0.02:      # silence between words
                continue
            frame = frame - frame.mean()
            corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
            if hi >= len(corr):
                continue
            k = int(np.argmax(corr[lo:hi])) + lo
            if corr[k] > 0.3 * corr[0]:                  # voiced, not noise
                f0s.append(sr / k)
        return float(np.median(f0s)) if f0s else 0.0

    # Rendered lines sit in a voice-keyed subfolder, so that changing voice and
    # re-running doesn't replay the previous voice's cached audio.
    #
    # The newest, not the alphabetically first. This test runs the job twice
    # with different voices, so a second folder appears — and on a scratch
    # directory that has been used before, whichever name sorted first won,
    # which is whatever a previous run happened to leave behind. The check then
    # measured audio from a different voice than the run it had just done, and
    # failed with two speakers three hertz apart on a working pipeline.
    lines_root = JOBS / job.id / "lines"
    subdirs = sorted((d for d in lines_root.iterdir() if d.is_dir()),
                     key=lambda d: d.stat().st_mtime) if lines_root.is_dir() else []
    lines_dir = subdirs[-1] if subdirs else lines_root
    pitches: dict[int, list[float]] = {}
    for idx, s_ in enumerate(segs):
        p = lines_dir / f"{idx:05d}.wav"
        if p.exists():
            pitches.setdefault(s_["speaker"], []).append(fundamental(p))

    if len(pitches) == 2:
        a_, b_ = (float(np.median(v)) for v in pitches.values())
        check("the two speakers sound different", abs(a_ - b_) > 20,
              f"{a_:.0f} Hz vs {b_:.0f} Hz")
    else:
        check("captured audio for both speakers", False, str(list(pitches)))


if __name__ == "__main__":
    test_plan()
    test_voice_assignment()
    test_diarization()
    test_separation()
    test_music_mix()
    test_clone_fallback()
    test_balanced_end_to_end()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("All checks passed.")
