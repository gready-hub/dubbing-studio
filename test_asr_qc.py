"""Tests for app/steps/asr_qc.py — the pre-translation repetition-loop safety net.

Run it from anywhere:  python3 test_asr_qc.py

Every fixture segment below is transcribed word-for-word (and, where it
matters, timestamp-for-timestamp) from the real full-52-minute mlx_whisper
dumps saved under ~/.claude/scratch/dubbing-studio-asr-hallucination/fixtures/
— full_video_default_decode.json (condition_on_previous_text=True, the
hallucinated decode) and full_video_nocond_decode.json (=False, the clean
one). None of it is invented. Where a fixture's absolute timestamps have been
shifted to start near 0 (only in the second-opinion tests, which need a real
backing wav file), the comment says so; the words and relative durations are
untouched.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Same dance as test_pipeline.py / test_upgrade.py: each run gets its own
# scratch home (removed on exit) unless one is named explicitly, and the
# scratch model cache is symlinked to the real one so the one real-engine
# integration test below doesn't re-fetch models it already has on disk.
_explicit_home = os.environ.get("DUBBING_STUDIO_HOME")
if _explicit_home:
    SCRATCH = Path(_explicit_home)
else:
    SCRATCH = Path(tempfile.mkdtemp(prefix="dubbing-studio-test-"))
    os.environ["DUBBING_STUDIO_HOME"] = str(SCRATCH)
    atexit.register(shutil.rmtree, SCRATCH, ignore_errors=True)

_real_models = Path.home() / "Library" / "Caches" / "DubbingStudio" / "models"
_scratch_models = SCRATCH / "cache" / "models"
if _real_models.is_dir() and not _scratch_models.is_symlink():
    if _scratch_models.is_dir():
        shutil.rmtree(_scratch_models, ignore_errors=True)
    _scratch_models.parent.mkdir(parents=True, exist_ok=True)
    _scratch_models.symlink_to(_real_models)

FIXTURES = (Path.home() / ".claude" / "scratch" / "dubbing-studio-asr-hallucination"
           / "fixtures")

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def _silence_wav(path: Path, seconds: float, sr: int = 16000) -> Path:
    """A backing wav for tests that exercise the slice-and-cross-check plumbing
    without caring what the audio actually contains — the engine call itself
    is monkeypatched in those tests, so only the framing (duration, format)
    needs to be real."""
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)
    return path


# ============================================ 1. intra-segment thresholds
def test_intra_segment_thresholds():
    """One segment whose own words are almost entirely one repeated token or
    short phrase — checked against the real numbers on both sides of the gap,
    not round ones picked in advance.
    """
    print("\n[1] Intra-segment repetition thresholds (real numbers)")
    from app.steps.asr_qc import _is_degenerate

    # The four "ellos" segments, 384.54-413.22s in full_video_default_decode.json:
    # a single token, 100% of the segment, word counts 16/14/13/9 — never
    # reproduced in the clean decode of the same span.
    for count in (16, 14, 13, 9):
        text = " ".join(["ellos"] * count)
        flagged, info = _is_degenerate(text)
        check(f"'ellos'x{count} (share={info.get('share')}) is flagged", flagged, info)

    # The "ya toda cadeneta" segments, 1942.48-1966.50s in the same file: a
    # three-word phrase looping, so no single token clears 40% (share stays
    # 0.33) — only the distinct-word count gives it away (3 distinct words
    # over 18-21 slots).
    for reps in (6, 7):
        text = " ".join(["ya", "toda", "cadeneta"] * reps)
        flagged, info = _is_degenerate(text)
        check(f"'ya toda cadeneta'x{reps} (unique_ratio={info.get('unique_ratio')}) "
             "is flagged", flagged, info)

    # Real, legitimate lines from the SAME two files, at or above the same
    # word-count floor, that must not trip either measure. These are the
    # highest share/lowest unique_ratio found anywhere in either decode once
    # the confirmed hallucination windows are excluded.
    legit = [
        # full_video_default_decode.json, 1375.20-1392.42 — real stitch count.
        "2 y 3 puntos altos, 3 cadenetas y 3 puntos altos, 1, 2 y 3 puntos altos y 3 cadenetas",
        # full_video_default_decode.json, 2272.84-2274.84 — real speech.
        "es la misma, la misma, la misma vuelta",
        # full_video_nocond_decode.json, 125.10-131.56 — real, repeated for emphasis.
        "Si la queréis hacer larga, si la queréis hacer corta, si la queréis hacer media",
    ]
    for text in legit:
        flagged, info = _is_degenerate(text)
        check(f"real line not flagged: {text[:40]!r}", not flagged, info)

    # A short real interjection must not even reach the word-count floor.
    flagged, info = _is_degenerate("¿vale? ¿vale?")
    check("a two-word segment is below the floor, not evaluated",
         not flagged and info["n"] < 6, info)


# ============================================ 2. cross-segment consecutive runs
def test_cross_segment_consecutive_duplicates():
    """Several consecutive raw segments carrying the exact same text — the
    shape each individual segment is too short to trip check 1 on its own.
    """
    print("\n[2] Cross-segment consecutive-duplicate runs (real spans)")
    from app.steps.asr_qc import _find_flags

    # "¿Vale?"x10 in 0.4s, 147.88-148.28s — real numbers, real timestamps.
    vale = (
        [{"start": 145.94, "end": 147.88, "text": "Eso ya dependerá de vosotras"}]
        + [{"start": round(147.88 + i * 0.04, 2), "end": round(147.92 + i * 0.04, 2),
           "text": "¿Vale?"} for i in range(10)]
        + [{"start": 148.28, "end": 150.48, "text": "Esta es una lana stop perlé número 5"}]
    )
    units = _find_flags(vale)
    check("the 10-rep 'Vale' run is found as one unit", len(units) == 1, units)
    if units:
        check("it covers exactly the 10 duplicate lines, not the real ones either side",
             (units[0]["lo"], units[0]["hi"]) == (1, 10), (units[0]["lo"], units[0]["hi"]))

    # "¡Chao!"x3 spread across 36s of near-silence at the very end of the
    # file — individually far apart, but no OTHER segment sits between them,
    # so they are still a consecutive run in index terms. The opposite
    # extreme from Vale (implausibly slow, not fast), which is why detection
    # here does not gate on words-per-second at all.
    chao = [
        {"start": 3088.60, "end": 3091.22, "text": "Bueno, chicas, nada, un saludo y que os veo en el próximo."},
        {"start": 3091.64, "end": 3091.98, "text": "¡Chao!"},
        {"start": 3118.60, "end": 3119.60, "text": "¡Chao!"},
        {"start": 3125.64, "end": 3127.64, "text": "¡Chao!"},
    ]
    units = _find_flags(chao)
    check("the 3-rep 'Chao' run is found despite the 27s+7s gaps inside it",
         len(units) == 1 and (units[0]["lo"], units[0]["hi"]) == (1, 3), units)

    # Real, non-consecutive repeats of the same real phrase must NOT form a
    # run: full_video_nocond_decode.json, 389.84-429.16s. "1, 2 y 3
    # cadenetas" and "punto alto" each occur twice in this stretch, but each
    # time inside a different full sentence with unrelated lines between the
    # two occurrences — never as back-to-back identical segments.
    crochet_counting = [
        {"start": 389.84, "end": 393.00, "text": "Mirad, empezamos dando 3 cadenetas"},
        {"start": 393.00, "end": 397.14, "text": "venga, 1, 2 y 3 cadenetas"},
        {"start": 397.14, "end": 400.00, "text": "hecho hebra y aquí en el primer puntito que dimos"},
        {"start": 400.00, "end": 401.44, "text": "bueno, pues un punto alto, ¿vale?"},
        {"start": 402.12, "end": 405.64, "text": "venga, la primera siempre queda así un poquito porque no he querido apretar"},
        {"start": 405.64, "end": 409.92, "text": "otra vez, mirad, 1, 2 y 3 cadenetas"},
        {"start": 409.92, "end": 413.14, "text": "hecho hebra y aquí, a ver si lo podéis ver bien"},
        {"start": 413.14, "end": 414.96, "text": "vamos a tener dos hilos"},
        {"start": 414.96, "end": 418.34, "text": "bueno, pues desde esos dos hilos saco y hago un punto alto"},
        {"start": 418.34, "end": 429.16, "text": "¿Vale? Mirad, 1, 2 y 3, echo hebra y desde aquí el primer puntito, un punto alto."},
    ]
    units = _find_flags(crochet_counting)
    check("real repeated stitch-counting, never back-to-back, forms no run",
         units == [], units)


# ============================================ 3. the four originally-confirmed
#    hallucinations, end to end, plus the false-positive guard
def test_confirmed_hallucinations_flagged_and_crochet_counting_spared():
    """The four hallucination instances spec.md names as directly confirmed,
    run through the real check() (with a stubbed second opinion so this stays
    a fast unit test), each asserted flagged — and, in the same pass, the
    real crochet-counting lines from the clean decode of the neighbouring
    span asserted untouched.
    """
    print("\n[3] The four confirmed hallucinations, flagged; crochet counting, spared")
    from app.steps import asr_qc

    real_ellos = [16, 14, 13, 9]
    segments = [
        # ---- "ellos" x52, 380.22-418.32s (default decode)
        {"start": 380.22, "end": 384.42,
         "text": "labor vamos a empezarla de una forma muy especial a mí me gusta muchísimo la verdad así creo que"},
        *[{"start": s, "end": e, "text": " ".join(["ellos"] * n)} for s, e, n in [
            (384.54, 389.20, 16), (389.20, 403.36, 14), (403.36, 409.06, 13), (409.06, 413.22, 9)]],
        {"start": 414.54, "end": 418.32, "text": "Bueno, pues desde esos dos hilos saco y hago un punto alto."},

        # ---- real crochet-counting lines that must survive untouched — the
        # false-positive case this module exists not to break (real content,
        # from full_video_nocond_decode.json's clean version of this span).
        {"start": 418.34, "end": 429.16,
         "text": "¿Vale? Mirad, 1, 2 y 3, echo hebra y desde aquí el primer puntito, un punto alto."},

        # ---- "¿Vale?"x10, 147.88-148.28s (default decode)
        {"start": 145.94, "end": 147.88, "text": "Eso ya dependerá de vosotras"},
        *[{"start": round(147.88 + i * 0.04, 2), "end": round(147.92 + i * 0.04, 2),
          "text": "¿Vale?"} for i in range(10)],
        {"start": 148.28, "end": 150.48, "text": "Esta es una lana stop perlé número 5"},

        # ---- "Y si no, pues no se vea"x5, 267.06-267.48s (default decode)
        {"start": 266.20, "end": 266.96, "text": "Porque si no, para la gente que no se vea"},
        {"start": 266.96, "end": 267.06, "text": "Pues no se vea"},
        {"start": 267.06, "end": 267.10, "text": "Y si no, pues no se vea"},
        {"start": 267.10, "end": 267.18, "text": "Y si no, pues no se vea"},
        {"start": 267.18, "end": 267.20, "text": "Y si no, pues no se vea"},
        {"start": 267.20, "end": 267.26, "text": "Y si no, pues no se vea"},
        {"start": 267.26, "end": 267.48, "text": "Y si no, pues no se vea"},
        {"start": 267.50, "end": 268.34, "text": "Que sabe hacer"},

        # ---- "¡Chao!"x3 across 36s of near-silence (default decode, end of file)
        {"start": 3088.60, "end": 3091.22, "text": "Bueno, chicas, nada, un saludo y que os veo en el próximo."},
        {"start": 3091.64, "end": 3091.98, "text": "¡Chao!"},
        {"start": 3118.60, "end": 3119.60, "text": "¡Chao!"},
        {"start": 3125.64, "end": 3127.64, "text": "¡Chao!"},
    ]
    del real_ellos  # only used to document the source counts above

    crochet_text = segments[6]["text"]
    real_texts_untouched = [segments[0]["text"], segments[5]["text"], crochet_text,
                            segments[7]["text"], segments[-4]["text"]]

    audio = Path(tempfile.mkdtemp(prefix="asr-qc-test-")) / "silence.wav"
    _silence_wav(audio, 3130.0)

    # A stub, not the real engine — this test is about detection and about
    # real content surviving, not about what any particular ASR says back.
    # The real-engine claim gets its own integration test below.
    asr_qc._transcribe_mlx = lambda path, progress=None: []
    asr_qc._transcribe_onnx = lambda path, progress=None: []

    report = asr_qc.check(segments, audio, True)

    check("all four confirmed hallucinations are found (one unit each)",
         report["flagged"] == 4, report)
    check("with nothing recoverable from a silent stub, all four are silenced",
         report["silenced"] == 4 and report["recovered"] == 0, report)

    texts = [s["text"] for s in segments]
    for kept in real_texts_untouched:
        check(f"real line survives untouched: {kept[:40]!r}", kept in texts)
    check("the crochet-counting line specifically was not silenced",
         crochet_text in texts and crochet_text != "")


# ============================================ 4. the second opinion itself
def test_second_opinion_recovers_and_silences():
    """The mandatory cross-check: a flagged span whose second opinion looks
    sane gets its text replaced; one whose second opinion is equally
    degenerate (or empty) gets silenced instead. This is the asymmetry the
    whole module exists for — a stub stands in for Parakeet, chosen by call
    order (check() processes flagged units in ascending start order, so the
    first call answers for the earlier span and the second for the later).
    """
    print("\n[4] Mandatory second opinion: recover one, silence the other")
    from app.steps import asr_qc

    # Same "ellos" shape as above, timestamps shifted to start near 0 so the
    # backing wav stays a few seconds instead of covering the real 400s mark
    # — the words and relative durations are exactly as decoded.
    segments = [
        {"start": 0.00, "end": 4.20,
         "text": "labor vamos a empezarla de una forma muy especial a mí me gusta muchísimo la verdad así creo que"},
        {"start": 4.32, "end": 8.98, "text": " ".join(["ellos"] * 16)},
        {"start": 8.98, "end": 23.14, "text": " ".join(["ellos"] * 14)},
        {"start": 23.14, "end": 28.84, "text": " ".join(["ellos"] * 13)},
        {"start": 28.84, "end": 33.00, "text": " ".join(["ellos"] * 9)},
        {"start": 34.32, "end": 38.10, "text": "Bueno, pues desde esos dos hilos saco y hago un punto alto."},
        # a second, separate cluster further along — the real "Vale" shape.
        {"start": 40.00, "end": 40.14, "text": "¿Vale?"},
        {"start": 40.14, "end": 40.16, "text": "¿Vale?"},
        {"start": 40.16, "end": 40.18, "text": "¿Vale?"},
    ]

    audio = Path(tempfile.mkdtemp(prefix="asr-qc-test-")) / "silence.wav"
    _silence_wav(audio, 45.0)

    calls = {"n": 0}

    def fake_engine(path, progress=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # A sane second opinion for the "ellos" span: plausible word
            # count for ~28.7s, no repetition.
            return [{"start": 0.0, "end": 1.0,
                     "text": "Vamos a ver cómo quedan estos hilos en la labor."}]
        # A degenerate second opinion for the "Vale" span: Parakeet finding
        # the same kind of nothing Whisper did.
        return [{"start": 0.0, "end": 1.0, "text": " ".join(["ellos"] * 8)}]

    real_mlx, real_onnx = asr_qc._transcribe_mlx, asr_qc._transcribe_onnx
    asr_qc._transcribe_mlx = fake_engine
    asr_qc._transcribe_onnx = fake_engine
    try:
        report = asr_qc.check(segments, audio, True)
    finally:
        asr_qc._transcribe_mlx, asr_qc._transcribe_onnx = real_mlx, real_onnx

    check("two units flagged, one recovered, one silenced",
         (report["flagged"], report["recovered"], report["silenced"]) == (2, 1, 1), report)
    check("segment count collapsed: 9 raw -> 4 (5 duplicates folded away)",
         len(segments) == 4, [s["text"] for s in segments])

    recovered = next((s for s in segments if "hilos en la labor" in s["text"]), None)
    check("the ellos span was replaced with the sane second opinion",
         recovered is not None, [s["text"] for s in segments])
    if recovered:
        check("the replacement keeps the original span's timing, not the stub's",
             (recovered["start"], recovered["end"]) == (4.32, 33.00),
             (recovered["start"], recovered["end"]))

    silenced = [s for s in segments if s["start"] == 40.00]
    check("the Vale span was silenced, not replaced with more 'ellos'",
         len(silenced) == 1 and silenced[0]["text"] == "", silenced)

    untouched = [s["text"] for s in segments if s["start"] in (0.00, 34.32)]
    check("the two real lines either side were never touched",
         "labor vamos a empezarla" in " ".join(untouched)
         and "Bueno, pues desde" in " ".join(untouched), untouched)

    # A second opinion can be too short to reach the flagging floor and still
    # be obvious nonsense. Judged at that floor it counted as a recovery, so a
    # smaller helping of the same repetition went into the dub in place of the
    # silence it had earned; the sanity test looks from three words up for
    # exactly this. A genuinely short answer must still be allowed through.
    from app.steps.asr_qc import _is_sane
    check("a short but repetitive second opinion is not sane",
         not _is_sane("vale vale vale", 2.0))
    check("while a short real one still is",
         _is_sane("un punto alto", 2.0))


# ============================================ 5. the real Parakeet cross-check
def test_real_parakeet_cross_check_on_real_hallucination_audio():
    """A monkeypatched cross-check proves nothing about whether the real one
    would work. This calls the actual _transcribe_mlx/_transcribe_onnx
    backend against real audio from the fixture clip covering the "¿Vale?"
    hallucination's own timestamps, and checks its answer is not itself
    degenerate by the same measure — the empirical claim behind the whole
    design, not just the plumbing.
    """
    print("\n[5] Real Parakeet cross-check against real hallucination audio")
    clip = FIXTURES / "clip_145_155.wav"
    if not clip.exists():
        check("fixture clip available (skipped — not found on this machine)",
             True, str(clip))
        return

    from app.config import detect_machine
    from app.steps.asr_qc import _is_degenerate
    from app.backends.asr import _transcribe_mlx, _transcribe_onnx

    machine = detect_machine()
    engine = _transcribe_mlx if machine.fast_path else _transcribe_onnx
    out = engine(clip, None)
    text = " ".join(s["text"] for s in out if (s.get("text") or "").strip())

    check("Parakeet returned something for 10s of real audio containing "
         "the '¿Vale?'x10 hallucination window", bool(text.strip()), text[:200])
    degenerate, info = _is_degenerate(text)
    check("and it is not itself degenerate by the same measure this module uses",
         not degenerate, {"text": text[:200], **info})


if __name__ == "__main__":
    test_intra_segment_thresholds()
    test_cross_segment_consecutive_duplicates()
    test_confirmed_hallucinations_flagged_and_crochet_counting_spared()
    test_second_opinion_recovers_and_silences()
    test_real_parakeet_cross_check_on_real_hallucination_audio()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("All checks passed.")
