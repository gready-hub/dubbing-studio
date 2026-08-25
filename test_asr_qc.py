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
    real_mlx, real_onnx = asr_qc._transcribe_mlx, asr_qc._transcribe_onnx
    asr_qc._transcribe_mlx = lambda path, progress=None: []
    asr_qc._transcribe_onnx = lambda path, progress=None: []
    try:
        report = asr_qc.check(segments, audio, True)
    finally:
        asr_qc._transcribe_mlx, asr_qc._transcribe_onnx = real_mlx, real_onnx

    check("all four confirmed hallucinations are found (one unit each)",
         report["flagged"] == 4, report)
    # The engine offering nothing is the case that used to delete the line.
    # Now it keeps one instance of whatever was being repeated, and the report
    # has no way to say "silenced" because nothing here silences anything.
    check("with nothing to be had from the stub, all four collapse rather than vanish",
         report["collapsed"] == 4 and report["recovered"] == 0
         and "silenced" not in report, report)

    texts = [s["text"] for s in segments]
    for kept in real_texts_untouched:
        check(f"real line survives untouched: {kept[:40]!r}", kept in texts)
    check("the crochet-counting line specifically was not silenced",
         crochet_text in texts and crochet_text != "")
    check("and no segment anywhere was emptied out",
         all((s["text"] or "").strip() for s in segments),
         [s for s in segments if not (s["text"] or "").strip()])
    # What each loop became: the thing itself, once.
    check("the ellos loop reads as one 'ellos', not 52 and not nothing",
         "ellos" in texts, [t for t in texts if "ellos" in t])
    check("the ¡Chao! run is one goodbye, and it is still a goodbye",
         "¡Chao!" in texts, [t for t in texts if "Chao" in t])


# ============================================ 4. the second opinion itself
def test_second_opinion_recovers_or_collapses():
    """The cross-check: a flagged span whose second opinion looks sane takes
    that text; one whose second opinion is equally degenerate keeps its own
    words with the repetition collapsed to one instance. Neither outcome
    deletes anything — a stub stands in for Parakeet, chosen by call order
    (check() processes flagged units in ascending start order, so the first
    call answers for the earlier span and the second for the later).
    """
    print("\n[4] Second opinion: re-read one, collapse the other, delete neither")
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

    check("two units flagged, one re-read, one collapsed",
         (report["flagged"], report["recovered"], report["collapsed"]) == (2, 1, 1), report)
    check("segment count collapsed: 9 raw -> 4 (5 duplicates folded away)",
         len(segments) == 4, [s["text"] for s in segments])

    recovered = next((s for s in segments if "hilos en la labor" in s["text"]), None)
    check("the ellos span was replaced with the sane second opinion",
         recovered is not None, [s["text"] for s in segments])
    if recovered:
        check("the replacement keeps the original span's timing, not the stub's",
             (recovered["start"], recovered["end"]) == (4.32, 33.00),
             (recovered["start"], recovered["end"]))

    # The Vale span: the stub answers with the same kind of repetition, so
    # there is nothing better to be had. It used to be deleted on exactly this
    # evidence — a second engine agreeing that the audio is repetitive is not
    # evidence that the audio is empty. It keeps one "¿Vale?" instead.
    kept = [s for s in segments if s["start"] == 40.00]
    check("the Vale span keeps one instance rather than being emptied",
         len(kept) == 1 and kept[0]["text"].strip() == "¿Vale?", kept)

    untouched = [s["text"] for s in segments if s["start"] in (0.00, 34.32)]
    check("the two real lines either side were never touched",
         "labor vamos a empezarla" in " ".join(untouched)
         and "Bueno, pues desde" in " ".join(untouched), untouched)

    # A second opinion can be too short to reach the flagging floor and still
    # be obvious nonsense; the sanity test looks from three words up for exactly
    # that. A genuinely short answer must still be allowed through.
    from app.steps.asr_qc import _is_sane
    check("a short but repetitive second opinion is not preferred",
         not _is_sane("vale vale vale", 2.0))
    check("while a short real one still is",
         _is_sane("un punto alto", 2.0))
    # The words-per-second *floor* used to live here and it deleted real
    # speech: a flagged run's span is unbounded, so a floor measured against it
    # demanded arbitrarily many words back. The real "¡Chao!" at the end of the
    # source video is one word over 36.6s, and the floor called the speaker's
    # own goodbye insane. Only an implausibly fast answer is rejected now.
    check("one real word over a long span is a fine answer",
         _is_sane("Ciao.", 36.6))
    check("and an impossibly fast one is still not",
         not _is_sane(" ".join(f"w{i}" for i in range(60)), 2.0))


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


# ======================================== 6. the metronome, and never deleting
def test_grid_and_never_deletes():
    """The shape that nearly got away, and the property that must always hold.

    An independent review of the first version of this module found that the
    largest hallucination in the evidence file was invisible to it: 158
    consecutive segments from 2685.84s to 3001.84s, every one exactly 2.00s
    long, counting numbers that skip as they climb. The texts differ by a word
    so no run of identical lines forms, and each is two words so the
    intra-segment floor never admits them. It was 71% of the fabricated
    timeline and the module flagged none of it.

    Run against the real dumps rather than a reconstruction, because the
    signal being tested here *is* the timing, and hand-typed timestamps would
    be testing the test.
    """
    print("\n[6] The fixed-grid counting loop, and never deleting anything")
    import json
    from app.steps import asr_qc

    dirty = FIXTURES / "full_video_default_decode.json"
    clean = FIXTURES / "full_video_nocond_decode.json"
    if not dirty.exists() or not clean.exists():
        print("      SKIP — decode dumps not on this machine")
        return

    def spoken(p):
        return [s for s in json.loads(p.read_text())["segments"]
                if (s.get("text") or "").strip()]

    d, c = spoken(dirty), spoken(clean)

    d_units = asr_qc._find_flags(d)
    grid = [u for u in d_units if any("fixed grid" in r for r in u["reasons"])]
    in_blind_spot = [u for u in d_units
                     if d[u["lo"]]["start"] >= 2680 and d[u["hi"]]["end"] <= 3010]
    covered = sum(d[u["hi"]]["end"] - d[u["lo"]]["start"] for u in in_blind_spot)
    check("the fixed-grid counting loop is seen at all", len(grid) > 0, len(grid))
    check("and most of its 316 seconds is covered, not a token slice of it",
         covered > 250, f"{covered:.0f}s of 316s across {len(in_blind_spot)} units")

    # The whole point of a third signal is more reach without more noise.
    c_units = asr_qc._find_flags(c)
    check("and the clean decode of the same audio still flags nothing at all",
         len(c_units) == 0,
         [(round(c[u["lo"]]["start"], 1), (c[u["lo"]]["text"] or "")[:40])
          for u in c_units[:5]])

    # The invariant, asserted over the whole real file rather than a fixture:
    # every flagged unit must come out of check() carrying words.
    audio = Path(tempfile.mkdtemp(prefix="asr-qc-grid-")) / "silence.wav"
    _silence_wav(audio, 3140.0)
    real_mlx, real_onnx = asr_qc._transcribe_mlx, asr_qc._transcribe_onnx
    asr_qc._transcribe_mlx = lambda path, progress=None: []   # no help available
    asr_qc._transcribe_onnx = lambda path, progress=None: []
    try:
        segs = [dict(s) for s in d]
        report = asr_qc.check(segs, audio, True)
    finally:
        asr_qc._transcribe_mlx, asr_qc._transcribe_onnx = real_mlx, real_onnx

    check("with no second opinion to be had, nothing is emptied",
         all((s["text"] or "").strip() for s in segs),
         [s for s in segs if not (s["text"] or "").strip()][:3])
    check("every flagged unit is accounted for",
         report["flagged"] == report["recovered"] + report["collapsed"]
         + report["left_alone"], report)
    check("and the report cannot even say 'silenced'", "silenced" not in report, report)

    # "Nothing was dropped" stated as arithmetic over the whole real file rather
    # than as a promise in a comment. With no second engine to prefer, every
    # distinct word that went in has to come back out — collapsing may remove
    # repetitions of a word, never the only copy of one. This is the assertion
    # that catches the failure a code review found: a unit whose lines each said
    # a different number was being replaced by the first of them, so one 44-second
    # segment reading "y 3" stood in for twenty-two distinct lines while the
    # report still promised nothing had gone.
    def words_of(ss):
        return {w for s in ss for w in
                (asr_qc._normalize_word(t) for t in (s["text"] or "").split()) if w}

    check("no distinct word anywhere in the file is lost",
         words_of(d) <= words_of(segs), sorted(words_of(d) - words_of(segs))[:12])
    # A span whose lines differ from one another has nothing to collapse to, so
    # it must survive intact rather than being folded onto its first line.
    grid_kept = [s for s in segs if 2700 <= s["start"] <= 2760]
    check("a counting grid keeps its separate lines instead of folding to one",
         len(grid_kept) > 10, len(grid_kept))
    # And a run that really is one word over and over does collapse, even when
    # the lines differ in *length* — "ellos"x16 beside "ellos"x14 is two strings
    # with one word between them.
    ellos = [s for s in segs if 384 <= s["start"] <= 414]
    check("but a one-word loop still collapses to that word",
         len(ellos) == 1 and ellos[0]["text"].strip() == "ellos", ellos)

    # check() walks the units in order and moves a cursor past each, so any
    # overlap emits the same loop twice and any unit ending behind the cursor
    # winds it backwards and re-emits what was already consumed. On this file
    # the span cap produced exactly that: four overlapping lines all reading
    # "4 cadenetas." for one window.
    units = asr_qc._find_flags(d)
    check("units never overlap and never run backwards",
         all(b["lo"] > a["hi"] and b["hi"] >= a["hi"]
             for a, b in zip(units, units[1:])),
         [(a["lo"], a["hi"], b["lo"], b["hi"]) for a, b in zip(units, units[1:])
          if not (b["lo"] > a["hi"] and b["hi"] >= a["hi"])][:3])
    dupes = [s["start"] for s in segs if sum(
        1 for t in segs if t["start"] == s["start"]) > 1]
    check("and no two output segments start at the same moment", not dupes, dupes[:5])

    # Uniform timing alone must not be enough. Real counting at a steady pace
    # looks like a metronome in every respect but one: it leaves air between the
    # numbers, where a generated grid runs gapless because nothing is finding
    # speech boundaries any more. Both fixtures below were flagged and collapsed
    # in full before the gap test existed — the first is the suite's own
    # translate-resume fixture, which is how this was found.
    paced = [{"start": n * 1.5, "end": n * 1.5 + 1.0, "text": f"linea {n}"}
             for n in range(26)]
    check("26 uniformly paced lines with gaps between them are left alone",
         asr_qc._find_flags(paced) == [], asr_qc._find_flags(paced)[:2])
    stitches = [{"start": n * 2.0, "end": n * 2.0 + 1.6, "text": f"punto {n}"}
                for n in range(30)]
    check("and so is someone counting stitches at a steady pace",
         asr_qc._find_flags(stitches) == [], asr_qc._find_flags(stitches)[:2])
    gapless = [{"start": n * 2.0, "end": n * 2.0 + 2.0, "text": f"y {n}"}
               for n in range(30)]
    check("while the same counting with no air in it at all is caught",
         len(asr_qc._find_flags(gapless)) > 0)

    # Collapsing keeps what was said, once — including the original disaster.
    check("221 copies of one word collapse to that word",
         asr_qc._collapse("como " * 221) == "como")
    check("a repeated phrase collapses to the phrase",
         asr_qc._collapse("ya toda cadeneta " * 6) == "ya toda cadeneta")
    check("and a real line is left exactly as it was",
         asr_qc._collapse("un punto alto en el siguiente arco")
         == "un punto alto en el siguiente arco")


if __name__ == "__main__":
    test_intra_segment_thresholds()
    test_cross_segment_consecutive_duplicates()
    test_confirmed_hallucinations_flagged_and_crochet_counting_spared()
    test_second_opinion_recovers_or_collapses()
    test_real_parakeet_cross_check_on_real_hallucination_audio()
    test_grid_and_never_deletes()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("All checks passed.")
