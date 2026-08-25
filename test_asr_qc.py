"""Tests for app/steps/asr_qc.py — the pre-translation repetition-loop safety net.

Fixture segments are taken verbatim (text and timing) from the real
mlx_whisper dumps under
~/.claude/scratch/dubbing-studio-asr-hallucination/fixtures/ —
full_video_default_decode.json (condition_on_previous_text=True, hallucinated)
and full_video_nocond_decode.json (=False, clean). Where timestamps are
shifted to start near 0 (second-opinion tests, which need a real backing wav),
the comment says so; words and relative durations are otherwise untouched.
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

# Same dance as test_pipeline.py / test_upgrade.py: isolated scratch home,
# real model cache symlinked in so the real-engine test below doesn't re-fetch.
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
    """Backing wav for tests where the engine call is monkeypatched — only the
    framing (duration, format) needs to be real, not the content."""
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)
    return path


# ============================================ 1. intra-segment thresholds
def test_intra_segment_thresholds():
    """Checked against real numbers on both sides of the gap, not round ones
    picked in advance."""
    print("\n[1] Intra-segment repetition thresholds (real numbers)")
    from app.steps.asr_qc import _is_degenerate

    # The four "ellos" segments, 384.54-413.22s in full_video_default_decode.json
    # (word counts 16/14/13/9) — never reproduced in the clean decode.
    for count in (16, 14, 13, 9):
        text = " ".join(["ellos"] * count)
        flagged, info = _is_degenerate(text)
        check(f"'ellos'x{count} (share={info.get('share')}) is flagged", flagged, info)

    # "ya toda cadeneta" (1942.48-1966.50s, same file): a 3-word loop keeps
    # share at 0.33 (never clears 40%), so only unique_ratio catches it.
    for reps in (6, 7):
        text = " ".join(["ya", "toda", "cadeneta"] * reps)
        flagged, info = _is_degenerate(text)
        check(f"'ya toda cadeneta'x{reps} (unique_ratio={info.get('unique_ratio')}) "
             "is flagged", flagged, info)

    # Real lines from the same two files, at/above the word-count floor: the
    # highest share / lowest unique_ratio found once hallucination windows are
    # excluded. Must not trip either measure.
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
    """Consecutive raw segments repeating the same text — each individual
    segment is too short to trip check 1 on its own."""
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

    # "¡Chao!"x3 spread across 36s of near-silence, end of file — far apart in
    # time but still consecutive in index terms (no other segment between
    # them). Opposite extreme from Vale (slow, not fast) — why detection here
    # doesn't gate on words-per-second.
    chao = [
        {"start": 3088.60, "end": 3091.22, "text": "Bueno, chicas, nada, un saludo y que os veo en el próximo."},
        {"start": 3091.64, "end": 3091.98, "text": "¡Chao!"},
        {"start": 3118.60, "end": 3119.60, "text": "¡Chao!"},
        {"start": 3125.64, "end": 3127.64, "text": "¡Chao!"},
    ]
    units = _find_flags(chao)
    check("the 3-rep 'Chao' run is found despite the 27s+7s gaps inside it",
         len(units) == 1 and (units[0]["lo"], units[0]["hi"]) == (1, 3), units)

    # full_video_nocond_decode.json, 389.84-429.16s: "1, 2 y 3 cadenetas" and
    # "punto alto" each occur twice here, but never back-to-back — must NOT
    # form a run.
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
    """The four hallucinations spec.md names as confirmed, run through the
    real check() (second opinion stubbed, for speed) and asserted flagged —
    alongside the real crochet-counting lines from the neighbouring span,
    asserted untouched."""
    print("\n[3] The four confirmed hallucinations, flagged; crochet counting, spared")
    from app.steps import asr_qc

    real_ellos = [16, 14, 13, 9]
    segments = [
        {"start": 380.22, "end": 384.42,
         "text": "labor vamos a empezarla de una forma muy especial a mí me gusta muchísimo la verdad así creo que"},
        *[{"start": s, "end": e, "text": " ".join(["ellos"] * n)} for s, e, n in [
            (384.54, 389.20, 16), (389.20, 403.36, 14), (403.36, 409.06, 13), (409.06, 413.22, 9)]],
        {"start": 414.54, "end": 418.32, "text": "Bueno, pues desde esos dos hilos saco y hago un punto alto."},

        # Real crochet-counting line that must survive untouched — the
        # false-positive case this module exists not to break.
        {"start": 418.34, "end": 429.16,
         "text": "¿Vale? Mirad, 1, 2 y 3, echo hebra y desde aquí el primer puntito, un punto alto."},

        {"start": 145.94, "end": 147.88, "text": "Eso ya dependerá de vosotras"},
        *[{"start": round(147.88 + i * 0.04, 2), "end": round(147.92 + i * 0.04, 2),
          "text": "¿Vale?"} for i in range(10)],
        {"start": 148.28, "end": 150.48, "text": "Esta es una lana stop perlé número 5"},

        {"start": 266.20, "end": 266.96, "text": "Porque si no, para la gente que no se vea"},
        {"start": 266.96, "end": 267.06, "text": "Pues no se vea"},
        {"start": 267.06, "end": 267.10, "text": "Y si no, pues no se vea"},
        {"start": 267.10, "end": 267.18, "text": "Y si no, pues no se vea"},
        {"start": 267.18, "end": 267.20, "text": "Y si no, pues no se vea"},
        {"start": 267.20, "end": 267.26, "text": "Y si no, pues no se vea"},
        {"start": 267.26, "end": 267.48, "text": "Y si no, pues no se vea"},
        {"start": 267.50, "end": 268.34, "text": "Que sabe hacer"},

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

    # Stubbed, not the real engine — this test is about detection and survival
    # of real content, not what a particular ASR says back (see test 5).
    real_mlx, real_onnx = asr_qc._transcribe_mlx, asr_qc._transcribe_onnx
    asr_qc._transcribe_mlx = lambda path, progress=None: []
    asr_qc._transcribe_onnx = lambda path, progress=None: []
    try:
        report = asr_qc.check(segments, audio, True)
    finally:
        asr_qc._transcribe_mlx, asr_qc._transcribe_onnx = real_mlx, real_onnx

    check("all four confirmed hallucinations are found (one unit each)",
         report["flagged"] == 4, report)
    # The engine offering nothing used to delete the line; now it keeps one
    # instance of whatever was repeated, so "silenced" never appears.
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
    """A sane second opinion replaces a flagged span's text; a degenerate one
    leaves the span's own words collapsed to one instance. Neither deletes.
    The stub answers by call order, since check() processes flagged units in
    ascending start order."""
    print("\n[4] Second opinion: re-read one, collapse the other, delete neither")
    from app.steps import asr_qc

    # Same "ellos" shape as above, timestamps shifted to start near 0 so the
    # backing wav is a few seconds instead of ~400s; words/durations unchanged.
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
            # Sane second opinion for the "ellos" span: plausible word count
            # for ~28.7s, no repetition.
            return [{"start": 0.0, "end": 1.0,
                     "text": "Vamos a ver cómo quedan estos hilos en la labor."}]
        # Degenerate second opinion for the "Vale" span: Parakeet finds the
        # same kind of nothing Whisper did.
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

    # The Vale span used to be deleted on exactly this evidence — a second
    # engine agreeing the audio is repetitive isn't evidence it's empty. It
    # keeps one "¿Vale?" instead.
    kept = [s for s in segments if s["start"] == 40.00]
    check("the Vale span keeps one instance rather than being emptied",
         len(kept) == 1 and kept[0]["text"].strip() == "¿Vale?", kept)

    untouched = [s["text"] for s in segments if s["start"] in (0.00, 34.32)]
    check("the two real lines either side were never touched",
         "labor vamos a empezarla" in " ".join(untouched)
         and "Bueno, pues desde" in " ".join(untouched), untouched)

    # A second opinion can be too short to reach the flagging floor yet still
    # be obvious nonsense; _is_sane checks from three words up. A genuinely
    # short answer must still be allowed through.
    from app.steps.asr_qc import _is_sane
    check("a short but repetitive second opinion is not preferred",
         not _is_sane("vale vale vale", 2.0))
    check("while a short real one still is",
         _is_sane("un punto alto", 2.0))
    # A words-per-second floor used to live here and deleted real speech: a
    # flagged run's span is unbounded, so it demanded arbitrarily many words
    # back — the real "¡Chao!" (one word over 36.6s) failed it. Only an
    # implausibly fast answer is rejected now.
    check("one real word over a long span is a fine answer",
         _is_sane("Ciao.", 36.6))
    check("and an impossibly fast one is still not",
         not _is_sane(" ".join(f"w{i}" for i in range(60)), 2.0))


# ============================================ 5. the real Parakeet cross-check
def test_real_parakeet_cross_check_on_real_hallucination_audio():
    """Calls the actual backend against real audio from the "¿Vale?"
    hallucination window and checks its answer isn't itself degenerate — the
    empirical claim behind the design, not just the plumbing a stub proves."""
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
    """Regression guard: the first version of this module missed a 158-segment
    fixed-2.00s-grid hallucination (71% of the fabricated timeline) because the
    texts differ word-to-word and each is too short for the intra-segment
    floor. Run against the real dumps, not hand-typed timestamps, since the
    signal under test is the timing itself."""
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

    # "Nothing was dropped" as arithmetic, not a comment's promise: with no
    # second engine, every distinct word must come back out — collapsing may
    # remove repeats of a word, never its only copy. Regression: a unit whose
    # lines each named a different number was being collapsed to the first
    # line, silently losing the rest.
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

    # check() walks units in order past a cursor, so an overlapping unit would
    # re-emit already-consumed lines. Regression: the span cap once produced
    # four overlapping units all reading "4 cadenetas." for one window.
    units = asr_qc._find_flags(d)
    check("units never overlap and never run backwards",
         all(b["lo"] > a["hi"] and b["hi"] >= a["hi"]
             for a, b in zip(units, units[1:])),
         [(a["lo"], a["hi"], b["lo"], b["hi"]) for a, b in zip(units, units[1:])
          if not (b["lo"] > a["hi"] and b["hi"] >= a["hi"])][:3])
    dupes = [s["start"] for s in segs if sum(
        1 for t in segs if t["start"] == s["start"]) > 1]
    check("and no two output segments start at the same moment", not dupes, dupes[:5])

    # Uniform timing alone must not be enough: real counting leaves air
    # between numbers, a generated grid runs gapless. Both fixtures below were
    # wrongly flagged before the gap test existed.
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
