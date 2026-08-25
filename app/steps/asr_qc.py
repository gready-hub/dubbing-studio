"""Safety net for Whisper's repetition-loop hallucination, before translation
or diarization ever sees the transcript.

Known openai-whisper decoder failure (openai/whisper discussions #679,
#2442): once stuck, the model re-decodes its own last output as fresh audio,
and `condition_on_previous_text=True` (Whisper's default) carries the bad
context into the next window too. Observed in production: one 30s segment
came back as "like"x221 in a finished dub.

Three failure shapes, found by scanning a real 52-minute source decoded both
hallucinated and clean (see fixtures/README.md):

  * intra-segment — a segment's own words are almost entirely one repeated
    token or short phrase. Top-word share alone misses a repeated *phrase*
    (no single token dominates), so this also checks distinct-word ratio.
  * cross-segment — several consecutive segments share identical text, at
    widely varying real-world rates (some far faster than speech, one
    spread 30+s apart), so a words-per-second gate alone can't catch all of
    it — see RUN_MIN.
  * grid/metronome — many consecutive segments of exactly the same duration,
    text differing (e.g. counting), where confidence scores stay
    unremarkable. Real speech doesn't land on one duration repeatedly — see
    GRID_RUN_MIN.

avg_logprob / compression_ratio / no_speech_prob are logged but never used to
flag: Whisper stamps every segment from one ~30s decode window with the same
three numbers, so a real line sharing a window with a hallucination looks
just as suspicious. Parakeet doesn't emit them at all.

**Nothing in here deletes speech.** A flagged span is either replaced by a
second engine's reading or collapsed to a single instance of the repeated
text. An earlier version silenced a span when the second engine couldn't do
better; that deleted real speech whenever the flag was a false positive,
because the second engine then faithfully returns the same "repetition" and
gets judged degenerate too. Collapsing is worse than a good recovery and
better than any deletion.
"""
from __future__ import annotations

import re
import tempfile
import wave
from pathlib import Path
from typing import Callable, Optional

from .. import logs
from ..backends.asr import _transcribe_mlx, _transcribe_onnx

Progress = Optional[Callable[[float, str], None]]

# Thresholds picked from real decode data: both measures sit in a wide gap
# between confirmed hallucinations (share up to 1.0, unique_ratio down to
# 0.14) and the worst real narration seen (share 0.38, ratio 0.350).
MIN_WORDS_INTRA = 6
SHARE_THRESH = 0.6
RATIO_THRESH = 0.25

# A length-2 run of identical text isn't enough to distinguish a hallucination
# from someone repeating a number for emphasis; every run of 3+ found in a
# real decode checked out as fabricated.
RUN_MIN = 3

# The metronome shape: a fixed-duration run whose *text* differs, so neither
# check above sees it. Confidence scores (avg_logprob, compression_ratio) are
# unremarkable across such a run, so timing alone is the tell. Set well above
# the longest identical-duration run seen in a clean decode (2), and below the
# shortest seen in a real hallucinated one (226). The grid can start before
# the text actually goes bad, so segments inside it are still judged on their
# own merits rather than flagged wholesale.
GRID_RUN_MIN = 8
GRID_TOLERANCE_S = 0.02

# The run also has to be gapless: a metronome's boundaries are generated, not
# detected, so consecutive segments touch exactly. Real paced counting leaves
# audible gaps — without this, a fixture of genuine steady-paced narration
# (uniform 1s segments, half-second gaps) was flagged as a false positive.
GRID_MAX_GAP_S = 0.02

# A hallucination's own timestamps can be too short to hold a spoken answer
# (a real run spanned 0.4s). Padding the audio sent to the second engine (not
# the span reported outward) gives it a fair chance without reaching into a
# neighbour's audio.
PAD_S = 0.3

# Ceiling only, deliberately with no floor: a minimum-words-per-second floor
# was tried and deleted real speech (a genuine one-word goodbye spoken slowly
# over 36s). Only an implausibly *fast* answer is treated as suspect.
MAX_WPS_SANE = 8.0
MIN_DURATION_FOR_RATE_CHECK = 1.0

# Below this, a flagged span's timestamps are too short to give a second
# engine anything real to answer (a 0.4s clip produced fluent nonsense in the
# wrong language). Skip the second opinion and collapse instead.
MIN_SPAN_FOR_OPINION_S = 1.5

# Cap on how much a single merged flag may swallow, so a long stretch of
# separately-flagged segments doesn't collapse into one re-transcribe
# standing in for minutes of video.
MAX_UNIT_SPAN_S = 45.0
MAX_UNIT_SEGMENTS = 24

_STALE_STATS = ("avg_logprob", "compression_ratio", "no_speech_prob")


def _normalize_word(word: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE).casefold()


def _is_degenerate(text: str, min_words: int = MIN_WORDS_INTRA) -> tuple[bool, dict]:
    """One segment's own words, checked for repetition two ways at once.

    share catches one token standing in for almost the whole segment
    ("ellos"). unique_ratio catches a short *phrase* looping instead of a
    single word ("ya toda cadeneta") — share alone stays low there (0.33,
    since the loop is three words wide) while unique_ratio does not (0.14-
    0.17, since only three distinct words fill 18-21 slots).
    """
    words = [w for w in (_normalize_word(w) for w in (text or "").split()) if w]
    n = len(words)
    if n < min_words:
        return False, {"n": n}
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    top_word, top_count = max(counts.items(), key=lambda kv: kv[1])
    share = top_count / n
    unique_ratio = len(counts) / n
    flagged = share >= SHARE_THRESH or unique_ratio <= RATIO_THRESH
    return flagged, {"n": n, "top_word": top_word, "share": round(share, 3),
                      "unique_ratio": round(unique_ratio, 3)}


def _normalize_line(text: str) -> str:
    """casefold + punctuation-insensitive, so "¿Vale?" and "vale" group together."""
    t = re.sub(r"[^\w\s]", "", (text or "").strip().casefold(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _grid_runs(segments: list[dict]) -> list[tuple[int, int]]:
    """Stretches where consecutive segments all share one duration, as indices.

    See GRID_RUN_MIN. Compared with a tolerance rather than for equality because
    the durations arrive as floats that have been through a rounding or two, and
    a metronome that ticks 2.0000001 on one segment is still a metronome.
    """
    runs, i, n = [], 0, len(segments)
    while i < n:
        j = i
        span = segments[i]["end"] - segments[i]["start"]
        while j + 1 < n and span > 0 and abs(
                (segments[j + 1]["end"] - segments[j + 1]["start"]) - span
                ) <= GRID_TOLERANCE_S and abs(
                segments[j + 1]["start"] - segments[j]["end"]) <= GRID_MAX_GAP_S:
            j += 1
        if span > 0 and j - i + 1 >= GRID_RUN_MIN:
            runs.append((i, j))
        i = j + 1
    return runs


def _find_flags(segments: list[dict]) -> list[dict]:
    """Every span worth a second opinion, as (lo, hi) indices into `segments`.

    Three independent passes — a segment can trip more than one, which is why
    overlapping and touching hits are merged before anything acts on them,
    rather than the same audio being sliced and cross-checked twice.
    """
    hits: list[tuple[int, int, str, dict]] = []

    for i, seg in enumerate(segments):
        degenerate, info = _is_degenerate(seg.get("text") or "")
        if degenerate:
            hits.append((i, i, "one segment repeats itself", info))

    # A grid run can be wider than the actual damage (real text preceding the
    # fabricated part), so each segment inside it is still judged on its own,
    # against a lower word-count floor than the intra-segment check uses.
    for lo, hi in _grid_runs(segments):
        for i in range(lo, hi + 1):
            words = [w for w in (_normalize_word(w) for w in
                                 (segments[i].get("text") or "").split()) if w]
            if not words:
                continue
            neighbours = [_normalize_line(segments[k].get("text") or "")
                          for k in range(max(lo, i - 2), min(hi, i + 2) + 1)]
            # A two-word line can't show internal repetition, so judge it by
            # words shared with its near-identical neighbours instead.
            shared = sum(1 for t in neighbours if t and words[0] in t.split())
            if len(words) <= 3 and shared >= 3:
                hits.append((i, i, "counting on a fixed grid, no two lines alike",
                             {"n": len(words), "grid_s": round(
                                 segments[i]["end"] - segments[i]["start"], 2)}))

    i, n = 0, len(segments)
    while i < n:
        j = i
        key = _normalize_line(segments[i].get("text") or "")
        while j + 1 < n and key and _normalize_line(segments[j + 1].get("text") or "") == key:
            j += 1
        run_len = j - i + 1
        if key and run_len >= RUN_MIN:
            span = segments[j]["end"] - segments[i]["start"]
            reps_words = len(key.split()) * run_len
            wps = reps_words / span if span > 0 else float("inf")
            hits.append((i, j, f"{run_len} consecutive identical lines",
                        {"run_len": run_len, "span_s": round(span, 2),
                         "implied_wps": round(wps, 1) if wps != float("inf") else wps}))
        i = j + 1

    hits.sort(key=lambda h: h[0])
    units: list[dict] = []
    for lo, hi, reason, info in hits:
        # Merge touching hits into one unit, capped so one flag can't swallow
        # minutes of video as a single re-transcribe.
        if units and lo <= units[-1]["hi"] + 1:
            grown_hi = max(units[-1]["hi"], hi)
            span = segments[grown_hi]["end"] - segments[units[-1]["lo"]]["start"]
            count = grown_hi - units[-1]["lo"] + 1
            if span <= MAX_UNIT_SPAN_S and count <= MAX_UNIT_SEGMENTS:
                units[-1]["hi"] = grown_hi
                units[-1]["reasons"].append(reason)
                units[-1]["info"].append(info)
                continue
            # Cap hit: clamp lo past what the previous unit already claimed.
            # check() walks units in order advancing a cursor, so leaving an
            # overlap here re-emits the same segments more than once.
            lo = units[-1]["hi"] + 1
            if lo > hi:
                continue
        units.append({"lo": lo, "hi": hi, "reasons": [reason], "info": [info]})
    return units


# --------------------------------------------------------- second opinion

def _slice_wav(src: Path, start: float, end: float, dst: Path) -> bool:
    """Copy [start, end) out of a wav by frames, not by re-encoding.

    speech16k.wav is already the exact format Parakeet wants, so this is a
    plain read/write rather than a second trip through ffmpeg.
    """
    try:
        with wave.open(str(src), "rb") as w:
            sr = w.getframerate()
            params = w.getparams()
            total = w.getnframes()
            start_frame = max(0, min(total, int(start * sr)))
            end_frame = max(start_frame, min(total, int(round(end * sr))))
            if end_frame <= start_frame:
                return False
            w.setpos(start_frame)
            frames = w.readframes(end_frame - start_frame)
    except (OSError, wave.Error):
        return False
    with wave.open(str(dst), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return True


def _parakeet_span(audio: Path, start: float, end: float, use_mlx: bool) -> str:
    """Ask the other engine what it hears in [start, end] of `audio`.

    No progress callback reaches the engine here: this runs inside the
    transcribe stage after it has already reported completion, and forwarding
    progress into a handful of few-second re-decodes would move the bar
    backwards as each one restarts near 0%.
    """
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "span.wav"
        if not _slice_wav(audio, start, end, clip):
            return ""
        try:
            engine = _transcribe_mlx if use_mlx else _transcribe_onnx
            out = engine(clip, None)
        except Exception:                                        # noqa: BLE001
            return ""
    return " ".join(s["text"] for s in out if (s.get("text") or "").strip())


def _is_sane(text: str, duration: float) -> bool:
    """Whether a second opinion is worth preferring to what is already there.

    Uses a lower word floor (3, not 6) than the flagging check: the question
    here is narrower — did the other engine do better than a loop.

    Failing here no longer deletes the line (it used to): on a false-positive
    flag the second engine faithfully returns the same "repetition" and would
    get real speech dropped. Failing now just leaves the original text.
    """
    text = (text or "").strip()
    if not text:
        return False
    degenerate, _ = _is_degenerate(text, min_words=3)
    if degenerate:
        return False
    # A ceiling only — see MAX_WPS_SANE. A slow answer is not a wrong one.
    if duration >= MIN_DURATION_FOR_RATE_CHECK:
        if len(text.split()) / duration > MAX_WPS_SANE:
            return False
    return True


def _words_in(segments: list[dict], lo: int, hi: int) -> set[str]:
    return {w for k in range(lo, hi + 1)
            for w in (_normalize_word(t) for t in
                      (segments[k].get("text") or "").split()) if w}


def _collapse_keeps_everything(segments: list[dict], lo: int, hi: int,
                               collapsed: str) -> bool:
    """Whether collapsing lo..hi to `collapsed` throws away only repetition.

    Tests the invariant directly: every distinct word in the span must
    survive. True for repeated identical text; false for a counting grid
    ("y 60", "y 61", "y 62"), where collapsing to the first line would drop
    the rest as data, not repetition.
    """
    return _words_in(segments, lo, hi) <= {
        w for w in (_normalize_word(t) for t in collapsed.split()) if w}


def _collapse(text: str) -> str:
    """The repeated thing said once, keeping the original's spelling.

    Conservative both ways: a real utterance repeated for emphasis survives
    as itself, and a fabricated loop shrinks to a word or two.
    """
    tokens = (text or "").split()
    if len(tokens) < 2:
        return (text or "").strip()
    norm = [_normalize_word(t) for t in tokens]
    n = len(tokens)
    for period in range(1, n // 2 + 1):
        unit = norm[:period]
        if all(norm[i] == unit[i % period] for i in range(n)):
            return " ".join(tokens[:period])
    # Not a clean cycle: drop runs of the same word where they butt together,
    # which is what a loop that drifted mid-phrase leaves behind.
    out = [tokens[0]]
    for i in range(1, n):
        if norm[i] != norm[i - 1]:
            out.append(tokens[i])
    return " ".join(out)


def check(segments: list[dict], audio: Path, use_mlx: bool,
         asr_model: str = "whisper", progress: Progress = None) -> dict:
    """Flag, cross-check, and repair — mutates `segments` in place.

    Must run before diarization or `merge_adjacent()`: merging would fuse a
    run of degenerate segments into one blob and hide the timing signature
    (many short identical segments) the cross-segment check looks for.

    **Nothing here ever deletes speech.** A flagged span ends up either
    carrying another engine's transcript, or its own text with the repetition
    collapsed to one instance — never silenced, since a false-positive flag
    means the second engine faithfully returns the same "repetition" too, and
    deleting on that agreement previously deleted real speech.
    """
    report = {"segments": len(segments), "flagged": 0, "recovered": 0,
              "collapsed": 0, "left_alone": 0, "skipped": 0}
    if not segments:
        return report

    units = _find_flags(segments)
    report["flagged"] = len(units)
    if not units:
        return report

    log = logs.get()
    if progress:
        # 1.0, not 0.99: the progress bar isn't held monotonic, so any lower
        # fraction here would walk it backwards after transcribe already
        # reported completion.
        progress(1.0, f"Double-checking {len(units)} suspect "
                      f"line{'s' if len(units) != 1 else ''}")

    new_segments: list[dict] = []
    cursor = 0
    for unit in units:
        lo, hi = unit["lo"], unit["hi"]
        new_segments.extend(segments[cursor:lo])

        start, end = segments[lo]["start"], segments[hi]["end"]
        left_bound = segments[lo - 1]["end"] if lo > 0 else 0.0
        right_bound = segments[hi + 1]["start"] if hi + 1 < len(segments) else end + PAD_S
        pad_start = max(left_bound, start - PAD_S)
        pad_end = min(right_bound, end + PAD_S)

        original = " / ".join(dict.fromkeys(
            (segments[k].get("text") or "").strip() for k in range(lo, hi + 1)))
        reason = ", ".join(dict.fromkeys(unit["reasons"]))
        base = {k: v for k, v in segments[lo].items() if k not in _STALE_STATS}

        # Skip the second engine if the span is too short, or if Parakeet is
        # the engine that was asked for (checking Parakeet with Parakeet gets
        # agreement, not an opinion). Note: this checks the *requested* engine,
        # not whichever one actually ran after transcribe()'s fallback ladder —
        # left this way because the only cost of getting it wrong is a
        # pointless re-transcribe that ends up collapsed or left alone anyway.
        span_ok = (pad_end - pad_start) >= MIN_SPAN_FOR_OPINION_S
        different_engine = asr_model != "parakeet"
        candidate = (_parakeet_span(audio, pad_start, pad_end, use_mlx)
                     if span_ok and different_engine else "")

        if _is_sane(candidate, end - start):
            report["recovered"] += 1
            log.warning("a looping transcript line was re-read by a second engine",
                       extra={"at": round(start, 1), "why": reason,
                              "was": original[:160], "now": candidate[:160]})
            new_segments.append({**base, "start": start, "end": end, "text": candidate})
        elif _collapse_keeps_everything(
                segments, lo, hi, _collapse(segments[lo].get("text") or "")):
            collapsed = _collapse(segments[lo].get("text") or "")
            report["collapsed"] += 1
            if not (span_ok and different_engine):
                report["skipped"] += 1
            log.warning("a looping transcript line was collapsed to one instance",
                       extra={"at": round(start, 1), "why": reason,
                              "was": original[:160], "now": collapsed[:160],
                              "asked_second_engine": bool(span_ok and different_engine)})
            new_segments.append({**base, "start": start, "end": end, "text": collapsed})
        else:
            # A counting grid: each line is its own distinct number, so there's
            # no single instance to collapse to without deleting the rest as
            # data. Leaving fabricated counting in place is a smaller wrong than
            # deleting real speech, and the only one this module is allowed.
            report["left_alone"] += 1
            log.warning("a suspect transcript span was left as it was: its lines "
                       "differ from each other, so there was nothing to collapse "
                       "and no second reading to prefer",
                       extra={"at": round(start, 1), "why": reason,
                              "lines": hi - lo + 1, "was": original[:160]})
            new_segments.extend(segments[lo:hi + 1])
        cursor = hi + 1
    new_segments.extend(segments[cursor:])
    segments[:] = new_segments

    if progress:
        progress(1.0, f"Transcribed {len(segments)} lines")
    return report


def summarise(report: dict) -> str:
    """One sentence for the finished-job report, or nothing when all is well."""
    if not report.get("flagged"):
        return ""
    parts = []
    if report.get("recovered"):
        parts.append(f"{report['recovered']} re-read with a second speech engine")
    if report.get("collapsed"):
        parts.append(f"{report['collapsed']} shortened to a single instance of "
                     "what was being repeated")
    # Named explicitly rather than omitted, so a reader knows which moments
    # still need a spot-check rather than assuming everything was fixed.
    if report.get("left_alone"):
        parts.append(f"{report['left_alone']} left as they were, having no "
                     "repetition to shorten and no better reading available")
    if not parts:
        return ""
    return ("The transcript repeated itself in places, and was patched: "
            + ", and ".join(parts) +
            ". Nothing was dropped. This is a known Whisper failure on long "
            "recordings, not a setting to change.")
