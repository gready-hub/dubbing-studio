"""Safety net for Whisper's repetition-loop hallucination, before translation
or diarization ever sees the transcript.

This is a known openai-whisper decoder failure (openai/whisper discussions
#679, #2442): once the model gets stuck it starts re-decoding its own last
output as though it were fresh audio, and `condition_on_previous_text=True`
(Whisper's own default) carries the bad context into the next 30-second
window too. On the real job this project exists to fix, one 30.00s segment —
Whisper's fixed internal decode-window length — came back as "like"×221 in
the finished English dub: about 8.6s of garbled audio, then 19s of dead air,
in place of a real sentence about YouTube's playback-speed control.

Run against the actual failing decode of the real 52-minute source (see
fixtures/README.md), the failure showed up in two shapes, not one:

  * intra-segment — a single segment's own words are almost entirely one
    repeated token or short phrase ("ellos" was the only word in four
    consecutive segments, 9-16 times each; "ya toda cadeneta" filled six
    segments the same way). A plain "top word share" catches the first kind
    but not the second, where no single token clears even a third of the
    segment — the loop is on a 3-word phrase, not a 1-word one — so this
    checks the phrase-level repetition too, via how few *distinct* words a
    segment has for its length.
  * cross-segment — several back-to-back segments carry the exact same text.
    "¿Vale?"×10 in 0.4s and "Y si no, pues no se vea"×5 in 0.42s are
    impossibly fast for real speech, but "¡Chao!"×3 spread across 36s of
    near-silence at the very end of the file is the opposite — each rep is
    30+ seconds apart, not fast at all — so a words-per-second gate alone
    would have missed it.

Scanning the *entire* file both ways — hallucinated
(`condition_on_previous_text=True`) and clean (`=False`) — turned up zero
runs of 3+ verbatim-identical consecutive segments anywhere in the clean
decode, and confirmed (by checking what the clean decode actually says at
the same timestamps) that every run of that length in the hallucinated
decode is fabricated, including three never previously documented ("4
cadenetas"×29, "1 punto bajo"×8, "y ya tenemos el último arco"×3 — all at a
perfectly ordinary 2-3.7 words/sec, which is exactly why a rate gate would
have cleared them). So a run of 3 is treated as decisive on its own here;
no rate test is layered on top of it, because on the one file this has ever
been checked against, a rate test would have hidden more real instances
than it excluded false ones.

avg_logprob / compression_ratio / no_speech_prob are read where present and
logged for whoever reads the log, but never used to decide a flag: Whisper
stamps every segment carved out of one ~30s decode window with the same
three numbers, so a real line that happens to share a window with a
hallucination looks exactly as suspicious by these fields as the
hallucination itself. Parakeet backends don't emit them at all, so nothing
here may require them.

A third shape was found later, by review, and it is the one that nearly got
away: 158 consecutive segments counting numbers, every one exactly 2.00
seconds long, 316 seconds of it, where the clean decode holds ordinary
instruction. No text test here could see it — consecutive lines differ by a
word, and each is two words long — and Whisper's own confidence numbers were
unremarkable across the whole stretch. What gives it away is the clock: real
speech does not land on one duration over and over. See GRID_RUN_MIN.

**Nothing in here deletes speech.** A flagged span either takes another
engine's reading of the same audio, or keeps its own words with the
repetition collapsed to a single instance. An earlier version silenced a
span when the second engine could not do better, which reads as prudent and
is the opposite: the second engine only contradicts a loop when the audio is
*not* looping, so on a false positive it returns the repetition faithfully,
is judged degenerate in its turn, and real speech is deleted on the strength
of two engines agreeing. That was not hypothetical — checked against real
audio, a genuine one-word goodbye held over 36 seconds was deleted exactly
so. Collapsing is worse than a good recovery and better than any deletion:
the failure that started all this, one word spoken 221 times, becomes that
word spoken once.
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

# ---- intra-segment: one segment dominated by one repeated token or phrase.
#
# Real numbers behind these, from full_video_default_decode.json: the four
# "ellos" segments (9-16 words each) are 100% one token, share=1.0; the six
# "ya toda cadeneta" segments (18-21 words) are only 33% any single token
# (three-word loop) but just 3 distinct words spread over 18-21, ratio
# 0.143-0.167. The highest either measure reaches anywhere else in either
# decode, at this same word-count floor, is share=0.38 and ratio=0.350 (real
# narration, not hallucination) — so both thresholds below sit in the middle
# of a real, wide, empty gap rather than close to either edge.
MIN_WORDS_INTRA = 6
SHARE_THRESH = 0.6
RATIO_THRESH = 0.25

# ---- cross-segment: a run of consecutive segments with identical text.
#
# "several", per the spec this module was built from — and long enough that
# a length-2 run (found a handful of times in the real decode, e.g. "y 96"
# said twice 4s apart) isn't enough on its own to tell a hallucination from
# someone repeating a number for emphasis. Every run of 3+ found in the real
# decode, by contrast, checked out as fabricated against the clean decode.
RUN_MIN = 3

# ---- the metronome: consecutive segments all landing on exactly one duration.
#
# The third shape, and the one that nearly got away. Neither test above can see
# it: the texts differ by a word ("y 65", "y 67", "y 68") so no run of identical
# lines forms, and each is two words long so the intra-segment floor never
# admits them. It is 158 consecutive segments of counting, every single one
# exactly 2.00 seconds, from 2685.84s to 3001.84s of the real source — 316
# seconds, five minutes of fabricated numbers that skip as they climb, where the
# clean decode of the same window holds ordinary instruction. It was 71% of the
# fabricated timeline in that file and every text-shaped test here missed all of
# it. Whisper's own numbers missed it too: median avg_logprob -0.24 and
# compression_ratio 1.57 over the stretch, nowhere near its thresholds. The
# model was confident and wrong.
#
# What gives it away is not the words but the clock. Real speech does not land
# on one duration over and over; when Whisper's timestamp machinery gives up it
# emits a metronome. Longest identical-duration run in the hallucinated decode:
# 226 segments. In the clean decode of the same audio: 2.
#
# Deliberately well above that 2, and above any run a real cadence might
# produce, because this fires on text that looks perfectly reasonable — the only
# evidence is the timing, so it has to be evidence nothing else explains. Note
# it is a *precursor* as much as a symptom: the grid starts at 2549.84s, about
# 136 seconds before the text goes bad, so the region it marks is wider than the
# damage and the words inside it are checked on their own merits.
GRID_RUN_MIN = 8
GRID_TOLERANCE_S = 0.02

# And the run has to be *gapless*, which is the half of this that stops it
# firing on real speech. A metronome is not merely regular, it is unbroken:
# all 158 gaps inside the real one measure exactly 0.0, because nothing is
# detecting where speech starts and stops any more — the boundaries are being
# generated, not found. Somebody genuinely counting at a steady pace leaves air
# between the numbers. This was not a hypothesis: without the gap test, the
# suite's own 26-segment "linea 0 / linea 1 / …" fixture — uniform 1.0s
# durations, near-identical short text, half a second of silence between each,
# which is exactly what real paced counting looks like — was flagged and
# collapsed in full. Uniform timing alone is not enough evidence.
GRID_MAX_GAP_S = 0.02

# A hallucination's own timestamps can be as unreliable as its text — the
# "¿Vale?"×10 run spans only 0.40s start-to-end, not enough runway to say a
# single "¿vale?" aloud in. Padding the audio handed to the second engine
# (never the span reported outward, which stays exactly what Whisper said)
# gives it a fair chance without reaching into a neighbour's own already-
# trustworthy audio.
PAD_S = 0.3

# Sanity for the second opinion, as a ceiling only. There was a floor here too
# — a minimum words-per-second — and it deleted real speech. The run test puts
# no bound on how long a flagged span may be, so the span is arbitrary, and a
# floor measured against it demands arbitrarily many words back: six at 36s,
# forty-five at five minutes. The real "¡Chao!" at the end of the source video
# is one word over 36.6 seconds, and the floor called the speaker's actual
# goodbye insane and dropped it. Only an implausibly *fast* answer says anything
# reliable, and only that is tested now.
MAX_WPS_SANE = 8.0
MIN_DURATION_FOR_RATE_CHECK = 1.0

# Below this there is not enough audio to be worth a second opinion. A flagged
# run's own timestamps can be nonsense — the "¿Vale?"×10 run is 0.40s end to
# end — and asking an engine about four tenths of a second gets an answer with
# nothing behind it: Parakeet really returns "Yeah." for that span, and
# "Well fine." for the next one, English words for Spanish audio, which then get
# written into the transcript as a recovery. Below the floor the repetition is
# collapsed and no engine is troubled for an opinion it cannot have.
MIN_SPAN_FOR_OPINION_S = 1.5

# How much a single flag may swallow. Touching units are merged, so without a
# cap a long stretch of separately-flagged segments becomes one unit, one
# re-transcribe, and one line standing in for minutes of video.
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

    # The metronome, taken one segment at a time rather than as a block. The
    # grid marks a region the recogniser lost its footing in, and it runs wider
    # than the damage — 136 seconds of the real one carries text that matches
    # the clean decode word for word. Flagging the whole run would throw that
    # away, so inside a grid each segment still has to look wrong on its own,
    # just against a threshold it can actually reach: two words of counting can
    # never trip a six-word floor, and does not have to here.
    for lo, hi in _grid_runs(segments):
        for i in range(lo, hi + 1):
            words = [w for w in (_normalize_word(w) for w in
                                 (segments[i].get("text") or "").split()) if w]
            if not words:
                continue
            neighbours = [_normalize_line(segments[k].get("text") or "")
                          for k in range(max(lo, i - 2), min(hi, i + 2) + 1)]
            # A short line inside a metronome, next to near-copies of itself:
            # "y 65" beside "y 67" beside "y 68". Judged on the words it shares
            # with its neighbours rather than on repetition inside itself, which
            # a two-word line cannot show.
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
        # Touching hits become one, so a single re-read covers a stretch that
        # tripped several tests — but only up to a point. Merging without a cap
        # lets one flag swallow minutes of video and stand in for all of it,
        # which is how a safety net turns into the biggest edit in the file.
        if units and lo <= units[-1]["hi"] + 1:
            grown_hi = max(units[-1]["hi"], hi)
            span = segments[grown_hi]["end"] - segments[units[-1]["lo"]]["start"]
            count = grown_hi - units[-1]["lo"] + 1
            if span <= MAX_UNIT_SPAN_S and count <= MAX_UNIT_SEGMENTS:
                units[-1]["hi"] = grown_hi
                units[-1]["reasons"].append(reason)
                units[-1]["info"].append(info)
                continue
            # The cap said no. The hit still has to start after everything the
            # previous unit already claimed, or the two overlap — and check()
            # walks these in order, replacing each span and moving a cursor past
            # it, so an overlap emits the same loop twice and a unit whose end
            # falls short of the cursor winds it *backwards* and re-emits the
            # segments behind it. On the real file that produced four overlapping
            # lines all reading "4 cadenetas." for one window: the loop this
            # exists to remove, delivered four times over.
            lo = units[-1]["hi"] + 1
            if lo > hi:
                continue                    # nothing of it left to claim
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

    Judged from three words up, not six. The floor on flagging exists so a real
    short interjection is never condemned on its own, which is the right caution
    when all that is known about a segment is its own text. Here the question is
    narrower — did the other engine do better than a loop — and "vale vale vale"
    answers no.

    Note what this is *not* used for any more. It used to decide whether to
    delete the line, and that was backwards: when the flag is a false positive
    the audio really is repetitive, so the second engine faithfully returns the
    repetition, this calls it degenerate, and real speech gets dropped on the
    strength of an engine agreeing with the recogniser. Failing here now means
    only that the original text stands.
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

    Asked as the invariant itself rather than as a proxy for it: every distinct
    word the span contained has to still be there afterwards. That is what
    separates a collapse from a deletion, and it is true of more than just
    identical lines — "ellos" sixteen times beside "ellos" fourteen times is two
    different strings holding one word between them, and keeping that word keeps
    all of it. A grid of "y 60", "y 61", "y 62" is not: collapsing to "y 60"
    keeps two words out of thirteen, and no amount of counting it as a repair
    makes the other eleven not deleted.
    """
    return _words_in(segments, lo, hi) <= {
        w for w in (_normalize_word(t) for t in collapsed.split()) if w}


def _collapse(text: str) -> str:
    """The repeated thing said once, keeping the original's spelling.

    What replaces a loop when no second opinion is available or preferable.
    Whisper said one thing over and over; this keeps the one thing. It is the
    conservative move in both directions — a real utterance repeated for
    emphasis survives as itself, and a fabricated loop shrinks to a word or two
    instead of filling half a minute.
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

    Runs on the raw list straight out of `asr_backend.transcribe()`, before
    diarization or `merge_adjacent()`: the latter would fuse a run of
    degenerate segments into one longer blob and hide the very timing
    signature — many short identical segments — that the cross-segment check
    looks for.

    **Nothing here ever deletes speech.** A flagged span ends up either carrying
    another engine's transcript of the same audio, or carrying its own text with
    the repetition collapsed to one instance. That is a deliberate retreat from
    an earlier version that silenced a span when the second engine could not do
    better, which sounded prudent and was not: the second engine only disagrees
    with a loop when the audio *is not* looping, so on a false positive it
    faithfully returns the repetition, gets called degenerate in turn, and real
    speech goes in the bin with a log line claiming nothing was there. Verified
    against real audio — a genuine one-word goodbye over 36 seconds was deleted
    exactly this way. Collapsing instead is worse than a perfect recovery and
    better than any deletion: the original disaster, one word spoken 221 times,
    becomes that word spoken once.
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
        # Said at 1.0, not at 0.99. The recogniser has already reported this
        # stage finished, and _stage()'s reporter turns whatever fraction it is
        # handed straight into job.overall with nothing holding it monotonic —
        # so a fraction below the one already sent walks the bar backwards. The
        # transcribing really is done; this is a second look at what came back,
        # and the message is what should change, not the position.
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

        # Worth asking a second engine at all? Not if the span is too short to
        # hold an answer, and not if Parakeet is the engine that was *asked* for
        # in the first place — checking Parakeet's own work with Parakeet gets
        # agreement rather than an opinion.
        #
        # This reads the requested engine, not the one that actually ran, and
        # those can differ: transcribe()'s ladder falls back through both
        # Whisper engines to Parakeet, so a "whisper" job on a machine where
        # Whisper will not load is second-guessed by the engine that produced
        # it. Left as is deliberately — knowing which rung answered means
        # changing what transcribe() returns, and every caller and test with it,
        # for a case where the cost is now merely a pointless re-transcribe: the
        # engine agrees, no reading is preferred, and the span is collapsed or
        # left alone exactly as it would have been. It cannot delete anything.
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
            # Every distinct word in the span survives, so this throws away
            # repetition and nothing else.
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
            # The segments say *different* things — a counting grid, where each
            # line is its own number. There is no single instance to keep, and
            # standing the first one in for all of them is a deletion however it
            # is counted: on the real file one 44-second segment reading "y 3"
            # replaced twenty-two distinct lines, under a report still promising
            # that nothing had been dropped. With no better reading to put here,
            # the honest move is to leave it exactly as the recogniser left it.
            # Fabricated counting left in place is a worse dub and a smaller
            # wrong than real speech thrown away, and it is the one this module
            # is allowed to make.
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
    # Said rather than quietly omitted: these are the places the app noticed
    # something wrong and could not improve on it, so the dub still carries
    # whatever was there. Somebody skimming the result deserves to know which
    # moments to spot-check, and a sentence that only lists successes reads as
    # if the whole thing were handled.
    if report.get("left_alone"):
        parts.append(f"{report['left_alone']} left as they were, having no "
                     "repetition to shorten and no better reading available")
    if not parts:
        return ""
    return ("The transcript repeated itself in places, and was patched: "
            + ", and ".join(parts) +
            ". Nothing was dropped. This is a known Whisper failure on long "
            "recordings, not a setting to change.")
