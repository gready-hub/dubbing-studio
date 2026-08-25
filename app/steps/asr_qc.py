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

On a flag, nothing is silenced outright. The flagged span is re-transcribed
with Parakeet — a transducer model, structurally not prone to this
particular decoder failure — and the segment's text is only replaced with
Parakeet's answer if that answer itself looks sane; only when Parakeet's own
attempt is equally degenerate (or empty) does the segment go silent. This
mirrors qc.py's "silence is a safer failure than garbage" rule, but applied
one engine later: deleting a real stitch-count outright is a worse mistake
on this content than re-transcribing it in a different engine's words.
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

# A hallucination's own timestamps can be as unreliable as its text — the
# "¿Vale?"×10 run spans only 0.40s start-to-end, not enough runway to say a
# single "¿vale?" aloud in. Padding the audio handed to the second engine
# (never the span reported outward, which stays exactly what Whisper said)
# gives it a fair chance without reaching into a neighbour's own already-
# trustworthy audio.
PAD_S = 0.3

# Sanity band for Parakeet's second opinion: generous enough that no real
# line in either decode of the source video falls outside it (the fastest
# real, non-hallucinated stretch found was ~7.5 words/sec; short flagged
# spans given only a little padding are the main reason the floor is this
# forgiving rather than tuned to a "normal" speaking pace).
MIN_WPS_SANE = 0.15
MAX_WPS_SANE = 8.0
# Below this, a span's own duration is too short for words-per-second to mean
# anything (a single word can legitimately "clock" at 20+ wps in a 0.1s slice
# once padding is added) — sanity then rests on non-degeneracy alone.
MIN_DURATION_FOR_RATE_CHECK = 1.0

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


def _find_flags(segments: list[dict]) -> list[dict]:
    """Every span worth a second opinion, as (lo, hi) indices into `segments`.

    Two independent passes — a segment can trip either, or in principle both,
    which is why overlapping/touching hits are merged before anything acts on
    them, rather than the same audio being sliced and cross-checked twice.
    """
    hits: list[tuple[int, int, str, dict]] = []

    for i, seg in enumerate(segments):
        degenerate, info = _is_degenerate(seg.get("text") or "")
        if degenerate:
            hits.append((i, i, "one segment repeats itself", info))

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
        if units and lo <= units[-1]["hi"] + 1:
            units[-1]["hi"] = max(units[-1]["hi"], hi)
            units[-1]["reasons"].append(reason)
            units[-1]["info"].append(info)
        else:
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
    text = (text or "").strip()
    if not text:
        return False
    # Judged from three words up, not six. The floor above exists so that a
    # real short interjection is never *flagged* on its own, which is the right
    # caution when the only thing known about a segment is its own text. Here
    # the caution runs the other way: Whisper has already produced garbage over
    # this span, and the only question left is whether the second engine did
    # any better. At the six-word floor "vale vale vale" answers that question
    # with a no and still counts as a recovery, which puts a smaller piece of
    # the same nonsense into the dub instead of the silence it had earned.
    degenerate, _ = _is_degenerate(text, min_words=3)
    if degenerate:
        return False
    if duration >= MIN_DURATION_FOR_RATE_CHECK:
        wps = len(text.split()) / duration
        if not (MIN_WPS_SANE <= wps <= MAX_WPS_SANE):
            return False
    return True


def check(segments: list[dict], audio: Path, use_mlx: bool,
         progress: Progress = None) -> dict:
    """Flag, cross-check, and repair or silence — mutates `segments` in place.

    Runs on the raw list straight out of `asr_backend.transcribe()`, before
    diarization or `merge_adjacent()`: the latter would fuse a run of
    degenerate segments into one longer blob and hide the very timing
    signature — many short identical segments — that the cross-segment check
    looks for.
    """
    report = {"segments": len(segments), "flagged": 0, "recovered": 0, "silenced": 0}
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

        candidate = _parakeet_span(audio, pad_start, pad_end, use_mlx)
        base = {k: v for k, v in segments[lo].items() if k not in _STALE_STATS}

        if _is_sane(candidate, end - start):
            report["recovered"] += 1
            log.warning("ASR repetition-loop segment recovered with a second engine",
                       extra={"at": round(start, 1), "why": reason,
                              "was": original[:160], "now": candidate[:160]})
            new_segments.append({**base, "start": start, "end": end, "text": candidate})
        else:
            report["silenced"] += 1
            log.warning("ASR repetition-loop segment silenced; the second engine "
                       "found nothing usable there either",
                       extra={"at": round(start, 1), "why": reason, "was": original[:160]})
            new_segments.append({**base, "start": start, "end": end, "text": ""})
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
    if report["recovered"]:
        parts.append(f"{report['recovered']} where Whisper looped on its own words, "
                     "fixed by re-transcribing that moment with a second speech engine")
    if report["silenced"]:
        parts.append(f"{report['silenced']} where the second engine could not find "
                     "real speech either, left silent")
    if not parts:
        return ""
    return ("The transcript needed patching: " + ", and ".join(parts) +
            ". This is a known Whisper failure on long recordings, not a setting to change.")
