"""Reshape the recogniser's lines into units worth dubbing.

Speech recognition returns lines sized for reading, not speaking: on
conversational material that means many short lines butted hard together,
each becoming a slot a synthesised line must fit inside. Measured on a real
10.5-minute run: 113 lines, median slot 2.00s, 41 of 113 under 1.5s — Kokoro's
pace meant most needed compression and 15 didn't fit even at the cap, ending
4.7s of accumulated drift.

Joining consecutive same-speaker lines that run straight into each other gives
the translation room to breathe: on that run, 113 lines became 62, median slot
2.00s -> 3.60s, tight slots 41 -> 7.

Nothing is lost — the pair occupied one continuous stretch of speech, so the
merged line spans exactly the time used. Real pauses (this app's target
instructional material) are far wider than the threshold and untouched.
"""
from __future__ import annotations

MAX_GAP = 0.25          # silence between lines that still counts as continuous
MAX_SPAN = 12.0         # never build a slot longer than this

# How to combine each per-window decode diagnostic when lines merge, and which
# direction "worse" runs. These are stamped once per decode window, so a merged
# line can straddle several with no single value describing the whole.
#
# Worst-of-group is kept, not dropped: dropping was tried first and was wrong,
# because a repetition loop produces exactly the short, gapless lines this
# function fuses — a later check would then find these diagnostics present on
# every clean line but missing from the hallucinated ones. Keeping the worst
# keeps the merged line at least as suspect as its worst part, regardless of
# how many windows it spans.
_WORST = {"avg_logprob": min,           # lower is a less confident decode
          "compression_ratio": max,     # higher is more repetitive text
          "no_speech_prob": max}        # higher is less likely to be speech


def merge_adjacent(segments: list[dict], max_gap: float = MAX_GAP,
                   max_span: float = MAX_SPAN) -> list[dict]:
    """Join consecutive same-speaker lines separated by less than `max_gap`.

    Requires the speaker labels to have been applied already; without them every
    line looks like the same speaker and unrelated turns would be run together.
    """
    if not segments:
        return segments

    merged: list[dict] = []
    current = dict(segments[0])
    for seg in segments[1:]:
        gap = seg["start"] - current["end"]
        same_speaker = seg.get("speaker") == current.get("speaker")
        would_span = seg["end"] - current["start"]
        if gap <= max_gap and same_speaker and would_span <= max_span:
            current["end"] = seg["end"]
            current["text"] = f"{current['text'].rstrip()} {seg['text'].lstrip()}"
            for key, worse_of in _WORST.items():
                # Missing (Parakeet) or None (Whisper declining to say) is
                # skipped rather than compared — absence isn't a good value.
                known = [v for v in (current.get(key), seg.get(key)) if v is not None]
                if known:
                    current[key] = worse_of(known)
        else:
            merged.append(current)
            current = dict(seg)
    merged.append(current)

    # The ids are positional and the translator keys its replies off them.
    for n, seg in enumerate(merged):
        seg["i"] = n
    return merged
