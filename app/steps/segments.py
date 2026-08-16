"""Reshape the recogniser's lines into units worth dubbing.

Speech recognition returns lines sized for reading, not for speaking. On
conversational material that means a lot of very short ones butted hard against
each other, and each becomes a slot that a synthesised line has to fit inside.
Measured on a real 10.5-minute run: 113 lines, median slot 2.00s, median gap to
the next line 0.08s, and 41 of the 113 slots under 1.5s. Kokoro speaks at a
measured pace, so most lines had to be compressed and 15 could not fit even at
the cap — the run ended with 61 of 113 compressed and 4.7s of accumulated drift.

Joining consecutive lines from one speaker that run straight into each other
gives the translation room to breathe. On that same run: 113 lines become 62,
the median slot goes 2.00s → 3.60s, and the tight slots drop from 41 to 7.

Nothing is lost. The pair occupied one continuous stretch of speech to begin
with, so the merged line spans exactly the time the original speaker used, and a
video with real pauses — the instructional material this app was built for —
has gaps far wider than the threshold and is left alone entirely.
"""
from __future__ import annotations

MAX_GAP = 0.25          # silence between lines that still counts as continuous
MAX_SPAN = 12.0         # never build a slot longer than this


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
        else:
            merged.append(current)
            current = dict(seg)
    merged.append(current)

    # The ids are positional and the translator keys its replies off them.
    for n, seg in enumerate(merged):
        seg["i"] = n
    return merged
