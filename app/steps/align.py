"""Fit synthesised lines into the timeline of the original speech.

The rule is simple: a line starts where the original speaker started, unless the
previous line is still going, in which case it starts as soon as that finishes.
If a line is too long for the gap before the next one, it is compressed with
ffmpeg's atempo filter, which shortens audio without shifting pitch.

Compression is capped (default 1.55x) because past roughly 1.6x it starts to
sound hurried. Beyond the cap the line is allowed to run over and the following
lines absorb it in their natural pauses.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

Progress = Optional[Callable[[float, str], None]]

FADE_S = 0.015          # de-click ramp on each line
BREATH_S = 0.06         # gap left before the next line starts
MIN_SLOT_S = 0.35


def _atempo_chain(factor: float) -> str:
    """atempo only accepts 0.5-2.0 per instance, so chain them for bigger factors."""
    stages, remaining = [], factor
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    stages.append(remaining)
    return ",".join(f"atempo={s:.4f}" for s in stages)


def compress(samples: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    if factor <= 1.001:
        return samples
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(src, samples, sample_rate)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                        "-filter:a", _atempo_chain(factor), str(dst)], check=True)
        out, _ = sf.read(dst, dtype="float32")
    return out


def assemble(lines: list[dict], total_duration: float, sample_rate: int,
             max_stretch: float = 1.55, progress: Progress = None) -> tuple[np.ndarray, dict]:
    """lines: [{"start", "end", "samples"}] sorted by start. Returns (track, stats)."""
    track = np.zeros(int(total_duration * sample_rate) + sample_rate, dtype=np.float32)
    stats = {"lines": len(lines), "compressed": 0, "max_factor": 1.0,
             "over_cap": 0, "max_drift": 0.0}

    cursor = 0.0
    for n, line in enumerate(lines):
        audio = np.asarray(line["samples"], dtype=np.float32).reshape(-1)
        if audio.size == 0:
            continue

        start = max(line["start"], cursor)
        stats["max_drift"] = max(stats["max_drift"], start - line["start"])

        next_start = lines[n + 1]["start"] if n + 1 < len(lines) else total_duration
        slot = max(MIN_SLOT_S, next_start - start - BREATH_S)
        length = len(audio) / sample_rate

        if length > slot:
            factor = length / slot
            if factor > max_stretch:
                stats["over_cap"] += 1
                factor = max_stretch
            audio = compress(audio, sample_rate, factor)
            stats["compressed"] += 1
            stats["max_factor"] = max(stats["max_factor"], factor)

        fade = int(FADE_S * sample_rate)
        if len(audio) > 2 * fade:
            audio[:fade] *= np.linspace(0, 1, fade)
            audio[-fade:] *= np.linspace(1, 0, fade)

        at = int(start * sample_rate)
        if at + len(audio) > len(track):
            audio = audio[:max(0, len(track) - at)]
        if audio.size:
            track[at:at + len(audio)] += audio
        cursor = start + len(audio) / sample_rate

        if progress and n % 25 == 0:
            progress(n / max(1, len(lines)), f"Fitting line {n} of {len(lines)}")

    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0:
        track = track / peak * 0.89
    stats["max_factor"] = round(stats["max_factor"], 3)
    stats["max_drift"] = round(stats["max_drift"], 3)
    return track, stats


def write_srt(segments: list[dict], dst: Path) -> Path:
    def stamp(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out = []
    for n, seg in enumerate(segments, 1):
        text = seg.get("translation") or seg.get("text", "")
        if not text:
            continue
        out.append(f"{n}\n{stamp(seg['start'])} --> {stamp(seg['end'])}\n{text}\n")
    dst.write_text("\n".join(out), encoding="utf-8")
    return dst
