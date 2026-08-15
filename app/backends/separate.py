"""Split the soundtrack into speech and everything else.

This is what makes dubbing work on real videos rather than just talking heads.
Without it, replacing the audio throws away the music and effects along with the
speech. With it, only the voices are swapped and the rest of the mix survives.

Uses Demucs (htdemucs_ft), run as a subprocess because its Python API changes
between releases while the command line stays stable.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

Progress = Optional[Callable[[float, str], None]]

MODEL = "htdemucs_ft"          # the fine-tuned variant: ~8.9 dB SDR vs 8.1 plain

# htdemucs_ft is a bag of four models run in sequence, so demucs counts 0-100%
# four separate times. Without knowing that, the progress bar slams back to zero
# three times during the stage.
MODEL_PASSES = {"htdemucs_ft": 4}


def available() -> bool:
    try:
        import demucs                                          # noqa: F401
        return True
    except ImportError:
        return False


def _device(prefer_gpu: bool) -> str:
    if not prefer_gpu:
        return "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:                                            # noqa: BLE001
        pass
    return "cpu"


def separate(audio: Path, workdir: Path, prefer_gpu: bool = True,
             progress: Progress = None) -> tuple[Path, Path] | None:
    """Returns (speech_wav, background_wav), or None if separation isn't possible.

    Falls back to None rather than raising: a video with no music loses nothing
    by skipping this, and a missing dependency shouldn't kill the job.
    """
    if not available():
        if progress:
            progress(1.0, "Skipping separation — Demucs isn't installed")
        return None

    out_root = workdir / "stems"
    speech = out_root / MODEL / audio.stem / "vocals.wav"
    background = out_root / MODEL / audio.stem / "no_vocals.wav"
    if speech.exists() and background.exists():
        return speech, background

    device = _device(prefer_gpu)
    if progress:
        where = {"mps": "the GPU", "cuda": "the GPU"}.get(device, "the processor")
        progress(0.05, f"Separating speech from music using {where}")

    cmd = [sys.executable, "-m", "demucs",
           "--two-stems", "vocals", "-n", MODEL,
           "-d", device, "-o", str(out_root), str(audio)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail: list[str] = []
    passes = MODEL_PASSES.get(MODEL, 1)
    done_passes, last_pct = 0, -1
    for line in proc.stdout:                                     # type: ignore[union-attr]
        tail.append(line)
        tail[:] = tail[-30:]
        # Demucs prints a percentage bar; surface it rather than looking stalled.
        if progress and "%" in line:
            digits = "".join(c for c in line.split("%")[0][-4:] if c.isdigit())
            if digits:
                pct = min(100, int(digits))
                if pct < last_pct:            # the bar restarted: next model in the bag
                    done_passes = min(done_passes + 1, passes - 1)
                last_pct = pct
                overall = (done_passes + pct / 100) / passes
                shown = min(99, int(overall * 100))
                progress(0.05 + 0.94 * overall, f"Separating speech from music — {shown}%")
    proc.wait()

    if proc.returncode != 0 or not speech.exists():
        if progress:
            progress(1.0, "Separation didn't work; carrying on with the full soundtrack")
        return None

    if progress:
        progress(1.0, "Speech separated from the soundtrack")
    return speech, background


def cleanup(workdir: Path) -> None:
    shutil.rmtree(workdir / "stems", ignore_errors=True)
