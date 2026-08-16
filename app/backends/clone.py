"""Voice cloning with Chatterbox.

Given a few seconds of someone speaking, this produces new English speech in
their voice. For dubbing that means the dubbed track keeps the original
speaker's identity and delivery instead of substituting a stranger.

Chatterbox is MIT-licensed and about 0.5B parameters, so it is meaningfully
slower than Kokoro — it is offered as a quality option, never as the default.

Note on accent: cloning carries the speaker's accent across languages. That is
usually what you want, but for instructional content a neutral built-in voice is
often easier to follow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

Progress = Optional[Callable[[float, str], None]]


def available() -> bool:
    try:
        import chatterbox                                       # noqa: F401
        return True
    except ImportError:
        return False


def prefetch(progress: Progress = None) -> None:
    """Fetch the Chatterbox checkpoint before a job needs it.

    Warming up never fetched this at all, so the Best preset — the only one that
    clones, and the one with the largest download — still stopped mid-job for
    it, which is the exact stall the warm-up exists to remove.
    """
    from chatterbox.tts import ChatterboxTTS
    ChatterboxTTS.from_pretrained(device=_device())


def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:                                            # noqa: BLE001
        pass
    return "cpu"


class CloneTTS:
    """Same interface as the other TTS engines, plus per-speaker references."""

    name = "Cloned voices (Chatterbox)"

    def __init__(self, progress: Progress = None) -> None:
        from chatterbox.tts import ChatterboxTTS

        device = _device()
        if progress:
            where = "the GPU" if device in ("mps", "cuda") else "the processor"
            progress(0.0, f"Loading the cloning model on {where}")
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = int(getattr(self.model, "sr", 24000))
        self._refs: dict[int, str] = {}

    def add_reference(self, speaker: int, samples: np.ndarray, rate: int,
                      workdir: Path) -> None:
        """Store a reference clip for one speaker. Must be called before say()."""
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / f"ref_speaker_{speaker}.wav"
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.95          # cloning is sensitive to level
        sf.write(path, audio, rate)
        self._refs[speaker] = str(path)

    @property
    def speakers(self) -> list[int]:
        return sorted(self._refs)

    def say(self, text: str, voice: str = "", speed: float = 1.0, speaker: int = 0):
        ref = self._refs.get(speaker) or (next(iter(self._refs.values()), None))
        if ref is None:
            raise RuntimeError("No reference audio was captured for cloning.")
        wav = self.model.generate(text, audio_prompt_path=ref)
        audio = np.asarray(getattr(wav, "cpu", lambda: wav)(), dtype=np.float32).reshape(-1)
        return audio, self.sample_rate
