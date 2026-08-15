"""Fetch the models a preset needs before the first job, not during it.

The first dub stopped for several gigabytes of downloading in the middle of
whichever stage happened to need them, reported as a fraction of that stage's
progress. A progress bar that sits still for twenty minutes is indistinguishable
from a hang, and the one thing the user can see — the stage name — says
"Listening to the original", which is not what is happening.

Run at the end of installation, so the wait happens once, while the installer is
already visibly downloading things, and the first real job goes straight to
work. Nothing here is required: every fetch is allowed to fail, and the pipeline
still downloads on demand exactly as before.

    python -m app.warmup [preset]
"""
from __future__ import annotations

import sys
from typing import Callable

from .config import PRESETS, Settings, detect_machine

Step = Callable[[str], None]


def _say(message: str) -> None:
    print(f"  {message}", flush=True)


def warm(preset: str = "balanced", on_step: Step = _say) -> list[str]:
    """Fetch what `preset` will need. Returns the names of anything that failed."""
    settings = Settings().apply_preset(preset)
    machine = detect_machine()
    failed: list[str] = []

    def attempt(label: str, fn) -> None:
        on_step(f"{label}…")
        try:
            fn()
        except Exception as exc:                                 # noqa: BLE001
            failed.append(label)
            on_step(f"{label} — couldn't fetch it now ({type(exc).__name__})")

    if machine.fast_path:
        # The Apple-GPU path is what this machine will actually use, so fetch
        # those rather than the portable fallbacks it will never reach.
        def parakeet() -> None:
            from parakeet_mlx import from_pretrained
            from .backends.asr import MLX_MODEL
            from_pretrained(MLX_MODEL)

        def kokoro_mlx() -> None:
            from mlx_audio.tts.utils import load_model
            from .backends.tts import MLX_TTS_MODEL
            load_model(MLX_TTS_MODEL)

        if settings.asr_model == "parakeet":
            attempt("Speech recognition (Parakeet)", parakeet)
        attempt("Voice (Kokoro)", kokoro_mlx)
    else:
        def onnx_asr() -> None:
            from .backends.asr import _ensure_onnx_models
            _ensure_onnx_models()

        def onnx_tts() -> None:
            from .backends.tts import OnnxTTS
            OnnxTTS()

        attempt("Speech recognition (Parakeet)", onnx_asr)
        attempt("Voice (Kokoro)", onnx_tts)

    if settings.asr_model == "whisper":
        def whisper() -> None:
            if machine.fast_path:
                import mlx_whisper                               # noqa: F401
                from huggingface_hub import snapshot_download
                from .backends.asr import WHISPER_MLX
                snapshot_download(WHISPER_MLX)
            else:
                from faster_whisper import WhisperModel
                WhisperModel("large-v3", device="cpu", compute_type="int8")

        attempt("Whisper transcription (about 3 GB)", whisper)

    if settings.diarize:
        def diar() -> None:
            from .backends.diarize import _ensure_models
            _ensure_models()

        attempt("Telling speakers apart", diar)

    if settings.separate_audio:
        def demucs_weights() -> None:
            from demucs.pretrained import get_model
            from .backends.separate import MODEL
            get_model(MODEL)

        attempt("Separating speech from music (Demucs)", demucs_weights)

    return failed


def main() -> int:
    preset = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    if preset not in PRESETS:
        preset = "balanced"
    failed = warm(preset)
    if failed:
        print(f"  {len(failed)} model(s) will download on first use instead.",
              flush=True)
    return 0                      # never fatal: the app fetches on demand anyway


if __name__ == "__main__":
    raise SystemExit(main())
