"""Fetch the models a preset needs before the first job, not during it.

Otherwise gigabytes downloaded mid-stage show as stalled progress under a
stage name ("Transcribing") that doesn't explain the wait.

Each backend says what it needs via its own `prefetch()`, built from the same
fallback ladder used at run time — this module only decides which backends a
preset reaches. (A prior version reconstructed per-backend model lists itself
and only fetched the primary engine, so fallbacks still stalled the job.)

Nothing here is required: any fetch may fail and the pipeline still downloads
on demand as before.

    python -m app.warmup [preset]
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
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

    @contextmanager
    def attempt(label: str):
        on_step(f"{label}…")
        try:
            yield
        except Exception as exc:                                 # noqa: BLE001
            failed.append(label)
            on_step(f"{label} — couldn't fetch it now ({type(exc).__name__})")

    with attempt("Speech recognition"):
        from .backends import asr
        asr.prefetch(machine.fast_path, settings.asr_model)

    with attempt("Voices"):
        from .backends import tts
        tts.prefetch(machine.fast_path)

    if settings.diarize:
        with attempt("Telling speakers apart"):
            from .backends import diarize
            diarize.prefetch()

    if settings.separate_audio:
        with attempt("Separating speech from music"):
            from .backends import separate
            separate.prefetch()

    if settings.voice_mode == "clone":
        with attempt("Cloning the original voices"):
            from .backends import clone
            clone.prefetch()

    return failed


def main() -> int:
    # The preset actually in use, not a hardcoded guess.
    preset = sys.argv[1] if len(sys.argv) > 1 else Settings.load().preset
    if preset not in PRESETS:
        preset = "balanced"                       # "custom" has no model list
    failed = warm(preset)
    if not failed:
        return 0
    print(f"  {len(failed)} of these will download on first use instead.", flush=True)
    # Non-zero so the installer can warn, but not fatal — it carries on, and
    # the app still fetches on demand as before.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
