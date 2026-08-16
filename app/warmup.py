"""Fetch the models a preset needs before the first job, not during it.

The first dub stopped for several gigabytes of downloading in the middle of
whichever stage happened to need them, reported as a fraction of that stage's
progress. A progress bar that sits still for twenty minutes is indistinguishable
from a hang, and the one thing the user can see — the stage name — says
"Listening to the original", which is not what is happening.

Each backend says what *it* needs, through its own `prefetch()`, built from the
same fallback ladder it uses at run time. This module only decides which
backends a preset will reach. It used to reconstruct the per-backend model lists
itself, reaching into private helpers to do it, and the two copies promptly
disagreed — it fetched only the engine expected to win, so the fallbacks, which
are reached precisely when the primary is failing, still stalled the job.

Nothing here is required: every fetch may fail, and the pipeline still downloads
on demand exactly as before.

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
    # Whatever is actually selected, not a hardcoded guess — otherwise the one
    # preset that gets warmed is the one the user may not be using.
    preset = sys.argv[1] if len(sys.argv) > 1 else Settings.load().preset
    if preset not in PRESETS:
        preset = "balanced"                       # "custom" has no model list
    failed = warm(preset)
    if not failed:
        return 0
    print(f"  {len(failed)} of these will download on first use instead.", flush=True)
    # Non-zero so the installer can say so. Not fatal to the install: the caller
    # warns and carries on, and the app still fetches on demand exactly as
    # before. Returning 0 regardless meant the installer promised the models
    # were ready even when none of them had arrived.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
