"""A side channel for fallbacks that changed the result.

Progress messages scroll past and are gone by the time anyone reads the report,
and the orchestrator can only record fallbacks it can observe from a return
value — which missed engines silently falling back to portable versions. Notes
travel on the same callback channel instead, written without the backend
knowing or caring whether anyone is collecting them.
"""
from __future__ import annotations

from typing import Callable, Optional

Progress = Optional[Callable[[float, str], None]]


def note(progress: Progress, message: str) -> None:
    """Record something the finished report should mention, if anyone is listening."""
    recorder = getattr(progress, "note", None)
    if recorder is not None:
        try:
            recorder(message)
        except Exception:                                        # noqa: BLE001
            pass                      # a note is never worth failing a job over


def info(message: str) -> dict:
    """Tag a note as information — an explanation or good news, not a fault.

    Anything else stays a bare string, which is what every note already was
    before this and what an old run's recorded notes still are; the done
    panel treats an untagged note as a warning, exactly as it always has.
    """
    return {"kind": "info", "text": message}
