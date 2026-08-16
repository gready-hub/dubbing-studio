"""A side channel for fallbacks that changed the result.

Every backend already reports progress through a callback, and a fallback that
changed the outcome was visible only as a progress message — which scrolls past
while the job runs and is gone by the time anyone reads the report. The
orchestrator can only write down the fallbacks it can *observe* from a return
value, which is why separation and cloning were recorded and the ASR and TTS
engines quietly falling back to their portable versions were not.

So the note travels on the channel that already reaches every layer. A backend
records one without knowing or caring whether anybody is collecting them.
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
