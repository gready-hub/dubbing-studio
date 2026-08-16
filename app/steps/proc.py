"""Run a subprocess and stream its output, one line at a time.

yt-dlp and Demucs are both driven this way: spawn, read stdout line by line,
pull a percentage out of it and forward that to the progress callback. Both had
their own copy of the loop, and the copies had already drifted in buffering and
in how much output they kept for the error message.

The part that matters is the ending. The progress callback is how a cancel
reaches these stages — it raises, unwinding the job — and neither copy told the
child. A cancelled separation carried on holding the GPU for the rest of its
four-model pass, and a cancelled download carried on pulling the video. Here
that property holds by construction rather than by remembering.
"""
from __future__ import annotations

import subprocess
from typing import Callable, Optional

OnLine = Optional[Callable[[str], None]]


def stream(cmd: list[str], on_line: OnLine = None, tail_lines: int = 30) -> tuple[int, str]:
    """Run `cmd`, calling `on_line` per line of output.

    Returns (returncode, tail) where tail is the last `tail_lines` of output,
    kept for building an error message. If `on_line` raises — which is how a
    cancel arrives — the child is killed and reaped before the exception
    continues on its way.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail: list[str] = []
    try:
        for line in proc.stdout:                                 # type: ignore[union-attr]
            tail.append(line)
            del tail[:-tail_lines]
            if on_line is not None:
                on_line(line)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    proc.wait()
    return proc.returncode, "".join(tail)
