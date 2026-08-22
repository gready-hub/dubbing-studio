"""One log file, in JSON, for whoever has to work out what went wrong.

What was here before was not a log. The app bundle redirected stderr to
`~/Library/Logs/DubbingStudio.log` and nothing else ever wrote to it, so on a
machine that had dubbed real videos the file held twenty-two lines: the same
`pkg_resources is deprecated` warning over and over, no timestamps, no job ids,
no errors. The README told people to send it. It would have told them nothing.

Deliberately one file rather than one per job. A job id is a field on the record,
which is all it needs to be — per-job files mean deciding which file to read,
which is a branch in every caller and a question for the user at exactly the
moment they are least able to answer it.

JSON here, plain text in the pasteable report: this side is read by a machine
that can filter on a field, that side is read by a person in a chat window.
"""
from __future__ import annotations

import json
import logging
import logging.config
import logging.handlers
import os
import platform
import sys
import warnings
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

# The job this thread is working on, set once when a job starts. Jobs run one at
# a time on a worker thread, so everything a job touches — including backends
# several layers down that know nothing about jobs — logs against the right one.
current_job: ContextVar[str] = ContextVar("current_job", default="")

# The path the README already names and Uninstall.command already deletes.
# Overridable so a test run does not write into the real one.
LOG_DIR = Path(os.environ.get(
    "DUBBING_STUDIO_LOGS",
    str(Path.home() / "Library" / "Logs" if platform.system() == "Darwin"
        else Path.home() / ".dubbing-studio" / "logs")))
LOG_FILE = LOG_DIR / "DubbingStudio.log"

# Sized so a whole job's records land in one file. A run split across a rotation
# boundary makes the report incoherent — it would quote the tail of a job and
# silently drop the head — and that matters more than the disk this costs, which
# at the ceiling is 20 MB against the gigabytes a single job already moves.
MAX_BYTES = 5 * 1024 ** 2
BACKUPS = 3

# Third-party chatter that made the old file unreadable. Filtered rather than
# left to be scrolled past: a log nobody can skim is a log nobody reads.
# httpx and httpcore are the loud ones in practice: loading a single speech model
# logged three "HTTP Request: HEAD https://huggingface.co/..." lines at INFO, none
# of which say anything the surrounding stage records do not.
_NOISY = ("huggingface_hub", "httpx", "httpcore", "urllib3", "requests",
          "filelock", "matplotlib", "numba", "asyncio", "multipart", "watchfiles")

_ready = False


def log_before_ready(message: str) -> None:
    """For the handful of things that can go wrong before setup() has run —
    config.py resolves paths and migrates old ones at import time, which is
    earlier than anything here can run.

    Written straight to the file rather than queued for setup() to drain,
    because the process that hits this is often a short-lived one — a
    subprocess, a test run — that never calls setup() at all, and a queue
    only that process ever reads is no better than the silence it replaces.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                  "level": "WARNING", "name": "app", "event": message}
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def setup(level: int = logging.INFO) -> Path:
    """Wire up the file. Safe to call more than once; only the first does work."""
    global _ready
    if _ready:
        return LOG_FILE
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.json.JsonFormatter",
                # asctime is renamed rather than reformatted so the key reads as
                # what it is to someone scanning the file.
                "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "rename_fields": {"asctime": "time", "levelname": "level",
                                  "message": "event"},
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "plain": {"format": "%(levelname)s: %(message)s"},
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(LOG_FILE),
                "maxBytes": MAX_BYTES,
                "backupCount": BACKUPS,
                "encoding": "utf-8",
                "formatter": "json",
            },
            # Warnings and worse also go to the terminal. Someone who started
            # the app from "Start Dubbing Studio" is watching that window, and
            # sending every problem exclusively to a file they have not been
            # told about would be a step backwards from printing to stderr.
            "console": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "plain",
                "stream": "ext://sys.stderr",
            },
        },
        "root": {"level": level, "handlers": ["file", "console"]},
        "loggers": {name: {"level": "WARNING"} for name in _NOISY},
    })

    # The single biggest contributor to the old file, emitted on every import of
    # a voice model and carrying nothing anyone can act on.
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

    # The case that leaves nothing behind today: the app dies and the window
    # simply closes, with no failed job to press a button on.
    def crashed(kind, value, tb):
        logging.getLogger("app").critical(
            "uncaught exception", exc_info=(kind, value, tb))
        sys.__excepthook__(kind, value, tb)

    sys.excepthook = crashed
    _ready = True
    return LOG_FILE


# Names LogRecord already owns. Passing one in `extra` does not shadow it, it
# raises — so a field called "message" turned a failed job into a second failure
# inside the logging call that was reporting the first. Renamed rather than
# dropped, because the value is usually the one worth having.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime"}


class _WithJob(logging.LoggerAdapter):
    """Adds the job id without taking the call site's own fields away.

    LoggerAdapter.process replaces kwargs["extra"] wholesale until Python 3.13,
    where merge_extra was added for exactly this. Merging here instead, so
    `log.info("downloaded", extra={"bytes": n})` keeps its bytes on 3.12.
    """

    def process(self, msg, kwargs):
        merged = {**self.extra, **(kwargs.get("extra") or {})}
        kwargs["extra"] = {(f"{k}_" if k in _RESERVED else k): v
                           for k, v in merged.items()}
        return msg, kwargs


def get(job_id: str = "") -> logging.LoggerAdapter:
    """A logger whose every record carries the job it belongs to.

    The adapter is what keeps this out of the call sites: `log.info("separated")`
    inside a job reads the same as anywhere else, and the id is attached without
    anyone having to remember to pass it.

    Falls back to the job this thread is running, so the places that matter most
    — the translation retry, the quality check — are tagged without threading a
    job id down through backends that have no other use for one.
    """
    job_id = job_id or current_job.get("")
    return _WithJob(logging.getLogger("app"), {"job_id": job_id} if job_id else {})


def recent(limit: int = 200) -> list[dict]:
    """The last records, newest last, for the pasteable report.

    Reads only the live file, not the rotated ones — MAX_BYTES is chosen so a
    whole job fits in it, and stitching backups together to chase a boundary
    would be work in aid of a case that should not arise.
    """
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            # A line ffmpeg or a model wrote straight to the handle. Kept rather
            # than dropped: an unparseable line is still evidence.
            out.append({"event": line})
    return out
