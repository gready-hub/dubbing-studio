"""One log file, in JSON, for whoever has to work out what went wrong.

Previously stderr was redirected to a file nothing else wrote to — no
timestamps, no job ids, no errors, just repeated deprecation warnings.

Deliberately one file rather than one per job: a job id is just a field on the
record, so no caller has to decide which file to read. JSON here, plain text in
the pasteable report — this side is machine-filtered, that side is human-read.
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

# Jobs run one at a time on a worker thread, so setting this once lets backends
# several layers down log against the right job without knowing about jobs.
current_job: ContextVar[str] = ContextVar("current_job", default="")

# The path the README already names and Uninstall.command already deletes.
# Overridable so a test run does not write into the real one.
LOG_DIR = Path(os.environ.get(
    "DUBBING_STUDIO_LOGS",
    str(Path.home() / "Library" / "Logs" if platform.system() == "Darwin"
        else Path.home() / ".dubbing-studio" / "logs")))
LOG_FILE = LOG_DIR / "DubbingStudio.log"

# Sized so a whole job's records land in one file — a rotation mid-job would
# quote the tail and silently drop the head.
MAX_BYTES = 5 * 1024 ** 2
BACKUPS = 3

# Third-party chatter that made the old file unreadable — httpx/httpcore are
# the worst, logging HTTP request lines at INFO that add nothing over the
# surrounding stage records.
_NOISY = ("huggingface_hub", "httpx", "httpcore", "urllib3", "requests",
          "filelock", "matplotlib", "numba", "asyncio", "multipart", "watchfiles")

_ready = False


def log_before_ready(message: str) -> None:
    """For failures before setup() has run (config.py migrates paths at import time).

    Written straight to the file rather than queued for setup() to drain,
    since the calling process (a subprocess, a test run) may never call
    setup() at all.
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
            # Also to the terminal: someone launching via "Start Dubbing Studio"
            # is watching that window and shouldn't lose visibility into a file.
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

    # Emitted on every voice-model import; carries nothing actionable.
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

    # Otherwise an app crash leaves nothing behind — no failed job to inspect.
    def crashed(kind, value, tb):
        logging.getLogger("app").critical(
            "uncaught exception", exc_info=(kind, value, tb))
        sys.__excepthook__(kind, value, tb)

    sys.excepthook = crashed
    _ready = True
    return LOG_FILE


# Names LogRecord already owns — passing one in `extra` raises rather than
# shadowing it, so these get renamed (suffixed) instead of dropped.
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

    Falls back to the job this thread is running, so call sites don't need to
    thread a job id down through backends that have no other use for one.
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
