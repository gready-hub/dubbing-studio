"""One block of text a non-technical user can paste into a message.

Plain text, not JSON: this gets pasted into a chat window, where JSON is
mangled by the client and looks alarming. The log itself stays JSON.

No per-job variant: the job that just failed is already in the recent log
lines, tagged with its id, so picking a job would add nothing but a question.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict

from . import logs, storage
from .config import OUTPUT_DIR, Settings, detect_machine, mac_generation

# Reduced to whether one is set: tolerable for /api/state over localhost, not
# for a clipboard headed to a chat window.
SECRETS = ("anthropic_key", "openai_key")

LONG_VALUE = 120


def _tool_version(*cmd: str) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return (out.stdout or out.stderr).strip().splitlines()[0][:80]
    except Exception:                                            # noqa: BLE001
        return "not found"


def _ytdlp_version() -> str:
    """The version of the yt-dlp the downloader will really run.

    Bare "yt-dlp" resolves whatever PATH finds first, which under the activated
    venv is not necessarily the copy actually doing the downloading.
    """
    from .steps.download import _ytdlp_cmd
    return _tool_version(*_ytdlp_cmd(), "--version")


def _settings_lines(settings) -> list[str]:
    rows = []
    # Everything except secrets, not an allowlist of "interesting" ones — an
    # allowlist drifts silently when a new setting is added.
    for key, value in sorted(asdict(settings).items()):
        if key in SECRETS:
            rows.append(f"  {key}: {'set' if value else 'not set'}")
            continue
        # Flattened: a pasted glossary carries newlines that would otherwise
        # turn one row into a wall of text.
        text = " ".join(str(value).split()) if value != "" else "(not set)"
        if len(text) > LONG_VALUE:
            # Size matters here, not content — that's on the user's own screen.
            text = f"{text[:LONG_VALUE]}… ({len(text)} characters)"
        rows.append(f"  {key}: {text}")
    return rows


def _log_lines(limit: int) -> list[str]:
    out = []
    for record in logs.recent(limit):
        time = record.get("time", "")
        level = record.get("level", "")
        event = record.get("event", "")
        extra = {k: v for k, v in record.items()
                 if k not in ("time", "level", "name", "event")}
        tail = f"  {json.dumps(extra, default=str)}" if extra else ""
        out.append(f"  {time} {level:<8} {event}{tail}")
    return out


def report(limit: int = 200) -> str:
    """The whole thing, ready to paste."""
    machine = detect_machine()
    settings = Settings.load()
    free = storage.free_bytes()

    lines: list[str] = ["Dubbing Studio — diagnostics", ""]

    lines.append("This Mac")
    lines.append(f"  macOS {platform.mac_ver()[0] or platform.release()} on {machine.arch}")
    generation = mac_generation()
    lines.append(f"  chip: {'Apple silicon' if machine.apple_silicon else 'Intel'}"
                 + (f", M{generation} generation" if generation else ""))
    lines.append(f"  memory: {machine.ram_gb} GB")
    lines.append(f"  engine: {'Apple GPU (MLX)' if machine.fast_path else 'Portable (CPU)'}")
    lines.append(f"  free disk: {storage.human_size(free)}")
    lines.append(f"  videos saved to: {OUTPUT_DIR}")
    lines.append("")

    lines.append("Versions")
    lines.append(f"  app: {_installed_version()}")
    lines.append(f"  python: {sys.version.split()[0]}")
    lines.append(f"  ffmpeg: {_tool_version('ffmpeg', '-version')}")
    lines.append(f"  yt-dlp: {_ytdlp_version()}")
    lines.append(f"  ollama: {_tool_version('ollama', '--version')}")
    lines.append("")

    lines.append("Setup check")
    for check in _checks():
        mark = "ok  " if check.get("ok") else "MISSING"
        optional = " (optional)" if check.get("optional") else ""
        lines.append(f"  [{mark}] {check.get('name', '?')}{optional}")
    lines.append("")

    lines.append("Settings")
    lines.extend(_settings_lines(settings))
    lines.append("")

    # A native crash (model, ffmpeg) never reaches the logger, so the app
    # bundle keeps stderr in a separate file; empty unless something crashed.
    crash = _crash_tail()
    if crash:
        lines.append("Crashes (stderr, most recent last)")
        lines.extend(crash)
        lines.append("")

    recent = _log_lines(limit)
    lines.append(f"Recent activity (last {len(recent)} entries)")
    lines.extend(recent or ["  (nothing logged yet)"])
    return "\n".join(lines) + "\n"


def _crash_tail(keep: int = 20) -> list[str]:
    path = logs.LOG_DIR / "DubbingStudio-crash.log"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [f"  {line}" for line in text.splitlines()[-keep:] if line.strip()]


def _installed_version() -> str:
    try:
        stamp = (storage.APP_DIR / ".version").read_text().strip()
        return stamp[:7] if len(stamp) == 40 else "unknown"
    except OSError:
        return "not recorded (installed by hand?)"


def _checks() -> list[dict]:
    """The setup check, asked of the server rather than restated here.

    Imported locally (not at module level) since server also imports this
    module — avoids a circular import.
    """
    try:
        from .server import doctor
        return doctor().get("checks", [])
    except Exception as exc:                                     # noqa: BLE001
        return [{"name": f"setup check unavailable ({exc})", "ok": False}]
