"""One block of text a non-technical user can paste into a message.

The app already knows everything needed to diagnose most failures — which Mac,
which versions, what the setup check thinks, what the last job did and what the
failing tool said. None of it was reachable without a terminal, so the best
advice available was "expand that grey triangle and send a screenshot".

Plain text rather than JSON on purpose. This gets pasted into a chat window,
where JSON is mangled by the client, looks alarming to the person sending it, and
is no easier to read at the other end. The log stays JSON; that is machine-read.

No per-job variant and no arguments. The job that just failed is in the recent
log lines by definition, tagged with its id, so choosing a job would add a
question for the user and a branch here in exchange for nothing.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict

from . import logs, storage
from .config import OUTPUT_DIR, Settings, detect_machine, mac_generation

# Never leaves the machine in the clear. /api/state hands these back in full,
# which is tolerable over localhost and is not tolerable on a clipboard bound for
# a chat window. Reduced to whether one is set, which is the only part that helps
# anybody diagnose anything.
SECRETS = ("anthropic_key", "openai_key")

# Everything except the secrets, rather than a list of the interesting ones.
# An allowlist drifts: a setting added later is simply missing from every report
# and nobody notices until it is the one that explains a failure. That is the
# same shape of bug as the download's format list, which was fetched and then
# dropped on the way out of the function that fetched it.
LONG_VALUE = 120


def _tool_version(*cmd: str) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return (out.stdout or out.stderr).strip().splitlines()[0][:80]
    except Exception:                                            # noqa: BLE001
        return "not found"


def _settings_lines(settings) -> list[str]:
    rows = []
    for key, value in sorted(asdict(settings).items()):
        if key in SECRETS:
            rows.append(f"  {key}: {'set' if value else 'not set'}")
            continue
        # Flattened first: a pasted glossary carries newlines, and one setting
        # spilling over several lines turns a scannable list into a wall.
        text = " ".join(str(value).split()) if value != "" else "(not set)"
        if len(text) > LONG_VALUE:
            # A pasted glossary can run to pages. Its size is what matters here;
            # the content is on the user's screen if it is ever needed.
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
    lines.append(f"  yt-dlp: {_tool_version('yt-dlp', '--version')}")
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

    # Whatever died below Python — a native crash in a model or in ffmpeg — never
    # reaches the logger, so the app bundle keeps stderr in a file of its own.
    # Empty on a healthy machine, and the only evidence there is on a sick one.
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

    Imported inside the function because server imports this module; asking at
    call time keeps that from being a circular import at load time.
    """
    try:
        from .server import doctor
        return doctor().get("checks", [])
    except Exception as exc:                                     # noqa: BLE001
        return [{"name": f"setup check unavailable ({exc})", "ok": False}]
