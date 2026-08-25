"""What this app is keeping on the disk, and how to get rid of it selectively.

Speech models, the Ollama translation model, working files and finished videos
can add up to tens of gigabytes; a full boot disk makes macOS itself unusable,
not just the job, so clearing needs to be per-category rather than all-or-nothing.
"""
from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

from .config import CACHE, JOBS, MODELS, OUTPUT_DIR, PREVIEWS, ollama_host

# Larger than everything under Application Support, so must be in the breakdown.
APP_DIR = Path(__file__).resolve().parent.parent
VENV = APP_DIR / ".venv"

# Shared with any other tool using Hugging Face; only our own repos (below) are
# counted or removed.
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
OUR_MODEL_REPOS = (
    "models--mlx-community--parakeet-tdt",
    "models--mlx-community--Kokoro-82M",
    "models--mlx-community--whisper",
    "models--prince-canuma--Kokoro-82M",
    "models--adefossez--HTDemucs",
    "models--ResembleAI--chatterbox",
)


def model_cache_dirs() -> list[Path]:
    """The Hugging Face repositories this app is responsible for."""
    if not HF_HUB.is_dir():
        return []
    return [p for p in HF_HUB.iterdir()
            if p.is_dir() and p.name.startswith(OUR_MODEL_REPOS)]

# Peak working-file cost per second of video, measured (not guessed) from a
# real job before it pruned itself; higher qualities scale up from there.
BYTES_PER_SECOND = {"720": 620_000, "1080": 1_100_000, "best": 1_800_000}
# The part of that which is not the source video: the 16 kHz wav, the four
# separated stems and the assembled track. Used when the real download size is
# known, so only the guessed half of the estimate is replaced.
DERIVED_PER_SECOND = 500_000
# Models load, ffmpeg works in temporary files, and a very short video still
# needs somewhere to put all that.
MINIMUM_NEED = 500 * 1024 ** 2
# Below this the machine itself starts misbehaving, whatever this app is doing.
LOW_DISK = 5 * 1024 ** 3
# Smallest job folder worth its own row in the breakdown.
LISTABLE = 1024 ** 2


def dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def free_bytes(path: Path = CACHE) -> int:
    """Room left on the volume this path lives on."""
    try:
        return shutil.disk_usage(path if path.exists() else path.parent).free
    except OSError:
        return 0


def human_size(count: int) -> str:
    """Bytes, written the way someone reading a message needs them.

    One formatter for every caller, so small sizes never round to "0.0 GB"
    the way a fixed "%.1f GB" format used to.
    """
    if count >= 1024 ** 3:
        return f"{count / 1024 ** 3:.1f} GB"
    if count >= 100 * 1024 ** 2:       # past 100 MB a decimal is just noise
        return f"{round(count / 1024 ** 2)} MB"
    if count >= 1024 ** 2:
        return f"{count / 1024 ** 2:.1f} MB"
    return f"{max(1, round(count / 1024))} KB"


def estimate_needed(duration: float, quality: str = "best",
                    source_bytes: int = 0, local_source: bool = False) -> int:
    """Roughly what a job of this length will occupy while it runs.

    Deliberately the peak, not the pruned-down end state: source video, wav,
    separated stems and assembled track can all exist at once mid-job.

    source_bytes (the real download size, when known) overrides the per-second
    guess, which is worst exactly on a high-bitrate video and on a short preview
    sample (duration is the sample length, but the whole video is still fetched).

    local_source means the source isn't being copied in, so the fetch drops out
    of the estimate but the written-back dub does not — see below.
    """
    if local_source:
        # Source costs nothing (read where it lies), but mux keeps the finished
        # file about the same size, and a non-H.264 source is first converted
        # into a second full-size file beside it — hence x2, as for a download,
        # minus the fetch. source_bytes is scaled by the caller (full vs sample).
        if source_bytes > 0:
            return max(MINIMUM_NEED,
                       int(source_bytes * 2 + max(0.0, duration) * DERIVED_PER_SECOND))
        # No size to go on — a file that could not be stat'd. Fall back to the
        # same per-second guess a download uses rather than to the derived
        # figure alone, which would leave no room for the dub at all.
        rate = BYTES_PER_SECOND.get(quality, BYTES_PER_SECOND["best"])
        return max(MINIMUM_NEED, int(max(0.0, duration) * rate))
    rate = BYTES_PER_SECOND.get(quality, BYTES_PER_SECOND["best"])
    need = max(0.0, duration) * rate
    if source_bytes > 0:
        # x2: yt-dlp writes video and audio streams separately, then merges
        # them into a third file.
        need = max(need, source_bytes * 2 + max(0.0, duration) * DERIVED_PER_SECOND)
    return max(MINIMUM_NEED, int(need))


def _ollama_bytes() -> int:
    """What Ollama is holding for us. Often the largest single item by far.

    Read-only here on purpose: a 20 GB translation model is worth showing
    someone hunting for space, but it belongs to Ollama, deleting it silently
    breaks translation, and re-fetching it is not a thing to trigger by
    accident from a tidy-up panel.
    """
    try:
        with urllib.request.urlopen(f"{ollama_host()}/api/tags", timeout=2) as r:
            models = json.loads(r.read()).get("models", [])
        return sum(int(m.get("size") or 0) for m in models)
    except Exception:                                            # noqa: BLE001
        return 0


def job_folders(title_for=None) -> list[dict]:
    """One entry per job folder, largest first.

    title_for maps a folder name to something a human recognises; without it
    the rows are hashes, which is what the folder names are.
    """
    if not JOBS.is_dir():
        return []
    rows = []
    for folder in JOBS.iterdir():
        if not folder.is_dir():
            continue
        size = dir_size(folder)
        # A failed-early job is just an error log; listing it as "0 KB" reads as
        # broken, not small. Clearing all jobs still sweeps these up.
        if size < LISTABLE:
            continue
        rows.append({"id": folder.name,
                     "title": (title_for(folder.name) if title_for else "") or folder.name,
                     "bytes": size})
    return sorted(rows, key=lambda r: r["bytes"], reverse=True)


def groups() -> list[dict]:
    """Every place this app puts bytes, and whether it is safe to empty."""
    return [
        {"key": "jobs", "label": "Working files", "bytes": dir_size(JOBS) if JOBS.is_dir() else 0,
         "clearable": True, "path": str(JOBS),
         "hint": "Part-finished audio, kept so a repeated link doesn't redo the "
                 "expensive steps. Safe to delete; the next run rebuilds it."},
        {"key": "models", "label": "Speech models", "bytes": dir_size(MODELS) if MODELS.is_dir() else 0,
         "clearable": True, "path": str(MODELS),
         "hint": "The voices and the recogniser. Safe to delete, but the next "
                 "video will spend several minutes downloading them again."},
        {"key": "previews", "label": "Voice samples",
         "bytes": dir_size(PREVIEWS) if PREVIEWS.is_dir() else 0,
         "clearable": True, "path": str(PREVIEWS),
         "hint": "One short clip per voice you've listened to."},
        {"key": "hfmodels", "label": "Downloaded AI models",
         "bytes": sum(dir_size(p) for p in model_cache_dirs()),
         "clearable": True, "path": str(HF_HUB) if HF_HUB.is_dir() else "",
         "hint": "The recogniser, the voices and the music separator, from "
                 "Hugging Face. Safe to delete; the next video downloads what it "
                 "needs again. Other apps' models here are left alone."},
        {"key": "venv", "label": "Python environment",
         "bytes": dir_size(VENV) if VENV.is_dir() else 0,
         "clearable": False, "path": str(APP_DIR),
         "hint": "The libraries the app runs on, inside its own folder. Removed "
                 "by Uninstall rather than from here — it is what is running."},
        {"key": "ollama", "label": "Translation model", "bytes": _ollama_bytes(),
         "clearable": False, "path": "",
         "hint": "Held by Ollama, not by this app. Remove it there if you need "
                 "the room — translation stops working until it's back."},
        {"key": "output", "label": "Finished videos",
         "bytes": dir_size(OUTPUT_DIR) if OUTPUT_DIR.is_dir() else 0,
         "clearable": False, "path": str(OUTPUT_DIR),
         "hint": "Never deleted by this app."},
    ]


def summary(title_for=None) -> dict:
    free = free_bytes(CACHE)
    return {"free": free, "low": free < LOW_DISK,
            "groups": groups(), "jobs": job_folders(title_for),
            "path": str(JOBS)}


def clear(what: str, keep: set[str] | None = None) -> int:
    """Empty one group, or one job folder. Returns the bytes reclaimed.

    Anything not named here is refused rather than interpreted, so a typo in a
    request cannot reach the finished videos.
    """
    keep = keep or set()
    if what == "jobs":
        if not JOBS.is_dir():
            return 0
        freed = 0
        for folder in JOBS.iterdir():
            if not folder.is_dir() or folder.name in keep:
                continue
            freed += dir_size(folder)
            shutil.rmtree(folder, ignore_errors=True)
        return freed
    if what == "hfmodels":
        freed = 0
        for folder in model_cache_dirs():
            freed += dir_size(folder)
            shutil.rmtree(folder, ignore_errors=True)
        return freed
    if what in ("models", "previews"):
        target = MODELS if what == "models" else PREVIEWS
        # A symlinked folder (e.g. models kept on an external drive) is not ours
        # to empty; without this check, size was measured through the link and
        # reported freed while rmtree silently refused to follow it.
        if not target.is_dir() or target.is_symlink():
            return 0
        freed = dir_size(target)
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        return freed
    if what.startswith("job:"):
        name = what[4:]
        # Resolved and checked, not trusted: this arrives over HTTP, and
        # "job:../.." would otherwise reach outside the jobs folder.
        folder = (JOBS / name).resolve()
        if name in keep or folder.parent != JOBS.resolve() or not folder.is_dir():
            return 0
        freed = dir_size(folder)
        shutil.rmtree(folder, ignore_errors=True)
        return freed
    raise ValueError(f"Nothing called {what!r} can be cleared.")
