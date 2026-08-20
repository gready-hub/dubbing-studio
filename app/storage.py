"""What this app is keeping on the disk, and how to get rid of it selectively.

The app writes to four places and only ever reported one number for one of them.
Between the speech models, the translation model Ollama holds on its behalf, the
working files and the finished videos, a few hours of use is comfortably tens of
gigabytes — and the only control was a single Clear button that emptied every
job's working files at once.

Filling the boot disk does not merely fail the job. macOS becomes unusable at
the same moment, which is not a state to leave someone in.
"""
from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

from .config import CACHE, JOBS, MODELS, OUTPUT_DIR, PREVIEWS, ollama_host

# The app's own folder, and the Python environment inside it. Between them these
# are larger than everything under Application Support, and neither was visible
# anywhere — a breakdown that omits the biggest items is not a breakdown.
APP_DIR = Path(__file__).resolve().parent.parent
VENV = APP_DIR / ".venv"

# Where the model libraries put their downloads. Shared with any other tool on
# the machine that uses Hugging Face, so only the repositories this app actually
# fetches are counted or removed — someone else's models are not ours to total
# up, let alone delete.
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

# Peak working-file cost per second of video. Measured rather than guessed: a
# 10m35s job at 720p peaked at 390 MB before it pruned itself, which is roughly
# 615 KB for every second of video. The higher qualities carry more pixels
# through the same pipeline, so they scale with it.
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

    One formatter rather than one per caller: the download used to print its own
    fixed "%.1f GB", which read "about 0.0 GB" for a short video and "0 MB" for
    anything under half a megabyte. A size nobody can act on is worse than no
    size, and the small end is exactly where a first, cautious try lands.
    """
    if count >= 1024 ** 3:
        return f"{count / 1024 ** 3:.1f} GB"
    if count >= 100 * 1024 ** 2:       # past 100 MB a decimal is just noise
        return f"{round(count / 1024 ** 2)} MB"
    if count >= 1024 ** 2:
        return f"{count / 1024 ** 2:.1f} MB"
    return f"{max(1, round(count / 1024))} KB"


def estimate_needed(duration: float, quality: str = "best",
                    source_bytes: int = 0) -> int:
    """Roughly what a job of this length will occupy while it runs.

    Deliberately the peak rather than the leftover. A job prunes itself down to
    a few megabytes when it succeeds, but it holds the source video, a
    full-band wav, the separated stems and the assembled track all at once
    before it gets there — and it is that moment, not the tidy end state, that
    fills a disk.

    source_bytes is the real download size when the site published one, and it
    is worth more than the per-second guess in the two cases the guess is worst:
    an unusually high-bitrate video, where the guess is simply too low, and a
    30-second preview, where duration is the *sample* length but the whole video
    is still fetched to take the sample out of — so the estimate was 500 MB for
    a download of well over a gigabyte, on the check that exists to stop a full
    disk taking the machine down with it.
    """
    rate = BYTES_PER_SECOND.get(quality, BYTES_PER_SECOND["best"])
    need = max(0.0, duration) * rate
    if source_bytes > 0:
        # Twice the finished size: yt-dlp writes the video and audio streams
        # side by side and then merges them into a third file. The derived rate
        # is what is left of the measurement above once the source is taken out
        # of it — the wav, the four separated stems and the assembled track.
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
        # A job that failed early leaves a folder holding little more than its
        # error log. Sub-megabyte entries are not where anybody's disk went, and
        # a row reading "0 KB" beside a Clear button reads as something broken
        # rather than as something small. Clearing all of them still sweeps
        # these up.
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
         "hint": "Part-finished audio kept so a repeated link doesn't redo the "
                 "expensive parts. Safe to delete; the next run rebuilds it."},
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
         "hint": "The recogniser, the voices and the music separator, as fetched "
                 "from Hugging Face. Safe to delete; the next video downloads "
                 "what it needs again. Other apps' models here are left alone."},
        {"key": "venv", "label": "Python environment",
         "bytes": dir_size(VENV) if VENV.is_dir() else 0,
         "clearable": False, "path": str(APP_DIR),
         "hint": "The libraries the app runs on, inside its own folder. Removed "
                 "by Uninstall rather than from here — it is what is running."},
        {"key": "ollama", "label": "Translation model", "bytes": _ollama_bytes(),
         "clearable": False, "path": "",
         "hint": "Held by Ollama, not by this app. Remove it from there if you "
                 "need the room — translation will stop working until it's back."},
        {"key": "output", "label": "Finished videos",
         "bytes": dir_size(OUTPUT_DIR) if OUTPUT_DIR.is_dir() else 0,
         "clearable": False, "path": str(OUTPUT_DIR),
         "hint": "The videos you asked for. This app will never delete these."},
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
        # A symlinked folder — someone keeping the models on an external drive,
        # which is exactly what a person short of space might do — is not ours
        # to empty. rmtree refuses to follow it anyway, so without this the size
        # was measured through the link and reported as freed while nothing was.
        if not target.is_dir() or target.is_symlink():
            return 0
        freed = dir_size(target)
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        return freed
    if what.startswith("job:"):
        name = what[4:]
        # Resolved and checked rather than joined and trusted: this arrives over
        # HTTP, and "job:../.." would otherwise reach outside the jobs folder.
        folder = (JOBS / name).resolve()
        if name in keep or folder.parent != JOBS.resolve() or not folder.is_dir():
            return 0
        freed = dir_size(folder)
        shutil.rmtree(folder, ignore_errors=True)
        return freed
    raise ValueError(f"Nothing called {what!r} can be cleared.")
