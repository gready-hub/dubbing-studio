"""Small helpers for pulling model files down and unpacking them."""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional


def fetch(url: str, dest: Path, progress: Optional[Callable[[float, str], None]] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def hook(block_num: int, block_size: int, total: int) -> None:
        if progress and total > 0:
            done = min(1.0, block_num * block_size / total)
            progress(done, f"Downloading {dest.name} — {done * 100:.0f}%")

    urllib.request.urlretrieve(url, tmp, reporthook=hook if progress else None)
    tmp.replace(dest)
    return dest


def extract(tarball: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tf:
        # Guard against path traversal in archive members.
        for member in tf.getmembers():
            target = (into / member.name).resolve()
            if not str(target).startswith(str(into.resolve())):
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tf.extractall(into)
    return into
