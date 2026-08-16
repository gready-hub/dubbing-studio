"""Fetch a video with yt-dlp and pull its audio out."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from .proc import stream

Progress = Optional[Callable[[float, str], None]]

_PCT = re.compile(r"\[download\]\s+([\d.]+)%")


def _ytdlp_cmd() -> list[str]:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return ["python3", "-m", "yt_dlp"]


def probe(url: str) -> dict:
    out = subprocess.run(_ytdlp_cmd() + ["-J", "--no-warnings", "--skip-download", url],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(_friendly(out.stderr))
    data = json.loads(out.stdout)
    return {
        "title": data.get("title", "video"),
        "duration": float(data.get("duration") or 0),
        "uploader": data.get("uploader", ""),
        "thumbnail": data.get("thumbnail", ""),
    }


# Failures that mean "ask again in a moment" rather than "this link is no good".
_TRANSIENT = ("403", "429", "too many requests", "temporarily unavailable",
              "timed out", "connection reset", "connection aborted")


def _looks_transient(message: str) -> bool:
    s = message.lower()
    return any(marker in s for marker in _TRANSIENT)


def _friendly(stderr: str) -> str:
    s = stderr.lower()
    if "403" in s and "forbidden" in s:
        return ("The site refused to hand over the video data, even though it "
                "described the video happily. This is nearly always temporary — "
                "try the same link again in a minute.")
    if "private video" in s:
        return "That video is private, so it can't be downloaded."
    if "sign in to confirm your age" in s or "age" in s and "restricted" in s:
        return "That video is age-restricted and can't be fetched without signing in."
    if "video unavailable" in s:
        return "That video is unavailable — check the link is still live."
    if "unsupported url" in s:
        return "That link isn't one yt-dlp recognises."
    if "http error 429" in s or "too many requests" in s:
        return "The site is rate-limiting downloads. Wait a few minutes and try again."
    tail = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    return tail[-1] if tail else "The download failed."


def download(url: str, workdir: Path, quality: str = "best",
             progress: Progress = None, info: dict | None = None) -> tuple[Path, dict]:
    """info: a probe() result the caller already has, to save asking twice.

    The preview needs the duration before it can weight the progress bar, which
    means probing before this stage rather than inside it.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    info = info or probe(url)
    if progress:
        progress(0.02, f"Found “{info['title']}”")

    fmt = "bv*+ba/b"
    if quality in ("1080", "720"):
        fmt = f"bv*[height<={quality}]+ba/b[height<={quality}]/b"

    target = workdir / "source.%(ext)s"
    cmd = _ytdlp_cmd() + [
        "-f", fmt, "--merge-output-format", "mp4",
        # Deliberately no --user-agent. YouTube ties a media URL's authorisation
        # to the client yt-dlp negotiated it as, so overriding the agent
        # *desyncs* the two: a hand-set desktop Chrome string turns a download
        # that works into a reliable 403. yt-dlp's own default is correct.
        "--retries", "10", "--fragment-retries", "10", "--extractor-retries", "3",
        "--newline", "--no-warnings", "-o", str(target), url,
    ]

    # A 403 on the media fetch, straight after a metadata probe that succeeded,
    # is throttling rather than a bad link — the same request goes through a
    # moment later. Retrying here beats making the user notice and re-paste.
    def show(line: str) -> None:
        m = _PCT.search(line)
        if m and progress:
            progress(0.02 + 0.96 * float(m.group(1)) / 100,
                     f"Downloading — {m.group(1)}%")

    attempts = 3
    for attempt in range(1, attempts + 1):
        code, problem = stream(cmd, show, tail_lines=25)
        if code == 0:
            break
        if attempt >= attempts or not _looks_transient(problem):
            raise RuntimeError(_friendly(problem))
        if progress:
            # Also the cancel check, so a stop during the wait is honoured.
            progress(0.02, f"The site refused that request; trying again "
                           f"({attempt} of {attempts - 1})")
        time.sleep(4 * attempt)

    files = sorted(workdir.glob("source.*"))
    if not files:
        raise RuntimeError("The download finished but no video file appeared.")
    video = files[0]
    if progress:
        progress(1.0, "Download complete")
    return video, info


def extract_audio(video: Path, dst: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
                    "-vn", "-ac", "1", "-ar", "16000", str(dst)], check=True)
    return dst


def media_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)
