"""Fetch a video with yt-dlp and pull its audio out."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

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
             progress: Progress = None) -> tuple[Path, dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    info = probe(url)
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
    attempts = 3
    for attempt in range(1, attempts + 1):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        tail: list[str] = []
        try:
            for line in proc.stdout:                             # type: ignore[union-attr]
                tail.append(line)
                tail[:] = tail[-25:]
                m = _PCT.search(line)
                if m and progress:
                    progress(0.02 + 0.96 * float(m.group(1)) / 100,
                             f"Downloading — {m.group(1)}%")
        except BaseException:
            # A cancel arrives through the progress callback; don't leave yt-dlp
            # pulling a gigabyte of video for a job that has already stopped.
            proc.kill()
            proc.wait()
            raise
        proc.wait()
        if proc.returncode == 0:
            break

        problem = "".join(tail)
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
