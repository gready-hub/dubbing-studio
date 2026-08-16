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
        raise DownloadError(_friendly(out.stderr), out.stderr)
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


class DownloadError(RuntimeError):
    """A failed download, with the tool's own words kept alongside the plain one.

    The friendly message is what somebody can act on; the raw tail is what makes
    the difference between "it didn't work" and knowing why, and it was only
    ever written to a log file in Application Support.
    """

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail.strip()


def _looks_transient(message: str) -> bool:
    s = message.lower()
    return any(marker in s for marker in _TRANSIENT)


def _friendly(stderr: str) -> str:
    s = stderr.lower()
    if "403" in s and "forbidden" in s:
        # By the time anyone reads this the download has already been attempted
        # several times, each as a different player client. Telling them to try
        # again in a minute at that point is advice that has already been taken
        # on their behalf and failed.
        return ("YouTube described the video but refused to send it, on every "
                "attempt. It usually wants a signed-in browser session: open "
                "Settings and set “Sign in as” to the browser you watch YouTube "
                "in. If that doesn't do it, the video may be private, "
                "age-restricted or members-only.")
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
    # A link that goes nowhere is one of the commonest ways this fails — a typo,
    # a stale bookmark, half a URL pasted — and it used to fall through to
    # yt-dlp's own words, which is where the interface got "ERROR: [generic]
    # does-not-exist: Unable to download webpage: HTTP Error 404: File not found
    # (caused by <HTTPError 404: File not found>)".
    if "http error 404" in s or "unable to download webpage" in s:
        return ("That link couldn't be opened. Check it is typed correctly and that "
                "the video is still there.")
    if ("name or service not known" in s or "nodename nor servname" in s
            or "temporary failure in name resolution" in s):
        return "That address couldn't be reached — check your internet connection."
    tail = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    return _tidy(tail[-1]) if tail else "The download failed."


def _tidy(line: str) -> str:
    """Make yt-dlp's last line readable without pretending to understand it.

    Whatever is left over after the cases above is genuinely unknown, so it is
    passed on rather than replaced — but the scaffolding around it is noise to
    everyone: the ERROR: prefix, the extractor name in brackets, and the wrapped
    Python exception repeated in parentheses at the end.
    """
    line = re.sub(r"^ERROR:\s*", "", line.strip())
    head = re.match(r"^\[[^\]]+\]\s*", line)
    if head:
        line = line[head.end():]
        # yt-dlp prints the video's own id straight after the extractor name.
        # Kept id-shaped deliberately: a message that merely happens to start
        # with a word and a colon — or with a URL, which is full of them — must
        # not lose its first clause to this.
        line = re.sub(r"^[\w.\-]{1,64}:\s+", "", line, count=1)
    line = re.sub(r"\s*\(caused by .*\)\s*$", "", line)
    return line.strip() or "The download failed."


def download(url: str, workdir: Path, quality: str = "best",
             progress: Progress = None, info: dict | None = None,
             cookies_from: str = "") -> tuple[Path, dict]:
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
        "--newline", "--no-warnings",
    ]
    # The fix for the stubborn case: YouTube increasingly wants a session, and a
    # signed-out request for some videos is refused whatever client asks. Off
    # unless the user names a browser, because reading their cookie store is not
    # something to do quietly on their behalf.
    if cookies_from:
        cmd += ["--cookies-from-browser", cookies_from]
    cmd += ["-o", str(target), url]

    # A 403 on the media fetch, straight after a metadata probe that succeeded,
    # is throttling rather than a bad link — the same request goes through a
    # moment later. Retrying here beats making the user notice and re-paste.
    def show(line: str) -> None:
        m = _PCT.search(line)
        if m and progress:
            progress(0.02 + 0.96 * float(m.group(1)) / 100,
                     f"Downloading — {m.group(1)}%")

    # yt-dlp negotiates a player client with YouTube before it is handed media
    # URLs, and a 403 on the media fetch straight after a metadata probe that
    # succeeded is usually that client being refused rather than the video being
    # unavailable. Asking again as a different one costs nothing, needs nothing
    # from the user, and is the retry most likely to land — repeating the same
    # request three times, which is what happened before, mostly just waits.
    #
    # Which clients YouTube accepts changes; this is a mitigation, not a
    # guarantee. None means "whatever yt-dlp would choose for itself", which is
    # kept first because it is the one its maintainers keep current.
    clients = (None, "web_safari", "tv", "ios")
    problem = ""
    for attempt, client in enumerate(clients, 1):
        run = list(cmd)
        if client:
            run += ["--extractor-args", f"youtube:player_client={client}"]
        code, problem = stream(run, show, tail_lines=25)
        if code == 0:
            break
        if attempt >= len(clients) or not _looks_transient(problem):
            raise DownloadError(_friendly(problem), problem)
        if progress:
            # Also the cancel check, so a stop during the wait is honoured.
            progress(0.02, f"The site refused that request — asking a different "
                           f"way ({attempt} of {len(clients) - 1})")
        time.sleep(3 * attempt)
    else:
        raise DownloadError(_friendly(problem), problem)

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
