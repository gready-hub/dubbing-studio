"""Fetch a video with yt-dlp and pull its audio out."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from ..storage import human_size
from .proc import stream

Progress = Optional[Callable[[float, str], None]]

# yt-dlp will emit progress in a shape we choose, rather than us reading the
# shape it chose for humans. The old regex matched "[download]  12.3%" out of
# text meant for a terminal, which is free to change between releases and says
# nothing but the percentage. This is a documented interface and carries the
# byte counts too.
_PROGRESS_PREFIX = "DUBPROG|"
_PROGRESS_TEMPLATE = (
    "download:" + _PROGRESS_PREFIX
    + "%(progress.downloaded_bytes)s|%(progress.total_bytes,progress.total_bytes_estimate)s"
)


def _ytdlp_cmd() -> list[str]:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return ["python3", "-m", "yt_dlp"]


# Which YouTube clients to ask, in one request rather than as a retry ladder of
# our own. yt-dlp merges the format lists of every client named here and knows
# which one can actually serve each format, which is the thing our ladder was
# guessing at: it tried one client, read the failure text, and decided from a
# list of substrings whether a different client was worth a go. That decision is
# yt-dlp's to make and it already makes it.
#
# The list is passed to the probe as well as the download on purpose. The probe
# is where choose_format() picks concrete format ids, so if the download asked a
# client the probe never spoke to, the ids it was told to fetch need not exist —
# which is precisely the "Requested format is not available" that ended the
# ladder. Same clients, same list, same ids.
_CLIENTS = "default,web_safari,tv"
_CLIENT_ARGS = ["--extractor-args", f"youtube:player_client={_CLIENTS}"]

# yt-dlp reads its own config files (portable, home, user, system) by default and
# silently merges whatever they contain into this command, which can add unwanted
# output files or override the flags chosen here. --ignore-config keeps the
# command fully specified; "Sign in as" still reaches --cookies-from-browser
# directly, with no config file needed either way.
_NO_CONFIG_ARGS = ["--ignore-config"]

# Retrying is yt-dlp's job and it does it properly: per-request, with real
# exponential backoff, without abandoning the bytes already on disk. Ours was a
# whole-command loop with a fixed 4n sleep that restarted the progress count
# every time. A 403 part way through a large media fetch — the failure actually
# measured here — is what --retries and http backoff exist for.
_RETRY_ARGS = [
    "--retries", "10", "--fragment-retries", "10", "--extractor-retries", "5",
    "--retry-sleep", "http:exp=1:60",
    "--retry-sleep", "fragment:exp=1:30",
    "--retry-sleep", "extractor:exp=1:30",
]


def probe(url: str, cookies_from: str = "") -> dict:
    """Ask the site what it has, without fetching any of it.

    cookies_from matters here and not only in download(): an age-restricted or
    members-only video refuses at the *lookup*, so a user who followed the 403
    advice and named their browser in Settings would still be turned away before
    the download that knows about their cookies was ever reached.

    --ignore-no-formats-error because this asks what exists, not for a stream.
    Without it yt-dlp resolves its default selector even under -J and fails the
    whole lookup when nothing satisfies it, which YouTube does intermittently to
    videos it will serve happily a minute later — a job dead before the download
    that would have coped was ever reached. The errors worth keeping still land:
    a private or age-restricted video fails to extract at all, which is a
    different error and still raises here with its own advice.
    """
    cmd = (_ytdlp_cmd() + ["-J", "--no-warnings", "--skip-download",
                           "--ignore-no-formats-error"]
           + _CLIENT_ARGS + _RETRY_ARGS + _NO_CONFIG_ARGS)
    if cookies_from:
        cmd += ["--cookies-from-browser", cookies_from]
    out = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=120)
    if out.returncode != 0 and _format_was_refused(out.stderr):
        # The same refusals the download stage sees, and for the same reason: the
        # site answers one negotiation differently from the next. Two of the three
        # failures in a real session died here, before the download that would
        # have coped was reached. Asked once more, which re-negotiates.
        out = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise DownloadError(_friendly(out.stderr), out.stderr)
    data = json.loads(out.stdout)
    return {
        "title": data.get("title", "video"),
        "duration": float(data.get("duration") or 0),
        "uploader": data.get("uploader", ""),
        "thumbnail": data.get("thumbnail", ""),
        # Kept rather than discarded: choose_format() reads this, and without it
        # every call returned None and quietly fell back to the selector string —
        # so the codec and the real download size were unknown until afterwards,
        # which is the whole thing choosing from the list was meant to fix. Stays
        # in memory; the caller writes its own small record to info.json.
        "formats": data.get("formats") or [],
        "is_live": bool(data.get("is_live")),
    }


class DownloadError(RuntimeError):
    """A failed download, with the tool's own words kept alongside the plain one.

    The friendly message is what somebody can act on; the raw tail is what makes
    the difference between "it didn't work" and knowing why, and it was only
    ever written to a log file in Application Support.
    """

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail.strip()


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
    # yt-dlp passes this one through from YouTube, and its own words are advice
    # that cannot work: there is no page here to reload. It means the exchange
    # was declined — the player challenge failed, or enough requests have gone
    # out recently that the next one is refused on sight. Both clear on their
    # own, and both come back faster for being left alone.
    if "page needs to be reloaded" in s:
        return ("YouTube turned the app away rather than the video down — it does "
                "this after a burst of downloads, and to the same link it will "
                "serve a few minutes later. Wait a little and press Try again. "
                "Repeating it straight away makes the wait longer, not shorter.")
    # yt-dlp's own words here are "Requested format is not available. Use
    # --list-formats for a list of available formats", which tells somebody who
    # has never opened a terminal to pass a command-line flag. It also sounds
    # like a fault in the video, and it is not: the video is listed and then
    # nothing in the listing is offered, which on YouTube is a defensive response
    # to being asked repeatedly.
    if "requested format is not available" in s:
        return ("YouTube listed the video but then wouldn't offer any version of "
                "it to download. That is usually temporary — wait a few minutes "
                "and try the same link again. If it keeps happening, open Settings "
                "and set “Sign in as” to the browser you watch YouTube in.")
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


def _format_was_refused(detail: str) -> bool:
    """Whether the *pinned formats* were the problem, so a fresh pick may not be.

    choose_format() reads its ids from probe(), a separate yt-dlp run and so a
    separate negotiation. The site answers the lookup and the fetch from
    different format lists often enough that a pinned id is a guess by the time
    it is used, and it comes back either as a 403 on a stream the listing had
    just offered or as the ids simply not being in the second list. Choosing
    again, in the same exchange that fetches, is the answer to both.

    Deliberately not every refusal. "The page needs to be reloaded" is the site
    declining the exchange itself — asking again straight away is asking a
    rate-limited host to serve twice as fast, which earns a longer refusal, not
    a video. That one is left to fail with advice to wait.
    """
    s = detail.lower()
    return ("requested format is not available" in s
            or ("403" in s and "forbidden" in s)
            or "unable to download video data" in s)


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


# Matches both names YouTube gives the same codec: avc1.640028 on some formats,
# h264 on others. A prefix match on one silently falls through to the next rung.
_H264 = r"[vcodec~='^(avc|h264)']"


def format_selector(quality: str = "best") -> str:
    """Build the yt-dlp format chain, as fallbacks from most to least wanted.

    The video stream is copied rather than re-encoded, so whatever is chosen here
    is what has to play on the far end. Every rung exists for a case that has
    actually been seen, and the chain always ends in a bare "b", so no video can
    be refused for want of a preferred format.

    quality is "best", "1080" or "720".

        1. bv*[H.264][cap]+ba   what we want: H.264, under the height cap
        2. bv*[cap]+ba          any codec still under the cap
        3. b[cap]               one combined stream, for the sites that have no
                                separate audio to merge
        4. b                    anything at all

    The boundary cases, all of which land on a later rung rather than failing:

    * A video offered only in AV1 or VP9 — rung 2. The finished file's codec is
      recorded, and the report says so if it is one older Macs cannot play.
    * A site with no separate audio stream — rung 3, or 4 when uncapped.
    * Nothing at or below the cap, which happens on a video published only at
      1440p or above — rung 4, which drops the cap rather than returning nothing.
      A larger file beats no file.
    * An unrecognised quality string — treated as "best", so a settings file
      edited by hand cannot produce an empty selector.
    * No usable formats at all, which is a live stream or a members-only video —
      yt-dlp fails and the message is translated for the user; there is no
      selector that can rescue it.

    """
    cap = f"[height<={quality}]" if quality in ("1080", "720") else ""
    rungs = [f"bv*{_H264}{cap}+ba", f"bv*{cap}+ba"]
    if cap:
        rungs.append(f"b{cap}")
    rungs.append("b")
    return "/".join(rungs)


# H.264 video and AAC audio in an MP4 is the one combination that plays
# everywhere worth caring about: every Mac and iPhone, every Android device of
# the last decade and a half, every browser, VLC, QuickTime and the smart TVs.
# It is preferred unconditionally — not per machine — because the file outlives
# the machine that made it. It gets sent to people, and a dub nobody else can
# open is not finished.
#
# Deliberately not H.265: smaller at the same quality, and the case it fails is
# exactly the one that matters here — Android support varies by chip and player,
# and Chrome and Firefox largely will not touch it. YouTube does not offer it
# for this material anyway; the codecs on a typical video are avc1, vp9 and av01.
#
# When H.264 is not on offer at all, nothing left is universally playable — VP9
# will not play in an MP4 in QuickTime whatever the Mac, and AV1 needs an M3.
# Ranking those against each other is false precision, so the smallest is taken
# and the report says plainly that the file may not play elsewhere.
def _codec_rank(vcodec: str) -> int:
    v = (vcodec or "").lower()
    return 0 if v.startswith(("avc", "h264")) else 1


def _size(f: dict) -> int:
    return int(f.get("filesize") or f.get("filesize_approx") or 0)


def choose_format(info: dict, quality: str = "best") -> dict | None:
    """Pick concrete formats from the list the site actually published.

    The probe already fetches every format and their sizes, and that was being
    thrown away in favour of a selector string and hope. Choosing here means the
    codec and the download size are both known before anything is fetched — the
    codec so an unplayable file can be avoided rather than reported afterwards,
    and the size so the disk check can use a real number instead of an estimate.

    Returns None when the listing is not usable — a live stream, a members-only
    video, or a site that publishes no format table — and the caller falls back
    to the selector string, which is what yt-dlp does best on its own.
    """
    formats = [f for f in (info.get("formats") or []) if f.get("format_id")]
    if not formats:
        return None

    def has_video(f):
        return f.get("vcodec", "none") not in ("none", None)

    def has_audio(f):
        return f.get("acodec", "none") not in ("none", None)

    cap = int(quality) if quality in ("1080", "720") else None
    video = [f for f in formats if has_video(f) and not has_audio(f)]
    audio = [f for f in formats if has_audio(f) and not has_video(f)]
    combined = [f for f in formats if has_video(f) and has_audio(f)]

    def under_cap(fs):
        # Dropping the cap beats returning nothing: a video published only above
        # it should still be dubbed, just larger.
        fitting = [f for f in fs if cap is None or (f.get("height") or 0) <= cap]
        return fitting or fs

    if video and audio:
        best_v = min(under_cap(video),
                     key=lambda f: (_codec_rank(f.get("vcodec")),
                                    -(f.get("height") or 0), _size(f)))
        # m4a first: it drops into an mp4 without being re-encoded. Stereo
        # before surround, because "highest bitrate" on its own picks the 5.1
        # track where a video has one — three times the bytes (30.8 MB against
        # 10.3 MB on one measured video) for audio that is either replaced
        # outright or downmixed to mono for the transcriber.
        best_a = min(audio, key=lambda f: (0 if f.get("ext") == "m4a" else 1,
                                           1 if (f.get("audio_channels") or 2) > 2 else 0,
                                           -(f.get("tbr") or 0)))
        return {"spec": f"{best_v['format_id']}+{best_a['format_id']}",
                "height": best_v.get("height"), "vcodec": best_v.get("vcodec") or "",
                "bytes": _size(best_v) + _size(best_a)}

    if combined:
        best = min(under_cap(combined),
                   key=lambda f: (_codec_rank(f.get("vcodec")),
                                  -(f.get("height") or 0), _size(f)))
        return {"spec": best["format_id"], "height": best.get("height"),
                "vcodec": best.get("vcodec") or "", "bytes": _size(best)}

    return None


# Names yt-dlp and its postprocessors are known to leave scratch state under
# (.part, .ytdl, .part-Frag<n>, source.temp.<ext>) — none of it the finished video.
def _is_scratch(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(".part") or lower.endswith(".ytdl") or ".part-frag" in lower:
        return True
    suffixes = Path(lower).suffixes
    return len(suffixes) >= 2 and suffixes[-2] == ".temp"


# The demuxers ffmpeg reads a still image through — see `ffmpeg -demuxers`.
# Every one of them describes itself as "piped X sequence" and is named
# accordingly, which is a property of the file (what read it), not a guess
# about its codec or its duration. yuv4mpegpipe is deliberately not swept up
# by the "*_pipe" pattern below: raw video frames, not a still-image
# sequence, and spelled without the underscore every image demuxer uses.
#
# gif is deliberately not in this set. Unlike jpeg_pipe or png_pipe, ffmpeg
# reads a one-frame thumbnail and a genuinely animated GIF through the same
# "gif" demuxer — format_name alone cannot tell them apart, and yt-dlp's own
# Imgur extractor offers a real 'ext': 'gif' format as a fallback for posts
# with no video transcode, which this app's format chain (ending in a bare
# "b" — anything at all) would legitimately select and download. See
# _is_single_frame_gif() for how that case is told apart instead.
_IMAGE_DEMUXERS = {"image2", "image2pipe"}


def _is_still_image(format_name: str) -> bool:
    names = (format_name or "").split(",")
    return any(n in _IMAGE_DEMUXERS or n.endswith("_pipe") for n in names)


def _is_single_frame_gif(format_name: str, nb_frames: str) -> bool:
    """A GIF is rejected only when ffprobe positively counts exactly one
    frame in it, straight from the container's own frame count — never from
    `-count_frames`, which would mean decoding the whole thing just to find
    out whether it was worth decoding.

    An unparseable or missing count is not treated as one frame: some GIF
    encoders leave nb_frames as "N/A" in the header, and calling that a
    thumbnail would silently throw away a real animated GIF for the same
    reason a duration floor did — proving a negative that the container
    never promised to make provable.
    """
    if "gif" not in (format_name or "").split(","):
        return False
    try:
        return int(nb_frames) == 1
    except (TypeError, ValueError):
        return False


def _looks_like_media(path: Path) -> bool:
    """Ask ffprobe what kind of file this is, rather than trusting its name or
    size — yt-dlp's config can drop a thumbnail or info.json sidecar next to
    the video under the same "source." prefix, and neither looks like scratch.

    Only a positive match against ffmpeg's own image demuxers (or a
    single-frame GIF) counts as not-media; an unreadable probe, or a stream
    that is neither audio nor video, is assumed to be a real video so a
    finished download is never thrown away for want of proof.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,nb_frames:format=format_name",
             "-of", "default=noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        # No answer from ffprobe is not evidence that this is a thumbnail —
        # it is not evidence of anything. Accepting it and letting the real
        # ffmpeg invocation downstream fail on its own, loudly, with its own
        # error attached, is strictly better than silently discarding a
        # completed download because the probe that was only ever meant to
        # catch a sidecar happened to time out.
        return True
    kinds, format_name, nb_frames = set(), "", ""
    for line in out.stdout.splitlines():
        key, _, value = line.strip().partition("=")
        if key == "codec_type":
            kinds.add(value)
        elif key == "format_name":
            format_name = value
        elif key == "nb_frames":
            nb_frames = value
    if "audio" in kinds:
        return True
    if _is_still_image(format_name) or _is_single_frame_gif(format_name, nb_frames):
        return False
    return "video" in kinds


def _finished_file(workdir: Path) -> Path | None:
    """The video that was actually downloaded, not whatever sorts first — a
    leftover .part fragment from an interrupted first try can sort ahead of the
    completed source.mp4 a retry then wrote.

    Excludes scratch state and anything ffprobe identifies as a still image,
    then prefers the merged filename, falling back to the largest candidate.
    """
    candidates = [p for p in workdir.glob("source.*")
                  if p.is_file() and not _is_scratch(p.name)]
    media = [p for p in candidates if _looks_like_media(p)]
    if not media:
        return None
    merged = [p for p in media if p.stem == "source"]
    return max(merged or media, key=lambda p: p.stat().st_size)


def _explain_missing_video(workdir: Path) -> str:
    """What download() puts in a DownloadError's detail when _finished_file()
    comes back empty, so "yt-dlp wrote nothing" can be told apart from
    "yt-dlp wrote something and it was turned away" — the difference between
    a bug here and a bug in whatever the user linked to.

    This only runs after download() has already decided to fail, and its own
    contract is best-effort: one guard round the whole body means a vanished
    file or an unreadable directory falls back to a vaguer detail instead of
    replacing the friendly error with a raw traceback.
    """
    try:
        if not workdir.is_dir():
            return f"{workdir} is not a directory that exists."
        everything = sorted(p.name for p in workdir.iterdir())
        if not everything:
            return f"{workdir} is empty — yt-dlp wrote nothing here."
        sourced = sorted((p for p in workdir.glob("source.*") if p.is_file()),
                          key=lambda p: p.name)
        if not sourced:
            return (f"nothing named source.* turned up. Everything actually in "
                    f"{workdir}: {', '.join(everything)}")
        notes = []
        for p in sourced:
            if _is_scratch(p.name):
                notes.append(f"{p.name} (yt-dlp scratch state, not a finished file)")
            elif not _looks_like_media(p):
                notes.append(f"{p.name} (ffprobe read this as a still image, not a video)")
            else:
                notes.append(f"{p.name} (looked like a finished video — this is a bug "
                              f"in _finished_file(), not in the download)")
        return f"considered and rejected: {'; '.join(notes)}"
    except OSError:
        return f"{workdir} couldn't be explained further."


def download(url: str, workdir: Path, quality: str = "best",
             progress: Progress = None, info: dict | None = None,
             cookies_from: str = "") -> tuple[Path, dict]:
    """info: a probe() result the caller already has, to save asking twice.

    The preview needs the duration before it can weight the progress bar, which
    means probing before this stage rather than inside it.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    info = info or probe(url, cookies_from)
    if progress:
        progress(0.02, f"Found “{info['title']}”")

    # Chosen from the published list where there is one, with the selector
    # string kept behind it as a fallback for anything that list cannot answer.
    picked = choose_format(info, quality)
    fmt = format_selector(quality)
    if picked:
        fmt = f"{picked['spec']}/{fmt}"
        if progress:
            size = f" — about {human_size(picked['bytes'])}" if picked["bytes"] else ""
            progress(0.03, f"Fetching {picked['height']}p{size}")

    target = workdir / "source.%(ext)s"
    cmd = _ytdlp_cmd() + [
        "-f", fmt, "--merge-output-format", "mp4",
        # Deliberately no --user-agent. YouTube ties a media URL's authorisation
        # to the client yt-dlp negotiated it as, so overriding the agent
        # *desyncs* the two: a hand-set desktop Chrome string turns a download
        # that works into a reliable 403. yt-dlp's own default is correct.
        "--newline", "--no-warnings",
        "--progress-template", _PROGRESS_TEMPLATE,
    ] + _CLIENT_ARGS + _RETRY_ARGS + _NO_CONFIG_ARGS
    # The fix for the stubborn case: YouTube increasingly wants a session, and a
    # signed-out request for some videos is refused whatever client asks. Off
    # unless the user names a browser, because reading their cookie store is not
    # something to do quietly on their behalf.
    if cookies_from:
        cmd += ["--cookies-from-browser", cookies_from]
    cmd += ["-o", str(target), url]

    # Picture and sound arrive as two separate downloads, each of which counts
    # its own bytes from zero. Reported as-is, the bar filled up and then
    # started again from nothing — which to anyone not watching the byte counts
    # looks like the job restarting itself near the end. Summed here instead,
    # against the total the format listing gave us, so it only ever goes forward.
    done = {"before": 0, "last": 0}
    expected = picked["bytes"] if picked else 0

    def fetched() -> int:
        """What is actually on disk for this download.

        The reported counter cannot be trusted on its own across a retry. A
        403 part way through leaves a part file behind, the next attempt is
        answered with "Resuming download at byte 288019941", and what the
        counter does from there depends on whether the format is served as one
        range request or as fragments. Measured on the video that failed for a
        user: the bar sat at "0% of 1.7 GB" while the file on disk passed 1.4 GB.

        The file has no opinion about any of that. Used as a floor rather than a
        replacement, so a site that streams into a single handle still reports
        normally.
        """
        try:
            return sum(f.stat().st_size for f in workdir.glob("source.*"))
        except OSError:
            return 0

    # yt-dlp's own words, with the progress records left out. --progress-template
    # emits a DUBPROG line per update and a large download emits thousands, so
    # the tail stream() keeps was all progress and no diagnosis. Seen in the
    # field: a real 403 reported with a detail that began "DUBPROG|64512|3" and
    # never reached the error at all. Collected here rather than filtered in
    # stream(), which drives Demucs too and has no business knowing what a
    # progress line looks like.
    said: list[str] = []

    def show(line: str) -> None:
        if not line.startswith(_PROGRESS_PREFIX):
            said.append(line)
            del said[:-25]
            return
        if not progress:
            return
        got, total = (line[len(_PROGRESS_PREFIX):].split("|") + ["", ""])[:2]
        try:
            got_b = int(got)
        except ValueError:
            return                      # "NA" until anything is known
        try:
            total_b = int(total)
        except ValueError:
            total_b = 0                 # some formats never publish one
        if got_b < done["last"]:        # counted down: the next stream began
            done["before"] += done["last"]
        done["last"] = got_b
        # Falls back to the stream's own total when the listing had no size, so
        # a site that publishes none still gets a moving bar rather than none.
        whole = expected or (done["before"] + total_b)
        if whole <= 0:
            return
        overall = max(done["before"] + got_b, fetched())
        share = max(0.0, min(1.0, overall / whole))
        progress(0.02 + 0.96 * share,
                 f"Downloading — {share * 100:.0f}% of {human_size(whole)}")

    code, problem = stream(cmd, show, tail_lines=25)
    # stream()'s own tail stays as the fallback, for a failure that produced
    # nothing but progress lines before dying.
    detail = "".join(said).strip() or problem
    if code != 0 and _format_was_refused(detail):
        # Every one of these is a refusal of the exchange rather than of the
        # video: the same argv, run again by hand a minute later, fetches it.
        # YouTube rate-limits and answers each negotiation differently, and
        # yt-dlp's own --extractor-retries does not count these as retryable, so
        # one refusal ended a job that a second ask would have got through.
        #
        # Asked again from scratch, which re-negotiates. When ids were pinned the
        # chain replaces them too: those came from probe(), a negotiation that
        # has already ended, and yt-dlp only walks the chain while it is
        # choosing — a format it chose and then could not fetch is the end of the
        # run rather than a reason to try the next rung.
        if progress:
            progress(0.02, "The site refused that; asking again")
        said.clear()
        done["before"] = done["last"] = 0
        expected = 0                     # possibly a different format, so a different size
        retry = list(cmd)
        if picked:
            retry[retry.index("-f") + 1] = format_selector(quality)
        code, problem = stream(retry, show, tail_lines=25)
        detail = "".join(said).strip() or problem
    if code != 0:
        raise DownloadError(_friendly(detail), detail)

    video = _finished_file(workdir)
    if video is None:
        # yt-dlp exited 0, so there is no stderr to relay — kept as a
        # DownloadError anyway so the UI's error pane and Copy details still
        # get something useful instead of nothing.
        raise DownloadError(
            "The download finished but no video file appeared.",
            f"yt-dlp reported success but _finished_file() found nothing usable "
            f"in {workdir} — {_explain_missing_video(workdir)}",
        )
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
