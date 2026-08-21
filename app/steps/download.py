"""Fetch a video with yt-dlp and pull its audio out."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
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

# yt-dlp reads its own configuration files by default — portable, home, user
# and system, in that order — and silently merges whatever it finds into every
# job this app runs. Demonstrated on a real job through the real UI: a
# --write-thumbnail left in a user's yt-dlp config turned an ordinary download
# into source.mp4 (633,710 bytes) *and* source.webp (23,038 bytes) sitting in
# the same working directory _finished_file() scans, which is how a thumbnail
# nearly got handed to the pipeline as the video. That particular hole is now
# closed by the media check in _looks_like_media(), but the cause is this: the
# command built here is not actually the whole command, because anything in
# that config rides along uninvited and can contradict the flags chosen above
# it — a different output template, a different format, a post-processor that
# writes files this app never asked for.
#
# The one thing worth weighing against this is that some configs hold a real
# proxy, rate limit or bound interface that a user would miss. It is not
# preserved. Settings already has its own route for the setting people
# actually reach for here — "Sign in as" passes --cookies-from-browser
# directly, with no config file involved either way — and a lost proxy fails
# loud: the fetch times out or is refused, which reads as "that address
# couldn't be reached" in _friendly() below, not as a wrong file quietly
# written to disk. A loud network error beats a silent extra file every time
# that trade has to be made.
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
    """
    cmd = (_ytdlp_cmd() + ["-J", "--no-warnings", "--skip-download"]
           + _CLIENT_ARGS + _RETRY_ARGS + _NO_CONFIG_ARGS)
    if cookies_from:
        cmd += ["--cookies-from-browser", cookies_from]
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


# Names yt-dlp and its own postprocessors are known to leave scratch state
# under, none of it the video. Not a claim that this is every artefact either
# could ever write — only the ones with a demonstrated reason to be here:
#   * .part      the download target while a stream is still arriving
#                (FileDownloader.temp_name in yt_dlp/downloader/common.py);
#                renamed away once complete.
#   * .ytdl      the resume bookkeeping FileDownloader.ytdl_filename writes
#                beside a fragmented download — progress state, not bytes of
#                the video.
#   * .part-Frag<n>  one fragment of a still-assembling fragmented download
#                (FragmentFD.fragment_filename in downloader/fragment.py),
#                joined into the real file and deleted once every fragment
#                has arrived; left behind by an attempt that was interrupted
#                before that happened, and sometimes still wearing its own
#                trailing .part while that fragment itself is mid-write.
#   * source.temp.<ext>  where FFmpegMergerPP builds the merged file before
#                renaming it over the final name (postprocessor/ffmpeg.py,
#                prepend_extension(filename, 'temp')) — present for as long
#                as the mux takes, on every job that downloads separate video
#                and audio.
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
    """Ask ffmpeg's own prober whether a file is a still image, rather than
    trusting its name, its size, or asking the harder question of whether it
    is definitely a video.

    Excluding scratch state is not enough on its own: yt-dlp's config can ask
    it to write sidecars — a thumbnail, an info.json — into this same
    directory under the same "source." prefix, and neither looks like scratch
    to the rules above. Demonstrated on a real job with --write-thumbnail set
    in the user's own yt-dlp config (which this app deliberately leaves
    readable — see download()): it left behind source.mp4 at 633,710 bytes
    and source.webp at 23,038 bytes, both matching "source.*" and neither
    a .part or a .ytdl. A size-only or name-only rule has no way to tell them
    apart; only actually looking at what is in the file does.

    Two things that look like they would tell a thumbnail from a video do
    not, and both were tried and measured wrong before this one:

    * codec_type alone. ffmpeg's own image demuxers hand a still image back
      as a one-frame "video" stream, because that is how ffmpeg represents
      every image format internally, so "is there a video stream" accepts a
      thumbnail exactly as readily as it accepts a real one.
    * a duration floor. A completely genuine, fully downloaded video-only
      remux — muxed to a non-seekable output, or a mux the writer never got
      to finalise, which is what a live-piped or interrupted-but-complete
      HLS/DASH source looks like — decodes every frame correctly and still
      reports no duration at all, because the container's duration field is
      normally patched in by seeking back after the last frame is known, and
      a stream that was never seekable, or never reached a clean close,
      never gets that pass. A duration check throws that file away exactly
      the way the container allowlist did, just one layer further in.

    What is reliable is what demuxer ffprobe actually read the file through:
    a thumbnail goes through one of ffmpeg's dedicated image demuxers
    (jpeg_pipe, png_pipe, ...) — see _is_still_image() — and nothing that
    demuxer produces is a video, whatever codec_type or duration it happens
    to report. GIF is the one demuxer name that is not by itself enough to
    say which: a thumbnail and a genuinely animated GIF both read as
    format_name=gif, and yt-dlp really does offer an animated GIF as a
    downloadable format on some sites (Imgur, when a post has no video
    transcode) — see _is_single_frame_gif(), which asks how many frames are
    actually in it instead. So a still image is the one thing this rejects on
    a positive finding rather than a failure to prove otherwise: an audio
    stream, a video stream — of any duration, including none reported at all
    — or a probe that could not run at all count as not-an-image, because the
    one thing this app must never do is throw away a video it actually
    finished downloading for want of proof. A probe that *did* run and found
    neither a video nor an audio stream, and no sign of an image demuxer
    either — an info.json sidecar, say, which is not a still image but is
    also not a video — is a genuine answer, not a missing one, and is
    rejected on it: "unknown" only describes the case where nothing was heard
    back at all.

    ffmpeg is already a hard dependency and ffprobe is already used for this
    kind of question in media_duration() below.
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
    """The video that was actually downloaded, not whatever sorts first.

    This was `sorted(workdir.glob("source.*"))[0]`, which is correct exactly
    until an attempt is interrupted. A refused first try leaves
    `source.f137.mp4.part` behind; the retry succeeds and writes `source.mp4`;
    and `source.f137.mp4.part` sorts first. Measured on the video that failed for
    a user: a complete 1.7 GB download, and the pipeline handed ffmpeg the 9.7 MB
    fragment beside it and reported that the video was in a format it could not
    read.

    This used to also require the suffix be one of a handful of named
    containers, which was the wrong shape of check: ffmpeg reads far more than
    the five that were named there, so a genuinely complete download in one of
    the others — a Wikimedia .ogv, an archive.org .mpg, both read by ffmpeg
    without complaint — was thrown away as if the fetch had failed, right after
    every byte of it had arrived. So the check is inverted twice over: exclude
    what is definitely still yt-dlp's own scratch state, exclude what ffprobe
    positively identifies as a still-image sidecar rather than requiring proof
    of the opposite, and apply the same two rules as before to what is left —
    the merged name preferred over a single format's, and the largest of what
    remains. Nothing here can pick a stub or a thumbnail, and nothing here can
    throw away a video for want of a container name or a duration it was never
    going to have.
    """
    candidates = [p for p in workdir.glob("source.*")
                  if p.is_file() and not _is_scratch(p.name)]
    media = [p for p in candidates if _looks_like_media(p)]
    if not media:
        return None
    merged = [p for p in media if p.stem == "source"]
    return max(merged or media, key=lambda p: p.stat().st_size)


def _safe_is_file(p: Path) -> bool:
    try:
        return p.is_file()
    except OSError:
        return False


def _explain_missing_video(workdir: Path) -> str:
    """What download() puts in a DownloadError's detail when _finished_file()
    comes back empty, so a pasted Copy details can tell "yt-dlp wrote nothing"
    from "yt-dlp wrote something and it was turned away" — the two look
    identical from the outside otherwise, and only one of them is a bug in
    this file rather than in whatever the user linked to.

    Re-walks the same directory _finished_file() just did rather than having
    that function collect notes as it goes, because this only ever runs on
    the rare failing path: the extra ffprobe calls it costs are calls that
    were never going to happen on a job that actually worked.

    Every filesystem access below is guarded, deliberately more defensively
    than _finished_file() itself needs to be. This only ever runs after
    download() has already decided to fail; turning that failure into an
    unhandled PermissionError or FileNotFoundError instead of the
    DownloadError it was in the middle of building would be strictly worse
    than the detail merely being vague — the friendly message would never
    reach the user at all, replaced by a raw traceback in the generic handler
    in app/pipeline.py. So nothing here may raise: not an unreadable workdir,
    not a workdir that was actually a file all along, and not a candidate
    that existed when it was listed and is gone by the time it is asked
    about. Path.is_dir() and Path.glob() already swallow OSError on their
    own — glob() skips an unreadable directory as if it had no matches,
    which is what let _finished_file() return a plain None on the same input
    that made the first version of this function raise — but Path.iterdir(),
    used below to list what is actually in the directory, does not, so this
    (deliberately unlike _finished_file()) does not use it without a guard.
    """
    try:
        is_dir = workdir.is_dir()
    except OSError:
        is_dir = False
    if not is_dir:
        return f"{workdir} is not a directory that exists."
    try:
        everything = sorted(p.name for p in workdir.iterdir())
    except OSError as exc:
        return f"{workdir} exists but could not be listed ({exc})."
    if not everything:
        return f"{workdir} is empty — yt-dlp wrote nothing here."
    try:
        sourced = sorted((p for p in workdir.glob("source.*") if _safe_is_file(p)),
                          key=lambda p: p.name)
    except OSError:
        sourced = []
    if not sourced:
        return (f"nothing named source.* turned up. Everything actually in "
                f"{workdir}: {', '.join(everything)}")
    notes = []
    for p in sourced:
        try:
            if _is_scratch(p.name):
                notes.append(f"{p.name} (yt-dlp scratch state, not a finished file)")
            elif not _looks_like_media(p):
                notes.append(f"{p.name} (ffprobe read this as a still image, not a video)")
            else:
                notes.append(f"{p.name} (looked like a finished video — this is a bug "
                              f"in _finished_file(), not in the download)")
        except OSError:
            notes.append(f"{p.name} (vanished or became unreadable while being checked)")
    return f"considered and rejected: {'; '.join(notes)}"


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
    if code != 0:
        # stream()'s own tail stays as the fallback, for a failure that produced
        # nothing but progress lines before dying.
        detail = "".join(said).strip() or problem
        raise DownloadError(_friendly(detail), detail)

    video = _finished_file(workdir)
    if video is None:
        # yt-dlp exited 0, so there is no stderr to hand back the way every
        # other failure in this file does — but the same DownloadError shape
        # is used anyway, because this is the one case that used to raise a
        # bare RuntimeError and leave the UI's "what the downloader actually
        # said" pane and Copy details with nothing in them at all. What is
        # actually sitting in the working directory is the only useful thing
        # left to report.
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
