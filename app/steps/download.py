"""Get hold of the video — fetched from a site with yt-dlp, or already on this
Mac — and pull its audio out."""
from __future__ import annotations

import datetime
import functools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

from ..storage import human_size
from .proc import stream

Progress = Optional[Callable[[float, str], None]]


# --------------------------------------------------- which kind of source is it

# Used only to tell a filename apart from a scheme-less web address: "clip.mov"
# and "youtube.com/watch" are both just a name with a dot in it.
MEDIA_SUFFIXES = frozenset({
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".wmv",
    ".flv", ".ts", ".m2ts", ".mts", ".3gp", ".ogv", ".mxf", ".vob", ".divx",
    ".asf", ".rm", ".rmvb", ".f4v", ".m2v",
    # Audio included too: "Mr.Robot/s01e01.mp3" is no more a web address than
    # "s01e01.mkv" is, and this set is what decides that.
    ".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".oga", ".opus", ".wma",
    ".aif", ".aiff", ".alac", ".ape", ".mka",
})

# A host with at least one dot in it, an optional port, and then the end or the
# start of a path. Anchored so it cannot match anything that begins the way a
# path does.
_BARE_LINK = re.compile(r"^(?![/~.])[\w-]+(\.[\w-]+)+(:\d+)?([/?#]|$)")

# Domain endings, consulted only to tell an address from a folder that also
# happens to look like a video name. Deliberately far short of the real
# registry — looks_like_bare_link() reads anything unlisted as a folder, the
# safer of the two wrong answers.
#
# Matched as typed, not lowercased: a domain is written lower case and a
# release tag upper, e.g. "The.Movie.2020.1080p.WEB.YTS.AM" vs anchor.fm.
# Comparing case-insensitively answered those folders with "put https:// on
# the front".
_LINK_ENDINGS = frozenset({
    "com", "net", "org", "info", "biz", "cloud", "online", "edu", "gov", "xyz",
    "io", "fm", "tv", "co", "me", "gg", "ly", "ai", "cc", "sh", "app", "dev",
    # Country codes, without which "bbc.co.uk/v/clip.mp4" reported a made-up
    # local path. Safe case-sensitively for the same reason as above: real
    # release tags that collide (.AM, .US, .NO) are written upper case.
    "uk", "de", "fr", "nl", "au", "jp", "eu", "es", "it", "ca", "br", "in",
    "se", "no", "dk", "fi", "ch", "at", "be", "pl", "ru", "nz", "za", "ie",
    "pt", "gr", "cz", "kr", "tw", "hk", "sg", "mx", "ar", "il", "ua", "ro",
    "hu", "tr", "cn", "us",
})


def normalise_source(raw: str) -> str:
    """Settle, once, what somebody put in the one box the app offers.

    A web link is left exactly as typed. yt-dlp is the authority on what it will
    accept, and rewriting a link on the way there can only get in its way.

    Anything else is a file on this Mac, and is resolved to a single absolute
    path here rather than being half-interpreted in each of the places
    downstream that would otherwise have to try. That is not tidiness: the job
    folder is named after a hash of this string, so "~/Movies/a.mp4" and
    "/Users/me/Movies/a.mp4" would otherwise be two jobs racing over one video.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    # Stripped before the http:// check: a quoted link ("...") was previously
    # sent down the file branch and reported as a missing file at
    # ".../https:/www.youtube.com/watch". Neither a real path nor a real URL
    # carries a matching quote pair, so removing it is always safe.
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
        if not text:
            return ""
    if text.lower().startswith(("http://", "https://")):
        return text
    if text.lower().startswith("file://"):
        # Finder and browsers both hand out percent-encoded file URLs. The host
        # part is either empty or "localhost"; neither belongs in a path.
        text = unquote(urlparse(text).path)
    try:
        return str(Path(text).expanduser().resolve())
    except (OSError, RuntimeError):
        # An unresolvable path is still the answer somebody gave, and the caller
        # has far better things to say about it than a traceback from here.
        return os.path.expanduser(text)


def is_local_source(source: str) -> bool:
    """True for a file on this Mac, false for something yt-dlp should fetch.

    Asked of a normalise_source() result, which has already turned any file://
    URL into a path — so the whole of the question is whether it is still a link.
    """
    return bool(source) and not source.lower().startswith(("http://", "https://"))


def looks_like_bare_link(typed: str) -> bool:
    """A web address with the scheme left off, told apart from a filename.

    Only worth asking once a path has been found not to exist. At that point the
    difference between "there is no file there" and "put https:// on the front"
    is the difference between advice that works and advice that doesn't.
    """
    text = (typed or "").strip().strip("\"'")
    if not text or not _BARE_LINK.match(text):
        return False
    if Path(text).suffix.lower() not in MEDIA_SUFFIXES:
        return True
    # Named like a video, which the pattern above cannot settle alone:
    # "clip.mov", "Season.1/ep01.mkv" and "cdn.example.com/talks/clip.mov" all
    # match it, and only the last is a link — told apart by whether the part
    # before the first separator is a *host*.
    cut = re.search(r"[/?#]", text)
    if not cut or not text[cut.end():]:
        # All name and no path, so a filename. "clip.mov/" still counts: Path()
        # drops the trailing slash before the suffix test above sees it.
        return False
    host = text[:cut.start()]
    # _BARE_LINK allows a port on the host, so it must be stripped here too.
    host = re.sub(r":\d+$", "", host)
    if re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host):
        return True                # nothing is named 192.168.1.5
    # Whether the last label is a domain ending, the only test that separates
    # "cdn.example.com/talks/clip.mov" from "Mr.Robot/s01e01.mkv" — both are
    # words either side of a dot. Unlisted endings read as a folder on
    # purpose (the safer wrong answer), which is also why ordinary words like
    # .video are left out of _LINK_ENDINGS.
    return host.rsplit(".", 1)[-1] in _LINK_ENDINGS


# --------------------------------------------------------- fetching from a site

# A machine-readable progress format, not yt-dlp's human-facing "[download]
# 12.3%" text, which is free to change between releases and carries no byte
# counts.
_PROGRESS_PREFIX = "DUBPROG|"
_PROGRESS_TEMPLATE = (
    "download:" + _PROGRESS_PREFIX
    + "%(progress.downloaded_bytes)s|%(progress.total_bytes,progress.total_bytes_estimate)s"
)


def _ytdlp_cmd() -> list[str]:
    """The yt-dlp inside this app's own environment, and no other.

    Deliberately not shutil.which("yt-dlp"): the launcher's .venv/bin leads
    PATH, so which() finds the pip-installed copy, while the installer's
    `brew update && brew upgrade yt-dlp` freshens a different binary further
    down PATH — the two drift apart, and the stale one is the one that ran.
    Measured: a 6-week-newer Homebrew copy fetched a video the pip copy died
    on with "HTTP Error 403", because it could negotiate a client the older
    one lacked entirely.

    sys.executable pins this to the interpreter already running — the venv the
    app ships and the installer upgrades — rather than whatever bare "python3"
    PATH happens to resolve to, which need not have yt_dlp installed at all.
    """
    return [sys.executable, "-m", "yt_dlp"]


# No player_client pin, on purpose. This used to name
# "default,web_safari,tv" so the probe and the download negotiated the same
# client list and agreed on format ids — but naming clients freezes a
# judgement that goes stale: YouTube has since retired all three, and a list
# written here cannot follow. Both stages simply not asking (relying on
# yt-dlp's own current defaults) preserves the same probe/download agreement
# without freezing anything.

# yt-dlp silently merges its own config files (portable, home, user, system)
# into this command by default, which can add unwanted outputs or override
# flags chosen here. --ignore-config keeps the command fully specified; "Sign
# in as" still reaches --cookies-from-browser directly either way.
_NO_CONFIG_ARGS = ["--ignore-config"]

# Real per-request exponential backoff, replacing a whole-command retry loop
# with a fixed 4n sleep that restarted the progress count each time and gave
# up the bytes already on disk. A 403 part way through a large fetch is what
# --retries and http backoff exist for.
_RETRY_ARGS = [
    "--retries", "10", "--fragment-retries", "10", "--extractor-retries", "5",
    "--retry-sleep", "http:exp=1:60",
    "--retry-sleep", "fragment:exp=1:30",
    "--retry-sleep", "extractor:exp=1:30",
]


# Deliberately no --no-warnings on either command. yt-dlp explains an
# impending failure in a warning that its error text does not repeat — a 403
# here was preceded by "challenge solving failed" and a skipped-solver note,
# neither of which reached the log while warnings were off. Costs nothing:
# stream() folds stderr into stdout anyway.


def probe(url: str, cookies_from: str = "") -> dict:
    """Ask the site what it has, without fetching any of it.

    cookies_from matters here too, not just in download(): an age-restricted
    or members-only video refuses at the *lookup*, so skipping it here would
    turn away a user who'd already set "Sign in as" before download() ever ran.

    --ignore-no-formats-error because this asks what exists, not for a stream:
    without it, yt-dlp resolves its default selector even under -J and fails
    the whole lookup when nothing satisfies it — something YouTube does
    intermittently to videos it serves fine a minute later. Errors worth
    keeping still land: a private/age-restricted video fails extraction
    entirely, a different error that still raises with its own advice.
    """
    cmd = (_ytdlp_cmd() + ["-J", "--skip-download",
                           "--ignore-no-formats-error"]
           + _RETRY_ARGS + _NO_CONFIG_ARGS)
    if cookies_from:
        cmd += ["--cookies-from-browser", cookies_from]
    out = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=120)
    if out.returncode != 0 and _format_was_refused(out.stderr):
        # Same refusals download() sees, for the same reason: the site answers
        # each negotiation differently. Asked once more, which re-negotiates.
        out = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise DownloadError(_friendly(out.stderr), out.stderr)
    data = json.loads(out.stdout)
    return {
        "title": data.get("title", "video"),
        "duration": float(data.get("duration") or 0),
        "uploader": data.get("uploader", ""),
        "thumbnail": data.get("thumbnail", ""),
        # Kept rather than discarded: choose_format() reads this to pick a
        # codec and know the real download size up front. Stays in memory
        # only; the caller writes its own small record to info.json.
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


# How old a yt-dlp has to be before it is the first thing to suspect. YouTube
# retires the player clients an older copy knows how to ask as, and the copy that
# produced the report this was written from was 51 days old.
YTDLP_STALE_DAYS = 30


@functools.lru_cache(maxsize=1)
def ytdlp_version() -> str:
    """The version string of the yt-dlp this app runs, or "" if it won't say.

    Cached: it costs a subprocess, _friendly() may ask for it several times over
    a run, and the answer cannot change without the app being restarted — an
    update re-execs the installer.
    """
    try:
        out = subprocess.run(_ytdlp_cmd() + ["--version"], capture_output=True,
                             text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = (out.stdout or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def ytdlp_age_days() -> int | None:
    """How many days old this yt-dlp is, or None when it cannot be dated.

    yt-dlp versions are release dates, so this needs no network. A build that is
    not three numbers — a distribution's own string, a git checkout — is not old,
    it is unknown, and unknown must not be reported as a fault.
    """
    try:
        year, month, day = (int(part) for part in ytdlp_version().split(".")[:3])
        return (datetime.date.today() - datetime.date(year, month, day)).days
    except (TypeError, ValueError):
        return None


def _stale_lead() -> str:
    """The sentence a refusal should open with when this yt-dlp is old enough to
    be the cause — and nothing at all when it is not.

    Exists because a public, browser-playable video was once refused with a
    403 and told the user to check cookies or the video's privacy — wrong on
    both counts, since every player client that copy knew had been retired.
    Confidently wrong advice is worse than none.
    """
    days = ytdlp_age_days()
    if days is None or days <= YTDLP_STALE_DAYS:
        return ""
    return (f"This copy of yt-dlp is {days} days old, and that is the likeliest "
            f"reason on its own: YouTube retires the players an older copy knows "
            f"how to ask as, and what that looks like is exactly this. Press "
            f"“Update yt-dlp” under Setup check, then try again. If it still "
            f"refuses: ")


def _why_refused(s: str) -> str:
    """The sentence a refusal opens with when something here explains it.

    Two things on this machine can cause a refusal a browser wouldn't get: a
    stale yt-dlp, or one that can't answer the JavaScript challenge media URLs
    are signed with. Both are named up front because both are fixed by the
    same click.

    Only ever one of them, as a lead rather than a branch of _friendly(): the
    challenge-failure text is emitted on every extraction on a machine with no
    JS runtime, so branching on it swallowed unrelated failures — a dropped
    network connection was once reported as "press Update yt-dlp" instead of
    "check your internet connection".
    """
    # Age first: a known number beats a diagnosis that has to hedge.
    stale = _stale_lead()
    if stale:
        return stale
    if "challenge solving failed" in s:
        # Doesn't promise the "update" click here: this copy isn't stale (or
        # the branch above would've fired), so it's usually a missing JS
        # runtime — which updating can't fix, and "already current" would
        # disprove the advice in one click.
        return ("yt-dlp couldn't answer YouTube's download check, and that is "
                "the likeliest reason on its own. Setup check will say whether "
                "it wants updating, and the error details below name what it "
                "could not load. Otherwise: ")
    return ""


def _friendly(stderr: str) -> str:
    s = stderr.lower()
    if "403" in s and "forbidden" in s:
        # By the time this is read, retries have already happened — so "try
        # again" isn't offered here. What this machine did wrong (if anything)
        # is named first via _why_refused(), since it's the one thing fixable
        # in a click; sign-in advice still follows as the fallback answer.
        return (_why_refused(s) +
                "YouTube described the video but refused to send it, on every "
                "attempt. It usually wants a signed-in session: set “Sign in as” "
                "in Settings to the browser you watch YouTube in. Otherwise the "
                "video may be private, age-restricted or members-only.")
    if "private video" in s:
        return "That video is private, so it can't be downloaded."
    # Named signals only. `"age" in s and "restricted" in s` used to match
    # here — those letters also occur in "package"/"page"/"message", and
    # yt-dlp's skipped-solver advisory says "NPM package" every time, so a
    # plain "restricted" video was misdiagnosed as age-restricted.
    if ("sign in to confirm your age" in s or "age-restricted" in s
            or "age restricted" in s or "inappropriate for some users" in s):
        return "That video is age-restricted and can't be fetched without signing in."
    if "video unavailable" in s:
        return "That video is unavailable — check the link is still live."
    if "unsupported url" in s:
        return "That link isn't one yt-dlp recognises."
    if "http error 429" in s or "too many requests" in s:
        return "The site is rate-limiting downloads. Wait a few minutes and try again."
    # yt-dlp passes this through from YouTube verbatim, but there's no page to
    # reload — it means the exchange was declined (challenge failed, or rate
    # limited) and clears on its own, faster for being left alone.
    #
    # Deliberately not led by _why_refused(): its leads end in "try again",
    # which directly contradicts the wait-don't-retry advice below.
    if "page needs to be reloaded" in s:
        return ("YouTube declined the request rather than the video. This happens "
                "after a burst of downloads and clears on its own. Wait a few "
                "minutes and press Try again — retrying straight away makes the "
                "wait longer.")
    # yt-dlp's own text tells someone to pass a --list-formats flag and reads
    # like a fault in the video. It isn't: the listing exists but nothing in it
    # is offered, usually a defensive response to repeated requests or the
    # sign the wanted formats were what an unanswered challenge withheld.
    if "requested format is not available" in s:
        return (_why_refused(s) +
                "YouTube listed the video but offered no version to download. "
                "That is usually temporary — wait a few minutes and try again. If "
                "it persists, set “Sign in as” in Settings to the browser you "
                "watch YouTube in.")
    # A dead link (typo, stale bookmark, half a URL) used to fall through to
    # yt-dlp's raw "ERROR: [generic] does-not-exist: Unable to download
    # webpage: HTTP Error 404: File not found (caused by ...)".
    if "http error 404" in s or "unable to download webpage" in s:
        return ("That link couldn't be opened. Check it is typed correctly and that "
                "the video is still there.")
    if ("name or service not known" in s or "nodename nor servname" in s
            or "temporary failure in name resolution" in s):
        return "That address couldn't be reached — check your internet connection."
    # Fallback for an unrecognised error, kept alongside the challenge lead
    # above (which only explains a refusal this function already recognises)
    # rather than instead of it — otherwise any unrecognised error went out as
    # raw yt-dlp text with no lead at all.
    tail = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    # Only fires when every line is a warning. Checking for a line *starting*
    # with ERROR was too narrow — neither an "OSError: ... No space left on
    # device" traceback nor "yt-dlp: error: unrecognized arguments" starts with
    # it. And the challenge-failure text alone is too broad to gate on: it's
    # emitted on every extraction with no JS runtime, even successful ones, so
    # testing it against stderr that also holds a real error hid that error —
    # a full disk once came back as "press Update yt-dlp". This only takes over
    # when there's genuinely no ERROR line, just warnings like "WARNING:
    # [youtube] [jsc] Remote components challenge solver script (deno) …".
    if (("challenge solving failed" in s or "challenge solver" in s)
            and all(ln.lstrip().upper().startswith("WARNING") for ln in tail)):
        return ("yt-dlp couldn't answer YouTube's download check, which is the "
                "likeliest reason this failed. Setup check will say whether it "
                "wants updating, and the error details below name what it could "
                "not load.")
    return _tidy(tail[-1]) if tail else "The download failed."


def _format_was_refused(detail: str) -> bool:
    """Whether the *pinned formats* were the problem, so a fresh pick may not be.

    choose_format() reads its ids from probe(), a separate yt-dlp run and so a
    separate negotiation — the site often answers the two with different
    format lists, making a pinned id a guess by the time it's used (a 403, or
    the id simply missing from the second list). Re-choosing in the same
    exchange that fetches answers both.

    Deliberately not every refusal: "the page needs to be reloaded" is the
    site declining the exchange itself, and asking again immediately just
    earns a longer refusal from a rate-limited host — left to fail with
    advice to wait instead.
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
        # Strips the video id yt-dlp prints after the extractor name. Kept
        # id-shaped so a message merely starting with "word:" — or a URL,
        # full of colons — doesn't lose its first clause.
        line = re.sub(r"^[\w.\-]{1,64}:\s+", "", line, count=1)
    line = re.sub(r"\s*\(caused by .*\)\s*$", "", line)
    return line.strip() or "The download failed."


# Matches both names YouTube gives the same codec: avc1.640028 on some formats,
# h264 on others. A prefix match on one silently falls through to the next rung.
_H264 = r"[vcodec~='^(avc|h264)']"


def format_selector(quality: str = "best") -> str:
    """Build the yt-dlp format chain, as fallbacks from most to least wanted.

    The video stream is copied rather than re-encoded, so whatever is chosen
    here is what has to play on the far end. quality is "best", "1080" or
    "720":

        1. bv*[H.264][cap]+ba   what we want: H.264, under the height cap
        2. bv*[cap]+ba          any codec still under the cap
        3. b[cap]               one combined stream, for sites with no
                                separate audio to merge
        4. b                    anything at all — never refused for want of a
                                preferred format

    Boundary cases, all landing on a later rung rather than failing: AV1/VP9-only
    video (rung 2, codec is recorded and flagged to the user if unplayable);
    no separate audio stream (rung 3/4); nothing under the height cap, e.g. a
    1440p-only source (rung 4, a larger file beats no file); an unrecognised
    quality string (treated as "best"); no usable formats at all — live stream
    or members-only video — which no selector can rescue.
    """
    cap = f"[height<={quality}]" if quality in ("1080", "720") else ""
    rungs = [f"bv*{_H264}{cap}+ba", f"bv*{cap}+ba"]
    if cap:
        rungs.append(f"b{cap}")
    rungs.append("b")
    return "/".join(rungs)


# H.264+AAC in MP4 is the one combination that plays everywhere worth caring
# about (every Mac/iPhone/Android/browser/VLC/QuickTime/smart TV), preferred
# unconditionally because the finished file gets sent to people and outlives
# the machine that made it.
#
# Not H.265: smaller at the same quality, but Android support varies by
# chip/player and Chrome/Firefox largely won't play it; YouTube doesn't offer
# it for this material anyway.
#
# When H.264 isn't offered at all, nothing left is universally playable — VP9
# won't play in an MP4 in QuickTime, AV1 needs an M3 — so ranking them against
# each other is false precision; the smallest is taken and the report flags
# that it may not play elsewhere.
def _codec_rank(vcodec: str) -> int:
    v = (vcodec or "").lower()
    return 0 if v.startswith(("avc", "h264")) else 1


def _size(f: dict) -> int:
    """Bytes if the site published a number, 0 for "it didn't say"."""
    return int(f.get("filesize") or f.get("filesize_approx") or 0)


def _size_key(f: dict) -> tuple[int, int]:
    """Sort smallest-first, with unpublished sizes *last* rather than first.

    _size() alone can't be the tiebreak: it reports unknown as 0, which sorts
    smallest — so an undeclared-size format would beat one that answered
    honestly (seen with a 300MB declared progressive-https 720p H.264 losing
    to a same-codec, same-height HLS format with no filesize at all).
    """
    size = _size(f)
    return (1, 0) if size == 0 else (0, size)


# Progressive https before HLS when a site offers both. HLS arrives as
# thousands of fragments with no total, so it publishes no filesize — leaving
# the disk check to guess and the progress bar with no total to count toward.
# https is one resumable ranged fetch that states its size up front.
def _protocol_rank(f: dict) -> int:
    return 1 if "m3u8" in (f.get("protocol") or "") else 0


# YouTube publishes some tracks twice: the original and a "-drc" (dynamic
# range compressed) variant, tied on container/channels/bitrate so whichever
# sorts first gets picked arbitrarily. The DRC copy is wrong for this app
# specifically: audio here is transcribed then dubbed over, and a transcriber
# does better on the unsquashed source.
def _is_drc(f: dict) -> bool:
    fid = str(f.get("format_id") or "")
    return fid.endswith("-drc") or "drc" in (f.get("format_note") or "").lower()


def _total_size(*picked: dict) -> int:
    """The download's size, or 0 meaning "not known".

    Summed straight, an unpublished size counted as nothing, so a video stream
    with no figure plus an audio stream that did have one came back as the
    audio's size alone — "100% of 48.4 MB" against a 353 MB finished file, and
    a disk check cleared a job 7x larger than it measured. One unknown part
    makes the total unknown; every caller already treats 0 that way.
    """
    sizes = [_size(f) for f in picked]
    return 0 if any(size == 0 for size in sizes) else sum(sizes)


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
                                    -(f.get("height") or 0),
                                    _protocol_rank(f), _size_key(f)))
        # m4a first: it drops into an mp4 without being re-encoded. Stereo
        # before surround, because "highest bitrate" on its own picks the 5.1
        # track where a video has one — three times the bytes (30.8 MB against
        # 10.3 MB on one measured video) for audio that is either replaced
        # outright or downmixed to mono for the transcriber.
        best_a = min(audio, key=lambda f: (_is_drc(f),
                                           0 if f.get("ext") == "m4a" else 1,
                                           1 if (f.get("audio_channels") or 2) > 2 else 0,
                                           _protocol_rank(f),
                                           -(f.get("tbr") or 0)))
        return {"spec": f"{best_v['format_id']}+{best_a['format_id']}",
                "height": best_v.get("height"), "vcodec": best_v.get("vcodec") or "",
                "bytes": _total_size(best_v, best_a)}

    if combined:
        best = min(under_cap(combined),
                   key=lambda f: (_codec_rank(f.get("vcodec")),
                                  -(f.get("height") or 0),
                                  _protocol_rank(f), _size_key(f)))
        return {"spec": best["format_id"], "height": best.get("height"),
                "vcodec": best.get("vcodec") or "", "bytes": _size(best)}

    return None


# Scratch state yt-dlp and its postprocessors leave behind — none of it the
# finished video.
def _is_scratch(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(".part") or lower.endswith(".ytdl") or ".part-frag" in lower:
        return True
    suffixes = Path(lower).suffixes
    return len(suffixes) >= 2 and suffixes[-2] == ".temp"


# The demuxers ffmpeg reads a still image through (see `ffmpeg -demuxers`);
# each names itself "piped X sequence", which "*_pipe" below matches without
# also catching yuv4mpegpipe (raw video frames, spelled without the
# underscore).
#
# gif is deliberately excluded: ffmpeg reads a one-frame thumbnail and a real
# animated GIF through the same "gif" demuxer, so format_name alone can't
# tell them apart, and yt-dlp's Imgur extractor can legitimately offer a real
# animated-gif format. See _is_single_frame_gif() for how that case is told
# apart instead.
_IMAGE_DEMUXERS = {"image2", "image2pipe"}


def _is_still_image(format_name: str) -> bool:
    names = (format_name or "").split(",")
    return any(n in _IMAGE_DEMUXERS or n.endswith("_pipe") for n in names)


def _is_single_frame_gif(format_name: str, nb_frames: str) -> bool:
    """A GIF is rejected only when ffprobe's own container frame count is
    exactly one — never via `-count_frames`, which would mean decoding the
    whole thing to decide whether it's worth decoding.

    An unparseable or missing count (some encoders leave nb_frames as "N/A")
    is not treated as one frame, since that would silently discard a real
    animated GIF the container never promised to prove itself.
    """
    if "gif" not in (format_name or "").split(","):
        return False
    try:
        return int(nb_frames) == 1
    except (TypeError, ValueError):
        return False


def _looks_like_media(path: Path) -> bool:
    """Ask ffprobe what kind of file this is, rather than trusting its name or
    size — yt-dlp's config can drop a thumbnail or info.json sidecar under the
    same "source." prefix, and neither looks like scratch.

    Only a positive match against ffmpeg's image demuxers (or a single-frame
    GIF) counts as not-media; an unreadable probe, or a stream that's neither
    audio nor video, is assumed to be a real video so a finished download is
    never discarded for want of proof.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,nb_frames:format=format_name",
             "-of", "default=noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        # No answer from ffprobe is not evidence of anything. Better to let
        # the downstream ffmpeg call fail loudly on its own than silently
        # discard a completed download because this sidecar-detecting probe
        # happened to time out.
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
    leftover .part fragment from an interrupted first try can sort ahead of a
    completed source.mp4 a retry then wrote.

    Excludes scratch state and still images, then prefers the merged filename,
    falling back to the largest candidate.
    """
    candidates = [p for p in workdir.glob("source.*")
                  if p.is_file() and not _is_scratch(p.name)]
    media = [p for p in candidates if _looks_like_media(p)]
    if not media:
        return None
    merged = [p for p in media if p.stem == "source"]
    return max(merged or media, key=lambda p: p.stat().st_size)


def _explain_missing_video(workdir: Path) -> str:
    """The DownloadError detail when _finished_file() comes back empty, so
    "yt-dlp wrote nothing" can be told apart from "yt-dlp wrote something and
    it was rejected" — a bug here vs. a bug in whatever the user linked to.

    Best-effort: the one guard round the whole body means a vanished file or
    unreadable directory falls back to a vaguer detail rather than a raw
    traceback.
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

    # picked, when available, is tried first; the selector string stays as a
    # fallback for anything the published list can't answer.
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
        # Deliberately no --user-agent: YouTube ties a media URL's
        # authorisation to the client yt-dlp negotiated it as, so overriding
        # the agent desyncs the two and turns a working download into a
        # reliable 403.
        "--newline",
        "--progress-template", _PROGRESS_TEMPLATE,
    ] + _RETRY_ARGS + _NO_CONFIG_ARGS
    # Off unless the user names a browser: reading their cookie store is not
    # something to do quietly on their behalf.
    if cookies_from:
        cmd += ["--cookies-from-browser", cookies_from]
    cmd += ["-o", str(target), url]

    # Picture and sound are two separate downloads, each counting its own
    # bytes from zero — reported as-is, the bar would fill up and restart near
    # the end. Summed here against the format listing's total instead, so it
    # only ever goes forward.
    done = {"before": 0, "last": 0}
    expected = picked["bytes"] if picked else 0

    def fetched() -> int:
        """What is actually on disk for this download.

        The reported counter can't be trusted across a retry: a 403 part way
        through leaves a part file, the resumed attempt's counter behaves
        differently depending on whether the format is one range request or
        fragments — measured once sitting at "0% of 1.7 GB" while the file on
        disk had passed 1.4 GB. Used as a floor, not a replacement, so a
        single-handle stream still reports normally.
        """
        try:
            return sum(f.stat().st_size for f in workdir.glob("source.*"))
        except OSError:
            return 0

    # yt-dlp's own words, with progress records filtered out — a large download
    # emits thousands of DUBPROG lines, so stream()'s own tail was all progress
    # and no diagnosis. Filtered here rather than in stream(), which also
    # drives Demucs and has no business knowing what a progress line looks like.
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
        # A refusal of the exchange, not the video: the same argv run again a
        # minute later fetches it, but yt-dlp's own --extractor-retries
        # doesn't count these as retryable. Re-negotiated from scratch, and
        # any pinned ids replaced too — those came from probe()'s already-
        # ended negotiation, and yt-dlp won't fall through its own chain once
        # it has chosen and failed to fetch a format.
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


# ------------------------------------------------- a video already on this Mac

def _seconds(value) -> float:
    """A duration ffprobe may have reported as "N/A", as nothing, or not at all."""
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _ffprobe_streams(path: Path) -> tuple[set[str], float, str]:
    """(what kinds of stream are in it, how long it is, what ffprobe complained
    about). Never raises — an unreadable file is an answer, not an exception.

    Streams are asked for their own duration as well as the container's, so a
    header-less file (a transport stream, a growing recording) still gets a
    length — the pipeline divides by this to weight a sample and find where
    speech starts.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,duration:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return set(), 0.0, str(exc)
    if out.returncode != 0:
        return set(), 0.0, out.stderr.strip()
    try:
        data = json.loads(out.stdout or "{}")
    except ValueError:
        return set(), 0.0, out.stdout[:400]
    streams = data.get("streams") or []
    kinds = {s.get("codec_type", "") for s in streams}
    duration = _seconds((data.get("format") or {}).get("duration"))
    if not duration:
        duration = max((_seconds(s.get("duration")) for s in streams), default=0.0)
    return kinds, duration, ""


def probe_local(path: Path | str) -> dict:
    """The same description of a video that probe() gets out of a site, for a
    file that is already here.

    Shaped identically on purpose: the pipeline reads title/duration/formats
    from one dictionary whichever end it came from. The empty format list is
    honest, not a stub — it means the same as choose_format() returning None.

    The two refusals here are the two the pipeline can't recover from further
    down: no picture means nothing to mux the soundtrack onto, no sound means
    nothing to transcribe. Both would otherwise die inside ffmpeg minutes
    later, in its words rather than ours.
    """
    path = Path(path)
    if not path.is_file():
        raise DownloadError(f"There's no file at {path} any more.",
                            f"{path} is not a file that exists.")
    kinds, duration, complaint = _ffprobe_streams(path)
    if not kinds:
        raise DownloadError(
            f"“{path.name}” isn't a video file this Mac can read.", complaint)
    if "video" not in kinds:
        raise DownloadError(
            f"“{path.name}” has no picture in it. Dubbing Studio puts a new "
            "soundtrack onto a video, so it needs a video to put it on.",
            f"ffprobe found only: {', '.join(sorted(k for k in kinds if k)) or 'nothing'}")
    if "audio" not in kinds:
        raise DownloadError(
            f"“{path.name}” has no sound in it, so there's nothing to dub.",
            f"ffprobe found only: {', '.join(sorted(k for k in kinds if k))}")
    return {
        "title": path.stem,
        "duration": duration,
        "uploader": "",
        "thumbnail": "",
        "formats": [],
        "is_live": False,
    }


def use_local(path: Path | str, progress: Progress = None,
              info: dict | None = None) -> tuple[Path, dict]:
    """download()'s counterpart for a file that never needed downloading.

    Deliberately no copy into the job folder: the source video is the largest
    thing a job touches, so duplicating a file already on the same disk just
    to read it once would be the most expensive no-op in the pipeline.
    Everything downstream treats it as an ffmpeg input and writes elsewhere,
    so reading it in place never touches the user's own file.
    """
    path = Path(path)
    # Re-checked even with a caller-supplied info: a file can be moved,
    # renamed or unmounted in the seconds between the probe and this call.
    if not path.is_file():
        raise DownloadError(f"There's no file at {path} any more.",
                            f"{path} was there when the job started and is not now.")
    info = info or probe_local(path)
    if progress:
        progress(1.0, f"Using “{info['title']}”")
    return path, info


def extract_audio(video: Path, dst: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
                    "-vn", "-ac", "1", "-ar", "16000", str(dst)], check=True)
    return dst


def media_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)
