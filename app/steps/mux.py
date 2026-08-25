"""Put the new soundtrack back onto the video."""
from __future__ import annotations

import subprocess
from pathlib import Path

OUTPUT_SAMPLE_RATE = 48000        # delivery audio rate used throughout this module
AAC_BITRATE = "160k"


def _ffmpeg_in(src: Path) -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", "-i", str(src)]


def mix_with_background(dub: Path, background: Path, dst: Path,
                        bed_gain_db: float = -6.0) -> Path:
    """Lay the dubbed speech over the music and effects kept from the original.

    The bed is pulled down a little because the original speech that used to sit
    on top of it is gone, so what remains reads louder than it did in the mix.
    """
    subprocess.run(_ffmpeg_in(dub) + [
        "-i", str(background),
        "-filter_complex",
        f"[0:a]aformat=channel_layouts=stereo:sample_rates={OUTPUT_SAMPLE_RATE}[speech];"
        f"[1:a]aformat=channel_layouts=stereo:sample_rates={OUTPUT_SAMPLE_RATE},"
        f"volume={bed_gain_db}dB[bed];"
        f"[speech][bed]amix=inputs=2:duration=longest:normalize=0[out]",
        "-map", "[out]", str(dst),
    ], check=True)
    return dst


def encode_track(wav: Path, dst: Path, duration: float) -> Path:
    """Normalise to broadcast-ish speech loudness and encode to AAC."""
    subprocess.run(_ffmpeg_in(wav) + [
        "-t", f"{duration:.6f}",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(OUTPUT_SAMPLE_RATE), "-c:a", "aac", "-b:a", AAC_BITRATE, str(dst),
    ], check=True)
    return dst


def mux(video: Path, dubbed: Path, dst: Path, mode: str = "replace",
        duck_db: float = -18.0, srt: Path | None = None) -> Path:
    """mode: replace (dub only) | duck (dub over quiet original) | dual (both tracks)."""
    # Input order matters: every -map below refers to these by position, so the
    # subtitle file has to be appended after the two media inputs, never before.
    cmd = _ffmpeg_in(video) + ["-i", str(dubbed)]
    sub_index = None
    if srt and srt.exists():
        sub_index = 2
        cmd += ["-i", str(srt)]

    if mode == "duck":
        cmd += [
            "-filter_complex",
            f"[0:a]volume={duck_db}dB[bed];[bed][1:a]amix=inputs=2:duration=first:"
            f"dropout_transition=0:normalize=0[out]",
            "-map", "0:v:0", "-map", "[out]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", AAC_BITRATE,
        ]
    elif mode == "dual":
        cmd += [
            "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0",
            # Both tracks re-encoded rather than copied: the original is
            # whatever the site served (often Opus on YouTube), which in an
            # MP4 fails on most Android players and QuickTime.
            "-c:v", "copy", "-c:a", "aac", "-b:a", AAC_BITRATE,
            "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=English (dubbed)",
            "-metadata:s:a:1", "title=Original",
            "-disposition:a:0", "default", "-disposition:a:1", "0",
        ]
    else:  # replace
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
                "-metadata:s:a:0", "language=eng"]

    if sub_index is not None:
        # Tagged like the audio beside it. Without a language the track lands as
        # "und", which players list as Unknown or Track 1 — the one thing the
        # menu is there to tell you.
        cmd += ["-map", f"{sub_index}:s:0", "-c:s", "mov_text",
                "-metadata:s:s:0", "language=eng"]

    # No -shortest here. It ends the output when the shortest *stream* ends, and
    # a subtitle track stops at its last cue — so a video whose final line of
    # speech lands before the picture ends was silently truncated to that cue,
    # losing both frames and audio. encode_track() already caps the dub at the
    # video duration, so the video is the longest stream and needs no trimming.
    cmd += ["-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def check_loudness(result: Path, total_duration: float,
                   silence_db: float = -50.0, min_run: float = 3.0) -> dict:
    """Confirm the finished track actually contains speech.

    verify() checks frame counts and durations, not content: a dub that is
    correctly muxed, right length and completely silent passes it anyway.
    Peak/mean level answer "is anything there". The no-line-seconds figure
    uses the same silencedetect pass but means something different here: this
    track holds only dubbed lines at their original timestamps, so a silent
    run is time nothing was dubbed there, not a defect in present audio.
    """
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-nostats", "-i", str(result), "-map", "0:a:0",
         "-af", f"volumedetect,silencedetect=noise={silence_db}dB:d={min_run}",
         "-f", "null", "-"],
        capture_output=True, text=True)
    text = proc.stderr

    def level(label: str) -> float | None:
        for line in text.splitlines():
            if label in line:
                try:
                    return float(line.split(label)[1].split("dB")[0].strip().rstrip(":").strip())
                except (IndexError, ValueError):
                    return None
        return None

    peak = level("max_volume:")
    mean = level("mean_volume:")
    silent = 0.0
    for line in text.splitlines():
        if "silence_duration:" in line:
            try:
                silent += float(line.split("silence_duration:")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass

    stats: dict = {"peak_db": peak, "mean_db": mean, "no_line_seconds": round(silent, 1)}
    if proc.returncode != 0 or peak is None:
        # The probe itself failed, which says nothing about the audio. Reporting
        # that as silence would put "the finished soundtrack is silent" on a
        # perfectly good dub — the one warning that must not cry wolf.
        stats["audio_present"] = None
        stats["audio_warning"] = ""
        return stats
    # -45 dBFS is well below anything audible as speech but comfortably above a
    # digitally silent track, so it separates "quiet" from "empty".
    stats["audio_present"] = peak > -45.0
    share = silent / total_duration if total_duration > 0 else 0.0
    stats["no_line_share"] = round(share, 3)

    if not stats["audio_present"]:
        stats["audio_warning"] = ("The finished soundtrack is silent — nothing was "
                                  "audible in the dubbed audio.")
    elif share > 0.5:
        stats["audio_warning"] = (
            f"{int(share * 100)}% of the video has no dubbed speech over it. "
            "That can be normal for a video with long wordless stretches, but it "
            "is also what a half-failed run looks like.")
    return stats


# Codecs any Mac in use can decode. Deliberately conservative: hevc is fine on
# Apple silicon but not on every older Intel Mac, and this list exists to answer
# "will it play for the person I send it to", not "will it play for me".
WIDELY_PLAYABLE = {"h264", "mpeg4"}

# VideoToolbox is Apple's hardware encoder: on Apple silicon a 1080p feature
# re-encodes in minutes rather than the best part of an hour, which is the
# difference between this being worth doing automatically and not. libx264 is
# the fallback for a Mac or an ffmpeg build without it.
_H264_ENCODERS = (
    ("h264_videotoolbox", ["-q:v", "60"]),
    ("libx264", ["-preset", "veryfast", "-crf", "20"]),
)


def transcode_h264(src: Path, dst: Path) -> str:
    """Re-encode the picture to H.264, leaving audio and subtitles untouched.

    Used only when the site offered nothing widely playable — rare on YouTube,
    but the alternative is a file that opens as sound with no picture.
    yuv420p is forced because AV1/VP9 10-bit is technically legal in H.264 but
    almost nothing can decode it. Returns the encoder used, or "" if none
    worked, in which case the original file is left alone.
    """
    for encoder, quality in _H264_ENCODERS:
        done = subprocess.run(
            _ffmpeg_in(src) + ["-map", "0", "-c", "copy", "-c:v", encoder, *quality,
                              "-profile:v", "high", "-pix_fmt", "yuv420p",
                              "-movflags", "+faststart", str(dst)],
            capture_output=True, text=True)
        if done.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return encoder
        dst.unlink(missing_ok=True)
    return ""


def verify(original: Path, result: Path) -> dict:
    """Confirm the video stream survived untouched and the audio spans the video."""
    def probe(path: Path, args: list[str]) -> str:
        return subprocess.run(["ffprobe", "-v", "error", *args, "-of", "csv=p=0", str(path)],
                              capture_output=True, text=True, check=True).stdout.strip()

    codec = probe(result, ["-select_streams", "v:0", "-show_entries", "stream=codec_name"])
    codec = codec.splitlines()[0] if codec else ""
    src_frames = probe(original, ["-select_streams", "v:0", "-show_entries", "stream=nb_frames"])
    out_frames = probe(result, ["-select_streams", "v:0", "-show_entries", "stream=nb_frames"])
    out_vdur = probe(result, ["-select_streams", "v:0", "-show_entries", "stream=duration"])
    out_adur = probe(result, ["-select_streams", "a:0", "-show_entries", "stream=duration"])

    def num(x: str) -> float:
        try:
            return float(x.splitlines()[0])
        except Exception:
            return 0.0

    return {
        "video_codec": codec,
        # H.264 plays on everything. AV1 needs an M3 or newer to decode on a Mac,
        # and QuickTime on anything older opens the file as audio with no
        # picture — a finished dub nobody can watch. The download is asked to
        # prefer H.264, so this is the check that the preference held.
        "widely_playable": codec in WIDELY_PLAYABLE,
        "frames_match": src_frames.splitlines()[:1] == out_frames.splitlines()[:1],
        "source_frames": src_frames.splitlines()[0] if src_frames else "?",
        "output_frames": out_frames.splitlines()[0] if out_frames else "?",
        "video_seconds": round(num(out_vdur), 2),
        "audio_seconds": round(num(out_adur), 2),
        "drift_seconds": round(abs(num(out_vdur) - num(out_adur)), 2),
    }
