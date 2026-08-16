"""Put the new soundtrack back onto the video."""
from __future__ import annotations

import subprocess
from pathlib import Path


def mix_with_background(dub: Path, background: Path, dst: Path,
                        bed_gain_db: float = -6.0) -> Path:
    """Lay the dubbed speech over the music and effects kept from the original.

    The bed is pulled down a little because the original speech that used to sit
    on top of it is gone, so what remains reads louder than it did in the mix.
    """
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(dub), "-i", str(background),
        "-filter_complex",
        f"[0:a]aformat=channel_layouts=stereo:sample_rates=48000[speech];"
        f"[1:a]aformat=channel_layouts=stereo:sample_rates=48000,"
        f"volume={bed_gain_db}dB[bed];"
        f"[speech][bed]amix=inputs=2:duration=longest:normalize=0[out]",
        "-map", "[out]", str(dst),
    ], check=True)
    return dst


def encode_track(wav: Path, dst: Path, duration: float) -> Path:
    """Normalise to broadcast-ish speech loudness and encode to AAC."""
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(wav),
        "-t", f"{duration:.6f}",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "48000", "-c:a", "aac", "-b:a", "160k", str(dst),
    ], check=True)
    return dst


def mux(video: Path, dubbed: Path, dst: Path, mode: str = "replace",
        duck_db: float = -18.0, srt: Path | None = None) -> Path:
    """mode: replace (dub only) | duck (dub over quiet original) | dual (both tracks)."""
    # Input order matters: every -map below refers to these by position, so the
    # subtitle file has to be appended after the two media inputs, never before.
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(dubbed)]
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
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
        ]
    elif mode == "dual":
        cmd += [
            "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0",
            # Both tracks re-encoded rather than copied. The dub is already AAC,
            # but the original is whatever the site served — on YouTube usually
            # Opus, which in an MP4 will not play on most Android players or in
            # QuickTime. Copying it produced a file that was universal in every
            # respect except its second audio track.
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=English (dubbed)",
            "-metadata:s:a:1", "title=Original",
            "-disposition:a:0", "default", "-disposition:a:1", "0",
        ]
    else:  # replace
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
                "-metadata:s:a:0", "language=eng"]

    if sub_index is not None:
        cmd += ["-map", f"{sub_index}:s:0", "-c:s", "mov_text"]

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

    verify() compares frame counts and durations, which catches structural
    mistakes but says nothing about content: a dub that is correctly muxed,
    exactly the right length and completely silent passes every check it makes.
    Peak and mean level answer "is there anything there at all"; the total time
    spent in silent runs of at least min_run seconds answers "is there anything
    there throughout", which is what catches a dub that died half way.

    Measured with ffmpeg's own volumedetect and silencedetect rather than by
    decoding the track here, so it costs one pass and no new dependency.
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

    stats: dict = {"peak_db": peak, "mean_db": mean, "silent_seconds": round(silent, 1)}
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
    stats["silent_share"] = round(share, 3)

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
