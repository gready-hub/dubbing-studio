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
            "-c:v", "copy", "-c:a", "copy",
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


def verify(original: Path, result: Path) -> dict:
    """Confirm the video stream survived untouched and the audio spans the video."""
    def probe(path: Path, args: list[str]) -> str:
        return subprocess.run(["ffprobe", "-v", "error", *args, "-of", "csv=p=0", str(path)],
                              capture_output=True, text=True, check=True).stdout.strip()

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
        "frames_match": src_frames.splitlines()[:1] == out_frames.splitlines()[:1],
        "source_frames": src_frames.splitlines()[0] if src_frames else "?",
        "output_frames": out_frames.splitlines()[0] if out_frames else "?",
        "video_seconds": round(num(out_vdur), 2),
        "audio_seconds": round(num(out_adur), 2),
        "drift_seconds": round(abs(num(out_vdur) - num(out_adur)), 2),
    }
