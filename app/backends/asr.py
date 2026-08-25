"""Speech recognition.

Two interchangeable backends behind one function:

  * MLX      - parakeet-mlx on Apple Silicon, runs on the GPU. Gives sentence
               timestamps directly, so no separate VAD pass is needed.
  * ONNX     - sherpa-onnx (Silero VAD + the same Parakeet model) on CPU. What
               a Mac without the GPU packages falls back to.

Both return the same thing: a list of {"start", "end", "text"} in seconds.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
import types
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import MODELS
from ..notes import note

Progress = Optional[Callable[[float, str], None]]

# Bumped when an engine's decode behaviour changes enough that the same audio
# would come back differently. pipeline.py folds this into segments.json's
# fingerprint so resumed jobs re-transcribe instead of replaying stale output.
# Keyed per engine (missing = 0) — a single global version would invalidate a
# finished, paid-for translation whenever an unrelated engine changed.
DECODE_VERSIONS = {"whisper": 2}


def decode_version(model: str) -> int:
    """How many times `model`'s decode behaviour has changed. 0 if never."""
    return DECODE_VERSIONS.get(model, 0)

MLX_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
ONNX_ASR_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2")
ONNX_VAD_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                "silero_vad.onnx")


def to_wav16k(src: Path, dst: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-ac", "1", "-ar", "16000", str(dst)], check=True)
    return dst


def _transcribe_mlx(audio: Path, progress: Progress = None) -> list[dict]:
    from parakeet_mlx import from_pretrained

    if progress:
        progress(0.05, "Loading speech model (first run downloads it)")
    model = from_pretrained(MLX_MODEL)

    if progress:
        progress(0.15, "Transcribing")
    # Chunked so memory stays flat on long videos; overlap avoids clipped words.
    result = model.transcribe(str(audio), chunk_duration=120.0, overlap_duration=15.0)

    out = []
    for s in getattr(result, "sentences", []) or []:
        text = (s.text or "").strip()
        if text:
            out.append({"start": float(s.start), "end": float(s.end), "text": text})
    if progress:
        progress(1.0, f"Transcribed {len(out)} lines")
    return out


def _ensure_onnx_models(progress: Progress = None) -> tuple[Path, Path]:
    from .download_util import fetch, extract

    asr_dir = MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    vad = MODELS / "silero_vad.onnx"
    if not asr_dir.exists():
        if progress:
            progress(0.02, "Downloading speech model (about 490 MB, one time)")
        tarball = MODELS / "asr.tar.bz2"
        fetch(ONNX_ASR_URL, tarball)
        extract(tarball, MODELS)
        tarball.unlink(missing_ok=True)
    if not vad.exists():
        fetch(ONNX_VAD_URL, vad)
    return asr_dir, vad


def _transcribe_onnx(audio: Path, progress: Progress = None) -> list[dict]:
    import sherpa_onnx

    asr_dir, vad_path = _ensure_onnx_models(progress)

    if progress:
        progress(0.08, "Loading speech model")
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(asr_dir / "encoder.int8.onnx"),
        decoder=str(asr_dir / "decoder.int8.onnx"),
        joiner=str(asr_dir / "joiner.int8.onnx"),
        tokens=str(asr_dir / "tokens.txt"),
        num_threads=max(2, _cpu_count() - 1),
        model_type="nemo_transducer",
    )

    with wave.open(str(audio)) as w:
        sr = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    data = samples.astype(np.float32) / 32768.0

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(vad_path)
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.35
    cfg.silero_vad.min_speech_duration = 0.20
    cfg.silero_vad.max_speech_duration = 18.0
    cfg.sample_rate = sr
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=180)

    if progress:
        progress(0.12, "Finding where speech starts and stops")
    chunks: list[tuple[float, np.ndarray]] = []
    window = 512
    for i in range(0, len(data), window):
        vad.accept_waveform(data[i:i + window])
        while not vad.empty():
            seg = vad.front
            chunks.append((seg.start / sr, np.array(seg.samples)))
            vad.pop()
    vad.flush()
    while not vad.empty():
        seg = vad.front
        chunks.append((seg.start / sr, np.array(seg.samples)))
        vad.pop()

    out = []
    total = max(1, len(chunks))
    for n, (start, buf) in enumerate(chunks):
        stream = rec.create_stream()
        stream.accept_waveform(sr, buf)
        rec.decode_stream(stream)
        text = stream.result.text.strip()
        if text:
            out.append({"start": round(start, 2),
                        "end": round(start + len(buf) / sr, 2),
                        "text": text})
        if progress and n % 10 == 0:
            progress(0.15 + 0.85 * n / total, f"Transcribing — {n} of {total} lines")
    if progress:
        progress(1.0, f"Transcribed {len(out)} lines")
    return out


def _cpu_count() -> int:
    import os
    return os.cpu_count() or 4


WHISPER_MLX = "mlx-community/whisper-large-v3-mlx"
WHISPER_CPU = "large-v3"
WHISPER_CPU_COMPUTE = "int8"


def _whisper_mlx_cached() -> bool:
    """Whether the model is already on this disk, asked without a network call.

    Only used to decide whether to warn about a 3 GB download. Anything that
    goes wrong here is answered "no": promising a download that then does not
    happen wastes a moment's worry, where staying silent about one that does
    leaves somebody watching an unexplained pause for several minutes.
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(WHISPER_MLX, local_files_only=True)
        return True
    except Exception:                                            # noqa: BLE001
        return False


@contextlib.contextmanager
def _mlx_whisper_progress(progress: Progress, first: float, last: float):
    """Report mlx_whisper's own decode progress while inside this block.

    mlx_whisper.transcribe() is one blocking call; without this it reported a
    single jump from 5% to done (a 6-minute frozen bar on a 52-minute video).
    It does track position, but via a tqdm bar rather than a callback — the
    decode loop does `with tqdm.tqdm(total=content_frames)` and calls
    pbar.update() per window, an exact fraction of the file. That tqdm is
    swapped for a stand-in that calls us instead of drawing, reached via
    sys.modules because mlx_whisper.transcribe is a function (the package's
    __init__ rebinds the name), so an attribute lookup on the module misses it.

    Reaches into another package's internals, so anything unexpected here
    (no callback, nothing to patch, a changed shape) falls back to the old
    quiet-bar behaviour rather than raising. Always restored, so the patch
    can't leak into later calls in the process.
    """
    module = sys.modules.get("mlx_whisper.transcribe")
    real = getattr(module, "tqdm", None)
    if progress is None or real is None:
        yield
        return

    class _Meter:
        """Enough of tqdm's surface for the one call site that builds it."""

        def __init__(self, *_args, total=None, **_kw):
            self.total = int(total or 0)
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def update(self, n=1):
            self.n += int(n or 0)
            if self.total <= 0:
                return                       # nothing to be a fraction of
            share = max(0.0, min(1.0, self.n / self.total))
            progress(first + (last - first) * share,
                     f"Transcribing — {share * 100:.0f}%")

        def close(self):
            pass

        def __getattr__(self, _name):
            # Unpinned mlx_whisper internals may probe other tqdm methods; a
            # no-op keeps failures limited to progress reporting instead of an
            # AttributeError aborting the transcription.
            return lambda *_args, **_kw: None

    module.tqdm = types.SimpleNamespace(tqdm=_Meter)
    try:
        yield
    finally:
        module.tqdm = real


def _transcribe_whisper_mlx(audio: Path, progress: Progress = None) -> list[dict]:
    import mlx_whisper

    # Checked separately because transcribe() loads+decodes in one call, giving
    # no moment to say "loading done" — without this the 3 GB download message
    # showed unconditionally, even once the model was already cached.
    cached = _whisper_mlx_cached()
    if progress:
        progress(0.02, "Loading Whisper" if cached
                 else "Loading Whisper (first run downloads about 3 GB)")
    if not cached:
        # Skipped when cached: snapshot_download() still makes a full pass and
        # prints its own "Fetching" bar even with nothing to download.
        _fetch_whisper_mlx()
    if progress:
        progress(0.05, "Transcribing — 0%")
    # condition_on_previous_text feeds each window's decode Whisper's own belief
    # about what it just said, which can lock in a self-reinforcing repetition
    # loop. Two independent 52-minute decodes of the same file with it on
    # produced 4 such loops between them; off, zero, with more words captured
    # and a faster decode. Details: ~/.claude/scratch/dubbing-studio-asr-hallucination/spec.md.
    with _mlx_whisper_progress(progress, 0.05, 0.98):
        result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=WHISPER_MLX,
                                        word_timestamps=False, verbose=None,
                                        condition_on_previous_text=False)
    out = []
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if text:
            # Kept (not just text) so a hallucinated repeat can be caught by
            # these fields even when not obvious by eye: one real repetition
            # had avg_logprob -2.12 / compression_ratio 21.3, both past
            # Whisper's own thresholds (-1.0 / 2.4).
            out.append({"start": float(s["start"]), "end": float(s["end"]), "text": text,
                        "avg_logprob": s.get("avg_logprob"),
                        "compression_ratio": s.get("compression_ratio"),
                        "no_speech_prob": s.get("no_speech_prob")})
    if progress:
        progress(1.0, f"Transcribed {len(out)} lines")
    return out


def _transcribe_whisper_cpu(audio: Path, progress: Progress = None) -> list[dict]:
    from faster_whisper import WhisperModel

    if progress:
        progress(0.05, "Loading Whisper (first run downloads about 1.5 GB)")
    model = WhisperModel(WHISPER_CPU, device="cpu", compute_type=WHISPER_CPU_COMPUTE,
                         cpu_threads=max(2, _cpu_count() - 1))
    # Same fix as the MLX path: condition_on_previous_text off avoids carrying a
    # bad decode into the next window's prompt. vad_filter=True already resets
    # at silence (protection the MLX path lacks); this hasn't had the MLX
    # path's full-length verification.
    segments, info = model.transcribe(str(audio), vad_filter=True, beam_size=1,
                                      condition_on_previous_text=False)
    out = []
    for n, s in enumerate(segments):
        text = (s.text or "").strip()
        if text:
            out.append({"start": float(s.start), "end": float(s.end), "text": text,
                        "avg_logprob": s.avg_logprob,
                        "compression_ratio": s.compression_ratio,
                        "no_speech_prob": s.no_speech_prob})
        if progress and n % 10 == 0:
            progress(min(0.98, 0.1 + s.end / max(1.0, info.duration)),
                     f"Transcribing — {len(out)} lines so far")
    if progress:
        progress(1.0, f"Transcribed {len(out)} lines")
    return out


def _fetch_whisper_mlx() -> None:
    import mlx_whisper                                           # noqa: F401
    from huggingface_hub import snapshot_download
    snapshot_download(WHISPER_MLX)


def _fetch_whisper_cpu() -> None:
    from faster_whisper import WhisperModel
    WhisperModel(WHISPER_CPU, device="cpu", compute_type=WHISPER_CPU_COMPUTE)


def _fetch_mlx() -> None:
    from parakeet_mlx import from_pretrained
    from_pretrained(MLX_MODEL)


def _ladder(use_mlx: bool, model: str) -> list[tuple[str, Callable, Callable]]:
    """The engines transcribe() will try, in order: (label, run, fetch).

    Shared with prefetch() deliberately. They were separate lists once, and the
    copies disagreed: warming up fetched only the engine expected to win, so a
    fallback — which is reached exactly when the primary is failing — still
    stopped the job mid-stage to download half a gigabyte.
    """
    ladder: list[tuple[str, Callable, Callable]] = []
    if model == "whisper":
        if use_mlx:
            ladder.append(("Whisper (Apple GPU)", _transcribe_whisper_mlx, _fetch_whisper_mlx))
        ladder.append(("Whisper (CPU)", _transcribe_whisper_cpu, _fetch_whisper_cpu))
    if use_mlx:
        ladder.append(("Parakeet (Apple GPU)", _transcribe_mlx, _fetch_mlx))
    ladder.append(("Parakeet (portable)", _transcribe_onnx, _ensure_onnx_models))
    return ladder


def prefetch(use_mlx: bool, model: str = "whisper", progress: Progress = None) -> None:
    """Fetch every engine transcribe() might reach, including its fallbacks.

    Each one is attempted independently. Letting the first failure abort the
    rest would skip exactly the fallbacks this exists to warm — they are reached
    when the primary is failing, which is when its download is likeliest to have
    failed too.
    """
    last_error: Exception | None = None
    fetched_any = False
    for label, _, fetch in _ladder(use_mlx, model):
        if progress:
            progress(0.0, f"Fetching {label}")
        try:
            fetch()
            fetched_any = True
        except Exception as exc:                                 # noqa: BLE001
            last_error = exc
    if not fetched_any and last_error is not None:
        raise last_error


def transcribe(audio_wav: Path, use_mlx: bool, model: str = "whisper",
               progress: Progress = None) -> list[dict]:
    """model: "parakeet" (fast) or "whisper" (more accurate, slower)."""
    ladder = _ladder(use_mlx, model)

    last_error = None
    for n, (label, attempt, _) in enumerate(ladder):
        try:
            return attempt(audio_wav, progress)
        except Exception as exc:                                 # noqa: BLE001
            last_error = exc
            if n + 1 < len(ladder):
                note(progress, f"{label} wouldn't run, so a different speech "
                               f"engine was used ({exc}).")
                if progress:
                    progress(0.0, f"Falling back to another engine ({exc})")
    raise RuntimeError(f"Transcription failed: {last_error}")
