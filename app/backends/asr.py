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

# How many times an engine's decode behaviour has changed enough that the same
# audio, under the same settings, would come back as a different transcript.
# pipeline.py folds this into segments.json's fingerprint, so a job resumed
# across such a change re-transcribes rather than replaying what the old
# behaviour produced — for condition_on_previous_text below, a transcript that
# could still carry the repetition-loop hallucination it exists to avoid.
#
# Keyed by engine, and an engine absent from here declares nothing. That matters
# because the fingerprint cascades: the terminology and the translation are both
# built on top of it, so one blanket version number would make a resumed
# Parakeet job throw away a finished translation — real money, on a paid
# translator — over a change that cannot have altered a Parakeet transcript at
# all. An engine left out keeps the exact fingerprint it had before any of this
# existed, and so keeps its cached work.
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


# ------------------------------------------------------------------ MLX

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


# ----------------------------------------------------------------- ONNX

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


# -------------------------------------------------------------- Whisper

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

    mlx_whisper.transcribe() is one blocking call that swallows the whole file,
    and it was reported as a single jump from 5% to done — six minutes of a
    frozen bar on a 52-minute video, under a label still saying "Loading
    Whisper". It does publish its position, just to a progress *bar* rather than
    to a callback: the decode loop runs inside `with tqdm.tqdm(total=
    content_frames)` and calls pbar.update() once per window, where the counter
    is the audio position. That is an exact fraction of the file, and better
    than the line counting the CPU path settles for.

    So the meter is swapped for one that calls us instead of drawing. Its module
    is reached through sys.modules deliberately: mlx_whisper.transcribe is the
    *function* — the package's __init__ rebinds the name — so the obvious
    attribute lookup finds something with no tqdm on it.

    This does reach into another package's internals, so it is written to fail
    into the old behaviour rather than to fail: no callback, no tqdm to replace,
    or a shape that no longer matches, and the transcription simply runs as it
    did before with a quiet bar. It is restored unconditionally, since leaving a
    foreign module monkeypatched would follow every later call in the process.
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
            # Anything else tqdm is asked for. Only the constructor, the
            # context manager and update() are used by the decode loop this
            # stands in for, but that is this version of mlx_whisper's business
            # and the requirement is not pinned. A method that does nothing
            # keeps the promise made above — that a shape which no longer
            # matches costs the progress reporting and nothing else — where an
            # AttributeError would be raised inside the transcription and drop
            # the whole job onto the portable engine instead.
            return lambda *_args, **_kw: None

    module.tqdm = types.SimpleNamespace(tqdm=_Meter)
    try:
        yield
    finally:
        module.tqdm = real


def _transcribe_whisper_mlx(audio: Path, progress: Progress = None) -> list[dict]:
    import mlx_whisper

    # Fetched as its own step so the download can be spoken about honestly. The
    # message here was unconditional — every run, for the life of the install,
    # announced a 3 GB first-run download — and transcribe() loads and decodes
    # in a single call, so there was no moment at which the app could have said
    # the loading had finished and the work had started.
    cached = _whisper_mlx_cached()
    if progress:
        progress(0.02, "Loading Whisper" if cached
                 else "Loading Whisper (first run downloads about 3 GB)")
    if not cached:
        # Only when there is really something to fetch. Asked unconditionally it
        # is a second full trip through snapshot_download() for a model already
        # on the disk, which costs a wasted pass and prints a second "Fetching"
        # bar into the window the user is told to leave open.
        _fetch_whisper_mlx()
    if progress:
        progress(0.05, "Transcribing — 0%")
    # condition_on_previous_text feeds each 30s window's decode the text Whisper
    # believes it just produced. On a real 52-minute recording that carried a
    # single garbled window into a self-reinforcing loop — one word repeated for
    # the rest of the window, sometimes hundreds of times — because a bad guess
    # became the next window's prompt instead of a fresh start. Two independent
    # full 52-minute decodes of this exact source file with the flag left on
    # produced 4 distinct repetition-loop instances between them; the same two
    # decodes with it off produced none, with *more* words captured overall
    # (nothing lost at the window boundaries this removes) and a faster decode
    # besides. Documented further, including what was and wasn't independently
    # verified, in ~/.claude/scratch/dubbing-studio-asr-hallucination/spec.md.
    with _mlx_whisper_progress(progress, 0.05, 0.98):
        result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=WHISPER_MLX,
                                        word_timestamps=False, verbose=None,
                                        condition_on_previous_text=False)
    out = []
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if text:
            # Kept rather than dropped: a hallucinated repeat isn't always this
            # cheap to catch by eye. One real instance of this same failure had
            # an avg_logprob of -2.12 and a compression_ratio of 21.3 — both far
            # past Whisper's own thresholds (-1.0, 2.4) — and would have been
            # visible from these fields alone, discarded here until now.
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
    # Same flag, same reason, as the MLX path above: condition_on_previous_text
    # carries a bad decode into the next window as its prompt. vad_filter=True
    # already resets segmentation at silence, which is real protection this
    # path had and the MLX one didn't — this is defense in depth alongside it,
    # not the primary fix, and hasn't had its own full-length real-audio
    # verification the way the MLX change did.
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


# ---------------------------------------------------------------- public

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
