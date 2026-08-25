"""Work out who is speaking when.

Without this, a two-person interview comes back as one narrator reading both
halves of the conversation. With it, each speaker gets their own voice.

Uses sherpa-onnx: pyannote segmentation 3.0 to find speech turns, and a speaker
embedding model to group turns belonging to the same person.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import MODELS

Progress = Optional[Callable[[float, str], None]]

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEG_URL = f"{BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL = f"{BASE}/speaker-recongition-models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"
EMB_NAME = "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"


def _ensure_models(progress: Progress = None) -> tuple[Path, Path]:
    from .download_util import fetch, extract

    seg_dir = MODELS / "sherpa-onnx-pyannote-segmentation-3-0"
    emb = MODELS / EMB_NAME
    if not seg_dir.exists():
        if progress:
            progress(0.02, "Downloading the speaker model (one time)")
        tarball = MODELS / "seg.tar.bz2"
        fetch(SEG_URL, tarball)
        extract(tarball, MODELS)
        tarball.unlink(missing_ok=True)
    if not emb.exists():
        fetch(EMB_URL, emb)
    return seg_dir / "model.onnx", emb


def prefetch(progress: Progress = None) -> None:
    """Fetch the segmentation and embedding models diarize() needs."""
    _ensure_models(progress)


def diarize(audio_wav: Path, threshold: float = 0.5,
            progress: Progress = None) -> list[dict]:
    """Returns [{"start", "end", "speaker"}]. Empty list if it can't run.

    The number of speakers is always worked out from the audio. Naming one
    instead makes the clustering produce exactly that many groups whether the
    voices support it or not, which is a guess dressed up as a measurement — and
    it leaves nothing for the over-segmentation check downstream to notice.
    """
    try:
        import sherpa_onnx
    except ImportError:
        return []

    try:
        seg_model, emb_model = _ensure_models(progress)
    except Exception:                                            # noqa: BLE001
        if progress:
            progress(1.0, "Couldn't fetch the speaker model; treating it as one speaker")
        return []

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model)),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=-1, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        return []

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    with wave.open(str(audio_wav)) as w:
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    data = samples.astype(np.float32) / 32768.0

    if rate != sd.sample_rate:
        # sherpa expects its own rate; resample cheaply by linear interpolation.
        ratio = sd.sample_rate / rate
        idx = np.linspace(0, len(data) - 1, int(len(data) * ratio))
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)

    def cb(processed: int, total: int) -> int:
        if progress and total:
            progress(min(0.99, processed / total), f"Identifying speakers — {processed*100//total}%")
        return 0

    try:
        result = sd.process(data, callback=cb).sort_by_start_time()
    except Exception:                                            # noqa: BLE001
        if progress:
            progress(1.0, "Couldn't tell speakers apart; treating it as one speaker")
        return []

    turns = [{"start": float(s.start), "end": float(s.end), "speaker": int(s.speaker)}
             for s in result]
    if progress:
        n = len({t["speaker"] for t in turns})
        progress(1.0, f"Found {n} speaker{'s' if n != 1 else ''}")
    return turns


def label_segments(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Tag each transcript segment with the speaker who talks through most of it."""
    if not turns:
        for seg in segments:
            seg["speaker"] = 0
        return segments

    for seg in segments:
        best, best_overlap = 0, 0.0
        for turn in turns:
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap, best = overlap, turn["speaker"]
        seg["speaker"] = best
    return segments


def median_pitch(samples: np.ndarray, rate: int,
                 lo_hz: float = 70.0, hi_hz: float = 350.0) -> float:
    """Median fundamental over the voiced frames, by autocorrelation. 0 if unsure.

    Framed rather than taken across the whole clip: one autocorrelation over a
    whole line also spans its silences and onsets, which moves the estimate by
    tens of hertz. An FFT peak is no good either — it finds harmonics, not the
    fundamental.
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    win, hop = int(0.04 * rate), int(0.02 * rate)
    lo, hi = int(rate / hi_hz), int(rate / lo_hz)
    if audio.size < win or hi >= win:
        return 0.0

    found: list[float] = []
    for start in range(0, audio.size - win, hop):
        frame = audio[start:start + win]
        if float(np.sqrt((frame ** 2).mean())) < 0.02:           # between words
            continue
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode="full")[win - 1:]
        if hi >= corr.size or corr[0] <= 0:
            continue
        peak = int(np.argmax(corr[lo:hi])) + lo
        if corr[peak] > 0.3 * corr[0]:                           # voiced, not noise
            found.append(rate / peak)
    return float(np.median(found)) if found else 0.0


def pick_reference(segments: list[dict], speaker: int, audio: np.ndarray, rate: int,
                   min_s: float = 6.0, max_s: float = 14.0) -> np.ndarray | None:
    """Longest continuous run of one speaker, for use as a voice-cloning reference.

    Prefers a single long segment over stitched-together fragments: cloning
    quality depends far more on the reference being clean and natural than long.
    """
    mine = sorted((s for s in segments if s.get("speaker") == speaker),
                  key=lambda s: s["end"] - s["start"], reverse=True)
    if not mine:
        return None

    best = mine[0]
    span = best["end"] - best["start"]
    if span >= min_s:
        end = best["start"] + min(span, max_s)
        return audio[int(best["start"] * rate):int(end * rate)]

    # Fall back: join several shorter segments with brief gaps between them.
    chunks, total = [], 0.0
    for seg in mine:
        piece = audio[int(seg["start"] * rate):int(seg["end"] * rate)]
        if piece.size == 0:
            continue
        chunks.append(piece)
        chunks.append(np.zeros(int(0.12 * rate), dtype=np.float32))
        total += seg["end"] - seg["start"]
        if total >= min_s:
            break
    if not chunks:
        return None
    return np.concatenate(chunks)[:int(max_s * rate)]
