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


def diarize(audio_wav: Path, expected_speakers: int = -1,
            threshold: float = 0.5, progress: Progress = None) -> list[dict]:
    """Returns [{"start", "end", "speaker"}]. Empty list if it can't run."""
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
            num_clusters=expected_speakers, threshold=threshold),
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

    # Nothing long enough on its own — join the best few with short gaps.
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
