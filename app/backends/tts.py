"""Speech synthesis (Kokoro-82M) with an MLX and an ONNX backend.

Both expose the same object:

    engine = load_tts(use_mlx=True)
    samples, sample_rate = engine.say("Hello there", voice="bf_emma", speed=1.0)
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..config import MODELS, ONNX_VOICE_IDS
from ..notes import note

Progress = Optional[Callable[[float, str], None]]

MLX_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
ONNX_TTS_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
                "kokoro-int8-multi-lang-v1_0.tar.bz2")
SAMPLE_RATE = 24000


class MlxTTS:
    name = "Apple GPU (MLX)"
    # Stated rather than left to be guessed at: the pipeline fixes one rate for
    # the whole track before the first line is spoken, and it used to reach for
    # this attribute, not find it on two of the three engines, and fall through
    # to a module constant that happened to agree.
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        from mlx_audio.tts.utils import load_model
        self.model = load_model(MLX_TTS_MODEL)
        # Kokoro builds its text front-end lazily, on the first generate() call
        # rather than at load. Without forcing it here a missing phonemiser looks
        # like a healthy engine and then raises mid-job, long past the point where
        # load_tts() could still have fallen back to the portable voice.
        self.say("Ready.")

    def say(self, text: str, voice: str = "bf_emma", speed: float = 1.0,
            speaker: int = 0):
        lang = "b" if voice.startswith(("bf_", "bm_")) else "a"
        pieces = []
        for result in self.model.generate(text=text, voice=voice, speed=speed, lang_code=lang):
            audio = result.audio
            pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not pieces:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        return np.concatenate(pieces), SAMPLE_RATE


class OnnxTTS:
    name = "Portable (CPU)"
    sample_rate = SAMPLE_RATE

    def __init__(self, threads: int = 0) -> None:
        import os
        import sherpa_onnx
        from .download_util import fetch, extract

        model_dir = MODELS / "kokoro-int8-multi-lang-v1_0"
        if not model_dir.exists():
            tarball = MODELS / "kokoro.tar.bz2"
            fetch(ONNX_TTS_URL, tarball)
            extract(tarball, MODELS)
            tarball.unlink(missing_ok=True)

        threads = threads or max(2, (os.cpu_count() or 4) - 1)
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(model_dir / "model.int8.onnx"),
                    voices=str(model_dir / "voices.bin"),
                    tokens=str(model_dir / "tokens.txt"),
                    data_dir=str(model_dir / "espeak-ng-data"),
                    dict_dir=str(model_dir / "dict"),
                    lexicon=f"{model_dir}/lexicon-gb-en.txt,{model_dir}/lexicon-zh.txt",
                ),
                num_threads=threads,
                provider="cpu",
            ),
            max_num_sentences=1,
        )
        self.tts = sherpa_onnx.OfflineTts(cfg)

    def say(self, text: str, voice: str = "bf_emma", speed: float = 1.0,
            speaker: int = 0):
        sid = ONNX_VOICE_IDS.get(voice, ONNX_VOICE_IDS["bf_emma"])
        audio = self.tts.generate(text, sid=sid, speed=speed)
        return np.asarray(audio.samples, dtype=np.float32), audio.sample_rate


def load_tts(use_mlx: bool, progress: Progress = None):
    if use_mlx:
        try:
            if progress:
                progress(0.0, "Loading the voice on the GPU")
            return MlxTTS()
        except Exception as exc:  # noqa: BLE001
            note(progress, "The Apple GPU voice wouldn't load, so the portable "
                           f"one was used ({exc}).")
            if progress:
                progress(0.0, f"GPU voice unavailable ({exc}); using the portable engine")
    if progress:
        progress(0.0, "Loading the voice")
    return OnnxTTS()


def prefetch(use_mlx: bool, progress: Progress = None) -> None:
    """Fetch and warm whatever load_tts() would pick, plus the fallback.

    Goes through load_tts rather than reproducing its choice, so MlxTTS's
    deliberate say("Ready.") — which forces Kokoro's lazily-built phonemiser —
    happens here too. Warming up by calling load_model() directly skipped
    exactly that, which is the stall this is meant to remove.
    """
    engine = load_tts(use_mlx, progress)
    if not isinstance(engine, OnnxTTS):
        # Still the engine a mid-job failure drops to, so it needs its weights.
        OnnxTTS()
