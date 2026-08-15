"""Job orchestration: link in, dubbed video out.

The stages that actually run depend on the quality preset, so the progress
weights are worked out per job rather than fixed. Every stage caches its output
in the job folder, so re-running a link that failed part way through picks up
where it left off instead of redoing the expensive parts.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .config import JOBS, OUTPUT_DIR, Settings, detect_machine
from .backends import asr as asr_backend
from .backends import clone as clone_backend
from .backends import diarize as diarize_backend
from .backends import separate as separate_backend
from .backends import tts as tts_backend
from .backends.translate import translate as run_translate, TranslationError
from .steps import align, download, mux

# (key, label, relative cost) — relative costs are turned into weights for
# whichever stages a given job actually runs.
ALL_STAGES = [
    ("download", "Fetching the video", 8),
    ("separate", "Separating speech from music", 14),
    ("diarize", "Identifying speakers", 5),
    ("transcribe", "Listening to the original", 16),
    ("translate", "Translating", 13),
    ("synthesize", "Speaking the new soundtrack", 34),
    ("assemble", "Fitting it to the picture", 6),
    ("finish", "Saving the finished video", 4),
]


def _job_id(url: str) -> str:
    """A stable id per link, so pasting the same one again resumes.

    This used to fold in the wall clock, which gave every submission a fresh
    folder and meant nothing was ever reused — the opposite of the intended
    behaviour. Python's hash() is no good here either: string hashing is
    randomised per process, so the id changed whenever the app restarted.
    """
    return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:12]


def _fingerprint(*parts) -> str:
    """Identify the settings a cached artefact was produced under."""
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _cache_valid(workdir: Path, key: str, fingerprint: str) -> bool:
    """True when a cached artefact was made under the settings now in force.

    Resuming has to be selective: reusing Parakeet's transcript for a job since
    switched to Whisper, or a translation made against a different glossary,
    would silently hand back the wrong work.
    """
    try:
        meta = json.loads((workdir / "cache-meta.json").read_text())
    except Exception:                                            # noqa: BLE001
        return False
    return meta.get(key) == fingerprint


def _cache_stamp(workdir: Path, key: str, fingerprint: str) -> None:
    path = workdir / "cache-meta.json"
    try:
        meta = json.loads(path.read_text())
    except Exception:                                            # noqa: BLE001
        meta = {}
    meta[key] = fingerprint
    path.write_text(json.dumps(meta, indent=1))


def _safe_name(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"\s+", "-", text)
    return (text or "dubbed")[:80]


def _resample_to(src: Path, dst: Path, rate: int, mono: bool = True) -> Path:
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ar", str(rate)]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)
    return dst


@dataclass
class Job:
    id: str
    url: str
    status: str = "queued"
    stage: str = ""
    stage_label: str = ""
    stage_progress: float = 0.0
    overall: float = 0.0
    message: str = "Waiting to start"
    title: str = ""
    duration: float = 0.0
    error: str = ""
    output: str = ""
    stats: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    engine: str = ""
    preset: str = ""
    speakers: int = 0

    def public(self) -> dict:
        d = asdict(self)
        d["elapsed"] = round((self.finished or time.time()) - self.started)
        return d


class JobRunner:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        self._cancel: set[str] = set()

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def unsubscribe(self, fn: Callable[[dict], None]) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _emit(self, job: Job) -> None:
        payload = job.public()
        for fn in list(self._listeners):
            try:
                fn(payload)
            except Exception:                                    # noqa: BLE001
                pass

    def submit(self, url: str, settings: Settings) -> Job:
        job_id = _job_id(url)
        with self._lock:
            existing = self.jobs.get(job_id)
            # The same link submitted twice is one job, not two racing over the
            # same folder.
            if existing and existing.status in ("queued", "running"):
                return existing
            job = Job(id=job_id, url=url, preset=settings.preset)
            self.jobs[job_id] = job
        self._cancel.discard(job_id)
        threading.Thread(target=self._run, args=(job, settings), daemon=True).start()
        return job

    def cancel(self, job_id: str) -> None:
        self._cancel.add(job_id)

    def _check_cancel(self, job: Job) -> None:
        if job.id in self._cancel:
            raise _Cancelled()

    # ------------------------------------------------------------ staging
    def _plan(self, settings: Settings) -> dict[str, tuple[float, float, str]]:
        """Map stage key -> (start_fraction, weight, label) for the stages in use."""
        active = []
        for key, label, cost in ALL_STAGES:
            if key == "separate" and not settings.separate_audio:
                continue
            if key == "diarize" and not settings.diarize:
                continue
            if key == "synthesize" and settings.voice_mode == "clone":
                cost = int(cost * 2.2)           # cloning is markedly slower
            if key == "transcribe" and settings.asr_model == "whisper":
                cost = int(cost * 2.0)
            active.append((key, label, cost))

        total = sum(c for _, _, c in active) or 1
        plan, run = {}, 0.0
        for key, label, cost in active:
            weight = cost / total
            plan[key] = (run, weight, label)
            run += weight
        return plan

    def _stage(self, job: Job, plan: dict, key: str):
        before, weight, label = plan[key]
        job.stage, job.stage_label = key, label

        def report(fraction: float, message: str = "") -> None:
            self._check_cancel(job)
            job.stage_progress = max(0.0, min(1.0, fraction))
            job.overall = round(before + weight * job.stage_progress, 4)
            if message:
                job.message = message
            self._emit(job)

        report(0.0, label)
        return report

    # ---------------------------------------------------------- execution
    def _run(self, job: Job, settings: Settings) -> None:
        machine = detect_machine()
        workdir = JOBS / job.id
        workdir.mkdir(parents=True, exist_ok=True)
        job.status = "running"
        job.engine = "Apple GPU (MLX)" if machine.fast_path else "Portable (CPU)"
        plan = self._plan(settings)
        self._emit(job)

        stats: dict = {"preset": settings.preset}

        try:
            # ---------------------------------------------------- download
            report = self._stage(job, plan, "download")
            video, info = download.download(job.url, workdir, settings.keep_video_quality, report)
            job.title = info["title"]
            job.duration = info["duration"] or download.media_duration(video)
            self._emit(job)

            # Full-band stereo copy is what Demucs wants; ASR wants 16k mono.
            full_wav = workdir / "full.wav"
            if not full_wav.exists():
                _resample_to(video, full_wav, 44100, mono=False)

            speech_wav = full_wav
            background: Path | None = None

            # ---------------------------------------------------- separate
            if settings.separate_audio:
                report = self._stage(job, plan, "separate")
                stems = separate_backend.separate(full_wav, workdir,
                                                  prefer_gpu=machine.fast_path,
                                                  progress=report)
                if stems:
                    speech_wav, background = stems
                    stats["separated"] = True
                else:
                    stats["separated"] = False

            speech16 = workdir / "speech16k.wav"
            if not speech16.exists():
                _resample_to(speech_wav, speech16, 16000, mono=True)

            # ----------------------------------------------------- diarize
            turns: list[dict] = []
            if settings.diarize:
                report = self._stage(job, plan, "diarize")
                turns = diarize_backend.diarize(speech16, settings.expected_speakers,
                                                progress=report)
                cache = workdir / "speakers.json"
                cache.write_text(json.dumps(turns, indent=1))

            # -------------------------------------------------- transcribe
            report = self._stage(job, plan, "transcribe")
            cache = workdir / "segments.json"
            asr_print = _fingerprint(settings.asr_model, settings.separate_audio,
                                     machine.fast_path)
            if cache.exists() and _cache_valid(workdir, "segments", asr_print):
                segments = json.loads(cache.read_text())
                report(1.0, "Reusing the transcription from last time")
            else:
                segments = asr_backend.transcribe(speech16, machine.fast_path,
                                                  settings.asr_model, report)
                cache.write_text(json.dumps(segments, ensure_ascii=False, indent=1))
                _cache_stamp(workdir, "segments", asr_print)
            if not segments:
                raise RuntimeError("No speech was found in that video.")

            segments = diarize_backend.label_segments(segments, turns)
            speaker_ids = sorted({s.get("speaker", 0) for s in segments})
            job.speakers = len(speaker_ids)
            stats["speakers"] = len(speaker_ids)

            # --------------------------------------------------- translate
            report = self._stage(job, plan, "translate")
            tcache = workdir / "translated.json"
            trans_print = _fingerprint(asr_print, settings.translator,
                                       settings.resolved_ollama_model(machine.ram_gb),
                                       settings.anthropic_model, settings.openai_model,
                                       settings.target_language, settings.glossary_text())
            if tcache.exists() and _cache_valid(workdir, "translated", trans_print):
                segments = json.loads(tcache.read_text())
                report(1.0, "Reusing the translation from last time")
            else:
                segments = run_translate(segments, settings, machine.ram_gb, report)
                tcache.write_text(json.dumps(segments, ensure_ascii=False, indent=1))
                _cache_stamp(workdir, "translated", trans_print)

            # -------------------------------------------------- synthesize
            report = self._stage(job, plan, "synthesize")
            engine, cloning = self._make_engine(settings, machine, segments,
                                                speech_wav, workdir, speaker_ids, report)
            stats["voices"] = engine.name

            # Keyed by voice: changing the voice and re-running must not replay
            # the previous voice's cached lines.
            segdir = workdir / "lines" / _fingerprint(settings.voice_mode, settings.voice,
                                                      settings.speed)
            segdir.mkdir(parents=True, exist_ok=True)
            spoken: list[dict] = []
            sample_rate = getattr(engine, "sample_rate", tts_backend.SAMPLE_RATE)
            total = len(segments)
            t0 = time.time()
            degraded = False          # already dropped to the portable voice
            failed = 0

            for n, seg in enumerate(segments):
                self._check_cancel(job)
                text = (seg.get("translation") or "").strip()
                if not text:
                    continue
                speaker = int(seg.get("speaker", 0))
                path = segdir / f"{n:05d}.wav"
                if path.exists():
                    audio, sample_rate = sf.read(path, dtype="float32")
                else:
                    try:
                        audio, sample_rate = engine.say(
                            text, settings.voice_for(speaker), settings.speed, speaker=speaker)
                    except Exception as exc:                     # noqa: BLE001
                        # A voice that breaks part way through must not cost the
                        # user the whole job. Drop to the portable engine once and
                        # carry on; if even that fails, skip the line and say so
                        # in the report rather than losing everything before it.
                        if not degraded:
                            report(n / max(1, total),
                                   f"The voice stopped working ({exc}); switching to the portable one")
                            try:
                                engine = tts_backend.OnnxTTS()
                                degraded = True
                                stats["voices"] = f"{engine.name} (fell back mid-job)"
                                audio, sample_rate = engine.say(
                                    text, settings.voice_for(speaker), settings.speed,
                                    speaker=speaker)
                            except Exception:                    # noqa: BLE001
                                failed += 1
                                continue
                        else:
                            failed += 1
                            continue
                    if getattr(audio, "size", 0):
                        sf.write(path, audio, sample_rate)
                if getattr(audio, "size", 0):
                    spoken.append({"start": seg["start"], "end": seg["end"], "samples": audio})
                if n % 5 == 0 or n == total - 1:
                    done = n + 1
                    rate = done / max(0.1, time.time() - t0)
                    report(done / total,
                           f"Speaking line {done} of {total} — about {_mins((total-done)/rate)} left")

            if not spoken:
                raise RuntimeError("Nothing was synthesised — the translation came back empty.")

            # ---------------------------------------------------- assemble
            report = self._stage(job, plan, "assemble")
            video_len = download.media_duration(video)
            track, astats = align.assemble(spoken, video_len, sample_rate,
                                           settings.max_stretch, report)
            stats.update(astats)
            raw = workdir / "dubbed.wav"
            sf.write(raw, track, sample_rate)

            report(0.75, "Mixing")
            speech_only = raw
            if background and settings.keep_music and settings.audio_mode == "replace":
                speech_only = mux.mix_with_background(raw, background, workdir / "mixed.wav")
                stats["music_kept"] = True
            else:
                stats["music_kept"] = False

            encoded = mux.encode_track(speech_only, workdir / "dubbed.m4a", video_len)

            # ------------------------------------------------------ finish
            report = self._stage(job, plan, "finish")
            srt = align.write_srt(segments, workdir / "subtitles.srt") if settings.write_srt else None

            stem = f"{_safe_name(job.title)}-{settings.target_language[:2].upper()}"
            out_path = OUTPUT_DIR / f"{stem}.mp4"
            counter = 2
            while out_path.exists():
                out_path = OUTPUT_DIR / f"{stem}-{counter}.mp4"
                counter += 1

            report(0.4, "Combining audio and video")
            mux.mux(video, encoded, out_path, settings.audio_mode, settings.duck_db, srt)

            report(0.8, "Checking the result")
            stats.update(mux.verify(video, out_path))
            if srt:
                shutil.copy(srt, out_path.with_suffix(".srt"))

            stats["engine"] = job.engine
            stats["lines_spoken"] = len(spoken)
            if failed:
                stats["lines_failed"] = failed
            separate_backend.cleanup(workdir)          # stems are large; drop them

            job.stats = stats
            job.output = str(out_path)
            job.status = "done"
            job.finished = time.time()
            job.overall = 1.0
            job.message = f"Saved to {out_path.parent.name}/{out_path.name}"
            self._emit(job)

        except _Cancelled:
            job.status = "cancelled"
            job.message = "Cancelled"
            job.finished = time.time()
            self._emit(job)
        except TranslationError as exc:
            self._fail(job, str(exc))
        except FileNotFoundError as exc:
            missing = getattr(exc, "filename", "") or str(exc)
            hint = ("ffmpeg is missing — re-run the installer."
                    if "ffmpeg" in str(missing) or "ffprobe" in str(missing)
                    else f"A required program is missing: {missing}")
            self._fail(job, hint)
        except Exception as exc:                                 # noqa: BLE001
            self._fail(job, str(exc), traceback.format_exc())

    # ------------------------------------------------------------- voices
    def _make_engine(self, settings, machine, segments, speech_wav: Path,
                     workdir: Path, speaker_ids: list[int], report):
        """Build the TTS engine, capturing a reference clip per speaker if cloning."""
        if settings.voice_mode != "clone":
            return tts_backend.load_tts(machine.fast_path, report), False

        if not clone_backend.available():
            report(0.0, "Cloning isn't installed; using a built-in voice instead")
            return tts_backend.load_tts(machine.fast_path, report), False

        try:
            engine = clone_backend.CloneTTS(report)
        except Exception as exc:                                 # noqa: BLE001
            report(0.0, f"Cloning unavailable ({exc}); using a built-in voice")
            return tts_backend.load_tts(machine.fast_path, report), False

        ref_wav = workdir / "speech24k.wav"
        if not ref_wav.exists():
            _resample_to(speech_wav, ref_wav, 24000, mono=True)
        audio, rate = sf.read(ref_wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        captured = 0
        for speaker in speaker_ids:
            ref = diarize_backend.pick_reference(segments, speaker, audio, rate)
            if ref is not None and ref.size > rate:              # at least a second
                engine.add_reference(speaker, ref, rate, workdir / "refs")
                captured += 1

        if not captured:
            report(0.0, "No clean reference audio found; using a built-in voice")
            return tts_backend.load_tts(machine.fast_path, report), False

        report(0.0, f"Cloned {captured} voice{'s' if captured != 1 else ''} from the original")
        return engine, True

    def _fail(self, job: Job, message: str, detail: str = "") -> None:
        job.status = "error"
        job.error = message
        job.message = message
        job.finished = time.time()
        if detail:
            (JOBS / job.id / "error.log").write_text(detail)
        self._emit(job)


class _Cancelled(Exception):
    pass


def _mins(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    return f"{seconds // 60} min"


runner = JobRunner()
