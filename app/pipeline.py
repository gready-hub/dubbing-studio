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
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .config import HISTORY_FILE, JOBS, OUTPUT_DIR, Settings, detect_machine
from .backends import asr as asr_backend
from .backends import clone as clone_backend
from .backends import diarize as diarize_backend
from .backends import separate as separate_backend
from .backends import tts as tts_backend
from .backends.translate import translate as run_translate, TranslationError
from .steps import align, download, mux
from .steps.segments import merge_adjacent

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


def _derived_dir(workdir: Path, *parts) -> Path:
    """A folder named after the fingerprint of everything that determines it.

    The JSON artefacts are fingerprinted, but the audio they derive from used to
    be guarded by existence alone. Since a job id is now a stable hash of the
    link, running the same link on Fast and then on Balanced found the Fast
    run's downmix of the *whole* soundtrack already sitting at speech16k.wav and
    skipped it: Demucs ran and was paid for, the report said separation had
    happened, and transcription, diarization and the cloning references were all
    still taken from the music-contaminated mix.

    Keying the folder rather than stamping the files makes a stale artefact
    unreachable instead of merely unlikely — there is no path by which the wrong
    settings can name the right folder.
    """
    path = workdir / "derived" / _fingerprint(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _match_rate(audio, src_rate: int, dst_rate: int) -> np.ndarray:
    """Bring one engine's output to the project rate.

    The rate used to be a single variable reassigned per line and then handed to
    assemble() once for all of them, so a mid-job fallback from a cloning engine
    to the portable one could leave two rates in the same list and time every
    line of one of them wrongly. Resampling on the way in makes a single rate an
    invariant rather than an observation. Costs an ffmpeg pass, but only on the
    lines that actually disagree, which on a healthy run is none of them.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.reshape(-1)
    if int(src_rate) == int(dst_rate) or audio.size == 0:
        return audio
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(src, audio, int(src_rate))
        _resample_to(src, dst, int(dst_rate), mono=True)
        out, _ = sf.read(dst, dtype="float32")
    return np.asarray(out, dtype=np.float32).reshape(-1)


FEMALE_ABOVE_HZ = 165.0     # conventional split; between typical male and female F0


def _voice_map(settings: Settings, segments: list[dict], speaker_ids: list[int],
               speech_wav: Path) -> dict[int, str]:
    """Give each speaker a voice roughly matching their own pitch.

    Additional speakers used to be handed voices round-robin from a pool, so a
    deep-voiced man could be dubbed by a bright female voice — the most obvious
    way a multi-speaker dub announces that nobody checked. The pitch is right
    there in the audio, and a median F0 over the voiced frames separates the two
    groups well enough to pick the right half of the pool.

    Only the longest clip per speaker is read, not the whole soundtrack.
    """
    voices: dict[int, str] = {}
    try:
        info = sf.info(str(speech_wav))
        rate = int(info.samplerate)
    except Exception:                                            # noqa: BLE001
        return voices

    for speaker in speaker_ids:
        mine = [s for s in segments if s.get("speaker") == speaker]
        if not mine:
            continue
        longest = max(mine, key=lambda s: s["end"] - s["start"])
        start = max(0, int(longest["start"] * rate))
        frames = min(int(6.0 * rate), int((longest["end"] - longest["start"]) * rate))
        if frames <= 0:
            continue
        try:
            clip, _ = sf.read(str(speech_wav), start=start, frames=frames,
                              dtype="float32", always_2d=False)
        except Exception:                                        # noqa: BLE001
            continue
        if getattr(clip, "ndim", 1) > 1:
            clip = clip.mean(axis=1)
        pitch = diarize_backend.median_pitch(clip, rate)
        if pitch <= 0:
            continue                      # unvoiced or too short to judge
        voices[speaker] = settings.voice_for(speaker, male=pitch < FEMALE_ABOVE_HZ)
    return voices


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
    cancelling: bool = False
    queue_position: int = 0              # 0 = running, or not waiting
    references: list[str] = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d["elapsed"] = round((self.finished or time.time()) - self.started)
        return d


HISTORY_LIMIT = 50

# What a job folder keeps once the job has succeeded. Everything else in there
# is bulk that can be rebuilt.
KEEP_SUFFIXES = {".json", ".srt", ".log"}


def dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def prune_workdir(workdir: Path) -> int:
    """Drop a finished job's bulky intermediates. Returns the bytes reclaimed.

    A job folder held source.mp4, full.wav, speech16k.wav, speech24k.wav, the
    Demucs stems, dubbed.wav, mixed.wav and dubbed.m4a — roughly 1.5 GB for an
    hour of video, kept for ever under Application Support where nobody would
    think to look for it.

    The JSON artefacts stay, and so do the rendered per-line WAVs. Those cover
    transcription, translation and synthesis, which are 63% of a job between
    them and a small share of the bulk; everything removed here is either a
    plain ffmpeg conversion or the download, and the pipeline already rebuilds
    whatever it finds missing.

    Only called after a job succeeds. A failed or cancelled job keeps
    everything, because that is precisely when the link gets run again.
    """
    # Walked once, before deleting anything: rglob walks lazily, and removing
    # entries from a directory that is still being scanned can skip its
    # siblings, which would leave some of the bulk behind at random.
    entries = list(workdir.rglob("*"))
    freed = 0
    for path in entries:
        if path.is_dir() or path.suffix in KEEP_SUFFIXES:
            continue
        # Relative to the job folder, so the check cannot be swayed by a
        # directory called "lines" somewhere above it in the user's home.
        if "lines" in path.relative_to(workdir).parts:
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            freed += size
        except OSError:
            pass
    # Deepest first, so a directory emptied by the pass above goes too.
    for path in sorted((p for p in entries if p.is_dir()),
                       key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass                          # not empty; something is still wanted
    return freed


def _load_history() -> list[dict]:
    try:
        data = json.loads(HISTORY_FILE.read_text())
    except Exception:                                            # noqa: BLE001
        return []
    return data if isinstance(data, list) else []


def _record_history(job: Job) -> None:
    """Keep a finished run once its in-memory record has been replaced.

    A job id is a stable hash of the link, so re-running a link built a fresh
    Job over the top of the old one and the previous run vanished from the
    history panel — and the whole panel emptied on restart regardless, since it
    was only ever an in-memory dict. One entry per file produced, which is the
    distinction the -2 suffix on the output name already makes.
    """
    entry = job.public()
    # Distinct from the live job id, so a re-run of the same link lists as its
    # own row rather than collapsing onto the previous one.
    entry["id"] = f"{job.id}-{int(job.finished)}"
    history = [h for h in _load_history() if h.get("output") != entry["output"]]
    history.append(entry)
    history.sort(key=lambda h: h.get("finished", 0))
    try:
        # Written whole, then moved into place: request threads read this file,
        # and one that caught it mid-write saw truncated JSON and reported no
        # history at all.
        tmp = HISTORY_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(history[-HISTORY_LIMIT:], indent=1))
        tmp.replace(HISTORY_FILE)
    except Exception:                                            # noqa: BLE001
        pass                      # history is a convenience, never worth a job


class JobRunner:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        self._cancel: set[str] = set()
        # One job at a time, and the rest wait their turn. Every heavy stage
        # already saturates the GPU and several gigabytes of model, so two jobs
        # at once do not finish in half the time — they contend and swap. This
        # used to start a thread per link, so pasting a second link silently ran
        # both at once and made each of them slower.
        self._pending: list[tuple[Job, Settings]] = []
        self._worker: threading.Thread | None = None

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
            self._pending.append((job, settings))
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, daemon=True)
                self._worker.start()
        self._renumber()
        return job

    def _work(self) -> None:
        """Drain the queue, one job at a time, then retire."""
        while True:
            with self._lock:
                if not self._pending:
                    self._worker = None
                    return
                job, settings = self._pending.pop(0)
                # Marked running under the same lock that removed it from the
                # queue: _renumber asks "is anything running?", and a submit
                # arriving between the pop and _run() setting the status would
                # otherwise be told the app was idle.
                job.status = "running"
            job.queue_position = 0
            if job.id in self._cancel:
                self._stopped(job)               # cancelled before it ever began
            else:
                try:
                    self._run(job, settings)
                except Exception as exc:                         # noqa: BLE001
                    # _run handles its own failures; this is so one bad job can
                    # never take the worker down and strand everything behind it.
                    self._fail(job, str(exc), traceback.format_exc())
            self._renumber()

    def _renumber(self) -> None:
        """Tell each waiting job where it now is in the queue."""
        with self._lock:
            waiting = [job for job, _ in self._pending]
        running = any(j.status == "running" for j in self.jobs.values())
        for position, job in enumerate(waiting, 1):
            job.queue_position = position
            if position == 1 and not running:
                # Submitted to an idle app: it is about to be picked up, and
                # telling it to wait for a video that does not exist made the
                # only job in the queue announce itself as second.
                continue
            job.message = ("Next — waiting for the current video to finish"
                           if position == 1 else
                           f"Waiting — {position - 1} ahead of it")
            self._emit(job)

    def _stopped(self, job: Job) -> None:
        job.status = "cancelled"
        job.cancelling = False
        job.queue_position = 0
        job.message = "Cancelled"
        job.finished = time.time()
        self._emit(job)

    def public_jobs(self) -> list[dict]:
        """Live jobs plus previously finished ones, newest first.

        A finished run is dropped if its file has since been deleted or moved,
        so the panel never offers to reveal something that isn't there.
        """
        live = [j.public() for j in self.jobs.values()]
        seen = {j["output"] for j in live if j.get("output")}
        past = [h for h in _load_history()
                if h.get("output") and h["output"] not in seen
                and Path(h["output"]).exists()]
        return sorted(live + past, key=lambda j: j.get("started", 0), reverse=True)

    def busy(self) -> bool:
        """True while anything is queued or running.

        The device, not the job, is what callers are really asking about — the
        voice preview wants to know whether loading a second speech model would
        contend with a job, and clearing the working files wants to know whether
        something is still writing to them. One predicate rather than three
        copies of the same comprehension.
        """
        return any(j.status in ("queued", "running") for j in self.jobs.values())

    def cancel(self, job_id: str) -> None:
        """Ask a job to stop, and say so straight away.

        The flag is only tested between stages and inside the synthesis loop, so
        a cancel during Demucs or a long Whisper pass genuinely does sit there
        until that stage finishes — minutes, on a long video. Those stages are
        subprocesses and model calls with no interruption point to offer, so
        rather than pretend otherwise, acknowledge the request immediately and
        name the stage being waited on.
        """
        self._cancel.add(job_id)
        with self._lock:
            keep = [(j, s) for j, s in self._pending if j.id != job_id]
            was_waiting = len(keep) != len(self._pending)
            self._pending = keep
        job = self.jobs.get(job_id)
        if job is not None and was_waiting:
            # Never started, so there is nothing to wind down and no reason to
            # make the user wait for a stage that isn't running.
            self._stopped(job)
            self._renumber()
            return
        if job and job.status in ("queued", "running") and not job.cancelling:
            job.cancelling = True
            stage = job.stage_label or "the current step"
            job.message = (f"Stopping — waiting for “{stage}” to finish first. "
                           "That step can't be interrupted part way.")
            self._emit(job)

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

    def _stage(self, job: Job, plan: dict, key: str, notes: list[str] | None = None):
        before, weight, label = plan[key]
        job.stage, job.stage_label = key, label

        def report(fraction: float, message: str = "") -> None:
            self._check_cancel(job)
            job.stage_progress = max(0.0, min(1.0, fraction))
            job.overall = round(before + weight * job.stage_progress, 4)
            if message:
                job.message = message
            self._emit(job)

        # The backends already receive this callback, so it is also how they
        # hand back a fallback worth keeping. Without it only the fallbacks the
        # orchestrator could observe from a return value made the report, and a
        # speech engine quietly dropping to its portable version did not.
        if notes is not None:
            report.note = notes.append                           # type: ignore[attr-defined]

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
        # Things the user should know happened but which are not failures. The
        # progress messages that carry them scroll past while a job runs, so a
        # fallback that changed the result was only visible to whoever happened
        # to be watching at the time; these survive into the finished report.
        notes: list[str] = []

        try:
            # ---------------------------------------------------- download
            # The downloaded video is keyed on the quality that fetched it too:
            # yt-dlp skips a file that is already there, so keying only the audio
            # derived from it would have re-derived everything downstream from
            # the previous quality's video and looked like a fix without being
            # one.
            source_dir = _derived_dir(workdir, "source", settings.keep_video_quality)
            report = self._stage(job, plan, "download", notes)
            video, info = download.download(job.url, source_dir,
                                            settings.keep_video_quality, report)
            job.title = info["title"]
            job.duration = info["duration"] or download.media_duration(video)
            self._emit(job)

            # Full-band stereo copy is what Demucs wants; ASR wants 16k mono.
            full_wav = source_dir / "full.wav"
            if not full_wav.exists():
                _resample_to(video, full_wav, 44100, mono=False)

            speech_wav = full_wav
            background: Path | None = None
            separated = False

            # ---------------------------------------------------- separate
            if settings.separate_audio:
                report = self._stage(job, plan, "separate", notes)
                # Stems live beside the audio they were cut from: they are cached
                # by existence too, and would otherwise outlive a quality change.
                stems = separate_backend.separate(full_wav, source_dir,
                                                  prefer_gpu=machine.fast_path,
                                                  progress=report)
                if stems:
                    speech_wav, background = stems
                    separated = True
                else:
                    notes.append("Speech and music could not be separated, so the "
                                 "original soundtrack was replaced rather than kept.")
                stats["separated"] = separated

            # Keyed on whether separation actually succeeded, not on whether it
            # was asked for: separation that was requested and then failed must
            # not be indistinguishable from separation that worked. The request
            # itself is deliberately *not* in the key — "not asked for" and
            # "asked for and failed" both leave the full downmix, so keying on
            # both would file one set of bytes under two names.
            audio_dir = _derived_dir(workdir, "speech", settings.keep_video_quality,
                                     separated)
            speech16 = audio_dir / "speech16k.wav"
            if not speech16.exists():
                _resample_to(speech_wav, speech16, 16000, mono=True)

            # ----------------------------------------------------- diarize
            turns: list[dict] = []
            if settings.diarize:
                report = self._stage(job, plan, "diarize", notes)
                turns = diarize_backend.diarize(speech16, settings.expected_speakers,
                                                progress=report)
                cache = workdir / "speakers.json"
                cache.write_text(json.dumps(turns, indent=1))
                if not turns:
                    # Falling back to one voice is right, but silence about it
                    # makes a two-person interview come back single-voiced and
                    # look like a bug in the app rather than a model that
                    # struggled.
                    stats["diarization_failed"] = True
                    notes.append("Speakers could not be told apart, so the whole "
                                 "video is dubbed in one voice. If you know how many "
                                 "people speak, set it in Settings and run it again.")
                    report(1.0, "Couldn't tell the speakers apart — using a single voice")

            # -------------------------------------------------- transcribe
            report = self._stage(job, plan, "transcribe", notes)
            cache = workdir / "segments.json"
            # Identify the audio by the folder it came out of rather than by the
            # separation *setting*: that also invalidates the transcript when
            # separation was requested but silently fell back to the full mix.
            asr_print = _fingerprint(settings.asr_model, machine.fast_path,
                                     audio_dir.name)
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

            # Labelled here so merging can respect who is speaking, and again
            # after the translation cache below, which carries its own copy.
            segments = diarize_backend.label_segments(segments, turns)
            if settings.merge_lines:
                segments = merge_adjacent(segments)

            # --------------------------------------------------- translate
            report = self._stage(job, plan, "translate", notes)
            tcache = workdir / "translated.json"
            trans_print = _fingerprint(asr_print, settings.merge_lines, settings.translator,
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

            # Labelled *after* the translation cache, not before it. The cache
            # file carries whatever speaker labels were current when it was
            # written, so reusing it replayed them — and changing "how many
            # people speak" then did nothing at all, which is precisely what the
            # note above tells the user to go and try. Translation does not
            # depend on who is speaking, so relabelling here costs nothing and
            # keeps the expensive artefact valid.
            segments = diarize_backend.label_segments(segments, turns)
            speaker_ids = sorted({s.get("speaker", 0) for s in segments})
            job.speakers = len(speaker_ids)
            stats["speakers"] = len(speaker_ids)

            # -------------------------------------------------- synthesize
            report = self._stage(job, plan, "synthesize", notes)
            engine, cloning = self._make_engine(settings, machine, segments,
                                                speech_wav, audio_dir, speaker_ids, report)
            stats["voices"] = engine.name
            # Offered for playback while the job runs: a reference with music
            # under it, or of the wrong person, colours every line that follows,
            # and this is the only moment it can still be caught cheaply.
            # Only when this run is actually cloning: Balanced and Best share an
            # audio_dir, so refs left behind by an earlier Best run would
            # otherwise be offered on a run that is not using them at all.
            job.references = ([str(p) for p in sorted((audio_dir / "refs").glob("*.wav"))]
                              if cloning else [])
            if job.references:
                self._emit(job)
            if settings.voice_mode == "clone" and not cloning:
                notes.append("The original speakers could not be cloned, so a "
                             "built-in voice was used instead.")

            # Voices matched to each speaker's own pitch, when there is more
            # than one of them and we are not cloning (a clone already sounds
            # like the speaker).
            voice_map: dict[int, str] = {}
            if not cloning and len(speaker_ids) > 1:
                voice_map = _voice_map(settings, segments, speaker_ids, speech16)
            if voice_map:
                stats["voice_match"] = "by pitch"

            # One rate for the whole track, fixed before the first line is spoken.
            project_rate = int(getattr(engine, "sample_rate", tts_backend.SAMPLE_RATE))
            stats["sample_rate"] = project_rate

            # Keyed by voice, by the translation the lines were spoken from, and
            # by the engine that spoke them: changing the voice must not replay
            # the previous voice's cached lines, a re-transcription that moved
            # the words behind an index must not leave the old audio in place,
            # and a run that fell back to a different engine must not read back
            # lines recorded at another rate. _match_rate would repair that last
            # one, but repairing on read is not the same as not being able to
            # reach it — and it costs an ffmpeg pass per line when it fires.
            # ...and by who each line was assigned to, so that changing the
            # speaker count really does re-render the voices rather than
            # replaying the previous run's.
            speaker_print = _fingerprint(
                settings.diarize, settings.expected_speakers,
                "".join(str(s.get("speaker", 0)) for s in segments),
                "".join(f"{k}{v}" for k, v in sorted(voice_map.items())))
            segdir = workdir / "lines" / _fingerprint(trans_print, settings.voice_mode,
                                                      settings.voice, settings.speed,
                                                      engine.name, project_rate,
                                                      speaker_print)
            segdir.mkdir(parents=True, exist_ok=True)
            spoken: list[dict] = []
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
                    audio, rate = sf.read(path, dtype="float32")
                else:
                    try:
                        audio, rate = engine.say(
                            text, voice_map.get(speaker) or settings.voice_for(speaker),
                            settings.speed, speaker=speaker)
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
                                audio, rate = engine.say(
                                    text,
                                    voice_map.get(speaker) or settings.voice_for(speaker),
                                    settings.speed, speaker=speaker)
                            except Exception:                    # noqa: BLE001
                                failed += 1
                                continue
                        else:
                            failed += 1
                            continue
                    if getattr(audio, "size", 0):
                        sf.write(path, audio, rate)
                if getattr(audio, "size", 0):
                    # Normalised here, on the way in, so a mid-job engine swap —
                    # or a resume that reads back a mixed set of cached lines —
                    # cannot put two rates into one track.
                    audio = _match_rate(audio, int(rate), project_rate)
                    spoken.append({"start": seg["start"], "end": seg["end"],
                                   "samples": audio, "rate": project_rate})
                if n % 5 == 0 or n == total - 1:
                    done = n + 1
                    rate = done / max(0.1, time.time() - t0)
                    report(done / total,
                           f"Speaking line {done} of {total} — about {_mins((total-done)/rate)} left")

            if not spoken:
                raise RuntimeError("Nothing was synthesised — the translation came back empty.")

            # ---------------------------------------------------- assemble
            report = self._stage(job, plan, "assemble", notes)
            video_len = download.media_duration(video)
            track, astats = align.assemble(spoken, video_len, project_rate,
                                           settings.max_stretch, report)
            stats.update(astats)
            raw = workdir / "dubbed.wav"
            sf.write(raw, track, project_rate)

            report(0.75, "Mixing")
            speech_only = raw
            if background and settings.keep_music and settings.audio_mode == "replace":
                speech_only = mux.mix_with_background(raw, background, workdir / "mixed.wav")
                stats["music_kept"] = True
            else:
                stats["music_kept"] = False

            encoded = mux.encode_track(speech_only, workdir / "dubbed.m4a", video_len)

            # ------------------------------------------------------ finish
            report = self._stage(job, plan, "finish", notes)
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
            # Measured on the assembled dub, not the finished file: with the
            # music bed kept, or in duck mode, the original sits underneath and
            # a completely silent dub still measures loud — which is precisely
            # the failure this check exists to catch. Cheaper too; no container
            # demux of a multi-gigabyte mp4.
            stats.update(mux.check_loudness(raw, video_len))
            if srt:
                shutil.copy(srt, out_path.with_suffix(".srt"))

            stats["engine"] = job.engine
            stats["lines_spoken"] = len(spoken)
            if failed:
                stats["lines_failed"] = failed
                notes.append(f"{failed} line{'s' if failed != 1 else ''} could not be "
                             "spoken and were left silent.")
            if stats.get("audio_warning"):
                notes.append(stats["audio_warning"])
            stats["notes"] = notes
            try:
                freed = prune_workdir(workdir)     # stems included; they are .wav
            except Exception:                                    # noqa: BLE001
                freed = 0        # the video is written; never fail a job over tidying

            if freed:
                stats["working_files_freed"] = round(freed / (1024 ** 2))

            job.stats = stats
            job.output = str(out_path)
            job.status = "done"
            job.finished = time.time()
            job.overall = 1.0
            job.message = f"Saved to {out_path.parent.name}/{out_path.name}"
            _record_history(job)
            self._emit(job)

        except _Cancelled:
            self._stopped(job)
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
                     audio_dir: Path, speaker_ids: list[int], report):
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

        # Under the keyed audio folder with the rest: this one feeds the clone
        # prompts, so a stale copy puts the original background music inside
        # every cloned voice.
        ref_wav = audio_dir / "speech24k.wav"
        if not ref_wav.exists():
            _resample_to(speech_wav, ref_wav, 24000, mono=True)
        audio, rate = sf.read(ref_wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Cleared first: the refs folder is shared by every run that clones this
        # audio, so clips for speakers that no longer exist would linger and be
        # offered for playback as if they belonged to this run.
        refs_dir = audio_dir / "refs"
        if refs_dir.is_dir():
            for stale in refs_dir.glob("*.wav"):
                stale.unlink(missing_ok=True)

        captured = 0
        for speaker in speaker_ids:
            ref = diarize_backend.pick_reference(segments, speaker, audio, rate)
            if ref is not None and ref.size > rate:              # at least a second
                engine.add_reference(speaker, ref, rate, audio_dir / "refs")
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
            try:
                path = JOBS / job.id
                path.mkdir(parents=True, exist_ok=True)
                (path / "error.log").write_text(detail)
            except OSError:
                pass          # raising here would kill the worker mid-queue
        self._emit(job)


class _Cancelled(Exception):
    pass


def _mins(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    return f"{seconds // 60} min"


runner = JobRunner()
