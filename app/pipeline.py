"""Job orchestration: link in, dubbed video out.

The stages that actually run depend on the quality preset, so the progress
weights are worked out per job rather than fixed. Every stage caches its output
in the job folder, so re-running a link that failed part way through picks up
where it left off instead of redoing the expensive parts.
"""
from __future__ import annotations

import hashlib
import json
import os
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

from . import logs, storage
from .config import HISTORY_FILE, JOBS, OUTPUT_DIR, VOICES, Settings, detect_machine
# Aliased: "info" below already names the probed video metadata.
from .notes import info as note_info
from .backends import asr as asr_backend
from .backends import clone as clone_backend
from .backends import diarize as diarize_backend
from .backends import separate as separate_backend
from .backends import tts as tts_backend
from .backends.translate import (describe_translator, extract_terms,
                                 translate as run_translate, TranslationError)
from .steps import align, download, mux, qc
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


def _link_id(url: str) -> str:
    """A stable id per link, so pasting the same one again resumes.

    This used to fold in the wall clock, which gave every submission a fresh
    folder and meant nothing was ever reused — the opposite of the intended
    behaviour. Python's hash() is no good here either: string hashing is
    randomised per process, so the id changed whenever the app restarted.
    """
    return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:12]


def _job_id(url: str, preview: bool = False) -> str:
    """Distinct from the work folder, which is keyed on the link alone.

    A preview and the full run of the same link have to be two jobs — they queue
    separately, they are cancelled separately, and submit() treats one id as one
    piece of work — while sharing one folder, because the entire value of
    previewing first is that the download is not paid for twice.
    """
    return _link_id(url) + ("-preview" if preview else "")


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


def _cache_clear(workdir: Path, key: str) -> None:
    """Drop a stamp whose artefact is gone, so cache-meta.json never claims a
    file that isn't there — harmless, since every read is guarded by an
    existence check, but confusing to whoever next reads the file."""
    path = workdir / "cache-meta.json"
    try:
        meta = json.loads(path.read_text())
    except Exception:                                            # noqa: BLE001
        return
    if meta.pop(key, None) is not None:
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

# Total audio read per speaker to estimate their pitch — bounds the cost on a
# speaker with many segments; no cap on any one segment within it.
PITCH_SAMPLE_SECONDS = 30.0


def _voice_map(settings: Settings, segments: list[dict], speaker_ids: list[int],
               speech_wav: Path) -> dict[int, str]:
    """Give each speaker a voice matching their own pitch.

    A deep voice dubbed by a bright female voice is the clearest sign nobody
    checked, so this reads up to PITCH_SAMPLE_SECONDS of a speaker's own
    segments, longest first, and takes one median F0 over the concatenation —
    so the majority of what they actually said decides the pitch, not
    whichever single clip was read.
    """
    voices: dict[int, str] = {}
    try:
        wav = sf.SoundFile(str(speech_wav))
    except Exception:                                            # noqa: BLE001
        return voices

    with wav:
        rate = wav.samplerate
        gap = np.zeros(int(0.1 * rate), dtype=np.float32)  # a join is not a voiced frame
        for speaker in speaker_ids:
            mine = sorted((s for s in segments if s.get("speaker") == speaker),
                          key=lambda s: s["end"] - s["start"], reverse=True)
            if not mine:
                continue
            clips: list[np.ndarray] = []
            budget = PITCH_SAMPLE_SECONDS
            for seg in mine:
                if budget <= 0:
                    break
                span = min(budget, seg["end"] - seg["start"])
                frames = int(span * rate)
                if frames <= 0:
                    continue
                try:
                    wav.seek(max(0, int(seg["start"] * rate)))
                    clip = wav.read(frames, dtype="float32", always_2d=False)
                except Exception:                                # noqa: BLE001
                    continue
                if getattr(clip, "ndim", 1) > 1:
                    clip = clip.mean(axis=1)
                clips.append(clip)
                clips.append(gap)
                budget -= span
            if not clips:
                continue
            pitch = diarize_backend.median_pitch(np.concatenate(clips), rate)
            if pitch <= 0:
                continue                      # unvoiced or too short to judge
            voices[speaker] = settings.voice_for(speaker, male=pitch < FEMALE_ABOVE_HZ)
    return voices


_VOICE_NAMES = {v["id"]: v["label"].split(" —")[0] for v in VOICES}


def _voice_names(settings: Settings, speaker_ids: list[int], voice_map: dict[int, str]) -> str:
    """Which built-in voices actually spoke, for the report — not which engine spoke them.

    Mirrors the same lookup the synthesis loop uses per line (voice_map, falling
    back to settings.voice_for), then folds speakers down to their distinct
    voices: the pool is small and reused, so several speakers commonly share one.
    """
    seen: list[str] = []
    for speaker in speaker_ids:
        vid = voice_map.get(speaker) or settings.voice_for(speaker)
        name = _VOICE_NAMES.get(vid, vid)
        if name not in seen:
            seen.append(name)
    return ", ".join(seen) if seen else _VOICE_NAMES.get(settings.voice, settings.voice)


UNREMARKABLE_SPEAKERS = 2   # two people in conversation; past that, ask


def _speaker_note(found: int, sample: bool) -> str:
    """What to say about a speaker count that is probably wrong; "" if it isn't.

    This step fails by over-segmentation — one person split across several
    clusters and dubbed in several voices — so only a count that is too high is
    worth a note. Nothing constrains the count for it to be judged against: the
    number of speakers is worked out from the audio, and the only answer the user
    can give is that there is one voice to find.
    """
    if found <= UNREMARKABLE_SPEAKERS:
        return ""
    later = "before dubbing the whole video" if sample else "and run it again"
    return (f"{found} different speakers were detected. Telling speakers apart is the "
            "least reliable step, and it can split one person into several voices. If "
            "the video has fewer people in it than that, set “Who's speaking?” "
            f"on the main page to one person {later}.")


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


# ------------------------------------------------------------------ preview

PREVIEW_SECONDS = 30.0
# Below this there is nothing to save: the preview would be most of the video,
# and the user would then wait through almost the same work twice.
PREVIEW_MIN_VIDEO = 45.0
# How far in to go looking for the first speech. Bounded because this is a full
# audio decode: unbounded, an hour of video costs a quiet twenty seconds before
# the preview has started, which is the opposite of the point.
PREVIEW_SCAN_SECONDS = 300.0
# Speech is rarely quieter than this, and title music rarely under it.
PREVIEW_SILENCE_DB = -40.0


def _speech_start(media: Path, duration: float, window: float = PREVIEW_SECONDS) -> float:
    """Where the talking starts, so a preview isn't 30 seconds of title card.

    Most videos open on a logo, a music sting or a held shot, and a preview of
    that tells the user nothing about the thing they are trying to judge. ffmpeg
    already knows where the quiet is — the same silencedetect the finished-audio
    check uses — so ask it rather than guessing at an offset.

    Only a leading silence counts. A gap in the middle of a conversation is not
    a lead-in, and treating it as one would drop the window into an arbitrary
    later part of the video for no reason.
    """
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-nostats", "-t", str(PREVIEW_SCAN_SECONDS),
         "-i", str(media), "-map", "0:a:0",
         "-af", f"silencedetect=noise={PREVIEW_SILENCE_DB}dB:d=1.0", "-f", "null", "-"],
        capture_output=True, text=True)

    start = 0.0
    lead_in = False
    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            try:
                begins = float(line.split("silence_start:")[1].strip().split()[0])
            except (IndexError, ValueError):
                break
            # Anything but a silence that is already running when the video
            # opens means the speech has started; stop looking.
            if begins > 0.5:
                break
            lead_in = True
        elif "silence_end:" in line and lead_in:
            try:
                start = float(line.split("silence_end:")[1].strip().split()[0])
            except (IndexError, ValueError):
                start = 0.0
            break

    # A lead-in that filled everything that was looked at is not a lead-in.
    # Without this, ninety seconds of silence reported its silence_end at the
    # end of the file and the window was placed in the last thirty seconds of a
    # video with nothing in it. Which of the two readings applies depends on
    # what stopped the scan: the end of the video means there is no speech to
    # find and the beginning is as good a place as any, while the scan limit
    # means the speech is somewhere past it and the boundary is the best guess
    # available.
    scanned = min(duration, PREVIEW_SCAN_SECONDS) if duration > 0 else PREVIEW_SCAN_SECONDS
    if start >= scanned - 0.5:
        start = scanned if duration > scanned else 0.0

    # Never past the end: a video whose speech begins in its last few seconds
    # would otherwise be handed a window with nothing in it. Guarded on a known
    # duration rather than on it exceeding the window, so that a video shorter
    # than the window — or a host that reports no duration at all — falls back
    # to the beginning instead of skipping past what little there is.
    if duration > 0:
        start = min(start, max(0.0, duration - window))
    return max(0.0, round(start, 2))


def _trim(src: Path, dst: Path, start: float, length: float) -> Path:
    """Cut a window out of the video, re-encoding so it stands on its own.

    Deliberately not -c copy. A stream copy can only cut on a keyframe, so the
    window would drift from the one that was chosen, and the file would open on
    a partial frame — which the player in the interface shows as a grey smear
    for the first second. Re-encoding thirty seconds costs a few seconds and
    removes both problems.

    -ss ahead of -i so ffmpeg seeks rather than decoding everything before the
    window; on an hour of video that is the difference between instant and not.
    """
    if dst.exists():
        return dst
    # Keeps the .mp4 on the end: ffmpeg picks the container from the extension,
    # and a bare ".part" leaves it with nothing to infer from and no output at
    # all.
    tmp = dst.with_suffix(".part.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}",
                    "-i", str(src), "-t", f"{length:.2f}",
                    "-map", "0:v:0", "-map", "0:a:0",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k", str(tmp)], check=True)
    # Moved into place only once ffmpeg has succeeded, so a cancel or a crash
    # part way through cannot leave a truncated file that the next run treats as
    # a finished cut.
    tmp.replace(dst)
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
    # What the failing tool actually said. Shown behind a disclosure rather than
    # in the message, so a failure is diagnosable without being frightening.
    error_detail: str = ""
    output: str = ""
    stats: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    began: float = 0.0                   # when it reached the front of the queue
    finished: float = 0.0
    engine: str = ""
    preset: str = ""
    speakers: int = 0
    cancelling: bool = False
    queue_position: int = 0              # 0 = running, or not waiting
    references: list[str] = field(default_factory=list)
    preview: bool = False                # a 30s taster, not the finished thing
    preview_from: float = 0.0            # where in the video the window starts
    # The stages this job actually runs, in order. The interface used to hold
    # its own fixed list, which omitted separation and speaker detection — so
    # for the whole of Demucs, often the longest part of a Balanced run, no chip
    # was highlighted and the row looked inert at exactly the moment the user is
    # asking whether anything is happening.
    stages: list[str] = field(default_factory=list)
    # Seconds each finished stage took. Nothing measured this before, so the only
    # figure anywhere was the whole run — which answers "was that slow" but never
    # "slow where", and on a 90-minute job that is the only useful question.
    # Stages run strictly in sequence, so _stage() closes the previous one's
    # clock as it opens the next and no separate bookkeeping is needed.
    stage_times: dict[str, float] = field(default_factory=dict)
    stage_began: float = 0.0

    def public(self) -> dict:
        d = asdict(self)
        # Time spent working, not time since the link was pasted. A job that sat
        # twenty minutes behind another and then ran for five reported
        # twenty-five, so "Took" was wrong — and an estimate of the time
        # remaining, which divides elapsed by the fraction done, would be wrong
        # by the same margin in the same direction. Nothing is elapsed while a
        # job is still waiting its turn.
        d["elapsed"] = (round((self.finished or time.time()) - self.began)
                        if self.began else 0)
        # Worked out here rather than from stage_began in the browser: that is an
        # epoch stamp from this machine's clock, and the two need not agree.
        d["stage_elapsed"] = (round(time.time() - self.stage_began, 1)
                              if self.stage_began else 0.0)
        # Answered every time it is asked, never stored. A finished video can be
        # moved, renamed or deleted while this app is not running, so a recorded
        # answer is only ever a claim about the past.
        d["output_exists"] = bool(self.output) and Path(self.output).is_file()
        return d


HISTORY_LIMIT = 50
# A failure only ever competes for room with other failures, never with a
# finished video — see _record_history(). Higher than the 6 the panel shows
# at once, so "…and N more" still means something, but nowhere near
# HISTORY_LIMIT: the audit this whole item answers to found six and ten
# consecutive failures in a single sitting, from a machine YouTube kept
# refusing, and this comfortably outlasts a run twice that bad without
# holding a really old failure hostage forever either.
FAILED_HISTORY_LIMIT = 20

# What a job folder keeps once the job has succeeded. Everything else in there
# is bulk that can be rebuilt.
KEEP_SUFFIXES = {".json", ".srt", ".log"}


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
    """Keep a run once its in-memory record has been replaced or lost — self.jobs
    is rebuilt empty on every launch, so this is the only durable record of
    what happened, on either success or failure.

    A success is superseded only by another success with the same output file,
    since two runs of one link can both produce a kept video; a failure is
    superseded by job_id alone, by a later success as much as a later failure,
    since either means it is no longer the last word on that link.
    """
    entry = job.public()
    # Derived at read time by public_jobs(); anything stored here would be
    # answering for a file the app cannot watch while it is not running.
    entry.pop("output_exists", None)
    # Distinct from the live job id, so a re-run of the same link lists as its
    # own row rather than collapsing onto the previous one.
    entry["id"] = f"{job.id}-{int(job.finished)}"
    entry["job_id"] = job.id

    def superseded(h: dict) -> bool:
        if not h.get("output"):
            # At most one failure record per link: whichever was written last,
            # whether this entry is itself another failure or the success that
            # finally followed it.
            return h.get("job_id") == entry["job_id"]
        # A success is only replaced by another success that produced the very
        # same file. A failure carries no output to collide on, so it can never
        # take a success's row — retrying a link and failing must not erase the
        # file that a previous attempt already produced.
        return bool(entry["output"]) and h.get("output") == entry["output"]

    history = [h for h in _load_history() if not superseded(h)]
    history.append(entry)
    # Trimmed as two lists, not one: before failures were recorded at all, this
    # was 50 finished videos every time. A run of failures sharing that same
    # budget would silently evict every dubbed video behind them — exactly
    # backwards, since a losing streak is precisely when the history panel
    # (and the finished file it points at) matters most. Each kind is capped
    # and sorted on its own and only interleaved again once both are within
    # their own limit, so public_jobs() and the next _load_history() still see
    # one newest-first list either way.
    successes = sorted((h for h in history if h.get("output")),
                       key=lambda h: h.get("finished", 0))
    failures = sorted((h for h in history if not h.get("output")),
                      key=lambda h: h.get("finished", 0))
    history = sorted(successes[-HISTORY_LIMIT:] + failures[-FAILED_HISTORY_LIMIT:],
                     key=lambda h: h.get("finished", 0))
    try:
        # Written whole, then moved into place: request threads read this file,
        # and one that caught it mid-write saw truncated JSON and reported no
        # history at all.
        tmp = HISTORY_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(history, indent=1))
        tmp.replace(HISTORY_FILE)
    except Exception:                                            # noqa: BLE001
        pass                      # history is a convenience, never worth a job


class KeepAwake:
    """Hold the Mac awake while there is work to do.

    An hour of video is an hour of work, and a laptop left alone sleeps long
    before that. The job does not fail when it happens — it simply stops, and
    resumes when somebody touches the trackpad, which to anyone who started it
    and walked away is indistinguishable from the app having hung.

    Deliberately not -d. Keeping the screen lit for an hour drains the battery
    and is not what was asked for; the work continues perfectly well with the
    display off.

    -w ties the assertion to this process, so a crash cannot leave a Mac that
    never sleeps again. The explicit stop covers the ordinary case.
    """

    FLAGS = ("-i",    # don't idle-sleep the system
             "-m",    # don't idle-sleep the disk
             "-s")    # don't sleep at all while on mains power

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Idempotent, and safe to call from the interface thread mid-job."""
        with self._lock:
            if self.active:
                return
            try:
                self._proc = subprocess.Popen(
                    ["caffeinate", *self.FLAGS, "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                self._proc = None  # not macOS, or no caffeinate — nothing to do

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:                                    # noqa: BLE001
                proc.kill()


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
        self.awake = KeepAwake()

    def sync_keep_awake(self, wanted: bool) -> bool:
        """Apply the setting to the assertion being held right now.

        A switch that only takes effect on the next job is a switch that lies
        for the hour it matters most, so this acts on the live one. Turning it
        on while nothing is running holds nothing — there is no work to protect.
        """
        if wanted and self.busy():
            self.awake.start()
        elif not wanted:
            self.awake.stop()
        return self.awake.active

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

    def submit(self, url: str, settings: Settings, preview: bool = False) -> Job:
        job_id = _job_id(url, preview)
        with self._lock:
            existing = self.jobs.get(job_id)
            # The same link submitted twice is one job, not two racing over the
            # same folder. A preview and a full run carry different ids, so they
            # are two jobs — but two previews, or two impatient clicks on "dub
            # the whole video", are still one.
            if existing and existing.status in ("queued", "running"):
                return existing
            job = Job(id=job_id, url=url, preset=settings.preset, preview=preview)
            self.jobs[job_id] = job
            self._cancel.discard(job_id)
            self._pending.append((job, settings))
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, daemon=True)
                self._worker.start()
        self._renumber()
        return job

    def _work(self) -> None:
        """Drain the queue, one job at a time, then retire.

        Held awake for the whole drain rather than per job, so a queue does not
        let the machine doze off in the gap between two of them.
        """
        if Settings.load().keep_awake:
            self.awake.start()
        try:
            self._drain()
        finally:
            self.awake.stop()

    def _drain(self) -> None:
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
                job.began = time.time()
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
        """Live jobs plus previously finished or failed ones, newest first.

        output_exists is worked out here rather than trusted from history, since
        a finished video can be moved or deleted while the app isn't running.

        A recorded failure is hidden only while the live job at that id is
        itself still in the error state — a retry that is running, cancelled,
        or since finished is a different event and doesn't make the earlier
        failure untrue.
        """
        live = [j.public() for j in self.jobs.values()]
        seen_outputs = {j["output"] for j in live if j.get("output")}
        live_error_ids = {j["id"] for j in live if j.get("status") == "error"}
        past = []
        for h in _load_history():
            if h.get("output"):
                if h["output"] in seen_outputs:
                    continue
                h = {**h, "output_exists": Path(h["output"]).is_file()}
            elif h.get("job_id") in live_error_ids:
                continue
            else:
                h = {**h, "output_exists": False}
            past.append(h)
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
    # A preview shrinks every stage except the download, which still fetches the
    # whole video. Weighted for a full run, the download's 8% left the bar
    # parked near zero for most of the wait and then sprinting — the single
    # worst thing a progress bar can do to someone deciding whether to give up.
    # Not the raw ratio, though: loading Kokoro and Parakeet costs the same
    # twenty seconds whether there are thirty seconds of audio or thirty
    # minutes, so the stages after the download never shrink as far as the
    # duration alone suggests. Capped where that fixed cost starts to dominate.
    PREVIEW_DOWNLOAD_CAP = 12.0

    def _plan(self, settings: Settings,
              download_weight: float = 1.0) -> dict[str, tuple[float, float, str]]:
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
            if key == "download":
                cost = cost * download_weight
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
        self._close_stage(job)
        job.stage, job.stage_label = key, label
        job.stage_began = time.time()
        logs.get(job.id).info("stage started", extra={"stage": key})

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

    def _close_stage(self, job: Job) -> None:
        """Stop the clock on whatever was running. Called by _stage and at the end."""
        if not job.stage or not job.stage_began:
            return
        spent = round(time.time() - job.stage_began, 1)
        # Added to, not replaced: a resumed link re-enters a stage it has already
        # been through, and two visits to "download" are one download's cost.
        job.stage_times[job.stage] = round(job.stage_times.get(job.stage, 0.0) + spent, 1)
        job.stage_began = 0.0
        logs.get(job.id).info("stage finished",
                              extra={"stage": job.stage, "seconds": spent})

    # ---------------------------------------------------------- execution
    def _run(self, job: Job, settings: Settings) -> None:
        # Set once, so every log record made anywhere beneath this — including in
        # backends that have no idea a job exists — says which job it belongs to.
        logs.current_job.set(job.id)
        machine = detect_machine()
        # Keyed on the link, not on the job: a preview and the full run of the
        # same video are two jobs sharing one folder, which is what lets the
        # full run skip a download the preview has already paid for.
        workdir = JOBS / _link_id(job.url)
        workdir.mkdir(parents=True, exist_ok=True)
        job.status = "running"
        job.engine = "Apple GPU (MLX)" if machine.fast_path else "Portable (CPU)"
        self._emit(job)

        # The settings this run was submitted with, not the ones selected by the
        # time anybody reads the result. Settings can be changed while a job is
        # queued, and are routinely changed before the next one is started.
        stats: dict = {"preset": settings.preset,
                       "settings": settings.run_snapshot()}
        # Things the user should know happened but which are not failures. The
        # progress messages that carry them scroll past while a job runs, so a
        # fallback that changed the result was only visible to whoever happened
        # to be watching at the time; these survive into the finished report.
        notes: list[str] = []

        try:
            # ---------------------------------------------------- download
            # Probed before the plan is drawn rather than inside the download,
            # because on a preview the plan has to know how much of the video
            # the download covers before it can weight it. Costs nothing: this
            # is the same call download() made for itself a moment later, now
            # handed to it.
            job.message = "Looking up the video…"
            self._emit(job)
            info = download.probe(job.url, settings.youtube_cookies)
            job.title = info["title"]
            job.duration = float(info["duration"] or 0)
            self._emit(job)

            # Left in the folder so the disk-space panel can name what it is
            # offering to delete. The folder is a hash of the link, and a sample
            # is deliberately never written to the history — which is exactly the
            # case someone is most likely to want to clear, and the one that
            # would otherwise be listed as twelve characters of SHA-1. A .json
            # suffix so prune_workdir keeps it.
            try:
                (workdir / "info.json").write_text(json.dumps(
                    {"title": job.title, "url": job.url}, ensure_ascii=False))
            except OSError:
                pass                        # a label; never worth failing a job

            window = PREVIEW_SECONDS if job.preview else 0.0
            if job.preview and 0 < job.duration <= PREVIEW_MIN_VIDEO:
                # Sampling most of a short video means waiting through
                # substantially the same work twice for no information.
                job.preview, window = False, 0.0
                notes.append(note_info(f"That video is only {_mins(job.duration)} long, so "
                             "the whole thing was dubbed rather than a sample of it."))
                self._emit(job)

            # Unweighted when the site reports no duration — live streams and a
            # few hosts do that, and a plausible-looking bar beats a division by
            # zero.
            # Checked here, where the duration is known and nothing has been
            # written yet. A job holds the source video, a full-band wav, the
            # separated stems and the assembled track at once before it prunes
            # itself, and filling the boot disk does not just fail the job — the
            # whole machine stops working at the same moment. Better to refuse
            # with a number the user can act on than to find out at 80%.
            # Asked here rather than left to the download, because the check
            # this feeds happens before anything is fetched. Pure dictionary
            # work on the listing the probe already returned; no second request.
            picked = download.choose_format(info, settings.keep_video_quality)
            need = storage.estimate_needed(window or job.duration,
                                           settings.keep_video_quality,
                                           source_bytes=(picked or {}).get("bytes", 0))
            free = storage.free_bytes(JOBS)
            if free < need:
                raise RuntimeError(
                    f"There isn't enough room on the disk. This video needs about "
                    f"{storage.human_size(need)} while it works, and there is "
                    f"{storage.human_size(free)} free. "
                    "Clear some working files in the app, empty the bin, and try again.")

            weight = (min(self.PREVIEW_DOWNLOAD_CAP, job.duration / window)
                      if window and job.duration else 1.0)
            plan = self._plan(settings, weight)
            job.stages = list(plan)

            # The downloaded video is keyed on the quality that fetched it too:
            # yt-dlp skips a file that is already there, so keying only the audio
            # derived from it would have re-derived everything downstream from
            # the previous quality's video and looked like a fix without being
            # one.
            source_dir = _derived_dir(workdir, "source", settings.keep_video_quality)
            report = self._stage(job, plan, "download", notes)
            video, _ = download.download(job.url, source_dir,
                                         settings.keep_video_quality, report, info,
                                         cookies_from=settings.youtube_cookies)
            if not job.duration:
                job.duration = download.media_duration(video)
            self._emit(job)

            # ----------------------------------------------------- preview
            # The window is cut here, after the whole video has been fetched,
            # rather than asked of yt-dlp with --download-sections. The download
            # is 8 of the 100 units of work in ALL_STAGES and everything that
            # scales with duration comes after it, so fetching only a section
            # would save that 8% once and then charge the entire download again
            # the moment the user says yes. Cutting locally leaves source.mp4 in
            # the shared folder and the full run skips its download outright.
            media_dir = source_dir
            if window:
                # Both at 1.0: the download has already reported itself finished,
                # and a bar that steps backwards to make room for a message is
                # worse than a bar that holds still while the message changes.
                report(1.0, "Finding where the speech starts")
                job.preview_from = _speech_start(video, job.duration, window)
                media_dir = _derived_dir(workdir, "preview", settings.keep_video_quality,
                                         job.preview_from, window)
                report(1.0, f"Taking {int(window)} seconds from {_clock(job.preview_from)}")
                video = _trim(video, media_dir / "source.mp4", job.preview_from, window)
                self._emit(job)

            # Full-band stereo copy is what Demucs wants; ASR wants 16k mono.
            # Derived from the trimmed video on a preview, so the whole-video
            # conversion — a real cost on an hour of source — never happens.
            full_wav = media_dir / "full.wav"
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
                stems = separate_backend.separate(full_wav, media_dir,
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
            # A preview's audio gets its own namespace rather than an extra
            # component on the full run's key, so that adding this feature does
            # not invalidate the transcript and translation of every job already
            # in the folder — those are keyed on audio_dir.name, and they are
            # the expensive ones.
            audio_dir = _derived_dir(
                workdir, *(("speech-preview", settings.keep_video_quality, separated,
                            job.preview_from, window) if window else
                           ("speech", settings.keep_video_quality, separated)))
            speech16 = audio_dir / "speech16k.wav"
            if not speech16.exists():
                _resample_to(speech_wav, speech16, 16000, mono=True)

            # ----------------------------------------------------- diarize
            turns: list[dict] = []
            if settings.diarize:
                report = self._stage(job, plan, "diarize", notes)
                turns = diarize_backend.diarize(speech16, progress=report)
                cache = workdir / "speakers.json"
                cache.write_text(json.dumps(turns, indent=1))
                if not turns:
                    # Falling back to one voice is right, but silence about it
                    # makes a two-person interview come back single-voiced and
                    # look like a bug in the app rather than a model that
                    # struggled.
                    stats["diarization_failed"] = True
                    notes.append("Speakers could not be told apart, so the whole "
                                 "video is dubbed in one voice.")
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
            pcache = workdir / "translated.partial.json"

            # Asked of this video before any of it is translated. The built-in
            # glossary only chooses between UK and US stitch names, and its
            # source side is Spanish, so on a Portuguese video it matched nothing
            # at all — while the same model, shown the transcript and asked
            # directly, names the vocabulary correctly and identically every
            # time. Cached alongside everything else, and folded into the
            # fingerprint below so that changing it re-translates rather than
            # replaying a translation made without it.
            gcache = workdir / "terminology.txt"
            gprint = _fingerprint(asr_print, settings.translator, settings.target_language,
                                  settings.resolved_ollama_model(machine.ram_gb))
            if gcache.exists() and _cache_valid(workdir, "terminology", gprint):
                found = gcache.read_text()
            else:
                found = extract_terms(segments, settings, machine.ram_gb, report)
                gcache.write_text(found)
                _cache_stamp(workdir, "terminology", gprint)
            if found:
                stats["terms_found"] = len(found.splitlines())

            trans_print = _fingerprint(asr_print, settings.merge_lines, settings.translator,
                                       settings.resolved_ollama_model(machine.ram_gb),
                                       settings.anthropic_model, settings.openai_model,
                                       settings.target_language, settings.glossary_text(),
                                       found)
            if tcache.exists() and _cache_valid(workdir, "translated", trans_print):
                segments = json.loads(tcache.read_text())
                report(1.0, "Reusing the translation from last time")
            else:
                # A prior attempt at these exact settings may have died part
                # way through translating. Guarded by the same fingerprint as
                # the finished cache, so a partial from before a settings
                # change is never mistaken for a head start on this one.
                resume: dict[int, str] = {}
                if pcache.exists() and _cache_valid(workdir, "translated_partial", trans_print):
                    try:
                        resume = {int(i): t for i, t in json.loads(pcache.read_text()).items()}
                    except Exception:                            # noqa: BLE001
                        resume = {}
                    if resume:
                        notes.append(note_info(f"Picking up where the last attempt left off — "
                                     f"{len(resume)} of {len(segments)} lines were "
                                     "already translated."))

                # Written after every batch, not just on failure: a kill -9 or
                # a power cut runs no exception handler, so only what has
                # already reached disk survives it. The fingerprint is the
                # same on every batch, so it is stamped once, up front, rather
                # than repeated on each write.
                _cache_stamp(workdir, "translated_partial", trans_print)

                def _save_partial(done: dict[int, str]) -> None:
                    pcache.write_text(json.dumps(done, ensure_ascii=False, indent=1))

                segments = run_translate(segments, settings, machine.ram_gb, report,
                                         extra_glossary=found, resume=resume,
                                         on_batch=_save_partial)
                tcache.write_text(json.dumps(segments, ensure_ascii=False, indent=1))
                _cache_stamp(workdir, "translated", trans_print)
                pcache.unlink(missing_ok=True)
                _cache_clear(workdir, "translated_partial")

            # Checked whether it was just produced or restored from cache. The
            # cache is the reason this cannot live in the parser: a translation
            # that went wrong once would otherwise be replayed on every re-run of
            # the same link, and the run that produced it is long gone.
            quality = qc.check(segments)
            stats["translation_check"] = quality
            trouble = qc.summarise(quality)
            if trouble:
                notes.append(trouble)

            # Recorded whether it ran or was reused, since the report describes
            # the file that exists rather than the work done this time.
            stats["translated_by"], weak = describe_translator(settings, machine.ram_gb)
            if weak:
                notes.append(weak)

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

            # More voices than a video can plausibly hold, judged against a
            # two-way conversation. Runtime is no guide to how many people
            # speak: a 90-minute craft tutorial has one or two, and diarization
            # reported 28 speakers for a ten-minute film with about seven
            # characters in it, while the report said "Speakers found: 28" with
            # no comment — so a dub in 28 different voices arrived looking like
            # what the app meant to do. There is already a lever for this, and
            # the person who can pull it is the one who watched the video.
            if settings.diarize:
                wrong = _speaker_note(len(speaker_ids), bool(window))
                if wrong:
                    notes.append(wrong)

            # -------------------------------------------------- synthesize
            report = self._stage(job, plan, "synthesize", notes)
            engine, cloning = self._make_engine(settings, machine, segments,
                                                speech_wav, audio_dir, speaker_ids, report)
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
            # Which voice spoke, not which engine spoke it — the two used to be
            # conflated here, so a built-in-voice run reported "Apple GPU (MLX)"
            # under "Voices used" instead of naming a single one of the voices.
            stats["voices"] = engine.name if cloning else _voice_names(
                settings, speaker_ids, voice_map)

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
                settings.diarize,
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
                                notes.append("The voice engine stopped working "
                                             "partway through, so the rest of the "
                                             "lines were spoken by the portable "
                                             "engine instead. The voices themselves "
                                             "didn't change.")
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
            if background and settings.keep_music and settings.keep_music_applies():
                speech_only = mux.mix_with_background(raw, background, workdir / "mixed.wav")
                stats["music_kept"] = True
            else:
                stats["music_kept"] = False

            encoded = mux.encode_track(speech_only, workdir / "dubbed.m4a", video_len)

            # ------------------------------------------------------ finish
            report = self._stage(job, plan, "finish", notes)
            srt = align.write_srt(segments, workdir / "subtitles.srt") if settings.write_srt else None

            if job.preview:
                # Deliberately not in the finished-videos folder. A sample is
                # not a deliverable, and thirty-second stubs sitting beside real
                # output — under names that differ only by a suffix — is exactly
                # the mess someone who does not read filenames closely cannot
                # get themselves out of. It plays in the app and nowhere else.
                out_path = media_dir / "preview.mp4"
                out_path.unlink(missing_ok=True)
            else:
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
            # A dub the recipient cannot open is not finished, so when the site
            # offered no H.264 the picture is converted rather than shipped with
            # a warning nobody can act on. Rare — YouTube publishes H.264 down to
            # 144p even on videos from 2005 — which is why it is worth the encode
            # on the occasions it does happen.
            if stats.get("video_codec") and not stats.get("widely_playable"):
                was = stats["video_codec"].upper()
                report(0.85, "Converting the picture so it plays anywhere")
                converted = out_path.with_name(out_path.stem + ".h264.mp4")
                encoder = mux.transcode_h264(out_path, converted)
                if encoder:
                    converted.replace(out_path)
                    stats.update(mux.verify(video, out_path))
                    stats["transcoded_from"] = was
                    notes.append(
                        f"This video isn't offered in H.264, only {was}, so the "
                        "picture was converted to H.264 after dubbing. That is why "
                        "the finished file took longer than usual. It now plays on "
                        "any Mac, phone or browser.")
                else:
                    notes.append(
                        f"The picture is {was}, because this video isn't offered in "
                        "H.264 at all, and converting it didn't work either. "
                        + ("It plays on this Mac, but not on Macs older than an M3, "
                           "and not reliably on phones — VLC opens it everywhere."
                           if machine.av1_ok else
                           "QuickTime opens the file with sound but no picture. VLC "
                           "will play it."))
            # The silence warning exists to catch a run that died half way, and
            # it decides on the share of the track that is quiet. That share is
            # also just what a time-fitted dub looks like: every line starts on
            # its original timestamp and English is shorter than most of what it
            # replaces, so the gaps are real and correct. Measured on a 98-minute
            # crochet tutorial — 92% speech in the original, 38% in the dub, all
            # 448 lines present and spoken, and the report told her it was "what
            # a half-failed run looks like".
            #
            # So the share alone no longer raises it. If nothing failed to speak
            # and the check found nothing to silence, every line is there and the
            # quiet is the shape of the video, not evidence of anything.
            checked = stats.get("translation_check") or {}
            everything_spoken = not failed and not checked.get("untranslated") \
                and not checked.get("template") and not checked.get("empty")
            if stats.get("audio_present") is False:
                # Distinct from the above and never suppressed: a track with
                # nothing audible anywhere is wrong however the lines counted.
                notes.append(stats["audio_warning"])
            elif stats.get("audio_warning") and not everything_spoken:
                notes.append(stats["audio_warning"])
            # Separate from that warning, and never suppressed by it: a run can
            # fail nothing at all — every line found gets translated and spoken —
            # and still have found almost no lines to dub, e.g. a largely-sung
            # short with one spoken line in it. That is a real success, not a
            # half-failed run, so it gets no "might be a failure" hedge; it just
            # gets said. 90% is well past the natural gaps timing alone produces
            # (38%, on the talky 98-minute tutorial above) — past it, there is
            # barely any dubbed video here, whatever the reason.
            if stats.get("no_line_share", 0) >= 0.9:
                n = stats["lines_spoken"]
                notes.append(
                    f"Only {n} line{'' if n == 1 else 's'} {'was' if n == 1 else 'were'} "
                    f"dubbed, leaving {int(stats['no_line_share'] * 100)}% of the video "
                    "with no dubbed line over it. That can be correct for a video with "
                    "very little speech in it — worth a look before sending it on.")
            if job.preview:
                stats["preview"] = True
                stats["preview_from"] = _clock(job.preview_from)
                # Said plainly, because a good sample invites exactly the wrong
                # conclusion. Timing is the thing that fails across a whole
                # video and the thing thirty seconds cannot speak to.
                notes.append(note_info(
                    f"This is a {int(window)}-second sample taken from "
                    f"{_clock(job.preview_from)}. It shows the voice, the wording and "
                    "the sound levels. It can't promise the timing holds for the whole "
                    "video, and a longer video may have more speakers in it."))
            stats["notes"] = notes

            freed = 0
            if not job.preview:
                try:
                    freed = prune_workdir(workdir)  # stems included; they are .wav
                except Exception:                                # noqa: BLE001
                    freed = 0    # the video is written; never fail a job over tidying
            # A preview never prunes: the bulk it would reclaim is the shared
            # download, which is the one thing the full run behind it is
            # counting on finding. The sample's own leftovers are thirty seconds
            # of audio, and the full run clears them along with everything else.

            if freed:
                stats["working_files_freed"] = round(freed / (1024 ** 2))

            self._close_stage(job)
            job.stats = stats
            job.output = str(out_path)
            job.status = "done"
            job.finished = time.time()
            job.overall = 1.0
            job.message = ("Sample ready — have a listen" if job.preview
                           else f"Saved to {out_path.parent.name}/{out_path.name}")
            if not job.preview:
                # Kept out of the history list for the same reason it is kept out
                # of the videos folder: it is not a thing the user asked to have.
                _record_history(job)
            logs.get(job.id).info("job finished", extra={
                "seconds": round(job.finished - job.began, 1),
                "duration": round(job.duration, 1), "preview": job.preview,
                "stage_times": job.stage_times, "codec": stats.get("video_codec"),
                "lines_spoken": stats.get("lines_spoken"),
                "notes": notes})
            self._emit(job)

        except _Cancelled:
            self._stopped(job)
        except download.DownloadError as exc:
            # Set before _fail() is called, not after: _fail() is also what
            # writes the record a reopened app reads a failure back from, and a
            # detail attached only once that record already exists would never
            # reach it.
            job.error_detail = exc.detail
            self._fail(job, str(exc), exc.detail)
        except TranslationError as exc:
            job.error_detail = exc.detail
            self._fail(job, str(exc), exc.detail)
        except FileNotFoundError as exc:
            missing = getattr(exc, "filename", "") or str(exc)
            hint = ("ffmpeg is missing — re-run the installer."
                    if "ffmpeg" in str(missing) or "ffprobe" in str(missing)
                    else f"A required program is missing: {missing}")
            self._fail(job, hint)
        except subprocess.CalledProcessError as exc:
            # str() on this is the entire command line, arguments and temporary
            # paths and all. Shown in the interface it reads as a crash rather
            # than as a video that could not be processed, and there is nothing
            # in it a non-technical user can act on. The full command still goes
            # to error.log, where it is worth having.
            tool = Path(str(exc.cmd[0])).name if exc.cmd else "a helper program"
            self._fail(job, f"{tool} couldn't process that video — it may be in a "
                            f"format that can't be read. (Error {exc.returncode}.)",
                       traceback.format_exc())
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
        self._close_stage(job)
        job.status = "error"
        job.error = message
        job.message = message
        job.finished = time.time()
        # "reason", not "message": LogRecord already owns that attribute and
        # raises rather than shadowing it, which turned every failed job into a
        # second failure inside the logging call.
        logs.get(job.id).error("job failed", extra={
            "stage": job.stage, "url": job.url, "reason": message,
            "detail": (job.error_detail or detail)[-2000:],
            "stage_times": job.stage_times})
        if detail:
            try:
                path = JOBS / job.id
                path.mkdir(parents=True, exist_ok=True)
                (path / "error.log").write_text(detail)
            except OSError:
                pass          # raising here would kill the worker mid-queue
        self._emit(job)
        # self.jobs is rebuilt empty on every launch, so without this a failure
        # would be reachable only for as long as this process keeps running.
        _record_history(job)


class _Cancelled(Exception):
    pass


def _mins(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    return f"{seconds // 60} min"


def _clock(seconds: float) -> str:
    """A position in the video, written the way a player writes it."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


runner = JobRunner()
