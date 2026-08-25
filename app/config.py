"""Settings, paths and machine detection for Dubbing Studio."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict, fields
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir

from . import logs

APP_NAME = "Dubbing Studio"

# ---------------------------------------------------------------- paths

# Two roots: settings/history are irreplaceable and go where backups look;
# models/working files are re-fetchable and belong in the purgeable cache,
# sparing backups the tens of gigabytes a speech model collection can run to.
def _base_dir() -> Path:
    env = os.environ.get("DUBBING_STUDIO_HOME")
    if env:
        return Path(env)
    return Path(user_data_dir("DubbingStudio", appauthor=False))


def _cache_dir(base: Path) -> Path:
    env = os.environ.get("DUBBING_STUDIO_CACHE")
    if env:
        return Path(env)
    # DUBBING_STUDIO_HOME means "keep it all here" (the test suite relies on
    # this); the base/cache split only applies when the location was left to us.
    if os.environ.get("DUBBING_STUDIO_HOME"):
        return base / "cache"
    return Path(user_cache_dir("DubbingStudio", appauthor=False))


BASE = _base_dir()
CACHE = _cache_dir(BASE)
JOBS = CACHE / "jobs"
MODELS = CACHE / "models"
PREVIEWS = CACHE / "previews"
SETTINGS_FILE = BASE / "settings.json"
HISTORY_FILE = BASE / "history.json"
OUTPUT_DIR = Path(os.environ.get("DUBBING_STUDIO_OUTPUT", str(Path.home() / "Movies" / "Dubbed")))


def ollama_host() -> str:
    """Where to reach Ollama.

    Accepts a bare host:port as well as a full URL, because that is the form the
    OLLAMA_HOST variable normally takes.
    """
    host = os.environ.get("OLLAMA_HOST", "").strip() or "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")

# Migrate jobs/models/previews left over from the old single-root layout into
# the cache, so multi-gigabyte models aren't re-downloaded. "Not already there"
# means empty, not merely present, since JOBS/MODELS are created unconditionally
# below — otherwise a failed rename (e.g. EXDEV) would never be retried.
for _name in ("jobs", "models", "previews"):
    _was, _now = BASE / _name, CACHE / _name
    _now_in_use = _now.is_dir() and any(_now.iterdir())
    if _was.is_dir() and not _was.is_symlink() and not _now_in_use:
        try:
            _now.parent.mkdir(parents=True, exist_ok=True)
            _was.rename(_now)
        except OSError as _exc:
            logs.log_before_ready(f"could not move {_was} to {_now}: {_exc}")

for _p in (BASE, CACHE, JOBS, MODELS, OUTPUT_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- machine

@dataclass
class Machine:
    system: str
    arch: str
    apple_silicon: bool
    ram_gb: int
    has_mlx: bool
    has_ffmpeg: bool
    has_ytdlp: bool
    has_ollama: bool
    av1_ok: bool

    @property
    def fast_path(self) -> bool:
        """True when we can use Apple's GPU via MLX."""
        return self.apple_silicon and self.has_mlx


def mac_generation() -> int:
    """Which Apple silicon generation this is, or 0 for anything else.

    "Apple M4 Pro" -> 4. Intel, or a machine that will not say, gives 0.
    """
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return 0
    import re
    m = re.search(r"Apple M(\d+)", brand)
    return int(m.group(1)) if m else 0


# AV1 decodes in hardware from the M3 onwards. Before that a Mac plays an AV1
# file as sound with no picture — QuickTime reports incompatible parts — which is
# how a finished 52-minute dub arrived unwatchable on an M1.
AV1_FROM_GENERATION = 3


def can_decode_av1() -> bool:
    return mac_generation() >= AV1_FROM_GENERATION


def _ram_gb() -> int:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            return max(1, int(out) // (1024 ** 3))
        return max(1, os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 ** 3))
    except Exception:
        return 8


def _module_available(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _ollama_up() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{ollama_host()}/api/tags", timeout=1.5)
        return True
    except Exception:
        return False


def detect_machine() -> Machine:
    system = platform.system()
    arch = platform.machine()
    apple_silicon = system == "Darwin" and arch == "arm64"
    return Machine(
        system=system,
        arch=arch,
        apple_silicon=apple_silicon,
        ram_gb=_ram_gb(),
        has_mlx=apple_silicon and _module_available("mlx"),
        has_ffmpeg=shutil.which("ffmpeg") is not None,
        # The downloader runs `sys.executable -m yt_dlp`, so the module inside
        # this environment is the thing whose absence would stop it. A yt-dlp on
        # PATH is somebody else's binary and no longer counts as having one.
        has_ytdlp=_module_available("yt_dlp"),
        has_ollama=_ollama_up(),
        av1_ok=can_decode_av1(),
    )


def suggest_ollama_model(ram_gb: int) -> str:
    """Pick a translation model that will actually fit in memory.

    "Fit" means alongside the OS and speech models, not on paper: a 12B model
    (~8.6 GB resident) drives a 16 GB Mac into swap, so 16 GB gets 8B instead.
    Capped at 14B regardless of RAM — the 32B's extra cost isn't worth it for
    line-by-line translation, where a glossary pins terminology anyway. A named
    model under Settings overrides this.
    """
    if ram_gb >= 24:
        return "qwen3:14b"
    if ram_gb >= 16:
        return "qwen3:8b"
    return "qwen3:4b"


# ---------------------------------------------------------------- settings

# Kokoro's own language codes, the letter each voice id starts with, spelled out.
LANGUAGE_NAMES = {"a": "English", "b": "English"}

VOICES = [dict(v, language=LANGUAGE_NAMES[v["lang"]]) for v in (
    {"id": "bf_emma", "label": "Emma — British female", "lang": "b"},
    {"id": "bf_alice", "label": "Alice — British female", "lang": "b"},
    {"id": "bf_isabella", "label": "Isabella — British female", "lang": "b"},
    {"id": "bf_lily", "label": "Lily — British female", "lang": "b"},
    {"id": "bm_george", "label": "George — British male", "lang": "b"},
    {"id": "bm_daniel", "label": "Daniel — British male", "lang": "b"},
    {"id": "af_bella", "label": "Bella — American female", "lang": "a"},
    {"id": "af_nicole", "label": "Nicole — American female", "lang": "a"},
    {"id": "am_michael", "label": "Michael — American male", "lang": "a"},
    {"id": "am_adam", "label": "Adam — American male", "lang": "a"},
)]

# Which languages a dub can be into. Only a voice can speak a translation, so
# the answer is whatever the voices above can say and nothing else: a language
# with no voice behind it would be translated correctly and then read aloud by
# an English one, which sounds fluent and is nonsense. Derived rather than
# listed, so it cannot claim a language the inventory cannot speak.
DUB_LANGUAGES = list(dict.fromkeys(v["language"] for v in VOICES))

# sherpa-onnx addresses Kokoro voices by index; this is the v1.0 alphabetical order.
ONNX_VOICE_IDS = {
    "af_alloy": 0, "af_aoede": 1, "af_bella": 2, "af_heart": 3, "af_jessica": 4,
    "af_kore": 5, "af_nicole": 6, "af_nova": 7, "af_river": 8, "af_sarah": 9,
    "af_sky": 10, "am_adam": 11, "am_echo": 12, "am_eric": 13, "am_fenrir": 14,
    "am_liam": 15, "am_michael": 16, "am_onyx": 17, "am_puck": 18, "am_santa": 19,
    "bf_alice": 20, "bf_emma": 21, "bf_isabella": 22, "bf_lily": 23,
    "bm_daniel": 24, "bm_fable": 25, "bm_george": 26, "bm_lewis": 27,
}

# The source side of every list here is Spanish, and the app is used mostly on
# Portuguese; the transcript vocabulary pass is what pins the terminology of the
# video actually in hand. What is left that no model can work out from a
# transcript is which side of the Atlantic the viewer's own patterns come from:
# the same stitch is a US single crochet and a UK double crochet, so a list
# built for one is wrong in the other and the video says nothing about which.
BUILTIN_GLOSSARIES = {
    "none": {"label": "Not a crochet video", "terms": ""},
    "crochet_uk": {
        "label": "UK terms — double crochet, treble crochet",
        "terms": (
            "cadeneta/cadenetas -> chain / chains\n"
            "punto bajo -> double crochet\n"
            "punto alto, vareta -> treble crochet\n"
            "punto medio alto, media vareta -> half treble crochet\n"
            "punto deslizado, punto raso -> slip stitch\n"
            "vuelta -> round (or row when worked flat)\n"
            "hilera -> row\n"
            "ganchillo, aguja -> hook\n"
            "marcador -> stitch marker\n"
            "arco / arcos -> arch / arches\n"
            "hebra -> strand or yarn; ovillo -> ball\n"
            "muestra -> tension swatch; talle -> bodice; sisa -> armhole\n"
            "manga -> sleeve; escote -> neckline; delantero -> front; espalda -> back\n"
            "saltar -> miss; aumento -> increase; disminucion -> decrease; rematar -> fasten off\n"
            "prenda -> garment; puntada/punto -> stitch"
        ),
    },
    "crochet_us": {
        "label": "US terms — single crochet, double crochet",
        "terms": (
            "cadeneta/cadenetas -> chain / chains\n"
            "punto bajo -> single crochet\n"
            "punto alto, vareta -> double crochet\n"
            "punto medio alto, media vareta -> half double crochet\n"
            "punto deslizado, punto raso -> slip stitch\n"
            "vuelta -> round (or row when worked flat)\n"
            "hilera -> row\n"
            "ganchillo, aguja -> hook\n"
            "marcador -> stitch marker\n"
            "arco / arcos -> arch / arches\n"
            "hebra -> strand or yarn; ovillo -> ball\n"
            "muestra -> gauge swatch; talle -> bodice; sisa -> armhole\n"
            "manga -> sleeve; escote -> neckline; delantero -> front; espalda -> back\n"
            "saltar -> skip; aumento -> increase; disminucion -> decrease; rematar -> fasten off\n"
            "prenda -> garment; puntada/punto -> stitch"
        ),
    },
}


# Voices handed out to additional speakers when a video has more than one person
# and cloning is off. Alternates timbre so two people never sound alike.
SPEAKER_VOICE_POOL = ["bf_emma", "bm_george", "bf_isabella", "am_michael",
                      "bf_alice", "bm_daniel", "af_nicole", "am_adam"]

PRESETS = {
    "fast": {
        "label": "Fast",
        "blurb": "Quickest. Good when there is no music or effects worth keeping.",
        "separate_audio": False,
        "asr_model": "whisper", "voice_mode": "fixed",
    },
    "balanced": {
        "label": "Balanced",
        "blurb": "Keeps the music and effects underneath the new voice.",
        "separate_audio": True,
        "asr_model": "whisper", "voice_mode": "fixed",
    },
    "best": {
        "label": "Best quality",
        "blurb": "Also clones each speaker's own voice. Much slower.",
        "separate_audio": True,
        "asr_model": "whisper", "voice_mode": "clone",
    },
}


@dataclass
class Settings:
    preset: str = "balanced"             # fast | balanced | best | custom
    voice: str = "bf_emma"
    speed: float = 1.0
    audio_mode: str = "replace"          # replace | duck | dual
    duck_db: float = -18.0
    target_language: str = DUB_LANGUAGES[0]      # only English has a voice
    # No source_language: it was declared here and never read by anything, and a
    # setting nothing consumes is worse than no setting at all. The recognisers
    # detect the source themselves and the translation prompt never named it.
    glossary: str = "none"
    custom_glossary: str = ""
    translator: str = "ollama"           # ollama | anthropic | openai
    ollama_model: str = ""               # blank = auto by RAM
    anthropic_key: str = ""
    openai_key: str = ""
    # Sonnet 4.5 is a legacy model; sonnet-5 is the current one and is markedly
    # better at exactly this job — specialist terminology and fitting a
    # translation into a fixed time slot. Same tier, same order of cost.
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o"
    keep_video_quality: str = "best"     # best | 1080 | 720
    # Blank, or a browser name yt-dlp understands. YouTube increasingly refuses
    # signed-out requests for some videos whatever client asks; borrowing the
    # session from a browser the user is already signed into is what fixes those.
    # Off unless named, because reading someone's cookie store is not a thing to
    # do quietly on their behalf.
    youtube_cookies: str = ""            # "" | safari | chrome | firefox | edge | brave
    keep_awake: bool = True              # hold the Mac awake while work is queued
    write_srt: bool = False
    max_stretch: float = 1.55

    # Set from the preset, but overridable individually (which flips preset to
    # "custom" in the interface).
    separate_audio: bool = True          # split speech from music before dubbing
    keep_music: bool = True              # mix the music/effects back underneath
    # Off by default. This app is used mostly on single-presenter instruction,
    # where telling speakers apart can only do harm: it is the least reliable
    # step in the chain, and when it over-segments it dubs one person in five
    # different voices — far worse than the alternative failure, which is two
    # people sharing one. Asked plainly on the front panel instead.
    diarize: bool = False                # detect multiple speakers
    merge_lines: bool = True             # join lines that run straight together
    asr_model: str = "whisper"           # parakeet | whisper
    voice_mode: str = "fixed"            # fixed | clone

    # The switches a preset is a name for. Deliberately not "diarize": how many
    # people are in a video is a fact about the video, like its subject, not a
    # quality-versus-cost trade-off. Leaving it in meant answering "just one
    # person" flipped the preset to custom, and picking a preset silently
    # overwrote the answer.
    PRESET_KEYS = ("separate_audio", "asr_model", "voice_mode")

    # A key is pasted in from somewhere outside this app and cannot be recovered
    # from inside it, which is why a finished job's record never carries one.
    SECRET_KEYS = ("anthropic_key", "openai_key")

    # What a blanket reset leaves alone. Everything else here can be chosen
    # again in seconds; a key cannot be recovered from inside this app at all,
    # and the custom glossary is hand-typed text of no fixed length with no undo
    # anywhere in the app. Naming one of these in the request still clears it,
    # which is how the panel that owns the field resets it deliberately.
    KEEP_ON_RESET = SECRET_KEYS + ("custom_glossary",)

    # Excluded so every other field is recorded by default. Keys are never
    # written down; keep_awake/youtube_cookies are preferences about this Mac,
    # not the video; the model/glossary fields reach the record in another form
    # via run_snapshot() (translator_model, has_custom_glossary).
    UNRECORDED_KEYS = SECRET_KEYS + ("keep_awake", "youtube_cookies",
                                     "ollama_model", "anthropic_model",
                                     "openai_model", "custom_glossary")

    def __post_init__(self) -> None:
        self.normalise()

    def normalise(self) -> None:
        """Drop values this build cannot honour (a settings file outlives its schema).

        Called on construction and again from save(), so a value assigned
        straight onto an existing instance cannot reach the disk unchecked.
        """
        if self.glossary not in BUILTIN_GLOSSARIES:
            self.glossary = "none"
        if self.target_language not in DUB_LANGUAGES:
            self.target_language = DUB_LANGUAGES[0]

    def apply_preset(self, name: str) -> "Settings":
        spec = PRESETS.get(name)
        if not spec:
            return self
        self.preset = name
        for key in self.PRESET_KEYS:
            setattr(self, key, spec[key])
        return self

    def matching_preset(self) -> str:
        """The preset these settings actually are, or "custom".

        Derived rather than stored, so toggling one setting away from a preset
        can't leave `preset` claiming a match the settings no longer have.
        """
        for name, spec in PRESETS.items():
            if all(getattr(self, key) == spec[key] for key in self.PRESET_KEYS):
                return name
        return "custom"

    def voice_for(self, speaker: int, male: bool | None = None) -> str:
        """First speaker gets the chosen voice; others get distinct ones.

        When `male` is known — measured from the speaker's own pitch — the
        alternatives are drawn from voices of that sex. Handing a deep-voiced
        man a bright female voice is the most obvious way a multi-speaker dub
        announces that nobody checked, and the pitch is right there in the audio.
        """
        if speaker <= 0:
            return self.voice
        pool = [v for v in SPEAKER_VOICE_POOL if v != self.voice]
        if male is not None:
            # Kokoro's ids encode it: bf_/af_ are female, bm_/am_ are male.
            wanted = [v for v in pool if v.split("_")[0].endswith("m" if male else "f")]
            pool = wanted or pool
        return pool[(speaker - 1) % len(pool)]

    @classmethod
    def defaults(cls) -> dict:
        """The values the app ships with, read off the dataclass itself.

        Constructed rather than listed, so it cannot drift from the declarations
        above the way a second copy of them would.
        """
        return asdict(cls())

    @classmethod
    def recorded_keys(cls) -> tuple[str, ...]:
        """Which fields a finished job records, read off the dataclass itself.

        The complement of UNRECORDED_KEYS rather than a list of its own, so
        adding a setting that changes the output records it without a second
        declaration to keep in step.
        """
        return tuple(f.name for f in fields(cls) if f.name not in cls.UNRECORDED_KEYS)

    def keep_music_applies(self) -> bool:
        """Whether the keep-the-music choice has any bearing on this run.

        There is only music and effects to put back when speech was separated
        from them, and only room for them underneath when the original audio is
        being replaced — ducking it and keeping it as a second track both leave
        the original soundtrack in the file already.
        """
        return self.separate_audio and self.audio_mode == "replace"

    def run_snapshot(self, local_source: bool = False) -> dict:
        """The settings that shaped a run, kept with its result.

        Settings can change after a job finishes, so the job carries its own
        copy. The custom glossary is recorded as present/absent, not verbatim
        (it's unbounded text). A setting with no bearing on this run is omitted
        rather than recorded as whatever it happened to say.
        """
        snap = {key: getattr(self, key) for key in self.recorded_keys()}
        snap["translator_model"] = {
            "anthropic": self.anthropic_model,
            "openai": self.openai_model,
        }.get(self.translator, self.ollama_model.strip())
        snap["has_custom_glossary"] = bool(self.custom_glossary.strip())
        if not self.keep_music_applies():
            snap.pop("keep_music", None)
        if local_source:
            # Which stream to fetch is a choice only a site offers. A file
            # picked off this Mac is the quality it is, and a report that listed
            # one anyway would be describing a decision nothing made.
            snap.pop("keep_video_quality", None)
        return snap

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text())
                known = {f for f in cls.__dataclass_fields__}
                return cls(**{k: v for k, v in raw.items() if k in known})
            except Exception as exc:                                # noqa: BLE001
                logs.get().warning("settings file unreadable, using defaults",
                                   extra={"error": str(exc)[:200]})
        s = cls()
        s.save()
        return s

    def save(self) -> None:
        """Write the settings, including any API keys, to SETTINGS_FILE.

        Keys are stored in plain text (Keychain would prompt on every job, and
        there's no server to hold a token instead) — a deliberate trade-off,
        disclosed in the settings panel and README. File is chmod'd owner-only.
        """
        self.normalise()
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2))
        try:
            SETTINGS_FILE.chmod(0o600)
        except OSError:
            pass                          # not fatal; the file is still written

    def glossary_text(self) -> str:
        parts = []
        builtin = BUILTIN_GLOSSARIES.get(self.glossary)
        if builtin and builtin["terms"]:
            parts.append(builtin["terms"])
        if self.custom_glossary.strip():
            parts.append(self.custom_glossary.strip())
        return "\n".join(parts)

    def resolved_ollama_model(self, ram_gb: int) -> str:
        return self.ollama_model.strip() or suggest_ollama_model(ram_gb)
