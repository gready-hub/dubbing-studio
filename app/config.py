"""Settings, paths and machine detection for Dubbing Studio."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path

APP_NAME = "Dubbing Studio"

# ---------------------------------------------------------------- paths

def _base_dir() -> Path:
    env = os.environ.get("DUBBING_STUDIO_HOME")
    if env:
        return Path(env)
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "DubbingStudio"
    return Path.home() / ".dubbing-studio"


BASE = _base_dir()
JOBS = BASE / "jobs"
MODELS = BASE / "models"
SETTINGS_FILE = BASE / "settings.json"
HISTORY_FILE = BASE / "history.json"
OUTPUT_DIR = Path(os.environ.get("DUBBING_STUDIO_OUTPUT", str(Path.home() / "Movies" / "Dubbed")))


def ollama_host() -> str:
    """Where to reach Ollama.

    The Docker build points this at the host machine, since the container has no
    Ollama of its own. Accepts a bare host:port as well as a full URL, because
    that is the form the OLLAMA_HOST variable normally takes.
    """
    host = os.environ.get("OLLAMA_HOST", "").strip() or "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")

for _p in (BASE, JOBS, MODELS, OUTPUT_DIR):
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
    in_docker: bool

    @property
    def fast_path(self) -> bool:
        """True when we can use Apple's GPU via MLX."""
        return self.apple_silicon and self.has_mlx and not self.in_docker


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
    in_docker = Path("/.dockerenv").exists() or os.environ.get("DUBBING_STUDIO_DOCKER") == "1"
    apple_silicon = system == "Darwin" and arch == "arm64" and not in_docker
    return Machine(
        system=system,
        arch=arch,
        apple_silicon=apple_silicon,
        ram_gb=_ram_gb(),
        has_mlx=apple_silicon and _module_available("mlx"),
        has_ffmpeg=shutil.which("ffmpeg") is not None,
        has_ytdlp=shutil.which("yt-dlp") is not None or _module_available("yt_dlp"),
        has_ollama=_ollama_up(),
        in_docker=in_docker,
    )


def suggest_ollama_model(ram_gb: int) -> str:
    """Pick a translation model that will actually fit in memory.

    "Fit" means alongside the operating system and the speech models, not just on
    paper. A 12B model needs about 8.6 GB resident, which on a 16 GB Mac drives
    the machine deep into swap and makes translation the slowest stage of the job
    by a wide margin — so 16 GB gets an 8B model instead.
    """
    if ram_gb >= 48:
        return "qwen3:32b"
    if ram_gb >= 24:
        return "qwen3:14b"
    if ram_gb >= 16:
        return "qwen3:8b"
    return "qwen3:4b"


# ---------------------------------------------------------------- settings

VOICES = [
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
]

# sherpa-onnx addresses Kokoro voices by index; this is the v1.0 alphabetical order.
ONNX_VOICE_IDS = {
    "af_alloy": 0, "af_aoede": 1, "af_bella": 2, "af_heart": 3, "af_jessica": 4,
    "af_kore": 5, "af_nicole": 6, "af_nova": 7, "af_river": 8, "af_sarah": 9,
    "af_sky": 10, "am_adam": 11, "am_echo": 12, "am_eric": 13, "am_fenrir": 14,
    "am_liam": 15, "am_michael": 16, "am_onyx": 17, "am_puck": 18, "am_santa": 19,
    "bf_alice": 20, "bf_emma": 21, "bf_isabella": 22, "bf_lily": 23,
    "bm_daniel": 24, "bm_fable": 25, "bm_george": 26, "bm_lewis": 27,
}

BUILTIN_GLOSSARIES = {
    "none": {"label": "None", "terms": ""},
    "crochet_us": {
        "label": "Crochet — US terms",
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
    "crochet_uk": {
        "label": "Crochet — UK terms",
        "terms": (
            "cadeneta/cadenetas -> chain / chains\n"
            "punto bajo -> double crochet\n"
            "punto alto, vareta -> treble crochet\n"
            "punto medio alto, media vareta -> half treble crochet\n"
            "punto deslizado, punto raso -> slip stitch\n"
            "vuelta -> round (or row when worked flat)\n"
            "ganchillo, aguja -> hook\n"
            "marcador -> stitch marker\n"
            "saltar -> miss; aumento -> increase; disminucion -> decrease; rematar -> fasten off"
        ),
    },
    "cooking": {
        "label": "Cooking & baking",
        "terms": (
            "sarten -> frying pan; olla -> pot; cazuela -> casserole dish\n"
            "batir -> whisk; amasar -> knead; sofreir -> saute; rehogar -> sweat\n"
            "harina de fuerza -> bread flour; levadura -> yeast; polvo de hornear -> baking powder\n"
            "a fuego lento -> on a low heat; punto de nieve -> stiff peaks\n"
            "Convert metric measures to words but keep the metric units."
        ),
    },
    "woodworking": {
        "label": "Woodworking & DIY",
        "terms": (
            "sierra -> saw; cepillo -> plane; formon -> chisel; lija -> sandpaper\n"
            "tornillo -> screw; clavo -> nail; taladro -> drill; broca -> drill bit\n"
            "ensamble -> joint; espiga -> tenon; mortaja -> mortise; ingletes -> mitres\n"
            "contrachapado -> plywood; tablero -> board; veta -> grain"
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
        "blurb": "Quickest. Best for a single person talking with no music.",
        "separate_audio": False, "diarize": False,
        "asr_model": "parakeet", "voice_mode": "fixed",
    },
    "balanced": {
        "label": "Balanced",
        "blurb": "Keeps music and effects, and gives each speaker their own voice.",
        "separate_audio": True, "diarize": True,
        "asr_model": "parakeet", "voice_mode": "fixed",
    },
    "best": {
        "label": "Best quality",
        "blurb": "Everything on, plus each speaker's own cloned voice. Around 3x slower.",
        "separate_audio": True, "diarize": True,
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
    target_language: str = "English"
    # No source_language: it was declared here and never read by anything, and a
    # setting nothing consumes is worse than no setting at all. The recognisers
    # detect the source themselves and the translation prompt never named it.
    glossary: str = "none"
    custom_glossary: str = ""
    translator: str = "ollama"           # ollama | anthropic | openai
    ollama_model: str = ""               # blank = auto by RAM
    anthropic_key: str = ""
    openai_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    openai_model: str = "gpt-4o"
    keep_video_quality: str = "best"     # best | 1080 | 720
    write_srt: bool = False
    max_stretch: float = 1.55

    # Set from the preset, but overridable individually (which flips preset to
    # "custom" in the interface).
    separate_audio: bool = True          # split speech from music before dubbing
    keep_music: bool = True              # mix the music/effects back underneath
    diarize: bool = True                 # detect multiple speakers
    expected_speakers: int = -1          # -1 = work it out automatically
    asr_model: str = "parakeet"          # parakeet | whisper
    voice_mode: str = "fixed"            # fixed | clone

    def apply_preset(self, name: str) -> "Settings":
        spec = PRESETS.get(name)
        if not spec:
            return self
        self.preset = name
        for key in ("separate_audio", "diarize", "asr_model", "voice_mode"):
            setattr(self, key, spec[key])
        return self

    def voice_for(self, speaker: int) -> str:
        """First speaker gets the chosen voice; others get distinct ones."""
        if speaker <= 0:
            return self.voice
        pool = [v for v in SPEAKER_VOICE_POOL if v != self.voice]
        return pool[(speaker - 1) % len(pool)]

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text())
                known = {f for f in cls.__dataclass_fields__}
                return cls(**{k: v for k, v in raw.items() if k in known})
            except Exception:
                pass
        s = cls()
        s.save()
        return s

    def save(self) -> None:
        """Write the settings, including any API keys, to SETTINGS_FILE.

        The Anthropic and OpenAI keys are stored here in plain text. That is a
        deliberate choice for a local single-user app — the Keychain would put a
        system prompt in front of every job, and the app has no server to hold a
        token for it — but it is only defensible if it is stated rather than
        assumed, so the settings panel and the README both say where the key
        goes. What can be done cheaply is narrowing who can read it: the file is
        owner-only, so another account on the same Mac cannot.
        """
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
