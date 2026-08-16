"""Web server for Dubbing Studio."""
from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import sys
import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .config import (BUILTIN_GLOSSARIES, OUTPUT_DIR, PRESETS, SETTINGS_FILE,
                     Settings, VOICES, detect_machine, in_container,
                     suggest_ollama_model)
from .pipeline import runner

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Dubbing Studio")


# ------------------------------------------------------------------ models

class JobRequest(BaseModel):
    url: str


class SettingsPatch(BaseModel):
    data: dict


# ------------------------------------------------------------------ routes

def _local_only(request: Request) -> None:
    """Refuse a state-changing request that came from a web page.

    The endpoints that take no body are CORS "simple requests": any site the
    user happens to have open can POST to them on loopback without a preflight,
    and while it cannot read the reply, the side effect still happens — wiping
    every job's working files, or cancelling a job halfway through an hour of
    video. The ones that take a JSON body are already protected, since that
    content type forces a preflight.

    A browser always sends Origin on a cross-origin POST. Our own page sends
    either none or its own origin, so this costs nothing and closes it.
    """
    origin = request.headers.get("origin")
    if not origin:
        return
    host = urlparse(origin).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise HTTPException(403, "That request didn't come from Dubbing Studio.")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/state")
def state() -> dict:
    machine = detect_machine()
    settings = Settings.load()
    return {
        "machine": {
            **asdict(machine),
            "engine": "Apple GPU (MLX)" if machine.fast_path else "Portable (CPU)",
            "suggested_model": suggest_ollama_model(machine.ram_gb),
        },
        "settings": asdict(settings),
        "voices": VOICES,
        "presets": PRESETS,
        "features": _feature_status(),
        "glossaries": {k: v["label"] for k, v in BUILTIN_GLOSSARIES.items()},
        "output_dir": str(OUTPUT_DIR),
        "settings_path": str(SETTINGS_FILE),
        "jobs": runner.public_jobs(),
    }


def _feature_status() -> dict:
    """Which optional capabilities are actually installed on this machine."""
    from .backends import clone, separate

    def has(name: str) -> bool:
        import importlib.util
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:                                        # noqa: BLE001
            return False

    return {
        "separation": separate.available(),
        "cloning": clone.available(),
        "whisper": has("mlx_whisper") or has("faster_whisper"),
        "diarization": has("sherpa_onnx"),
    }


@app.get("/api/doctor")
def doctor() -> dict:
    machine = detect_machine()
    checks = [
        {"name": "ffmpeg", "ok": machine.has_ffmpeg,
         "hint": "Install with: brew install ffmpeg"},
        {"name": "yt-dlp", "ok": machine.has_ytdlp,
         "hint": "Install with: brew install yt-dlp"},
    ]
    settings = Settings.load()
    if settings.translator == "ollama":
        checks.append({
            "name": "Ollama", "ok": machine.has_ollama,
            "hint": "Start the Ollama app, or switch to an API key in Settings.",
        })
        if machine.has_ollama:
            checks.append({
                "name": f"Model {settings.resolved_ollama_model(machine.ram_gb)}",
                "ok": _ollama_has(settings.resolved_ollama_model(machine.ram_gb)),
                "hint": f"Run: ollama pull {settings.resolved_ollama_model(machine.ram_gb)}",
            })
    elif settings.translator == "anthropic":
        checks.append({"name": "Anthropic API key", "ok": bool(settings.anthropic_key),
                       "hint": "Paste a key in Settings."})
    elif settings.translator == "openai":
        checks.append({"name": "OpenAI API key", "ok": bool(settings.openai_key),
                       "hint": "Paste a key in Settings."})
    if machine.apple_silicon:
        checks.append({"name": "MLX (Apple GPU)", "ok": machine.has_mlx,
                       "hint": "Optional. Without it everything still works, just slower."})

    features = _feature_status()
    optional = [
        ("Music separation (Demucs)", features["separation"],
         "Needed for the Balanced preset. Re-run the installer to add it."),
        ("Voice cloning (Chatterbox)", features["cloning"],
         "Needed for the Best preset. Re-run the installer to add it."),
        ("Whisper transcription", features["whisper"],
         "Optional accuracy upgrade. Re-run the installer to add it."),
    ]
    for name, ok_, hint in optional:
        checks.append({"name": name, "ok": ok_, "hint": hint, "optional": True})

    return {"checks": checks,
            "ready": all(c["ok"] for c in checks
                         if not c.get("optional") and not c["name"].startswith("MLX"))}


def _ollama_has(model: str) -> bool:
    import urllib.request
    from .config import ollama_host
    try:
        with urllib.request.urlopen(f"{ollama_host()}/api/tags", timeout=2) as r:
            tags = json.loads(r.read()).get("models", [])
        base = model.split(":")[0]
        return any(m.get("name", "").split(":")[0] == base for m in tags)
    except Exception:
        return False


@app.post("/api/settings")
def save_settings(patch: SettingsPatch) -> dict:
    settings = Settings.load()
    data = dict(patch.data)

    # A preset change rewrites the stage switches; explicit switches sent in the
    # same request win, which is how the interface expresses "custom".
    preset = data.pop("preset", None)
    if preset:
        settings.apply_preset(preset)
        if preset in PRESETS:
            settings.preset = preset

    for key, value in data.items():
        if hasattr(settings, key):
            current = getattr(settings, key)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, int):
                    value = int(value)
            except (TypeError, ValueError):
                continue
            setattr(settings, key, value)
    settings.save()
    return asdict(settings)


@app.post("/api/job")
def create_job(req: JobRequest) -> dict:
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "Paste a link first.")
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "That doesn't look like a web link.")
    machine = detect_machine()
    if not machine.has_ffmpeg:
        raise HTTPException(400, "ffmpeg isn't installed — re-run the installer.")
    if not machine.has_ytdlp:
        raise HTTPException(400, "yt-dlp isn't installed — re-run the installer.")
    return runner.submit(url, Settings.load()).public()


@app.post("/api/job/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    _local_only(request)
    if job_id not in runner.jobs:
        raise HTTPException(404, "No such job.")
    runner.cancel(job_id)
    return {"ok": True}


# A line long enough to judge a voice by, and in the register the app is
# actually used for.
PREVIEW_TEXT = "Right, let's carry on with the next round of the pattern."
_preview_lock = threading.Lock()


@app.get("/api/voice-preview")
def voice_preview(voice: str, speed: float = 1.0):
    """Speak one line in a voice, so choosing one isn't a guess that costs a run.

    Rendered once per voice and speed and then kept, so the second listen is
    instant and flicking between voices is quick enough to actually compare
    them.
    """
    from .config import BASE, VOICES
    if not any(v["id"] == voice for v in VOICES):
        raise HTTPException(404, "No such voice.")

    speed = max(0.5, min(2.0, float(speed)))
    out = BASE / "previews" / f"{voice}-{speed:.2f}.wav"
    if not out.exists():
        # Rendering a new one loads a second copy of the speech model. Doing
        # that while a job is running competes with it for the GPU and pushed
        # this machine deep into swap — the interface stopped answering for
        # minutes. An already-rendered preview costs nothing and is still served.
        if runner.busy():
            raise HTTPException(409, "Try that once the current video has finished — "
                                     "playing a new voice now would slow it down.")
        out.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        from .backends import tts as tts_backend
        # Serialised: two clicks in quick succession would otherwise load the
        # model twice at once. Released again afterwards — keeping it resident
        # would leave a second Kokoro in memory for the life of the process,
        # competing with every later job for exactly the resource the guard
        # above exists to protect. Previews are cached on disk, so the only
        # thing being given up is a faster first render of each voice.
        with _preview_lock:
            engine = tts_backend.load_tts(detect_machine().fast_path)
            audio, rate = engine.say(PREVIEW_TEXT, voice, speed)
            del engine
        if not getattr(audio, "size", 0):
            raise HTTPException(500, "The voice produced no sound.")
        sf.write(out, audio, rate)
    return FileResponse(out, media_type="audio/wav")


@app.get("/api/job/{job_id}/reference/{index}")
def job_reference(job_id: str, index: int):
    """The clip captured from the original speaker and used as the clone prompt.

    Hearing it is the quickest way to catch a bad reference — one with music
    under it, or the wrong person — before it colours every line of the dub.
    """
    from .config import JOBS
    job = runner.jobs.get(job_id)
    if not job or index < 0 or index >= len(job.references):
        raise HTTPException(404, "No such reference.")
    path = Path(job.references[index]).resolve()
    # Paths come from the job's own folder, but this is a filesystem read served
    # over HTTP, so confirm rather than assume.
    if not path.is_file() or JOBS.resolve() not in path.parents:
        raise HTTPException(404, "No such reference.")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/storage")
def storage() -> dict:
    """How much disk the job working files are holding.

    Nobody goes looking in Application Support, so without this the folder just
    grows until something else runs out of room.
    """
    from .config import JOBS
    from .pipeline import dir_size
    if not JOBS.is_dir():
        return {"bytes": 0, "jobs": 0, "path": str(JOBS)}
    return {"bytes": dir_size(JOBS),
            "jobs": sum(1 for p in JOBS.iterdir() if p.is_dir()),
            "path": str(JOBS)}


@app.post("/api/storage/clear")
def clear_storage(request: Request) -> dict:
    """Delete every job's working files. Finished videos are not touched.

    Refused while something is running, since the job being cleared out from
    under itself would fail in a way that looks like a bug in the pipeline.
    """
    import shutil
    from .config import JOBS
    from .pipeline import dir_size

    _local_only(request)
    if runner.busy():
        raise HTTPException(409, "Something is still running — wait for it to finish.")

    freed = dir_size(JOBS) if JOBS.is_dir() else 0
    live = {j.id for j in runner.jobs.values() if j.status in ("queued", "running")}
    for folder in (p for p in JOBS.iterdir() if p.is_dir()):
        # The busy() guard above is check-then-act; this makes the actual
        # deletion safe if a job starts in the window between the two.
        if folder.name in live:
            continue
        shutil.rmtree(folder, ignore_errors=True)
    return {"freed": freed}


@app.get("/api/job/{job_id}/video")
def job_video(job_id: str):
    job = runner.jobs.get(job_id)
    if not job or not job.output:
        raise HTTPException(404, "Not ready.")
    return FileResponse(job.output, media_type="video/mp4")


@app.post("/api/reveal")
def reveal(body: dict) -> dict:
    path = Path(body.get("path", str(OUTPUT_DIR)))
    try:
        if platform.system() == "Darwin":
            # An empty argument is not the same as no argument: `open "" <path>`
            # makes open try to launch a file called "", and it fails.
            cmd = ["open"] + (["-R"] if path.is_file() else []) + [str(path)]
            subprocess.run(cmd, check=False)
        elif platform.system() == "Windows":
            subprocess.run(["explorer", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path.parent if path.is_file() else path)],
                           check=False)
    except Exception:                                            # noqa: BLE001
        return {"ok": False}
    return {"ok": True}


@app.get("/api/events")
async def events():
    loop = asyncio.get_running_loop()
    # An asyncio queue, deliberately, not a thread-blocking one. Waiting on a
    # queue.Queue via run_in_executor leaked a worker thread on every keepalive:
    # wait_for cancels the await, but the thread stays parked in get() for ever.
    # FastAPI runs ordinary `def` endpoints on that same pool, so once it was
    # exhausted every other request hung too — leaving the window open for a few
    # minutes was enough to freeze the whole interface.
    q: asyncio.Queue = asyncio.Queue()

    def listener(payload: dict) -> None:
        loop.call_soon_threadsafe(q.put_nowait, payload)

    runner.subscribe(listener)

    async def stream():
        try:
            for job in sorted(runner.jobs.values(), key=lambda j: j.started):
                yield f"data: {json.dumps(job.public())}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=20)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            runner.unsubscribe(listener)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def main() -> None:
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

    # Inside a container, 127.0.0.1 is the container's own loopback. Docker
    # publishes a port by forwarding to the container's external interface, so
    # binding to loopback left the containerised app listening where nothing
    # could reach it — the README's localhost:8765 answered nothing at all.
    # Everywhere else, stay on loopback: this is a personal app with no
    # authentication, and it has no business being reachable from the network.
    # Deliberately not in_docker(): that is satisfied by DUBBING_STUDIO_DOCKER=1
    # alone, which is a reasonable thing to set on a workstation while testing
    # the portable path — and it would have published an unauthenticated API,
    # settings included, on every interface. Only an actual container qualifies.
    docker = in_container()
    host, shown = ("0.0.0.0", "localhost") if docker else ("127.0.0.1", "127.0.0.1")  # noqa: S104
    url = f"http://{shown}:{port}"

    print(f"\n  Dubbing Studio is running.\n  Open this in your browser:  {url}\n")
    if "--no-browser" not in sys.argv and not docker:
        try:
            webbrowser.open(url)
        except Exception:                                        # noqa: BLE001
            pass
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
