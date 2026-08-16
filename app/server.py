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
    preview: bool = False


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
    # Room to work in. A job holds the source video, a full-band wav, the
    # separated stems and the assembled track at once before it prunes itself,
    # and a boot disk that fills takes the whole machine down with it — so this
    # belongs beside ffmpeg and yt-dlp as something to know before starting,
    # not something to discover at 80%.
    from . import storage as store
    free = store.free_bytes()
    checks.append({
        "name": f"Disk space — {round(free / 1024 ** 3, 1)} GB free",
        "ok": free >= store.LOW_DISK,
        "hint": "Under 5 GB is not enough to dub a long video, and macOS itself "
                "starts to struggle. Clear some working files below.",
    })

    settings = Settings.load()
    if settings.translator == "ollama":
        checks.append({
            "name": "Ollama", "ok": machine.has_ollama,
            "hint": "Start the Ollama app, or switch to an API key in Settings.",
        })
        if machine.has_ollama:
            # Reports what will actually be used, not what was ideally wanted.
            # The suggested model comes from installed memory, so a machine whose
            # installer pulled a different one used to be told it was missing and
            # handed a terminal command — for a difference nobody outside this
            # code can see or cares about. Any Qwen translates instructional
            # speech; only having none at all is a problem.
            from .backends.translate import installed_models, usable_model
            wanted = settings.resolved_ollama_model(machine.ram_gb)
            picked, swapped = usable_model(wanted)
            have = bool(installed_models())
            checks.append({
                "name": f"Translation model — {picked}" if have else "Translation model",
                "ok": have,
                "hint": (f"None installed. Run:  ollama pull {wanted}" if not have
                         else swapped),
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


@app.post("/api/settings")
def save_settings(patch: SettingsPatch) -> dict:
    settings = Settings.load()
    data = dict(patch.data)

    # A preset change rewrites the stage switches; explicit switches sent in the
    # same request win.
    preset = data.pop("preset", None)
    if preset:
        settings.apply_preset(preset)

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
    # Read back off the switches rather than left as whatever was last named, so
    # that turning one of them off in Advanced shows as "custom" instead of
    # leaving the control claiming a preset the settings no longer are.
    settings.preset = settings.matching_preset()
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
    return runner.submit(url, Settings.load(), preview=req.preview).public()


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


def _folder_title(name: str) -> str:
    """Turn a job folder's hash back into the video it belongs to.

    The folder is named after the link, and a full run carries that same name as
    its id — so a live job matches outright, and a finished one is found in the
    history, whose ids are that name plus a timestamp.
    """
    from .config import JOBS
    from .pipeline import _load_history
    job = runner.jobs.get(name)
    if job and job.title:
        return job.title
    # Written by the job itself. Covers the case the history cannot: a sample is
    # never recorded there, and a sample is the most likely thing to be cleared.
    try:
        return json.loads((JOBS / name / "info.json").read_text())["title"]
    except Exception:                                            # noqa: BLE001
        pass
    for entry in reversed(_load_history()):
        if str(entry.get("id", "")).split("-")[0] == name and entry.get("title"):
            return entry["title"]
    return ""


@app.get("/api/storage")
def storage_summary() -> dict:
    """Everything this app is keeping, and how much room is left for more.

    It used to report one number for one of the four places it writes to, and
    nobody goes looking in Application Support — so between the speech models,
    the model Ollama holds on its behalf, the working files and the finished
    videos, the only visible figure was the smallest one.
    """
    from . import storage as store
    return store.summary(_folder_title)


class ClearRequest(BaseModel):
    what: str = "jobs"


@app.post("/api/storage/clear")
def clear_storage(req: ClearRequest, request: Request) -> dict:
    """Empty one group, or one job's working files. Never the finished videos.

    Refused while something is running, since the job being cleared out from
    under itself would fail in a way that looks like a bug in the pipeline.
    """
    from . import storage as store

    _local_only(request)
    if runner.busy():
        raise HTTPException(409, "Something is still running — wait for it to finish.")
    try:
        # The busy() guard above is check-then-act; naming the live jobs makes
        # the deletion itself safe if one starts in the window between the two.
        live = {j.id for j in runner.jobs.values() if j.status in ("queued", "running")}
        freed = store.clear(req.what, keep=live)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"freed": freed, **store.summary(_folder_title)}


@app.get("/api/job/{job_id}/video")
def job_video(job_id: str):
    job = runner.jobs.get(job_id)
    if not job or not job.output:
        raise HTTPException(404, "Not ready.")
    # A sample lives in the working folder, so clearing that folder — or the
    # full run of the same link tidying up after itself — can take it away while
    # the panel offering it is still on screen. Checked rather than assumed, so
    # the player gets a clean 404 to react to instead of a stalled request.
    if not Path(job.output).is_file():
        raise HTTPException(404, "That file has been cleared away.")
    return FileResponse(job.output, media_type="video/mp4")


REPO = "gready-hub/dubbing-studio"


@app.get("/api/version")
def version() -> dict:
    """Whether a newer version exists. Never blocks anything if it can't tell.

    The installer records the commit it fetched; this asks GitHub what is on the
    branch now. Both sides are best-effort — offline, rate-limited or installed
    some other way all mean the same thing, which is that no update is offered.
    """
    import urllib.request
    from . import storage as store

    stamp = store.APP_DIR / ".version"
    try:
        current = stamp.read_text().strip()
    except OSError:
        return {"known": False}
    if len(current) != 40:
        return {"known": False}
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{REPO}/commits/main",
            headers={"Accept": "application/vnd.github.sha", "User-Agent": "dubbing-studio"})
        with urllib.request.urlopen(req, timeout=4) as r:
            latest = r.read().decode().strip()
    except Exception:                                            # noqa: BLE001
        return {"known": False}
    return {"known": True, "current": current[:7], "latest": latest[:7],
            "update": len(latest) == 40 and latest != current}


@app.post("/api/update")
def run_update(request: Request) -> dict:
    """Hand the installer to Terminal. Same script as a first install."""
    _local_only(request)
    from . import storage as store
    script = store.APP_DIR / "Install.command"
    if not script.is_file():
        raise HTTPException(404, "The installer isn't in the app folder.")
    script.chmod(0o755)
    subprocess.run(["open", "-a", "Terminal", str(script)], check=False)
    return {"ok": True}


@app.post("/api/uninstall")
def open_uninstaller(request: Request) -> dict:
    """Open the uninstaller in Terminal.

    Not performed here. Removing the app means removing the Python environment
    this process is running inside, and the last thing it deletes is the folder
    holding the script doing the deleting — neither is something a web request
    should be halfway through. The script shows what it will remove, separates
    what belongs to this app from what the rest of the Mac shares, and asks.
    """
    _local_only(request)
    from . import storage as store
    script = store.APP_DIR / "Uninstall.command"
    if not script.is_file():
        raise HTTPException(404, "The uninstaller isn't in the app folder.")
    script.chmod(0o755)
    subprocess.run(["open", "-a", "Terminal", str(script)], check=False)
    return {"ok": True}


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
