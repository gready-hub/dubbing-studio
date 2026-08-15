"""Web server for Dubbing Studio."""
from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import sys
import webbrowser
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .config import (BUILTIN_GLOSSARIES, OUTPUT_DIR, PRESETS, Settings, VOICES,
                     detect_machine, suggest_ollama_model)
from .pipeline import runner

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Dubbing Studio")


# ------------------------------------------------------------------ models

class JobRequest(BaseModel):
    url: str


class SettingsPatch(BaseModel):
    data: dict


# ------------------------------------------------------------------ routes

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
        "jobs": [j.public() for j in sorted(runner.jobs.values(),
                                            key=lambda j: j.started, reverse=True)],
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
def cancel_job(job_id: str) -> dict:
    if job_id not in runner.jobs:
        raise HTTPException(404, "No such job.")
    runner.cancel(job_id)
    return {"ok": True}


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
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Dubbing Studio is running.\n  Open this in your browser:  {url}\n")
    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:                                        # noqa: BLE001
            pass
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
