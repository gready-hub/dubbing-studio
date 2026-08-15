"""Translation.

Three backends, one interface. Segments are translated in batches with the
neighbouring lines supplied as context, because a tutorial sentence often only
makes sense given the one before it.

The wire format is deliberately NOT JSON. Local models emit malformed JSON far
too often on long batches; a numbered-line format degrades gracefully and is
easy to repair.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Callable, Optional

Progress = Optional[Callable[[float, str], None]]

BATCH = 25
CONTEXT = 3

SYSTEM = """You translate speech for an AI-dubbed video soundtrack.

Rules you must follow every time:

1. TIMING. Each line is spoken into a fixed time slot given in seconds. Keep the
   translation short enough to fit: at most 2.6 words per second of the slot. Trim
   filler and hesitation rather than padding. Being slightly short is good; running
   long is a defect.
2. SPOKEN REGISTER. This is talking, not writing. Use contractions and natural
   phrasing. Keep the speaker's warmth, but do not invent content.
3. READ ALOUD BY A MACHINE. Write numbers, units and symbols as words: "2,5 mm"
   becomes "two point five millimetres", "5%" becomes "five percent". No digits, no
   symbols, no parentheses, no bullets.
4. IMPERFECT INPUT. The source came from speech recognition and contains
   mis-hearings and run-on sentences. Infer the intent from context and translate
   that. Never return an empty line.
5. FORMAT. Reply with one line per input, exactly as:
   <id>|<translation>
   No preamble, no commentary, no code fences, no blank lines."""


class TranslationError(RuntimeError):
    pass


# ------------------------------------------------------------- prompt build

def _build_prompt(batch: list[dict], context: list[str], target: str, glossary: str) -> str:
    parts = [f"Translate into {target}."]
    if glossary:
        parts.append("\nUse these terms exactly:\n" + glossary)
    if context:
        parts.append("\nPreceding lines, for continuity only — do NOT translate them:\n"
                     + "\n".join(f"  {c}" for c in context))
    parts.append("\nTranslate these lines. Reply with one `id|translation` line each:\n")
    for seg in batch:
        slot = seg["end"] - seg["start"]
        parts.append(f'{seg["i"]}|[{slot:.1f}s] {seg["text"]}')
    return "\n".join(parts)


def _parse(reply: str, batch: list[dict]) -> dict[int, str]:
    wanted = {s["i"] for s in batch}
    out: dict[int, str] = {}
    for line in reply.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        head, _, tail = line.partition("|")
        head = head.strip().lstrip("#").strip()
        if not head.isdigit():
            continue
        idx = int(head)
        if idx in wanted:
            text = tail.strip()
            text = re.sub(r"^\[[\d.]+s\]\s*", "", text)      # strip an echoed slot marker
            text = text.strip().strip('"')
            if text:
                out[idx] = text
    return out


# ----------------------------------------------------------------- backends

def _call_ollama(prompt: str, model: str, host: str = "") -> str:
    from ..config import ollama_host
    host = host or ollama_host()
    body = json.dumps({
        "model": model,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise TranslationError(
            f"Could not reach Ollama on {host}. Is it running?"
        ) from exc
    return data.get("message", {}).get("content", "")


def _call_anthropic(prompt: str, model: str, key: str) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": 4096,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"Content-Type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return "".join(b.get("text", "") for b in data.get("content", []))


def _call_openai(prompt: str, model: str, key: str) -> str:
    body = json.dumps({
        "model": model,
        "temperature": 0.2,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ------------------------------------------------------------------- public

def translate(segments: list[dict], settings, ram_gb: int, progress: Progress = None) -> list[dict]:
    """Returns segments with a "translation" key added to each."""
    backend = settings.translator
    glossary = settings.glossary_text()
    target = settings.target_language

    if backend == "ollama":
        model = settings.resolved_ollama_model(ram_gb)
        call = lambda p: _call_ollama(p, model)                      # noqa: E731
        label = f"local model {model}"
    elif backend == "anthropic":
        if not settings.anthropic_key:
            raise TranslationError("No Anthropic API key set in Settings.")
        call = lambda p: _call_anthropic(p, settings.anthropic_model, settings.anthropic_key)  # noqa: E731
        label = settings.anthropic_model
    elif backend == "openai":
        if not settings.openai_key:
            raise TranslationError("No OpenAI API key set in Settings.")
        call = lambda p: _call_openai(p, settings.openai_model, settings.openai_key)  # noqa: E731
        label = settings.openai_model
    else:
        raise TranslationError(f"Unknown translation backend: {backend}")

    for n, seg in enumerate(segments):
        seg["i"] = n

    done: dict[int, str] = {}
    batches = [segments[i:i + BATCH] for i in range(0, len(segments), BATCH)]

    for bn, batch in enumerate(batches):
        start = batch[0]["i"]
        context = [s["text"] for s in segments[max(0, start - CONTEXT):start]]
        prompt = _build_prompt(batch, context, target, glossary)

        got: dict[int, str] = {}
        for attempt in range(3):
            try:
                got = _parse(call(prompt), batch)
            except TranslationError:
                raise
            except Exception:                                        # noqa: BLE001
                got = {}
            if len(got) >= len(batch):
                break
            # Retry only the stragglers, one at a time — small prompts almost always land.
            missing = [s for s in batch if s["i"] not in got]
            if attempt == 2 or not missing:
                break
            if len(missing) <= 4:
                for s in missing:
                    single = _build_prompt([s], context, target, glossary)
                    try:
                        got.update(_parse(call(single), [s]))
                    except Exception:                                # noqa: BLE001
                        pass
                break
            prompt = _build_prompt(missing, context, target, glossary)
            batch = missing

        done.update(got)
        if progress:
            progress((bn + 1) / len(batches),
                     f"Translating with {label} — {min(len(done), len(segments))} of {len(segments)} lines")

    missing = [s for s in segments if s["i"] not in done]
    if len(missing) > len(segments) * 0.05:
        raise TranslationError(
            f"Translation only returned {len(done)} of {len(segments)} lines. "
            "If you are using a local model, try a larger one in Settings."
        )
    for seg in segments:
        seg["translation"] = done.get(seg["i"], "").strip()
    return segments
