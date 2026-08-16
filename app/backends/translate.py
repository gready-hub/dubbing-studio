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

from ..notes import note

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

def installed_models(host: str = "") -> list[dict]:
    """What Ollama actually has, largest first."""
    from ..config import ollama_host
    host = host or ollama_host()
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as r:
            models = json.loads(r.read()).get("models", [])
    except Exception:                                            # noqa: BLE001
        return []
    return sorted((m for m in models if m.get("name")),
                  key=lambda m: int(m.get("size") or 0), reverse=True)


def usable_model(wanted: str, host: str = "") -> tuple[str, str]:
    """The wanted model if Ollama has it, otherwise the best thing it does have.

    The suggested model is chosen from installed memory, so a machine whose
    installer pulled a different one — the download timed out, Ollama was slow to
    start its first time, the RAM tier moved — asked for a tag that was not there
    and the job died on a 404 part way through translating. The person this app
    is for does not know that model tags exist and should not have to: another
    Qwen of a different size translates instructional speech perfectly well, and
    using it is strictly better than refusing.

    Returns (model, note). The note is empty when nothing was substituted, and
    otherwise says what happened, because a quietly different model is a quietly
    different translation.
    """
    have = installed_models(host)
    if not have:
        return wanted, ""                 # unreachable; the caller reports that
    names = [m["name"] for m in have]
    if wanted in names:
        return wanted, ""

    family = wanted.split(":")[0]
    same = [n for n in names if n.split(":")[0] == family]
    picked = same[0] if same else names[0]
    return picked, (f"{wanted} isn't installed, so {picked} was used to translate "
                    f"instead. To use {wanted}, run:  ollama pull {wanted}")


def _call_ollama(prompt: str, model: str, host: str = "") -> str:
    from ..config import ollama_host
    host = host or ollama_host()

    def ask(think: bool | None) -> str:
        body: dict = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
        }
        if think is not None:
            body["think"] = think
        req = urllib.request.Request(f"{host}/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read()).get("message", {}).get("content", "")

    # Qwen3 — the model every default install ends up with — reasons at length
    # before answering, and for a line-by-line translation that reasoning is
    # pure cost. Measured against this exact endpoint: 92 generated tokens with
    # thinking on, 5 for the identical output with it off. Translation is the
    # slow stage on a local model, and most of it was the model talking to
    # itself.
    for think in (False, None):
        try:
            return ask(think)
        except urllib.error.HTTPError as exc:
            # Ollama older than 0.9, or a model with no thinking mode, rejects
            # the field outright — ask again without it before giving up.
            if think is False and exc.code in (400, 404, 422):
                continue
            raise TranslationError(
                f"Ollama refused the request ({exc.code}) for model {model}."
            ) from exc
        except urllib.error.URLError as exc:
            raise TranslationError(
                f"Could not reach Ollama on {host}. Is it running?"
            ) from exc
    return ""


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
        model, swapped = usable_model(settings.resolved_ollama_model(ram_gb))
        if swapped:
            # Reported the same way any other result-changing fallback is, so it
            # survives into the finished report rather than scrolling past.
            note(progress, swapped)
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
