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

from .. import logs
from ..notes import note

Progress = Optional[Callable[[float, str], None]]

BATCH = 25            # lines per request...
BATCH_CHARS = 2800    # ...unless they are long ones, which is when models buckle
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
4. NEVER COPY. Returning a line unchanged, or repeating its id or its slot
   marker back, is a defect — every line must come back in the target language
   and nothing else.
5. IMPERFECT INPUT. The source came from speech recognition and contains
   mis-hearings and run-on sentences. Infer the intent from context and translate
   that. Never return an empty line.
6. FORMAT. Reply with one line per input, exactly as:
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


# What the model sometimes puts back at the front of its own answer. The slot
# marker was already stripped; the id was not, so a reply of "63|id: 63 Blusa de
# verano..." was spoken aloud with the number in it.
# Deliberately narrow. An earlier, looser version also matched a bare number and
# a punctuation-free "line 5", which would have silently eaten the front of "3.
# Chain three stitches" and "Line 5 of the pattern" — real sentences in the
# material this app is pointed at. Corrupting a good translation is worse than
# leaving a rare piece of scaffolding, which the check downstream counts anyway.
_ECHOED = re.compile(
    r"^\s*(?:\[[\d.]+s\]"                     # an echoed slot marker
    r"|(?:id|line)\s*[:.]\s*\d+\s*[-:.|)]*"   # "id: 63"
    r"|(?:id|line)\s+\d+\s*[-:.|)]+"           # "id 63." — punctuation required
    r"|#\s*\d+\s*[-:.|)]*"                     # "#63"
    r")\s*", re.IGNORECASE)


def _strip_echo(text: str) -> str:
    """Remove any scaffolding the model repeated back into its translation."""
    for _ in range(3):                     # "63| id: 63 ..." has two layers
        stripped = _ECHOED.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return text.strip().strip('"').strip()


def _looks_untranslated(text: str, source: str) -> bool:
    """The model handed the source back instead of translating it.

    A local model that has lost the thread does this for a run of lines, and
    nothing downstream can tell — the dub simply speaks the original language in
    an English voice for several minutes, which is what happened on a real
    hour-long video.

    Short lines are exempt: "OK", a number, or a name legitimately survives
    translation unchanged, and rejecting those would throw away good work.
    """
    def bare(s: str) -> str:
        return re.sub(r"[^\w]+", "", s).casefold()

    a, b = bare(text), bare(source)
    return len(b) > 15 and a == b


def _parse(reply: str, batch: list[dict]) -> dict[int, str]:
    wanted = {s["i"]: s.get("text", "") for s in batch}
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
            text = _strip_echo(tail.strip())
            # Treated as a miss rather than a result, so the retry above gets a
            # go at it and it is counted if it never lands.
            if text and not _looks_untranslated(text, wanted[idx]):
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
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,
                # A small model under token pressure falls into repeating itself
                # — echoing the input back, or filling the budget with the same
                # fragment. It is a well-known failure of quantised local models
                # on long structured tasks, and it is what produced a dub that
                # read out "id: 63" and then several minutes of the original
                # language. A mild repetition penalty discourages the loop
                # without flattening legitimately repeated words, which matter
                # in a tutorial that says "chain three" forty times.
                "repeat_penalty": 1.1,
                "min_p": 0.05,
            },
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


# Small enough that its output on specialist material is a real risk to the
# result, rather than merely a bit worse.
WEAK_LOCAL_MODELS = ("qwen3:4b", "qwen3:1.7b", "qwen3:0.6b")


def describe_translator(settings, ram_gb: int) -> tuple[str, str]:
    """(what will translate, a warning if that is the weak link).

    Translation is the one stage whose failure cannot be heard as a failure: a
    bad dub with clean audio and perfect timing sounds finished and is useless.
    So what did it goes in the report next to everything else, and when it is a
    model small enough to be the risk, the report says so.
    """
    if settings.translator == "anthropic":
        return settings.anthropic_model, ""
    if settings.translator == "openai":
        return settings.openai_model, ""
    model, _ = usable_model(settings.resolved_ollama_model(ram_gb))
    if model in WEAK_LOCAL_MODELS:
        return model, (f"Translated by {model}, the smallest local model, because "
                       "that is what fits this Mac's memory. On specialist material "
                       "it is the weakest part of the chain — if the wording reads "
                       "badly, an API key under Settings costs a few pence a video "
                       "and is markedly better.")
    return model, ""


def _batches(segments: list[dict]) -> list[list[dict]]:
    """Group lines into requests, capped by count and by size.

    A fixed twenty-five was fine until the lines were long: joined run-on speech
    can be several hundred characters each, and a batch of those crowds the
    context window, which is precisely when a small model stops following the
    format and starts repeating itself.
    """
    out: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for seg in segments:
        cost = len(seg.get("text", "")) + 16          # the id and slot marker
        if current and (len(current) >= BATCH or size + cost > BATCH_CHARS):
            out.append(current)
            current, size = [], 0
        current.append(seg)
        size += cost
    if current:
        out.append(current)
    return out


def _ask(batch: list[dict], context: list[str], target: str, glossary: str,
         call) -> dict[int, str]:
    try:
        return _parse(call(_build_prompt(batch, context, target, glossary)), batch)
    except TranslationError:
        raise
    except Exception:                                            # noqa: BLE001
        return {}


def _translate_chunk(batch: list[dict], context: list[str], target: str,
                     glossary: str, call) -> dict[int, str]:
    """One batch, halved whenever the model cannot manage it whole.

    The retry used to break a batch down only when four or fewer lines were
    missing, on the reasoning that small prompts almost always land. That is
    true, and it was applied to the wrong case: a model that starts echoing its
    input back fails a whole batch at once, and twenty-five missing lines fell
    outside the rule, so it was asked the identical question twice and then
    given up on.

    Halving asks a different question each time, and by the time a piece is one
    line the prompt is small enough that almost anything lands. Bounded by the
    halving itself — a batch of twenty-five bottoms out in five levels.
    """
    got = _ask(batch, context, target, glossary, call)
    missing = [s for s in batch if s["i"] not in got]
    if not missing:
        return got

    # Logged because this is the machinery that stops a local model's bad patch
    # reaching the finished video, and until now it left no trace either way —
    # so after a run nobody could say whether it had saved the job or never
    # fired. On a long video these lines are the evidence.
    log = logs.get()
    log.warning("translation batch incomplete, halving", extra={
        "asked": len(batch), "missing": len(missing),
        "first_missing": missing[0].get("text", "")[:80]})

    if len(batch) == 1:
        got.update(_ask(batch, context, target, glossary, call))   # one more go
        if batch[0]["i"] not in got:
            log.error("line could not be translated even alone",
                      extra={"text": batch[0].get("text", "")[:120]})
        return got

    mid = max(1, len(missing) // 2)
    for half in (missing[:mid], missing[mid:]):
        if half:
            got.update(_translate_chunk(half, context, target, glossary, call))
    return got


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
    batches = _batches(segments)

    for bn, batch in enumerate(batches):
        start = batch[0]["i"]
        context = [s["text"] for s in segments[max(0, start - CONTEXT):start]]
        done.update(_translate_chunk(batch, context, target, glossary, call))
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
