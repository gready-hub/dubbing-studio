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

Progress = Optional[Callable[[float, str], None]]

BATCH = 25            # lines per request...
BATCH_CHARS = 2800    # ...unless they are long ones, which is when models buckle
CONTEXT = 3

SYSTEM = """You translate speech for an AI-dubbed video soundtrack.

Rules you must follow every time:

1. TIMING. Each line is spoken over the moment the line it translates was spoken
   in, so it has to take about as long to say. Keep it no longer than its own
   source line: trim filler and hesitation rather than padding. Being slightly
   short is good; running long is a defect.
2. SPOKEN REGISTER. This is talking, not writing. Use contractions and natural
   phrasing. Keep the speaker's warmth, but do not invent content.
3. READ ALOUD BY A MACHINE. Write numbers, units and symbols as words: "2,5 mm"
   becomes "two point five millimetres", "5%" becomes "five percent". No digits, no
   symbols, no parentheses, no bullets.
4. NEVER COPY. Returning a line unchanged, or repeating its id back, is a defect
   — every line must come back in the target language and nothing else.
5. IMPERFECT INPUT. The source came from speech recognition and contains
   mis-hearings and run-on sentences. Infer the intent from context and translate
   that. Never return an empty line.
6. FORMAT. Reply with exactly one line per input line, in the same order, each
   beginning with that input's own number and a vertical bar. An input numbered
   7 comes back looking like this and only like this:
   7|Nobody remembered to close the gate behind them.
   No preamble, no commentary, no code fences, no blank lines. Never put a line
   break inside a translation — anything after it is thrown away."""


# A concrete filled-in example is imitated more reliably than "<id>|<translation>"
# is filled in — but a copied example reads as a plausible sentence that neither
# _is_scaffolding nor _looks_untranslated can catch, so it must be deliberately
# alien to any real tutorial line rather than merely checked for.
EXAMPLE_LINE = "Nobody remembered to close the gate behind them."


def _is_example(text: str) -> bool:
    bare = lambda s: re.sub(r"[^\w]+", "", s).casefold()   # noqa: E731
    return bare(text) == bare(EXAMPLE_LINE)


class TranslationError(RuntimeError):
    """A translation failure, with the provider's own words kept in `.detail`.

    Shaped like download.DownloadError on purpose: pipeline.py already lifts
    `.detail` into job.error_detail and the UI already renders it, so no
    special-casing is needed here. Before this, a rejected key surfaced only
    as "translation batch incomplete, halving" repeated thirty times.
    """

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail.strip()


def _build_prompt(batch: list[dict], context: list[str], target: str, glossary: str) -> str:
    parts = [f"Translate into {target}."]
    if glossary:
        parts.append("\nUse these terms exactly:\n" + glossary)
    if context:
        parts.append("\nPreceding lines, for continuity only — do NOT translate them:\n"
                     + "\n".join(f"  {c}" for c in context))
    # Deliberately no placeholder here — this is the last instruction the model
    # reads before answering, so it's the nearest thing to copy. A placeholder
    # here once got copied verbatim into 16% of a video.
    parts.append("\nTranslate these lines. Reply with one line for each: its "
                 "number, a vertical bar, then the translation.\n")
    # No timing slot appended per line: "[5.1s] " used to ride along here and
    # came back spoken aloud on 89% of Japanese lines, since rule 3 turns
    # numbers into words. assemble() reads timings from the segments instead.
    for seg in batch:
        parts.append(f'{seg["i"]}|{seg["text"]}')
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
    r"^\s*(?:\[?[\d.]+\s*s\]"                 # an echoed slot marker, opening bracket optional
    r"|(?:id|line)\s*[:.]\s*\d+\s*[-:.|)]*"   # "id: 63"
    r"|(?:id|line)\s+\d+\s*[-:.|)]+"           # "id 63." — punctuation required
    r"|#\s*\d+\s*[-:.|)]*"                     # "#63"
    r")\s*", re.IGNORECASE)


# The reply format spelled back instead of filled in — a local model under
# pressure copies the format rule's own placeholder syntax rather than
# answering it. Rule 6's filled-in example is the actual fix; this stays to
# catch whatever shape a model hands back before it gets there.
_BRACKETED = r"[<\[{(][^>\]})]{0,40}[>\]})]"
_SCAFFOLD_PREFIX = re.compile(rf"^\s*{_BRACKETED}\s*\|\s*")
_ALL_SCAFFOLD = re.compile(rf"^(?:\s*{_BRACKETED}\s*[|:.\-]?\s*)+$")
# The id repeated after the bar the parser already ate: "63|63|Now we chain
# three." arrived as "63|Now we chain three." and was spoken as "sixty-three".
# _ECHOED deliberately will not match a bare number, to protect "3. Chain three
# stitches." — but a number followed by a bar, this deep in, is never a sentence.
_ECHOED_ID_BAR = re.compile(r"^\s*\d{1,5}\s*\|\s*")


def _is_scaffolding(text: str) -> bool:
    """True when the model returned the shape of an answer instead of one."""
    return bool(_ALL_SCAFFOLD.match(text))


def _slot_echo(slot: float) -> re.Pattern:
    """The marker this line was sent, whatever the model did to its brackets.

    Matches the value actually sent rather than guessing the echoed shape:
    "[2.0s]", "2.0s]", "2s" and "(2.0 s)" are all the same echo, none confusable
    with a sentence merely opening on a number.
    """
    exact = f"{slot:.1f}"
    whole = exact[:-2] if exact.endswith(".0") else exact
    # The decimal form stands alone; the bare form (".0" dropped) can also open
    # a real sentence — "60s is a long time" — so it's only an echo when bracketed.
    # Only closing punctuation may follow: an opening ¿/¡ or quote belongs to the
    # translation, and a greedier tail used to eat it.
    after = r"[\s\]\)}>|:.,\-]*"
    return re.compile(rf"^\W*(?:{re.escape(exact)}\s*s\b{after}"
                      rf"|{re.escape(whole)}\s*s\s*[\]\)}}>]{after})")


def _strip_echo(text: str, slot: float | None = None) -> str:
    """Remove any scaffolding the model repeated back into its translation."""
    if slot is not None:
        text = _slot_echo(slot).sub("", text, count=1)
    for _ in range(3):
        stripped = _ECHOED_ID_BAR.sub("", _SCAFFOLD_PREFIX.sub("", text, count=1),
                                      count=1)
        if stripped == text:
            break
        text = stripped
    for _ in range(3):                     # "63| id: 63 ..." has two layers
        stripped = _ECHOED.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return text.strip().strip('"').strip()


def _looks_untranslated(text: str, source: str) -> bool:
    """The model handed the source back instead of translating it.

    A local model that has lost the thread does this for a run of lines, and
    nothing downstream can tell — happened for real on an hour-long video, which
    spoke the original language in an English voice for several minutes.

    Short lines are exempt: "OK", a number, or a name legitimately survives
    translation unchanged.
    """
    def bare(s: str) -> str:
        return re.sub(r"[^\w]+", "", s).casefold()

    a, b = bare(text), bare(source)
    return len(b) > 15 and a == b


def _parse(reply: str, batch: list[dict]) -> tuple[dict[int, str], bool]:
    """Returns (translations, trustworthy).

    trustworthy is False when the reply's line numbering does not correspond to
    the batch's — an id never asked for, or repeated — the fingerprint of a
    model that renumbered rather than answered, which attaches the lines it
    *does* match to the wrong slots.

    Caught for real in a finished 98-minute dub: two lines carried the
    translation of the pair two slots earlier, so the voice described a corner
    while the picture showed a seam. Every content check passed it — fluent,
    unmatched to its own source, one line short of the batch (under the 5%
    ceiling). Nothing compared a translation to the slot it landed in.
    """
    wanted = {s["i"]: s.get("text", "") for s in batch}
    slots = {s["i"]: s["end"] - s["start"] for s in batch if "end" in s and "start" in s}
    out: dict[int, str] = {}
    seen: set[int] = set()
    trustworthy = True
    last: int | None = None
    # A model reasoning aloud writes draft "63|..." lines before its answer, and
    # the last match for an id would win. Older Ollama returns that inline.
    reply = reply.rsplit("</think>", 1)[-1]
    for line in reply.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            # A translation the model wrapped onto a second physical line.
            # Only a genuine wrap is rejoined, and the test is the first
            # character: a sentence continued onto a second line carries on in
            # lower case, while anything the model adds of its own — "Let me
            # know if you want any adjustments!", a closing code fence, a note
            # about the audio — starts with a capital or a symbol. `last in out`
            # also refuses a continuation after a *rejected* line, so it cannot
            # attach itself to an earlier, unrelated slot.
            if (last is not None and last in out and line[:1].islower()
                    and any(c.isalpha() for c in line)):
                out[last] = f"{out[last]} {line}".strip()
            else:
                last = None
            continue
        head, _, tail = line.partition("|")
        head = head.strip().lstrip("#").strip()
        # isdigit() is true for superscripts, which int() then refuses.
        if not (head.isascii() and head.isdigit()):
            continue
        idx = int(head)
        if idx not in wanted or idx in seen:
            trustworthy = False
        seen.add(idx)
        if idx in wanted:
            text = _strip_echo(tail.strip(), slots.get(idx))
            # Treated as a miss rather than a result, so the retry above gets a
            # go at it and it is counted if it never lands. A line of pure
            # template counts as a miss for the same reason: asking again is the
            # only thing that can turn it into a translation, and shipping it
            # puts the literal characters "<id>|<translation>" into the audio.
            if text and not _is_scaffolding(text) and not _is_example(text) \
                    and not _looks_untranslated(text, wanted[idx]):
                out[idx] = text
                last = idx
    return out, trustworthy


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

    A machine whose installer pulled a different model (timeout, slow first
    start, RAM tier moved) used to ask for a tag that wasn't there and die on a
    404 mid-translation. The user doesn't know model tags exist and shouldn't
    have to — substituting is strictly better than refusing.

    Returns (model, note); note is empty unless something was substituted,
    since a quietly different model is a quietly different translation.
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


def _call_ollama(prompt: str, model: str, host: str = "",
                 system: str = SYSTEM) -> str:
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
                # (echoing input, looping a fragment) — produced a dub that read
                # "id: 63" then minutes of the original language. Mild penalty
                # discourages the loop without flattening legitimate repeats
                # like a tutorial saying "chain three" forty times.
                "repeat_penalty": 1.1,
                "min_p": 0.05,
            },
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }
        if think is not None:
            body["think"] = think
        req = urllib.request.Request(f"{host}/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read()).get("message", {}).get("content", "")

    # Qwen3, the default install, reasons at length before answering — pure
    # cost for line-by-line translation. Measured: 92 generated tokens with
    # thinking on vs. 5 for the identical output off.
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


def _provider_message(body: bytes) -> str:
    """The provider's own explanation, if the body is shaped like one.

    Anthropic and OpenAI both nest it as {"error": {"message": ...}}; that's
    the only field read here, everything else in the body is left alone.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    err = data.get("error") if isinstance(data, dict) else None
    return str(err.get("message") or "").strip() if isinstance(err, dict) else ""


def _redact(text: str, key: str) -> str:
    """Strip the literal key before showing/logging text — OpenAI's own 401
    body echoes it back, so a provider's message isn't safe to relay raw.
    """
    return text.replace(key, "[key redacted]") if key.strip() and key in text else text


def _retry_hint(exc: urllib.error.HTTPError) -> str:
    """What the provider itself said to wait, when it said anything at all."""
    after = exc.headers.get("retry-after") if exc.headers else None
    try:
        seconds = int(float(after))
    except (TypeError, ValueError):
        return ""
    return f" It suggested waiting about {seconds}s before trying again." if seconds > 0 else ""


def _api_error(provider: str, exc: urllib.error.HTTPError, key: str) -> TranslationError:
    """Turn a rejected HTTP request into the thing a user can act on.

    The API paths used to raise a bare HTTPError, which _ask() swallowed as
    "no lines came back" — thirty retries and ninety seconds later blaming a
    local model that was never in use. Raising TranslationError here instead
    lets _ask() re-raise it so an unrecoverable request fails on the first try.
    """
    body = exc.read()
    said = _redact(_provider_message(body), key)
    quote = f" {provider} said: \u201c{said}\u201d." if said else ""
    detail = _redact(body.decode("utf-8", "replace"), key)[:2000]

    if exc.code in (401, 403):
        return TranslationError(
            f"{provider} rejected the API key in Settings \u2192 Translation.{quote} "
            "Check the key is correct and hasn't been revoked.", detail)
    if exc.code == 429:
        return TranslationError(
            f"{provider} is rate-limiting this key, or it's out of credit.{quote}"
            f"{_retry_hint(exc)}", detail)
    if exc.code >= 500:
        return TranslationError(
            f"{provider}'s API is having problems on its end (HTTP {exc.code})."
            f"{quote} This is usually temporary.", detail)
    return TranslationError(
        f"{provider} refused the request (HTTP {exc.code}).{quote}", detail)


def _unreachable(provider: str, exc: urllib.error.URLError) -> TranslationError:
    return TranslationError(
        f"Could not reach {provider}'s API. Check the internet connection and try again.",
        str(exc.reason))


def _call_anthropic(prompt: str, model: str, key: str,
                    system: str = SYSTEM) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": 8192,
        # Sonnet 5 runs adaptive thinking when this is omitted (Sonnet 4.5 did
        # not). Thinking is billed inside max_tokens, so a batch could reason
        # itself into a truncated reply — invisible here, arriving downstream
        # as missing lines and a halving retry that looks like a slow model.
        "thinking": {"type": "disabled"},
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"Content-Type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise _api_error("Anthropic", exc, key) from exc
    except urllib.error.URLError as exc:
        raise _unreachable("Anthropic", exc) from exc
    return "".join(b.get("text", "") for b in data.get("content", []))


def _call_openai(prompt: str, model: str, key: str,
                 system: str = SYSTEM) -> str:
    body = json.dumps({
        "model": model,
        "temperature": 0.2,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise _api_error("OpenAI", exc, key) from exc
    except urllib.error.URLError as exc:
        raise _unreachable("OpenAI", exc) from exc
    return data["choices"][0]["message"]["content"]


# Small enough that its output on specialist material is a real risk to the
# result, rather than merely a bit worse.
WEAK_LOCAL_MODELS = ("qwen3:4b", "qwen3:1.7b", "qwen3:0.6b")


def describe_translator(settings, ram_gb: int) -> tuple[str, str]:
    """(what will translate, a warning if that is the weak link).

    Translation is the one stage whose failure can't be heard: a bad dub with
    clean audio and perfect timing sounds finished and is useless. So this goes
    in the report, and flags the model when it's small enough to be the risk.
    """
    if settings.translator == "anthropic":
        return settings.anthropic_model, ""
    if settings.translator == "openai":
        return settings.openai_model, ""
    model, _ = usable_model(settings.resolved_ollama_model(ram_gb))
    if model in WEAK_LOCAL_MODELS:
        return model, (f"Translated by {model}, the smallest local model, which is "
                       "what fits this Mac's memory. On specialist material it is "
                       "the weakest part of the chain; an API key under Settings "
                       "gives better results.")
    return model, ""


def _batches(segments: list[dict]) -> list[list[dict]]:
    """Group lines into requests, capped by count and by size.

    A fixed count of 25 was fine until lines got long: joined run-on speech can
    run several hundred characters, crowding the context window right when a
    small model stops following the format and starts repeating itself.
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
        got, trustworthy = _parse(
            call(_build_prompt(batch, context, target, glossary)), batch)
    except TranslationError:
        raise
    except Exception:                                            # noqa: BLE001
        return {}
    if not trustworthy:
        # Discard the whole batch, not just the mismatched lines: when a model
        # renumbers, even the answers under ids we asked for are translations of
        # *other* lines — fluent, pass every check, attached to the wrong moment.
        # Discarding turns them into missing lines, which the halving retry can act on.
        logs.get().warning("reply did not line up with the batch, discarding it",
                           extra={"asked": len(batch), "matched": len(got),
                                  "first_id": batch[0]["i"]})
        return {}
    return got


def _translate_chunk(batch: list[dict], context: list[str], target: str,
                     glossary: str, call) -> dict[int, str]:
    """One batch, halved whenever the model cannot manage it whole.

    Used to retry only when four or fewer lines were missing (small prompts
    almost always land) — but a model that echoes its input fails a whole
    batch at once, so 25 missing lines fell outside that rule and got asked
    the identical question twice before being given up on.

    Halving asks a different question each time; by one line the prompt is
    small enough that almost anything lands. A batch of 25 bottoms out in
    five levels.
    """
    got = _ask(batch, context, target, glossary, call)
    missing = [s for s in batch if s["i"] not in got]
    if not missing:
        return got

    # Logged so a run leaves evidence of whether this machinery — which stops a
    # local model's bad patch reaching the finished video — actually fired.
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


# Asked of the same model, before any translating, with the whole transcript in
# front of it. A model that called "ponto amêndoa" amaranth, shell and cluster
# across one video answers correctly and identically here — it knows the
# vocabulary and loses it under per-line pressure.
EXTRACT_SYSTEM = """You extract specialist vocabulary from a tutorial transcript \
so that a translator renders it the same way every time.

Reply with one line per term, exactly:
term as it appears in the transcript | the standard English term

Only genuine craft, trade or technical vocabulary — never everyday words. Give
the term itself, not a phrase built around it. At most 20 lines. No preamble, no
commentary, no code fences."""

EXTRACT_RUNS = 2         # kept only if the runs agree; see _agreed()
EXTRACT_MAX = 20
EXTRACT_MAX_WORDS = 3    # a term, not a phrase built around one
EXTRACT_TARGET_WORDS = 6  # ...and its rendering is a term too, not a sentence
# Sized against the clock, not the context window: the model reads this twice
# before translating starts, and 18,000 chars cost 282s (14% of a 33-min job)
# vs. a third of that at 8,000. Kept contiguous — a strided sample made the
# vocabulary look unstable when it isn't.
EXTRACT_CHARS = 8000


def _extract_once(transcript: str, target: str, call) -> dict[str, str]:
    # Wording matters more than it should: "Give the standard English equivalent
    # for each specialist term" reads, to a small model looking at numbered
    # speech, as an instruction to translate the lines. First run came back
    # with "3 -> Today's lesson is about..." for every line it saw.
    prompt = (f"This transcript is from an instructional video. Identify the "
              f"specialist vocabulary in it and give the standard {target} term "
              f"for each one.\n\nTranscript:\n{transcript}")
    out: dict[str, str] = {}
    reply = call(prompt, system=EXTRACT_SYSTEM)
    for line in reply.rsplit("</think>", 1)[-1].splitlines():
        if "|" not in line:
            continue
        src, _, dst = line.partition("|")
        src = src.strip().strip("-*• ").casefold()
        dst = dst.strip()
        if src and dst and len(src) <= 60:
            out.setdefault(src, dst)
    return out


def _agreed(runs: list[dict[str, str]], transcript: str) -> dict[str, str]:
    """Terms every run returned identically, and that survive the filters.

    Agreement is the gate because a term the model is sure of comes back the
    same each time, while a guess wanders. On a real transcript, what varied
    across runs was compositional phrases ("iniciar com dois pontos altos"),
    not terms — the word cap below removes those; agreement alone did not.
    """
    if not runs:
        return {}
    low = transcript.casefold()
    keep: dict[str, str] = {}
    for src, dst in runs[0].items():
        if len(src.split()) > EXTRACT_MAX_WORDS:
            continue
        if len(dst.split()) > EXTRACT_TARGET_WORDS:
            continue                    # a sentence, so the model was translating
        if not any(c.isalpha() for c in src):
            continue                    # a line number, not a term
        if src not in low:              # never said it; the model invented it
            continue
        if src == dst.casefold():       # needs no pinning
            continue
        if any(r.get(src) != dst for r in runs[1:]):
            continue
        keep[src] = dst
    # Most-mentioned first, so a cap keeps what the video is actually about.
    ranked = sorted(keep.items(), key=lambda kv: -low.count(kv[0]))
    return dict(ranked[:EXTRACT_MAX])


def _merge_glossaries(known: str, extracted: str) -> str:
    """Known-good terms first; extracted ones only where they add something.

    Precedence is settled here, not left to the model: an extracted term
    already covered by the built-in or user glossary is dropped, so the two
    can never contradict each other in the prompt.
    """
    if not extracted.strip():
        return known
    covered = known.casefold()
    fresh = [ln for ln in extracted.splitlines()
             if ln.strip() and ln.split("->")[0].strip().casefold() not in covered]
    if not fresh:
        return known
    block = "\n".join(fresh)
    return f"{known}\n{block}" if known.strip() else block


def extract_terms(segments: list[dict], settings, ram_gb: int,
                  progress: Progress = None) -> str:
    """Terminology lifted from this video, as "source -> English" lines.

    Returns "" on any failure. A glossary is a nicety; a job that dies because
    the nicety failed is not, so nothing in here is allowed to raise.
    """
    try:
        transcript = "\n".join((s.get("text") or "") for s in segments)[:EXTRACT_CHARS]
        # Below a couple of batches there's no cross-batch drift to prevent, and
        # asking anyway is where junk comes from: a four-line French clip
        # returned "pas compliqué -> not complicated" and "mamies -> grandmas" —
        # everyday phrases it would then have pinned.
        if len(transcript) < 2 * BATCH_CHARS:
            return ""
        call, _ = backend_for(settings, ram_gb)
        runs = []
        for n in range(EXTRACT_RUNS):
            if progress:
                progress((n + 1) / (EXTRACT_RUNS + 1), "Collecting the video's terminology")
            runs.append(_extract_once(transcript, settings.target_language, call))
        terms = _agreed(runs, transcript)
        logs.get().info("terminology extracted", extra={
            "kept": len(terms), "seen": len(runs[0]) if runs else 0,
            "terms": terms})
        return "\n".join(f"{k} -> {v}" for k, v in terms.items())
    except Exception as exc:                                     # noqa: BLE001
        logs.get().warning("terminology pass failed, carrying on without it",
                           extra={"error": str(exc)[:200]})
        return ""


def backend_for(settings, ram_gb: int):
    """(call, label) for whichever translator is configured.

    Lifted out of translate() so the terminology pass uses the same backend and
    model rather than a second copy that could drift away from it.

    Takes no progress callback: it used to, only to record a substituted model,
    and since both the translation and terminology pass call this, the note
    was doubled on every finished job.
    """
    backend = settings.translator
    if backend == "ollama":
        # Substitution deliberately not noted on the finished job: which local
        # models are installed is a property of this Mac, not this video, so it
        # doesn't belong in a per-run report. Setup check already surfaces it,
        # on the row named for the model, where the fix is actionable.
        model, _ = usable_model(settings.resolved_ollama_model(ram_gb))
        return ((lambda p, system=SYSTEM: _call_ollama(p, model, system=system)),
                f"local model {model}")
    if backend == "anthropic":
        if not settings.anthropic_key:
            raise TranslationError("No Anthropic API key set in Settings \u2192 Translation.")
        return ((lambda p, system=SYSTEM: _call_anthropic(
                    p, settings.anthropic_model, settings.anthropic_key, system=system)),
                settings.anthropic_model)
    if backend == "openai":
        if not settings.openai_key:
            raise TranslationError("No OpenAI API key set in Settings \u2192 Translation.")
        return ((lambda p, system=SYSTEM: _call_openai(
                    p, settings.openai_model, settings.openai_key, system=system)),
                settings.openai_model)
    raise TranslationError(f"Unknown translation backend: {backend}")


def translate(segments: list[dict], settings, ram_gb: int, progress: Progress = None,
              extra_glossary: str = "", resume: dict[int, str] | None = None,
              on_batch: Callable[[dict[int, str]], None] | None = None) -> list[dict]:
    """Returns segments with a "translation" key added to each.

    extra_glossary is terminology lifted from this video's own transcript. It
    goes in behind the built-in and custom terms, which are known-good and win
    any collision.

    resume seeds already-translated lines from an attempt that failed part
    way, so a batch that was fully finished before is never re-asked for.
    on_batch, when given, is handed the accumulated results after every batch
    that actually made a request — the caller's chance to persist them before
    a later batch's failure (or the process dying outright) takes the rest.
    """
    glossary = _merge_glossaries(settings.glossary_text(), extra_glossary)
    target = settings.target_language
    call, label = backend_for(settings, ram_gb)

    for n, seg in enumerate(segments):
        seg["i"] = n

    done: dict[int, str] = dict(resume or {})
    batches = _batches(segments)

    for bn, batch in enumerate(batches):
        start = batch[0]["i"]
        context = [s["text"] for s in segments[max(0, start - CONTEXT):start]]
        todo = [s for s in batch if s["i"] not in done]
        if todo:
            done.update(_translate_chunk(todo, context, target, glossary, call))
            if on_batch:
                on_batch(dict(done))
        if progress:
            progress((bn + 1) / len(batches),
                     f"Translating with {label} — {min(len(done), len(segments))} of {len(segments)} lines")

    missing = [s for s in segments if s["i"] not in done]
    if len(missing) > len(segments) * 0.05:
        if settings.translator == "ollama":
            hint = "If you are using a local model, try a larger one in Settings \u2192 Translation."
        else:
            hint = (f"{label} isn't a local model, so a bigger one won't help — "
                    "try the job again; if it keeps happening, a handful of lines "
                    "are likely tripping it up rather than the whole batch.")
        # settings.translator, not label: label is free text for API backends
        # and shouldn't be parsed to decide the advice below — this used to
        # always blame a local model, telling a paid API key that had
        # translated all but a few lines to go find a bigger local model.
        raise TranslationError(
            f"Translation only returned {len(done)} of {len(segments)} lines, "
            f"translating with {label}. {hint}"
        )
    for seg in segments:
        seg["translation"] = done.get(seg["i"], "").strip()
    return segments
