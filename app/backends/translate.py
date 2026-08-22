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
from ..notes import info as note_info, note

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
6. FORMAT. Reply with exactly one line per input line, in the same order, each
   beginning with that input's own number and a vertical bar. An input numbered
   7 comes back looking like this and only like this:
   7|Nobody remembered to close the gate behind them.
   No preamble, no commentary, no code fences, no blank lines. Never put a line
   break inside a translation — anything after it is thrown away."""


# The one filled-in example above. A concrete example is imitated far more
# reliably than a placeholder is filled in, which is why it replaced
# "<id>|<translation>" — but that trade has a cost: a copied example is a
# plausible English sentence, and neither _is_scaffolding nor _looks_untranslated
# can see it. The first version of this said "12|Now chain three and turn your
# work.", which on a crochet tutorial is indistinguishable from a correct
# translation of a real line. So the sentence is deliberately alien to anything
# this app is pointed at, and is checked for rather than trusted to look wrong.
EXAMPLE_LINE = "Nobody remembered to close the gate behind them."


def _is_example(text: str) -> bool:
    bare = lambda s: re.sub(r"[^\w]+", "", s).casefold()   # noqa: E731
    return bare(text) == bare(EXAMPLE_LINE)


class TranslationError(RuntimeError):
    """A translation failure, with the provider's own words kept alongside the
    plain one.

    Shaped like download.DownloadError on purpose: pipeline.py already knows how
    to lift a `.detail` into job.error_detail, and the UI already knows how to
    render whatever lands there, so a translation failure needs no special-casing
    on either side to get the same treatment. Before this, a rejected key
    surfaced only as "translation batch incomplete, halving" repeated thirty
    times in Copy details — never the rejection itself.
    """

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail.strip()


# ------------------------------------------------------------- prompt build

def _build_prompt(batch: list[dict], context: list[str], target: str, glossary: str) -> str:
    parts = [f"Translate into {target}."]
    if glossary:
        parts.append("\nUse these terms exactly:\n" + glossary)
    if context:
        parts.append("\nPreceding lines, for continuity only — do NOT translate them:\n"
                     + "\n".join(f"  {c}" for c in context))
    # Deliberately no placeholder. This is the last instruction the model reads
    # before it answers, so it is the nearest thing to copy — and the copy that
    # produced 16% of a video was of a placeholder just like the one that used to
    # be here, in backticks, on this line. Fixing the rule in SYSTEM and leaving
    # this one moved the hazard closer to the generation point rather than away.
    parts.append("\nTranslate these lines. Reply with one line for each: its "
                 "number, a vertical bar, then the translation.\n")
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


def _strip_echo(text: str) -> str:
    """Remove any scaffolding the model repeated back into its translation."""
    for _ in range(3):
        stripped = _ECHOED_ID_BAR.sub("", _SCAFFOLD_PREFIX.sub("", text, count=1),
                                      count=1)
        if stripped != text:
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


def _parse(reply: str, batch: list[dict]) -> tuple[dict[int, str], bool]:
    """Returns (translations, trustworthy).

    trustworthy is False when the reply's line numbering does not correspond to
    the batch's — an id that was never asked for, or the same id twice. Both are
    the fingerprint of a model that renumbered rather than answered, and when it
    renumbers the lines it *does* match are attached to the wrong slots.

    That case is the reason this returns a flag rather than just a dict. Found in
    a finished 98-minute dub: two lines carried the translation of the pair two
    slots earlier, so the voice described a corner while the picture showed a
    seam. Every existing check passed it — each translation was fluent English,
    none matched its own source, and the batch was one line short, which is under
    the 5% ceiling by construction. Nothing in the pipeline compared a
    translation against the slot it landed in, so nothing could have seen it.
    """
    wanted = {s["i"]: s.get("text", "") for s in batch}
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
            # A translation the model wrapped onto a second physical line. This
            # used to be skipped, which silently deleted the rest of the
            # sentence: the id was already answered, so it was not a miss, not a
            # retry, and not counted — the dub simply spoke half the line, and
            # the half it spoke was a grammatical fragment of about the right
            # length. Most likely on the long run-on lines speech recognition
            # produces, which is where a model is likeliest to wrap.
            #
            # Only a genuine wrap is rejoined, and the test is the first
            # character: a sentence continued onto a second line carries on in
            # lower case, while anything the model adds of its own — "Let me
            # know if you want any adjustments!", a closing code fence, a note
            # about the audio — starts with a capital or a symbol. Without that
            # test the trailing pleasantry was glued onto the last translation
            # and spoken aloud, and a continuation after a *rejected* line
            # attached itself to an earlier, unrelated slot.
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
            text = _strip_echo(tail.strip())
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


# --------------------------------------------------------- provider HTTP errors

def _provider_message(body: bytes) -> str:
    """The provider's own explanation, if the body is shaped like one.

    Anthropic and OpenAI both nest it as {"error": {"message": ...}} once
    decoded, which is also the only field read here — anything else in the body
    is left alone rather than guessed at.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    err = data.get("error") if isinstance(data, dict) else None
    return str(err.get("message") or "").strip() if isinstance(err, dict) else ""


def _redact(text: str, key: str) -> str:
    """Strip the literal key out of anything about to be shown or logged —
    OpenAI's own 401 body echoes it back, so a provider's message isn't safe
    to relay unexamined.

    Matched as an exact, case-sensitive substring against the real key rather
    than a pattern; a whitespace-only key is treated as no key at all.
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

    The Ollama path below already does this; the two API paths used to do
    nothing at all, so a 401 raised urllib.error.HTTPError, which is not a
    TranslationError and was swallowed a few lines down in _ask() as "no lines
    came back" — thirty retries and ninety seconds later the report blamed a
    local model that was never in use. The distinction made here is what lets
    that swallow-and-retry loop be skipped rather than merely renamed: _ask()
    re-raises a TranslationError instead of eating it, so a request that can
    never succeed fails on the first try.
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
        # The same judgement the Ollama path makes below, and now it has to be
        # said out loud: Sonnet 5 runs adaptive thinking whenever this field is
        # omitted, where Sonnet 4.5 ran without it. Thinking is billed inside
        # max_tokens, so a batch of twenty-five lines could be reasoned about
        # until the reply truncated — and a truncated reply is not visible as an
        # error here. It arrives as missing lines and a halving retry, which
        # looks like a slow model rather than a misconfigured request.
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
        got, trustworthy = _parse(
            call(_build_prompt(batch, context, target, glossary)), batch)
    except TranslationError:
        raise
    except Exception:                                            # noqa: BLE001
        return {}
    if not trustworthy:
        # The whole batch goes back, not the lines that happened to match. When
        # a model renumbers, the answers it gives under the ids we asked for are
        # translations of *other* lines, so keeping them is worse than keeping
        # none: they are fluent, they pass every content check, and they are
        # attached to the wrong moment in the video. Discarding makes them missing
        # lines, which is the one condition the halving retry can act on.
        logs.get().warning("reply did not line up with the batch, discarding it",
                           extra={"asked": len(batch), "matched": len(got),
                                  "first_id": batch[0]["i"]})
        return {}
    return got


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

# ------------------------------------------------------- terminology pass

# Asked of the same model, before any translating, with the transcript in front
# of it. The model that called "ponto amêndoa" amaranth, shell and cluster across
# one video answers this correctly and identically on every run — it knows the
# vocabulary and loses it under per-line pressure, so it is worth asking once
# while it can see the whole thing.
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
# Contiguous, and sized against the clock rather than the context window: the
# model reads this twice before any translating starts, and a full 18,000
# characters cost 282 seconds — 14% of a 33-minute job — where 8,000 costs a
# third of that. Contiguous matters more than long: a thin strided sample made
# the vocabulary look unstable when it is not.
EXTRACT_CHARS = 8000


def _extract_once(transcript: str, target: str, call) -> dict[str, str]:
    # Wording matters more than it should. "Give the standard English equivalent
    # for each specialist term" reads, to a small model looking at numbered
    # speech, as an instruction to translate the lines — the first real run came
    # back with "3 -> Today's lesson is about..." for every line it saw.
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
    same each time, while a guess wanders. Measured on a real transcript: the
    core terms were unanimous over three runs, and what varied was compositional
    phrases — "iniciar com dois pontos altos", "pulo de correntinhas de espaço" —
    which are not terms at all. The word cap removes those; agreement alone left
    them in.
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

    Precedence is settled here rather than left to the model: an extracted term
    whose source is already covered by the built-in list or by the user's own
    terms is dropped, so the two can never contradict each other in the prompt.
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
        # Below a couple of batches there is no cross-batch drift to prevent —
        # the whole point of pinning terminology is that batch 14 cannot see what
        # batch 2 called a stitch, and a video this short is one batch. Asking
        # anyway is where the junk comes from: on a four-line French clip it
        # returned "pas compliqué -> not complicated" and "mamies -> grandmas",
        # everyday phrases it would then have pinned.
        if len(transcript) < 2 * BATCH_CHARS:
            return ""
        call, _ = backend_for(settings, ram_gb, progress)
        runs = []
        for n in range(EXTRACT_RUNS):
            if progress:
                progress((n + 1) / (EXTRACT_RUNS + 1), "Reading the subject's vocabulary")
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


def backend_for(settings, ram_gb: int, progress: Progress = None):
    """(call, label) for whichever translator is configured.

    Lifted out of translate() so the terminology pass uses the same backend, the
    same model and the same substitution note rather than a second copy that
    could drift away from it.
    """
    backend = settings.translator
    if backend == "ollama":
        model, swapped = usable_model(settings.resolved_ollama_model(ram_gb))
        if swapped:
            # The run still succeeded with a model that works, so this is
            # explanation, not a fault — tagged the same way as other good news,
            # not boxed like the warnings beside it.
            note(progress, note_info(swapped))
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
    call, label = backend_for(settings, ram_gb, progress)

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
        # settings.translator, not label: label is "local model X" for Ollama
        # and a free-text model id the user typed into Settings for the other
        # two, and parsing that string to decide the advice below couples the
        # message to a display convention when the definitive answer — which
        # backend actually ran — is sitting right here. This used to ignore
        # both and always blame a local model, so a paid API key that had
        # translated all but a few lines of the video was told, in the
        # failure it caused, to go find a bigger local model.
        if settings.translator == "ollama":
            hint = "If you are using a local model, try a larger one in Settings \u2192 Translation."
        else:
            hint = (f"{label} isn't a local model, so a bigger one won't help — "
                    "try the job again; if it keeps happening, a handful of lines "
                    "are likely tripping it up rather than the whole batch.")
        raise TranslationError(
            f"Translation only returned {len(done)} of {len(segments)} lines, "
            f"translating with {label}. {hint}"
        )
    for seg in segments:
        seg["translation"] = done.get(seg["i"], "").strip()
    return segments
