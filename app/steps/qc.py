"""A last look at the translation before any of it is spoken.

Cheap on purpose: string work over lines that are already in memory, no model
call, no audio. It exists because the expensive checks all happen too late — the
finished-audio check confirms there is sound, and the frame check confirms the
picture survived, and a dub that reads out "id: 63" followed by untranslated
Spanish passes both of them.

It also runs against a translation restored from cache, which the parser cannot:
a job whose translation went wrong once would otherwise replay it on every
re-run of the same link.
"""
from __future__ import annotations

from .. import logs
from ..backends.translate import _looks_untranslated, _strip_echo


def check(segments: list[dict]) -> dict:
    """Repair what can be repaired, drop what cannot, and say what happened.

    Repaired means scaffolding the model echoed back was removed and a real
    translation was underneath. Dropped means there was nothing usable — those
    lines are left silent, which is a worse dub than a correct line and a much
    better one than the original language read aloud in an English voice.
    """
    repaired = untranslated = empty = 0
    # Every repair is logged with the line it changed, not just counted. The
    # count says a long video needed patching; the lines say where, so they can
    # be listened to in the finished dub instead of taken on trust.
    log = logs.get()

    for seg in segments:
        text = (seg.get("translation") or "").strip()
        if not text:
            empty += 1
            continue

        cleaned = _strip_echo(text)
        if cleaned != text:
            repaired += 1
            log.warning("stray scaffolding removed from a translated line",
                        extra={"at": round(seg.get("start", 0.0), 1),
                               "was": text[:120], "now": cleaned[:120]})
            text = cleaned

        if not text:
            empty += 1
        elif _looks_untranslated(text, seg.get("text", "")):
            untranslated += 1
            log.warning("line came back untranslated, silenced",
                        extra={"at": round(seg.get("start", 0.0), 1),
                               "source": (seg.get("text") or "")[:120],
                               "returned": text[:120]})
            text = ""

        seg["translation"] = text

    spoken = sum(1 for s in segments if (s.get("translation") or "").strip())
    return {"lines": len(segments), "spoken": spoken, "repaired": repaired,
            "untranslated": untranslated, "empty": empty}


def summarise(report: dict) -> str:
    """One sentence for the finished report, or nothing when all is well."""
    parts = []
    if report["repaired"]:
        parts.append(f"{report['repaired']} had stray numbering removed")
    if report["untranslated"]:
        parts.append(f"{report['untranslated']} came back in the original language "
                     "and were left silent")
    if not parts:
        return ""
    return ("The translation needed patching up: " + ", and ".join(parts) +
            ". This happens when a local model loses the thread part way through "
            "a long video — a larger model, or an API key under Settings, avoids it.")
