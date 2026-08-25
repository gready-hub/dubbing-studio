"""A last look at the translation before any of it is spoken.

Cheap on purpose: string work over in-memory lines, no model call, no audio.
Exists because the other checks run too late — a dub reading "id: 63" followed
by untranslated Spanish passes both the audio-present and frame-count checks.

Also runs against a translation restored from cache, which the parser cannot,
so a job whose translation went wrong once won't replay the fault forever.
"""
from __future__ import annotations

from .. import logs
from ..backends.translate import (_is_example, _is_scaffolding,
                                  _looks_untranslated, _strip_echo)


def check(segments: list[dict]) -> dict:
    """Repair what can be repaired, drop what cannot, and say what happened.

    Repaired means scaffolding the model echoed back was removed and a real
    translation was underneath. Dropped means there was nothing usable — those
    lines are left silent, which is a worse dub than a correct line and a much
    better one than the original language read aloud in an English voice.
    """
    repaired = untranslated = empty = template = 0
    # Each repair is logged with the line it changed, not just counted, so the
    # affected lines can be checked in the finished dub instead of taken on trust.
    log = logs.get()

    for seg in segments:
        text = (seg.get("translation") or "").strip()
        if not text:
            empty += 1
            continue

        cleaned = _strip_echo(text)
        # A cached translated.json replays straight past the parser's guards
        # (those only run at translation time). Stripping "<id>|" off a
        # template reply like "<id>|<translation>" leaves non-empty text that
        # looks like a real translation and would otherwise be spoken and
        # miscounted as a repair.
        if cleaned and (_is_scaffolding(cleaned) or _is_example(cleaned)):
            template += 1
            log.warning("line came back as the reply format, not a translation",
                        extra={"at": round(seg.get("start", 0.0), 1),
                               "returned": text[:120]})
            seg["translation"] = ""
            continue

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
            "untranslated": untranslated, "template": template, "empty": empty}


def summarise(report: dict) -> str:
    """One sentence for the finished report, or nothing when all is well."""
    parts = []
    if report["repaired"]:
        parts.append(f"{report['repaired']} had stray numbering removed")
    if report["untranslated"]:
        parts.append(f"{report['untranslated']} came back in the original language "
                     "and were left silent")
    # Counted apart from the line above, because they are a different fault and
    # saying "came back in the original language" about them would be untrue.
    if report.get("template"):
        parts.append(f"{report['template']} came back as the reply format rather "
                     "than a translation and were left silent")
    if not parts:
        return ""
    return ("The translation needed patching: " + ", and ".join(parts) +
            ". A local model can lose the thread over a long video; a larger model, "
            "or an API key under Settings, avoids this.")
