"""End-to-end and unit tests for Dubbing Studio.

Run it from anywhere:  python test_pipeline.py

The job data is redirected to a scratch folder so a test run never disturbs the
real jobs under Application Support. Set DUBBING_TEST_SOURCE to a video with
speech in it to exercise transcription against real recorded audio; without one
the suite synthesises its own speech and is fully self-contained.
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Must be set before app.config is imported anywhere — it reads these at import
# time to decide where the job and output folders live. A fixed shared path let
# concurrent runs corrupt each other's job state, so each run now gets its own
# directory (removed on exit) unless the caller names one explicitly.
_explicit_home = os.environ.get("DUBBING_STUDIO_HOME")
if _explicit_home:
    SCRATCH = Path(_explicit_home)
else:
    SCRATCH = Path(tempfile.mkdtemp(prefix="dubbing-studio-test-"))
    os.environ["DUBBING_STUDIO_HOME"] = str(SCRATCH)
    atexit.register(shutil.rmtree, SCRATCH, ignore_errors=True)
os.environ.setdefault("DUBBING_STUDIO_OUTPUT", str(SCRATCH / "output"))
WORK = SCRATCH / "work"
WORK.mkdir(parents=True, exist_ok=True)

# Point the scratch model folder at the real one so a test run doesn't re-fetch
# 700 MB of speech models it already has. Jobs and output stay isolated.
#
# The scratch home lives under /var/folders, which macOS purges on its own
# schedule. It deletes the big model files but leaves the directories, so a
# plain "does it exist" check kept a hollowed-out models folder in place and the
# suite failed from deep inside sherpa-onnx with "File doesn't exist" — an
# environment problem wearing a code regression's clothes. A real directory that
# is not the symlink we intended is therefore replaced, not respected.
_real_models = Path.home() / "Library" / "Caches" / "DubbingStudio" / "models"
_scratch_models = SCRATCH / "cache" / "models"
if _real_models.is_dir() and not _scratch_models.is_symlink():
    if _scratch_models.is_dir():
        shutil.rmtree(_scratch_models, ignore_errors=True)
    _scratch_models.parent.mkdir(parents=True, exist_ok=True)
    _scratch_models.symlink_to(_real_models)

import numpy as np
import soundfile as sf

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def note_text(n):
    """A job.stats note may be a bare string (a warning, always) or an
    app.notes.info()-tagged dict. Read the words either way."""
    return n["text"] if isinstance(n, dict) else n


def stub_download(clip: Path, title: str, duration: float, replace: bool = False):
    """Stand in for the download step, probe included.

    The pipeline probes the link before it draws the progress plan, because a
    sample has to know how much of the whole video the download covers before it
    can weight it. A stub that replaces only download() therefore leaves the real
    yt-dlp being asked about a made-up URL, and every job dies on its first line.

    Returns (probe, download) for the caller to install; everything after the
    download is the real thing.
    """
    meta = {"title": title, "duration": duration, "uploader": "t", "thumbnail": ""}

    # **_ for the same reason as fake_download below: the pipeline now hands the
    # probe the browser to take cookies from, and a stub with a fixed signature
    # turns that into every job in the suite dying on its first line.
    def fake_probe(url, *_a, **_kw):
        return dict(meta)

    # **_ so a new pass-through argument is not a suite-wide failure.
    def fake_download(url, workdir, quality="best", progress=None, info=None, **_):
        workdir.mkdir(parents=True, exist_ok=True)
        dest = workdir / "source.mp4"
        if replace or not dest.exists():
            shutil.copy(clip, dest)
        if progress:
            progress(1.0, "Downloaded")
        return dest, dict(meta)

    return fake_probe, fake_download


def speech_wav(dst: Path, seconds: float = 75.0) -> Path:
    """Audio with real speech in it, for the transcription end of the pipeline.

    Prefers a real recording if one is configured, because recorded speech is a
    far harder and more representative test of the recogniser than synthesised
    speech is. Falls back to the app's own portable voice so the suite still runs
    on a machine that has no sample video to hand.
    """
    if dst.exists():
        return dst

    source = os.environ.get("DUBBING_TEST_SOURCE", str(Path.home() / "Downloads" / "videoplayback.mp4"))
    if source and Path(source).exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "20", "-t", f"{seconds:g}",
                        "-i", source, "-ac", "1", "-ar", "24000", str(dst)], check=True)
        return dst

    from app.backends.tts import OnnxTTS
    engine = OnnxTTS()
    lines = ["Right, let us carry on with the next round of the pattern.",
             "We chain three stitches and then turn the work over.",
             "Now work a single crochet into that very same stitch.",
             "Keep the tension nice and loose so the fabric drapes well.",
             "You can see how the pattern starts to build up along here."]
    pieces, rate = [], 24000
    while sum(p.size for p in pieces) / rate < seconds:
        for text in lines:
            audio, rate = engine.say(text, voice="bf_emma")
            pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            pieces.append(np.zeros(int(0.45 * rate), dtype=np.float32))
            if sum(p.size for p in pieces) / rate >= seconds:
                break
    sf.write(dst, np.concatenate(pieces)[:int(seconds * rate)], rate)
    return dst


# ======================================================= 1. alignment maths
def test_align():
    print("\n[1] Alignment maths")
    from app.steps import align

    sr = 24000

    def tone(seconds):
        return (0.2 * np.sin(np.linspace(0, seconds * 440 * 2 * np.pi, int(seconds * sr)))
                ).astype(np.float32)

    # Lines that comfortably fit their slots must not be touched at all.
    lines = [{"start": 0.0, "end": 2.0, "samples": tone(1.0)},
             {"start": 5.0, "end": 7.0, "samples": tone(1.0)},
             {"start": 10.0, "end": 12.0, "samples": tone(1.0)}]
    track, stats = align.assemble([dict(l) for l in lines], 15.0, sr)
    check("no compression when lines fit", stats["compressed"] == 0, str(stats["compressed"]))
    check("no drift when lines fit", stats["max_drift"] == 0.0)
    check("track is the right length", abs(len(track) / sr - 16.0) < 0.01,
          f"{len(track)/sr:.2f}s")

    # A line far too long for its slot must be compressed, but capped.
    lines = [{"start": 0.0, "end": 1.0, "samples": tone(6.0)},
             {"start": 2.0, "end": 3.0, "samples": tone(0.5)}]
    track, stats = align.assemble([dict(l) for l in lines], 10.0, sr, max_stretch=1.55)
    check("over-long line is compressed", stats["compressed"] >= 1)
    check("compression respects the cap", stats["max_factor"] <= 1.5501,
          f"{stats['max_factor']}")
    check("over-cap lines are counted", stats["over_cap"] == 1, str(stats["over_cap"]))

    # A line that overruns must push the next one later, not overlap it.
    lines = [{"start": 0.0, "end": 1.0, "samples": tone(4.0)},
             {"start": 1.0, "end": 2.0, "samples": tone(1.0)}]
    _, stats = align.assemble([dict(l) for l in lines], 12.0, sr, max_stretch=1.1)
    check("drift is recorded when a line overruns", stats["max_drift"] > 0,
          f"{stats['max_drift']}s")

    # Peak normalisation.
    loud = [{"start": 0.0, "end": 2.0, "samples": (tone(1.0) * 20).astype(np.float32)}]
    track, _ = align.assemble(loud, 5.0, sr)
    check("output is normalised, not clipped", 0.85 <= float(np.max(np.abs(track))) <= 0.9,
          f"peak {float(np.max(np.abs(track))):.3f}")

    # atempo chaining for large factors.
    check("atempo chains beyond 2x", align._atempo_chain(3.5).count("atempo") == 2,
          align._atempo_chain(3.5))
    check("atempo single stage under 2x", align._atempo_chain(1.4).count("atempo") == 1)

    # SRT output.
    segs = [{"start": 1.5, "end": 3.25, "translation": "Hello there"},
            {"start": 4.0, "end": 5.0, "translation": "Second line"}]
    out = WORK / "test.srt"
    align.write_srt(segs, out)
    body = out.read_text()
    check("SRT timestamps are formatted correctly", "00:00:01,500 --> 00:00:03,250" in body,
          body.splitlines()[1] if body else "empty")

    # A cue too long to read at a glance: one run-on merge span, ~220 characters
    # over 11 seconds, no punctuation to hint at a break.
    long_text = " ".join(f"word{n}" for n in range(40))  # 40 short words, ~220 chars
    long_segs = [{"start": 0.0, "end": 11.0, "translation": long_text}]
    out2 = WORK / "test_long.srt"
    align.write_srt(long_segs, out2)
    cues = out2.read_text().strip().split("\n\n")
    lines_per_cue = [c.split("\n")[2:] for c in cues]
    check("an over-long cue is split into more than one",
          len(cues) > 1, str(len(cues)))
    check("every line respects the character limit",
          all(len(line) <= align.LINE_CHARS for lines in lines_per_cue for line in lines))
    check("every cue keeps to two lines or fewer",
          all(len(lines) <= 2 for lines in lines_per_cue))
    check("the words survive the split, in order",
          " ".join(" ".join(lines).replace("\n", " ") for lines in lines_per_cue) == long_text)

    # A single token with no space in it anywhere near the line width — the
    # case _chunk_text used to wave through untouched, for textwrap to then
    # hard-break into as many lines as it took, inside one cue.
    blob = "a" * 300
    blob_segs = [{"start": 0.0, "end": 20.0, "translation": blob}]
    out_blob = WORK / "test_blob.srt"
    align.write_srt(blob_segs, out_blob)
    blob_cues = out_blob.read_text().strip().split("\n\n")
    blob_lines = [c.split("\n")[2:] for c in blob_cues]
    check("an unbreakable 300-char token still keeps every cue to two lines",
          all(len(lines) <= 2 for lines in blob_lines), str([len(l) for l in blob_lines]))
    check("and every line of it still respects the width",
          all(len(line) <= align.LINE_CHARS for lines in blob_lines for line in lines))

    # A CJK run has no spaces at all, so a naive word-splitter cannot touch it
    # either — same failure mode as the blob above, different alphabet.
    cjk = "这是一个很长的中文句子用来测试断行逻辑是否能够正确地把长文本切成不超过四十二个字符的行" * 3
    cjk_segs = [{"start": 0.0, "end": 15.0, "translation": cjk}]
    out_cjk = WORK / "test_cjk.srt"
    align.write_srt(cjk_segs, out_cjk)
    cjk_cues = out_cjk.read_text().strip().split("\n\n")
    cjk_lines = [c.split("\n")[2:] for c in cjk_cues]
    check("a CJK run with no spaces still keeps every cue to two lines",
          all(len(lines) <= 2 for lines in cjk_lines), str([len(l) for l in cjk_lines]))
    check("and every line of it still respects the width",
          all(len(line) <= align.LINE_CHARS for lines in cjk_lines for line in lines))

    # A short cue is left exactly as it was: no gratuitous split or wrap.
    short_segs = [{"start": 0.0, "end": 2.0, "translation": "Hello there"}]
    out3 = WORK / "test_short.srt"
    align.write_srt(short_segs, out3)
    check("a cue that already fits is not split or rewrapped",
          out3.read_text().strip().split("\n")[2:] == ["Hello there"])

    # A cue is left to run long when the speech genuinely is slow — no cap
    # invents a gap where the dub audio is still talking.
    slow_segs = [{"start": 0.0, "end": 23.6, "translation": "Just a few words, spoken slowly."}]
    out_slow = WORK / "test_slow.srt"
    align.write_srt(slow_segs, out_slow)
    check("a short cue over a long span is not artificially cut short",
          "00:00:23,600" in out_slow.read_text())

    # Timings a real pipeline should not produce, but nothing upstream is
    # guarded against it: write_srt is the one place left to catch it.
    zero_segs = [{"start": 5.0, "end": 5.0, "translation": "Frozen frame"}]
    out_zero = WORK / "test_zero.srt"
    align.write_srt(zero_segs, out_zero)
    check("a zero-duration segment still produces a valid, non-negative cue",
          "00:00:05,000 --> 00:00:05,000" in out_zero.read_text())

    backwards_segs = [{"start": 5.0, "end": 2.0, "translation": "Time ran backwards"}]
    out_back = WORK / "test_backwards.srt"
    align.write_srt(backwards_segs, out_back)
    back_stamps = out_back.read_text().splitlines()[1]
    back_start, back_end = (s.strip() for s in back_stamps.split("-->"))
    check("a segment with end before start is clamped, not passed through",
          back_start <= back_end, back_stamps)

    # Subtitles have to arrive as a track a player will offer, which means the
    # codec MP4 carries and a language on it. Untagged, the track lands as "und"
    # and the menu calls it Unknown — the one thing the menu is there to say.
    from app.steps import mux as _mux
    sub_dir = WORK / "subtitle-track"
    sub_dir.mkdir(parents=True, exist_ok=True)
    vid_in, dub_in = sub_dir / "v.mp4", sub_dir / "d.m4a"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "color=c=navy:s=160x120:d=4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(vid_in)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=4", "-c:a", "aac",
                    str(dub_in)], check=True)
    cue_srt = sub_dir / "cues.srt"
    align.write_srt([{"start": 0.5, "end": 2.0, "translation": "A spoken line."}], cue_srt)
    subbed = sub_dir / "out.mp4"
    _mux.mux(vid_in, dub_in, subbed, "replace", -18.0, cue_srt)
    sub_streams = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=codec_name:stream_tags=language",
         "-of", "default=nw=1", str(subbed)],
        capture_output=True, text=True).stdout
    check("subtitles are muxed as the codec an MP4 can carry",
          "mov_text" in sub_streams, sub_streams.replace("\n", " "))
    check("and the track says which language it is",
          "language=eng" in sub_streams, sub_streams.replace("\n", " "))


# ==================================================== 2. translation parsing
def test_translate():
    print("\n[2] Translation handling")
    from app.backends import translate as T

    batch = [{"i": 0, "start": 0, "end": 2, "text": "hola"},
             {"i": 1, "start": 2, "end": 4, "text": "adios"}]

    lines = lambda reply: T._parse(reply, batch)[0]        # noqa: E731
    trusted = lambda reply: T._parse(reply, batch)[1]      # noqa: E731

    check("parses clean output",
          lines("0|Hello\n1|Goodbye") == {0: "Hello", 1: "Goodbye"})
    check("ignores preamble and fences",
          lines("Sure!\n```\n0|Hello\n1|Goodbye\n```") == {0: "Hello", 1: "Goodbye"})
    check("strips an echoed time marker",
          lines("0|[2.0s] Hello") == {0: "Hello"})
    check("drops ids not in the batch",
          lines("0|Hello\n99|Nope") == {0: "Hello"})
    check("survives a missing line", lines("1|Goodbye") == {1: "Goodbye"})
    check("handles pipes inside the translation",
          lines("0|Hello | there") == {0: "Hello | there"})

    # A model that renumbers hands back fluent translations of *other* lines
    # under the ids we asked for. Found in a finished 98-minute dub: two slots
    # carried the pair from two slots earlier, so the voice described a corner
    # while the picture showed a seam. Every content check passed it, because
    # nothing compared a translation against the slot it landed in.
    check("a reply that lines up is trusted", trusted("0|Hello\n1|Goodbye"))
    check("an id we never asked for makes the whole batch untrustworthy",
          not trusted("1|Hello\n2|Goodbye"))
    check("and so does the same id twice",
          not trusted("0|The real one.\n0|Sorry, I meant this."))
    check("an untrustworthy batch is discarded rather than cherry-picked",
          T._ask(batch, [], "English", "",
                 lambda _p: "1|Hello\n2|Goodbye") == {})
    check("a clean batch still comes back from _ask",
          T._ask(batch, [], "English", "",
                 lambda _p: "0|Hello\n1|Goodbye") == {0: "Hello", 1: "Goodbye"})

    # A translation the model wrapped onto a second physical line. This used to
    # be dropped: the id was already answered, so it was not a miss, not a
    # retry, and not counted — the dub spoke a grammatical fragment of roughly
    # the right length and nothing downstream could tell.
    check("a wrapped translation is rejoined, not truncated",
          lines("0|Now chain three and turn,\nand then work two together.")
          == {0: "Now chain three and turn, and then work two together."})
    # Only a genuine wrap. A sentence continued onto a second line carries on in
    # lower case; anything the model adds of its own starts with a capital.
    # Without that test the pleasantry was glued on and spoken aloud.
    check("but the model's own sign-off is not glued onto the last line",
          lines("0|Hello\n1|Goodbye\nLet me know if you want any adjustments!")
          == {0: "Hello", 1: "Goodbye"})

    prompt = T._build_prompt(batch, ["contexto"], "English", "hola -> hi")
    # Nothing but the id and the line. A slot marker riding along here came back
    # spoken aloud on 89 of every 100 Japanese lines: rule 3 renders numbers as
    # words, and "four point zero seconds" is a translation as far as every
    # check downstream can tell. What is never sent cannot be echoed.
    check("prompt carries no slot marker", "s]" not in prompt and "2.0" not in prompt,
          prompt[-80:])
    check("prompt still carries the line and its id",
          all(f'{s["i"]}|{s["text"]}' in prompt for s in batch))
    check("prompt carries the glossary", "hola -> hi" in prompt)
    check("prompt marks context as not-for-translation", "do NOT translate" in prompt)

    # Full translate() with a stub backend, including a deliberately flaky one.
    from app.config import Settings
    settings = Settings()
    settings.translator = "ollama"

    segs = [{"start": i * 2.0, "end": i * 2.0 + 1.8, "text": f"linea {i}"} for i in range(60)]
    calls = {"n": 0}

    def flaky(prompt, model, host="x", **_):
        calls["n"] += 1
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        if calls["n"] == 1:                       # first batch drops two lines
            ids = ids[:-2]
        return "\n".join(f"{i}|English line {i}" for i in ids)

    T._call_ollama = flaky
    out = T.translate([dict(s) for s in segs], settings, 16)
    check("every segment gets a translation",
          all(s.get("translation") for s in out), f"{sum(1 for s in out if s.get('translation'))}/60")
    check("retry recovered the dropped lines", out[len(out) - 1]["translation"].startswith("English"))

    # resume/on_batch: what lets a caller (pipeline.py) survive a translation
    # that dies part way through, without redoing the batches that landed.
    saved_states: list[dict] = []
    calls2 = {"n": 0}

    def dies_second_batch(prompt, model, host="x", **_):
        calls2["n"] += 1
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        if calls2["n"] == 2:
            raise T.TranslationError("simulated failure")
        return "\n".join(f"{i}|English line {i}" for i in ids)

    T._call_ollama = dies_second_batch
    try:
        T.translate([dict(s) for s in segs], settings, 16,
                    on_batch=lambda d: saved_states.append(dict(d)))
        check("a mid-run failure raises rather than returning a truncated result", False)
    except T.TranslationError:
        check("a mid-run failure raises rather than returning a truncated result", True)
    check("on_batch captured the finished batch before the one that failed",
          bool(saved_states) and sorted(saved_states[-1]) == list(range(25)),
          str(sorted(saved_states[-1])) if saved_states else "nothing saved")

    # A retry seeded with that partial must not re-ask for lines it already has.
    requested: list[int] = []

    def strict(prompt, model, host="x", **_):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        requested.extend(ids)
        return "\n".join(f"{i}|English line {i}" for i in ids)

    T._call_ollama = strict
    resumed = T.translate([dict(s) for s in segs], settings, 16, resume=saved_states[-1])
    check("resuming only asks for the lines that weren't already done",
          sorted(requested) == list(range(25, 60)), str(sorted(requested)[:3]) + "…")
    check("a resumed translation still fills in every line",
          all(s.get("translation") for s in resumed))

    # A backend that returns nothing must raise, not ship a silent empty dub.
    T._call_ollama = lambda p, m, host="x", **_: ""
    try:
        T.translate([dict(s) for s in segs], settings, 16)
        check("empty backend raises", False)
    except T.TranslationError as exc:
        check("empty backend raises", True)
        check("a local model's own incomplete-translation advice is unchanged",
              "If you are using a local model" in str(exc), str(exc))

    # A tester put a bogus key in Settings, ran a job, and ninety seconds and
    # thirty retries later was told "Translation only returned 0 of 30 lines.
    # If you are using a local model, try a larger one in Settings" — despite
    # having said, ten seconds earlier in its own progress line, that it was
    # translating with claude-sonnet-5. The words "API key" appeared nowhere.
    # _call_anthropic() and _call_openai() had no exception handling at all, so
    # the 401 raised urllib.error.HTTPError, which is not a TranslationError and
    # was swallowed by _ask()'s "except Exception: return {}" as an ordinary
    # empty reply — indistinguishable from a slow model, and retried the same way.
    import io
    import urllib.error
    import urllib.request as _urlreq

    real_urlopen = _urlreq.urlopen
    api_segs = [{"start": i * 2.0, "end": i * 2.0 + 1.8, "text": f"line {i}"} for i in range(10)]

    def hit_provider(provider_key, build_error, status):
        """Run translate() against a stubbed HTTP layer that always answers
        with one status and body, and report (exception, request count, the
        canary key that was set, the settings used)."""
        calls = {"n": 0}
        s = Settings()
        s.translator = provider_key
        key = f"sk-{provider_key}-CANARY-SECRET"
        setattr(s, f"{provider_key}_key", key)

        def counted(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(
                req.full_url, status, "err", {}, io.BytesIO(json.dumps(build_error(key)).encode()))

        _urlreq.urlopen = counted
        try:
            T.translate([dict(x) for x in api_segs], s, 16)
            return None, calls["n"], key
        except T.TranslationError as exc:
            return exc, calls["n"], key
        finally:
            _urlreq.urlopen = real_urlopen

    # OpenAI's own 401 body echoes the key back ("Incorrect API key provided:
    # sk-..."); Anthropic's does not. Both are exercised so the redaction in
    # _redact() is tested against a body that actually contains the secret,
    # not just against one that never had it to begin with.
    for provider, build_error in (
            ("anthropic", lambda k: {"type": "error", "error": {
                "type": "authentication_error", "message": "invalid x-api-key"}}),
            ("openai", lambda k: {"error": {
                "message": f"Incorrect API key provided: {k}.",
                "type": "invalid_request_error", "code": "invalid_api_key"}})):
        exc, n_calls, key = hit_provider(provider, build_error, 401)
        label = "Anthropic" if provider == "anthropic" else "OpenAI"
        check(f"a 401 from {label} raises a TranslationError, not a silent miss",
              exc is not None)
        msg = str(exc) if exc else ""
        check(f"it names {label}", label in msg, msg)
        check("it points at Settings → Translation",
              "Settings" in msg and "Translation" in msg, msg)
        check("it says the key was rejected",
              "reject" in msg.lower() and "key" in msg.lower(), msg)
        check("it does not contain the key itself",
              key not in msg and key not in (exc.detail if exc else ""), msg)
        check(f"{label} fails on the first request rather than after thirty retries",
              n_calls == 1, f"{n_calls} calls")

    # A whitespace-only key is still truthy, so backend_for()'s "if not
    # settings.anthropic_key" guard never catches it, and matching it against a
    # message the ordinary way would replace every run of spaces in the text —
    # reachable by nothing more than a stray space pasted into Settings.
    check("a whitespace-only key is not redacted, and doesn't corrupt the message",
          T._redact("Anthropic said the key was invalid.", "   ")
          == "Anthropic said the key was invalid.")
    check("but a real key, however it's padded, still gets redacted",
          "[key redacted]" in T._redact("key: '  sk-test-123  ' is bad", "  sk-test-123  "))

    # A 429 must read differently from a 401 — one is fixed by a different key,
    # the other by waiting or paying, and conflating them sends someone to
    # Settings to fix a key that was never the problem.
    exc429, n429, _ = hit_provider(
        "anthropic", lambda k: {"error": {"message": "rate limit exceeded"}}, 429)
    check("a 429 is distinguished from a 401",
          exc429 is not None and "reject" not in str(exc429).lower()
          and ("rate" in str(exc429).lower() or "credit" in str(exc429).lower()), str(exc429))
    check("a 429 also fails fast rather than retrying thirty times",
          n429 == 1, f"{n429} calls")

    # The final incomplete-translation message used to hardcode local-model
    # advice regardless of which backend actually ran.
    settings4 = Settings()
    settings4.translator = "anthropic"
    settings4.anthropic_key = "sk-ant-whatever"
    settings4.anthropic_model = "claude-sonnet-5"
    T._call_anthropic = lambda *a, **kw: ""
    try:
        T.translate([dict(x) for x in api_segs], settings4, 16)
        check("an incomplete remote translation raises", False)
    except T.TranslationError as exc:
        msg = str(exc)
        check("an incomplete remote translation raises", True)
        check("the message names the translator that actually ran",
              "claude-sonnet-5" in msg, msg)
        check("and does not blame a local model that was never in use",
              "If you are using a local model" not in msg, msg)

    # anthropic_model and openai_model are free-text fields in Settings, so the
    # advice above is keyed off settings.translator rather than sniffing the
    # label string — a user is free to name their model "local model special"
    # and the advice must still be the remote one.
    settings5 = Settings()
    settings5.translator = "anthropic"
    settings5.anthropic_key = "sk-ant-whatever"
    settings5.anthropic_model = "local model special"
    T._call_anthropic = lambda *a, **kw: ""
    try:
        T.translate([dict(x) for x in api_segs], settings5, 16)
        check("an adversarial model name still raises", False)
    except T.TranslationError as exc:
        msg = str(exc)
        check("the advice is keyed off the backend, not the model's own name",
              "If you are using a local model" not in msg, msg)


# ============================================================ 3. HTTP layer
def test_server():
    print("\n[3] Web server")
    from fastapi.testclient import TestClient
    from app.server import app

    client = TestClient(app)

    r = client.get("/")
    check("home page loads", r.status_code == 200 and "Dubbing Studio" in r.text)

    r = client.get("/api/state")
    check("state endpoint responds", r.status_code == 200)
    body = r.json()
    check("state includes machine info", "engine" in body["machine"], body["machine"]["engine"])
    check("state includes voices", any(v["id"] == "bf_emma" for v in body["voices"]))
    check("state includes glossaries", "crochet_us" in body["glossaries"])

    r = client.post("/api/settings", json={"data": {"voice": "bf_lily", "speed": 1.1,
                                                    "write_srt": True}})
    check("settings save", r.status_code == 200 and r.json()["voice"] == "bf_lily")
    check("numeric settings coerce", r.json()["speed"] == 1.1)
    check("boolean settings coerce", r.json()["write_srt"] is True)

    r = client.post("/api/settings", json={"data": {"speed": "not-a-number"}})
    check("bad setting value is ignored, not fatal", r.status_code == 200)

    client.post("/api/settings", json={"data": {"voice": "bf_emma", "write_srt": False}})

    r = client.post("/api/job", json={"url": ""})
    check("empty link is rejected", r.status_code == 400)
    r = client.post("/api/job", json={"url": "not a url"})
    check("non-link is rejected", r.status_code == 400)

    # One box, two kinds of answer. "That doesn't look like a web link" was the
    # only thing this endpoint could say, and it is the wrong thing to say about
    # three of the four ways a file can be unusable.
    from app import pipeline as _pl
    from app import server as _srv
    tiny = SCRATCH / "server-clip.mp4"
    if not tiny.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-f", "lavfi", "-i", "sine=frequency=440",
                        "-t", "1", "-c:v", "libx264", "-preset", "ultrafast",
                        "-shortest", str(tiny)], check=True)

    r = client.post("/api/job", json={"url": str(SCRATCH / "nope.mp4")})
    check("a file that isn't there says so, and names it",
          r.status_code == 400 and "no file at" in r.json()["detail"],
          str(r.json())[:90])
    r = client.post("/api/job", json={"url": "youtube.com/watch?v=x"})
    check("a link with the https:// left off is told what's missing",
          r.status_code == 400 and "https://" in r.json()["detail"],
          str(r.json())[:90])
    r = client.post("/api/job", json={"url": str(SCRATCH)})
    check("a folder is refused as a folder",
          r.status_code == 400 and "folder" in r.json()["detail"],
          str(r.json())[:90])

    # Accepted, and normalised on the way in: the file chooser hands over a
    # tidy path but a person types "~/…", and those have to be one job.
    real_submit = _pl.runner.submit
    seen = []
    # The canonical path, not the one that happens to be typed: the scratch
    # folder lives under /var, which macOS symlinks to /private/var, and
    # resolving that is the whole point — two spellings of one video have to be
    # one job over one folder rather than two racing over it.
    settled = str(tiny.resolve())

    def fake_submit(source, settings, preview=False):
        seen.append(source)
        return _pl.Job(id="stub", url=source, local=_pl.download.is_local_source(source))

    _pl.runner.submit = fake_submit
    try:
        r = client.post("/api/job", json={"url": str(tiny)})
        check("a real video file is accepted", r.status_code == 200, str(r.json())[:90])
        check("and reaches the runner as itself", seen[-1:] == [settled], str(seen[-1:]))
        client.post("/api/job", json={"url": f"file://{tiny}"})
        check("a file:// URL arrives as the same path", seen[-1] == settled, seen[-1])
        client.post("/api/job", json={"url": f'"{tiny}"'})
        check("and so does one pasted with quotes round it", seen[-1] == settled, seen[-1])
    finally:
        _pl.runner.submit = real_submit

    # The file chooser, without a dialog: the endpoint's job is to run the
    # script and read what comes back, and both answers it has to cope with —
    # a path, and somebody pressing Cancel — can be produced without one.
    real_script = _srv._CHOOSE_FILE
    try:
        _srv._CHOOSE_FILE = 'return "/Users/x/Movies/a b.mp4"'
        r = client.post("/api/choose-file")
        check("the file chooser hands back the path it was given",
              r.status_code == 200 and r.json()["path"] == "/Users/x/Movies/a b.mp4",
              str(r.json()))
        _srv._CHOOSE_FILE = "error number -128"
        r = client.post("/api/choose-file")
        check("and Cancel is an answer, not a failure",
              r.status_code == 200 and r.json()["path"] == "", str(r.json()))
        _srv._CHOOSE_FILE = 'error "no such thing" number -1728'
        r = client.post("/api/choose-file")
        check("a chooser that genuinely fails says so", r.status_code == 500,
              str(r.json())[:90])
    finally:
        _srv._CHOOSE_FILE = real_script

    # A page on some other site can reach the endpoints that take no body.
    r = client.post("/api/choose-file", headers={"Origin": "https://evil.example"})
    check("and no other web page can open it", r.status_code == 403, str(r.json())[:60])

    r = client.get("/api/doctor")
    check("doctor endpoint responds", r.status_code == 200 and "checks" in r.json())

    # An old yt-dlp is the commonest reason a video describes itself happily and
    # then refuses to download, and the check used to say only that it existed.
    import datetime as _dt
    names = [c["name"] for c in r.json()["checks"]]
    check("the setup check names the yt-dlp version",
          any(n.startswith("yt-dlp — ") for n in names), str(names[:3]))
    # The advice a 403 now gives is "press Update yt-dlp under Setup check", so
    # the row has to carry that action and the route behind it has to exist.
    # Both halves are checked because the advice is only as good as the button:
    # naming a control that isn't there is worse than the vague message it
    # replaced. The endpoint is not *called* here — it shells out to pip and
    # would rewrite the environment the suite is running in.
    ytdlp_row = next((c for c in r.json()["checks"]
                      if str(c.get("name", "")).startswith("yt-dlp")), {})
    check("the yt-dlp row offers the fix the app can perform itself",
          ytdlp_row.get("action") == "update-ytdlp", str(ytdlp_row)[:120])
    routes = {getattr(rt, "path", "") for rt in _srv.app.routes}
    check("and the route that performs it is registered",
          "/api/ytdlp/update" in routes)
    check("the stale hint names that control, not the full reinstall",
          "Update yt-dlp" in ytdlp_row.get("hint", ""), ytdlp_row.get("hint", "")[:80])

    real_run = _srv.subprocess.run

    class _Out:
        def __init__(self, s): self.stdout = s

    def dated(days_old):
        d = _dt.date.today() - _dt.timedelta(days=days_old)
        return lambda *a, **k: _Out(f"{d.year}.{d.month:02d}.{d.day:02d}\n")

    try:
        _srv.subprocess.run = dated(5)
        check("a recent yt-dlp is not flagged", _srv._ytdlp_age()[0] is False)
        _srv.subprocess.run = dated(200)
        stale, shown = _srv._ytdlp_age()
        check("one from months ago is", stale is True, shown)
        # Genuinely undateable: a version that is not three numbers. The fixture
        # here used to be "2026.07.04.dev0+abc", which splits to 2026/07/04 and
        # dates perfectly well — it only looked "left alone" because it happened
        # to fall inside the staleness window, and tightening that window turned
        # a test that had been asserting nothing into a failure.
        _srv.subprocess.run = lambda *a, **k: _Out("stable@2026.07.04\n")
        check("a build with an undateable version is left alone",
              _srv._ytdlp_age() == (False, "stable@2026.07.04"))
        _srv.subprocess.run = lambda *a, **k: _Out("2020.01.01.dev0+abc\n")
        check("but a dev build that does carry a date is dated, not waved through",
              _srv._ytdlp_age() == (True, "2020.01.01.dev0+abc"))
        def boom(*a, **k): raise OSError("no yt-dlp")
        _srv.subprocess.run = boom
        check("and a missing yt-dlp is not reported as stale",
              _srv._ytdlp_age() == (False, ""))
    finally:
        _srv.subprocess.run = real_run

    # Open folder sends the path the panel is holding, and until the first state
    # arrives that is the empty string the store starts with. An empty string is
    # a key that exists, so a get() default never fired and Path("") is ".",
    # which opened whatever folder the app was running from.
    opened = []
    _srv.subprocess.run = lambda cmd, **k: opened.append([str(c) for c in cmd])
    try:
        from app.config import OUTPUT_DIR as _OUT
        for body in ({}, {"path": ""}, {"path": None}):
            opened.clear()
            client.post("/api/reveal", json=body)
            check(f"reveal falls back to the output folder for {body}",
                  opened and str(_OUT) in opened[-1], str(opened))
            check(f"and never opens the working directory for {body}",
                  opened and "." not in opened[-1], str(opened))
        opened.clear()
        client.post("/api/reveal", json={"path": str(_OUT)})
        check("a path the panel does send is the one opened",
              opened and str(_OUT) in opened[-1], str(opened))
    finally:
        _srv.subprocess.run = real_run

    # What a failed download actually says to the person reading it. The failed
    # panel shows this verbatim, so anything yt-dlp phrases for itself ends up
    # in front of someone who cannot act on it.
    from app.steps.download import _friendly
    dead = _friendly("ERROR: [generic] nope: Unable to download webpage: HTTP Error "
                     "404: File not found (caused by <HTTPError 404: File not found>)")
    check("a dead link is explained in plain English",
          "typed correctly" in dead and "ERROR" not in dead, dead)

    # A pinned format that cannot be fetched must not end the run. The ids come
    # from probe(), which is its own negotiation, so YouTube can list a stream to
    # the lookup and refuse it to the fetch — seen three different ways on one
    # video inside six minutes: a 403, the ids missing from the second list, and
    # the tv client refusing outright.
    import app.steps.download as _dl
    for label, failure in (
            ("a 403 on the pinned stream",
             "ERROR: unable to download video data: HTTP Error 403: Forbidden"),
            ("the ids not being offered the second time",
             "ERROR: [youtube] x: Requested format is not available."),
            ("the pinned stream not being offered at all",
             "ERROR: unable to download video data: HTTP Error 403: Forbidden")):
        seen = []
        real_stream, real_probe = _dl.stream, _dl.probe
        real_media = _dl._looks_like_media
        box = SCRATCH / f"reselect-{abs(hash(label))}"
        box.mkdir(parents=True, exist_ok=True)
        for leftover in box.glob("source.*"):
            leftover.unlink()

        def fake_stream(cmd, on_line=None, tail_lines=30, _seen=seen, _box=box):
            _seen.append(cmd[cmd.index("-f") + 1])
            if len(_seen) == 1:
                return 1, failure
            (_box / "source.mp4").write_bytes(b"\x00" * 2048)
            return 0, ""

        _dl.stream = fake_stream
        _dl.probe = lambda url, cookies_from="": {
            "title": "t", "duration": 1.0, "uploader": "", "thumbnail": "",
            "formats": [
                {"format_id": "137", "vcodec": "avc1.640028", "acodec": "none",
                 "height": 1080, "ext": "mp4", "filesize": 1024},
                {"format_id": "140", "vcodec": "none", "acodec": "mp4a.40.2",
                 "ext": "m4a", "filesize": 256}],
            "is_live": False}
        _dl._looks_like_media = lambda p: True
        try:
            got, _ = _dl.download("https://example.test/v", box, "best")
            check(f"a second format is asked for after {label}", len(seen) == 2, str(seen))
            check(f"and the retry drops the pinned ids after {label}",
                  len(seen) == 2 and seen[1] == _dl.format_selector("best"), str(seen[-1:]))
            check(f"and the video is returned rather than the run failing"
                  f" after {label}", got.name == "source.mp4", str(got))
        except Exception as exc:                                 # noqa: BLE001
            check(f"a refused format does not end the run after {label}", False, repr(exc))
        finally:
            _dl.stream, _dl.probe = real_stream, real_probe
            _dl._looks_like_media = real_media

    # A declined exchange is not a format problem. Asking again immediately asks
    # a rate-limited host to serve twice as fast, so this one fails on the spot,
    # and what it says has to be something a person can act on — yt-dlp's own
    # words tell them to reload a page that does not exist.
    reload_said = _friendly("ERROR: [youtube] x: The page needs to be reloaded.")
    check("a declined exchange is explained as a wait, not a reload",
          "reload" not in reload_said.lower() and "Try again" in reload_said,
          reload_said)
    check("and it says that repeating it straight away is worse",
          "longer" in reload_said, reload_said)
    seen3, real_stream = [], _dl.stream
    def once(cmd, on_line=None, tail_lines=30):
        seen3.append(cmd[cmd.index("-f") + 1])
        return 1, "ERROR: [youtube] x: The page needs to be reloaded."
    _dl.stream = once
    real_probe = _dl.probe
    _dl.probe = lambda url, cookies_from="": {
        "title": "t", "duration": 1.0, "uploader": "", "thumbnail": "",
        "formats": [
            {"format_id": "137", "vcodec": "avc1.640028", "acodec": "none",
             "height": 1080, "ext": "mp4", "filesize": 1024},
            {"format_id": "140", "vcodec": "none", "acodec": "mp4a.40.2",
             "ext": "m4a", "filesize": 256}],
        "is_live": False}
    try:
        box = SCRATCH / "reselect-declined"; box.mkdir(parents=True, exist_ok=True)
        try:
            _dl.download("https://example.test/v", box, "best")
            check("a declined exchange still fails", False, "no error raised")
        except _dl.DownloadError as exc:
            check("a declined exchange still fails", "Try again" in str(exc), str(exc)[:60])
        check("and is asked exactly once, not twice", len(seen3) == 1, str(seen3))
    finally:
        _dl.stream, _dl.probe = real_stream, real_probe

    # A failure a different format cannot fix must still fail, and at once.
    seen2, real_stream = [], _dl.stream
    def one_shot(cmd, on_line=None, tail_lines=30):
        seen2.append(cmd[cmd.index("-f") + 1])
        return 1, "ERROR: [youtube] x: Private video. Sign in if you've been granted access."
    _dl.stream = one_shot
    real_probe = _dl.probe
    _dl.probe = lambda url, cookies_from="": {
        "title": "t", "duration": 1.0, "uploader": "", "thumbnail": "",
            "formats": [
                {"format_id": "137", "vcodec": "avc1.640028", "acodec": "none",
                 "height": 1080, "ext": "mp4", "filesize": 1024},
                {"format_id": "140", "vcodec": "none", "acodec": "mp4a.40.2",
                 "ext": "m4a", "filesize": 256}],
        "is_live": False}
    try:
        box = SCRATCH / "reselect-private"; box.mkdir(parents=True, exist_ok=True)
        try:
            _dl.download("https://example.test/v", box, "best")
            check("a private video still fails", False, "no error raised")
        except _dl.DownloadError as exc:
            check("a private video still fails", "private" in str(exc).lower(), str(exc))
        check("and is not retried with another format", len(seen2) == 1, str(seen2))
    finally:
        _dl.stream, _dl.probe = real_stream, real_probe
    # Two files have to agree about which model a given Mac gets: config.py
    # suggests it and the setup check reports it, while Install.command is what
    # actually downloads it. A comment asks them to be kept in step; this is what
    # notices when they aren't.
    import re as _re
    from app.config import suggest_ollama_model
    ladder = _re.findall(r"RAM_GB >= (\d+) \)\); then LADDER=\(([^)]+)\)",
                         (ROOT / "Install.command").read_text())
    check("the installer names its memory tiers", len(ladder) >= 2, str(ladder))
    for ram, models in ladder:
        first = models.split()[0]
        check(f"at {ram} GB the installer fetches what the app expects",
              first == suggest_ollama_model(int(ram)),
              f"installs {first}, app wants {suggest_ollama_model(int(ram))}")
    # Each tier falls back to something smaller, so one failed download is not
    # the end of it.
    check("every tier but the smallest has something to fall back to",
          all(len(m.split()) > 1 for _, m in ladder), str(ladder))
    # Capped deliberately: the 32B is a 20 GB download and slower per line.
    check("no tier defaults to the 32B model",
          all("32b" not in suggest_ollama_model(r) for r in (8, 16, 24, 48, 64, 128)))

    # Nobody outside this code knows model tags exist, and a machine whose
    # installer pulled a different size used to die on a 404 mid-translation.
    from app.backends import translate as T
    real_installed = T.installed_models
    T.installed_models = lambda host="": [{"name": "qwen3:8b", "size": 5_000_000_000},
                                          {"name": "llama3:8b", "size": 4_000_000_000}]
    try:
        picked, note_ = T.usable_model("qwen3:8b")
        check("an installed model is used as-is", picked == "qwen3:8b" and not note_)
        picked, note_ = T.usable_model("qwen3:32b")
        check("a missing size falls back within the same family",
              picked == "qwen3:8b" and "isn't installed" in note_, picked)
        picked, note_ = T.usable_model("mistral:7b")
        check("a missing family falls back to whatever is there",
              picked in ("qwen3:8b", "llama3:8b") and note_, picked)
        T.installed_models = lambda host="": []
        picked, note_ = T.usable_model("qwen3:8b")
        check("with nothing installed it asks for what it wanted",
              picked == "qwen3:8b" and not note_)
    finally:
        T.installed_models = real_installed

    # A substituted model is off the finished run's report entirely. It began as
    # a warning there, was softened to information, and was still the wrong
    # place for it: which local models are installed is a property of this Mac
    # rather than of this video, it does not change from one dub to the next,
    # and it was restated on every finished job — twice, because the translation
    # and the terminology pass each resolve a backend and each recorded it. A
    # note is for what this run did differently. Setup check is the one place it
    # is said now, on the row named for the model, where the fix is actionable.
    import inspect as _inspect
    from app.config import Settings as _Settings, Machine as _Machine
    T.installed_models = lambda host="": [{"name": "qwen3:8b", "size": 5_000_000_000}]
    try:
        settings6 = _Settings()
        settings6.ollama_model = "qwen3:32b"
        recorded = []
        progress = type("P", (), {})()
        progress.note = recorded.append
        _, label6 = T.backend_for(settings6, 16)
        check("a substituted model is not recorded on the finished job at all",
              recorded == [], str(recorded))
        check("and there is no progress callback left to record it through",
              "progress" not in _inspect.signature(T.backend_for).parameters,
              str(_inspect.signature(T.backend_for)))
        check("while the substitute is still the model actually used",
              "qwen3:8b" in label6, label6)

        real_load, real_detect = _srv.Settings.load, _srv.detect_machine
        _srv.Settings.load = staticmethod(lambda: settings6)
        _srv.detect_machine = lambda: _Machine(
            system="Darwin", arch="arm64", apple_silicon=False, ram_gb=16,
            has_mlx=False, has_ffmpeg=True, has_ytdlp=True, has_ollama=True,
            av1_ok=True)
        try:
            report = _srv.doctor()
        finally:
            _srv.Settings.load, _srv.detect_machine = real_load, real_detect
        model_check = next(c for c in report["checks"]
                           if c["name"].startswith("Translation model"))
        check("Setup check still shows a green dot for a working substitute",
              model_check["ok"] is True, model_check)
        check("but names the substitution instead of just going green",
              "qwen3:32b" in model_check.get("note", ""), model_check)
        check("without also putting it in the fix-it hint a green row hides",
              model_check["hint"] == "", model_check)
    finally:
        T.installed_models = real_installed

    check("an offline machine is told so",
          "internet connection" in _friendly("ERROR: unable to open: "
                                             "nodename nor servname provided"))
    # Said honestly: by the time this reaches anyone the download has already
    # been retried as several different player clients, so "try again in a
    # minute" is advice that has been taken on their behalf and failed.
    forbidden = _friendly("ERROR: unable to download: HTTP Error 403: Forbidden")
    check("a 403 says what to actually do about it",
          "Sign in as" in forbidden and "every attempt" in forbidden, forbidden[:80])
    check("and no longer claims it is temporary", "temporary" not in forbidden.lower())
    # The advice that was confidently wrong. A public video which plays fine in a
    # browser was refused with a 403, and this message sent the user off to
    # configure browser cookies and suggested the video might be private or
    # members-only. It was none of those: the copy of yt-dlp asking was 51 days
    # old, every player client it knew had been retired, and the one that still
    # served the video was added after it was built. When the copy is old enough
    # to be the cause, it is named first — and it is a one-click fix, which the
    # cookie advice is not.
    import app.steps.download as _dlm
    _real_version = _dlm.ytdlp_version
    try:
        _dlm.ytdlp_version = lambda: "2020.01.01"
        stale_403 = _dlm._friendly("ERROR: unable to download: HTTP Error 403: Forbidden")
        check("an old yt-dlp is named first when a 403 arrives",
              stale_403.startswith("This copy of yt-dlp is")
              and "Update yt-dlp" in stale_403, stale_403[:70])
        check("and the sign-in advice still follows it, rather than being lost",
              "Sign in as" in stale_403)
        stale_fmt = _dlm._friendly("ERROR: Requested format is not available")
        check("the same for a listing that offers nothing, the other stale symptom",
              stale_fmt.startswith("This copy of yt-dlp is"), stale_fmt[:70])
        # An unknown version is not an old one, and guessing would put a wrong
        # first sentence on every failure a distribution build ever produced.
        _dlm.ytdlp_version = lambda: "stable@2026.07.04"
        undated = _dlm._friendly("ERROR: unable to download: HTTP Error 403: Forbidden")
        check("a yt-dlp whose version cannot be dated is not blamed",
              not undated.startswith("This copy of yt-dlp is"), undated[:70])
        _dlm.ytdlp_version = lambda: ""
        silent = _dlm._friendly("ERROR: unable to download: HTTP Error 403: Forbidden")
        check("nor is one that will not say its version at all",
              not silent.startswith("This copy of yt-dlp is"), silent[:70])
    finally:
        _dlm.ytdlp_version = _real_version
    odd = _friendly("ERROR: [youtube] abc: Something nobody anticipated "
                    "(caused by <SomeError: blah>)")
    check("anything unrecognised is passed on, minus the scaffolding",
          odd == "Something nobody anticipated", odd)

    r = client.get("/api/job/nope/video")
    check("missing job video 404s", r.status_code == 404)

    # The preset is read back off the switches rather than stored as an
    # assertion about them. It used to be the latter, so choosing Balanced and
    # then turning separation off left it saying "balanced" and the segmented
    # control in the interface claiming a preset the settings no longer were.
    from app.config import PRESETS, Settings
    for name in PRESETS:
        check(f"{name} settings identify as {name}",
              Settings().apply_preset(name).matching_preset() == name)
    # Balanced without separation is exactly Fast, and is named as such rather
    # than called custom — the presets are points in one space, not labels.
    off = Settings().apply_preset("balanced")
    off.separate_audio = False
    check("balanced minus separation is recognised as fast",
          off.matching_preset() == "fast", off.matching_preset())
    # A combination no preset describes.
    mixed = Settings().apply_preset("balanced")
    mixed.asr_model = "parakeet"
    check("a mix no preset describes reads as custom",
          mixed.matching_preset() == "custom", mixed.matching_preset())

    # How many people are speaking is a fact about the video, not a
    # quality-versus-cost setting, so it belongs to neither preset.
    for name in PRESETS:
        check(f"{name} does not dictate who is speaking", "diarize" not in PRESETS[name])
    spoke = Settings().apply_preset("best")
    spoke.diarize = True
    check("saying several people speak does not change the preset",
          spoke.matching_preset() == "best", spoke.matching_preset())

    r = client.post("/api/settings", json={"data": {"preset": "balanced"}})
    check("picking a preset saves it", r.json()["preset"] == "balanced")
    r = client.post("/api/settings", json={"data": {"asr_model": "parakeet"}})
    check("an off-preset switch flips the saved preset to custom",
          r.json()["preset"] == "custom", r.json()["preset"])
    r = client.post("/api/settings", json={"data": {"asr_model": "whisper"}})
    check("and putting it back returns to the named preset",
          r.json()["preset"] == "balanced", r.json()["preset"])
    r = client.post("/api/settings", json={"data": {"diarize": True}})
    check("but saying several people speak leaves the preset alone",
          r.json()["preset"] == "balanced", r.json()["preset"])

    # --- values the app cannot honour, on the way out as well as the way in
    #
    # A language with no voice behind it is translated into and then read aloud in
    # English, which sounds fluent and is nonsense; an unrecognised glossary
    # contributes no terms while still claiming to. Both were caught when a saved
    # file was read and nowhere else, so this endpoint assigned onto a live
    # instance, wrote the value to disk and echoed it back — and only the next
    # load put it right. Normalisation belongs to the model, so every write path
    # is covered by construction rather than by remembering.
    from app.config import SETTINGS_FILE
    r = client.post("/api/settings", json={"data": {"target_language": "French"}})
    check("a language no voice can speak is not echoed back as accepted",
          r.json()["target_language"] == Settings().target_language,
          r.json()["target_language"])
    check("and is not what lands in settings.json either",
          json.loads(SETTINGS_FILE.read_text())["target_language"]
          == Settings().target_language,
          json.loads(SETTINGS_FILE.read_text())["target_language"])
    r = client.post("/api/settings", json={"data": {"glossary": "crochet_atlantis"}})
    check("nor is a glossary this build does not have",
          r.json()["glossary"] == "none", r.json()["glossary"])
    # The next write path added — another endpoint, a preset, a command line —
    # gets this without knowing it exists, because save() is the one gate.
    straight = Settings.load()
    straight.target_language, straight.glossary = "Klingon", "not_a_glossary"
    straight.save()
    check("a value set straight onto the dataclass is corrected by the save",
          straight.target_language == Settings().target_language
          and straight.glossary == "none",
          f"{straight.target_language} {straight.glossary}")

    # --- what the app ships with, and putting it back
    # The settings panel marks every field the user has moved away from, which
    # needs the shipped values as well as the saved ones.
    from dataclasses import asdict as _asdict
    body = client.get("/api/state").json()
    check("state carries the shipped defaults alongside the saved settings",
          body["settings_defaults"] == _asdict(Settings()),
          str(len(body.get("settings_defaults", {}))) + " fields")
    check("and they are the defaults, not this user's values",
          body["settings_defaults"]["diarize"] is False
          and body["settings"]["diarize"] is True)

    client.post("/api/settings", json={"data": {
        "voice": "bf_lily", "speed": 1.4, "keep_video_quality": "720",
        "anthropic_key": "sk-ant-RESET-CANARY", "asr_model": "parakeet"}})

    # The panel resets one tab at a time, so a subset has to be resettable
    # without disturbing the rest.
    r = client.post("/api/settings/reset", json={"keys": ["voice", "speed"]})
    got = r.json()
    check("a named subset goes back to its defaults",
          got["voice"] == Settings().voice and got["speed"] == Settings().speed,
          f'{got["voice"]} {got["speed"]}')
    check("and leaves everything else exactly where it was",
          got["keep_video_quality"] == "720", got["keep_video_quality"])
    # Parakeet is still on, so the preset is still not one of the named ones.
    check("a partial reset still reads the preset back off the switches",
          got["preset"] == "custom", got["preset"])

    r = client.post("/api/settings/reset", json={})
    got = r.json()
    check("naming nothing resets everything",
          got["keep_video_quality"] == Settings().keep_video_quality
          and got["asr_model"] == Settings().asr_model
          and got["diarize"] is False, str(got["keep_video_quality"]))
    check("and the preset follows the switches back",
          got["preset"] == Settings().preset, got["preset"])
    # A key is pasted in from an account somewhere else and cannot be read back
    # out of this app. Everything else here can be chosen again in seconds, so
    # "reset" must not be able to destroy one by implication.
    check("a blanket reset does not throw away an API key",
          got["anthropic_key"] == "sk-ant-RESET-CANARY", got["anthropic_key"][:12])
    r = client.post("/api/settings/reset", json={"keys": ["anthropic_key"]})
    check("but asking for the key by name does clear it",
          r.json()["anthropic_key"] == "", r.json()["anthropic_key"])

    check("the reset is saved, not merely reported",
          client.get("/api/state").json()["settings"]["voice"] == Settings().voice)
    r = client.post("/api/settings/reset", json={"keys": ["not_a_setting"]})
    check("a field that does not exist is refused rather than ignored",
          r.status_code == 400, str(r.status_code))
    r = client.post("/api/settings/reset", json={"keys": []})
    check("and naming none of them resets none of them",
          r.status_code == 200
          and r.json() == client.get("/api/state").json()["settings"])

    # --- what a blanket reset is allowed to destroy
    #
    # Both confirm dialogs promise that API keys and finished videos survive, and
    # the promise has to be kept here rather than in the wording in front of it. A
    # glossary of someone's own terms is unbounded text typed by hand with no undo
    # anywhere in the app, which is not something that "can be chosen again in
    # seconds". Checked as the rule: everything moved off its default, then a
    # blanket reset, and exactly the spared fields are the ones still moved.
    moved = {"voice": "bf_lily", "speed": 1.4, "keep_video_quality": "720",
             "asr_model": "parakeet", "write_srt": True, "diarize": True,
             "translator": "openai", "keep_awake": False,
             "anthropic_key": "sk-ant-SPARE-CANARY", "openai_key": "sk-oai-SPARE-CANARY",
             "custom_glossary": "punto raso -> slip stitch"}
    client.post("/api/settings", json={"data": dict(moved)})
    got = client.post("/api/settings/reset", json={}).json()
    survived = {k for k, v in moved.items() if got[k] == v}
    check("a blanket reset spares exactly the settings that cannot be re-chosen",
          survived == set(Settings.KEEP_ON_RESET), str(sorted(survived)))
    check("the hand-typed glossary is one of them",
          got["custom_glossary"] == moved["custom_glossary"], got["custom_glossary"])
    check("and it is spared on disk, not merely in the reply",
          client.get("/api/state").json()["settings"]["custom_glossary"]
          == moved["custom_glossary"])
    # Naming one is the escape hatch that keeps a deliberate wipe possible, and it
    # is the same escape hatch the keys already have.
    for key in Settings.KEEP_ON_RESET:
        client.post("/api/settings", json={"data": {key: moved[key]}})
        got = client.post("/api/settings/reset", json={"keys": [key]}).json()
        check(f"but asking for {key} by name still clears it",
              got[key] == Settings.defaults()[key], repr(got[key])[:24])

    # keep_awake acts on the assertion being held right now, so a reset has to
    # go through the same path a save does rather than wait for the next job.
    from app import server as _srv3
    awoken = []
    real_sync = _srv3.runner.sync_keep_awake
    _srv3.runner.sync_keep_awake = lambda wanted: (awoken.append(wanted), False)[1]
    try:
        client.post("/api/settings", json={"data": {"keep_awake": False}})
        client.post("/api/settings/reset", json={"keys": ["keep_awake"]})
    finally:
        _srv3.runner.sync_keep_awake = real_sync
    check("a reset applies keep-awake the same way a save does",
          awoken == [False, True], str(awoken))

    # --- a settings file written by an earlier build
    #
    # settings.json outlives the build that wrote it, and this one names a setting
    # that no longer exists. Unknown keys are dropped before the constructor sees
    # them, so an old file loads with the fields it still shares rather than
    # raising and losing every one of them.
    from app.config import SETTINGS_FILE as _SF
    saved = _SF.read_text() if _SF.exists() else None
    try:
        _SF.write_text(json.dumps({"voice": "bm_daniel", "speed": 1.1,
                                   "expected_speakers": 4, "gone_entirely": True}))
        revived = Settings.load()
        check("a settings file naming a setting since removed still loads",
              revived.voice == "bm_daniel" and revived.speed == 1.1,
              f"{revived.voice} {revived.speed}")
        check("and the setting it named is not revived along with it",
              not hasattr(revived, "expected_speakers"))
        check("the app is still answerable while holding one",
              client.get("/api/state").status_code == 200)
    finally:
        if saved is None:
            _SF.unlink(missing_ok=True)
        else:
            _SF.write_text(saved)


# ======================================================= 4. full pipeline run
def test_end_to_end():
    print("\n[4] Full pipeline (real ASR, real TTS, real mux)")
    from app import pipeline
    from app.backends import translate as T
    from app.config import Settings, JOBS
    from app.steps import download as dl

    work = WORK / "e2e"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"

    if not clip.exists():
        audio = speech_wav(work / "speech.wav", seconds=75.0)
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15",
                        "-i", str(audio), "-map", "0:v", "-map", "1:a", "-t", "75",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)
    check("test clip exists", clip.exists())

    pipeline.download.probe, pipeline.download.download = stub_download(
        clip, "Pipeline Test Clip", 75.0, replace=True)

    # Stand in for the LLM, returning plausible English of a sensible length.
    PHRASES = ["Right, let's carry on with the next round.",
               "We chain three and turn the work.",
               "Now single crochet into the same stitch.",
               "Keep the tension loose so it drapes nicely.",
               "One, two, three. Then we skip one.",
               "You can see how the pattern builds up here."]

    def fake_llm(prompt, model=None, host=None, key=None, **_):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|{PHRASES[i % len(PHRASES)]}" for i in ids)

    T._call_ollama = fake_llm

    settings = Settings()
    settings.translator = "ollama"
    settings.voice = "bf_emma"
    settings.audio_mode = "replace"
    settings.write_srt = True

    events = []
    pipeline.runner.subscribe(lambda p: events.append((p["stage"], p["overall"])))

    job = pipeline.runner.submit("https://example.com/fake", settings)
    print("      running…", end="", flush=True)
    t0 = time.time()
    while job.status in ("queued", "running") and time.time() - t0 < 900:
        time.sleep(2)
        print(".", end="", flush=True)
    print()

    check("job completed", job.status == "done", f"{job.status}: {job.error}")
    if job.status != "done":
        log = JOBS / job.id / "error.log"
        if log.exists():
            print(log.read_text()[-1500:])
        return

    out = Path(job.output)
    check("output file was written", out.exists(), out.name)
    check("output has sensible size", out.stat().st_size > 50_000, f"{out.stat().st_size} bytes")

    s = job.stats
    check("video frames were preserved", s.get("frames_match") is True,
          f"{s.get('source_frames')} -> {s.get('output_frames')}")
    check("audio and video lengths agree", s.get("drift_seconds", 99) < 1.0,
          f"{s.get('drift_seconds')}s")
    check("lines were spoken", s.get("lines_spoken", 0) > 0, str(s.get("lines_spoken")))
    check("timing drift stayed small", s.get("max_drift", 99) < 2.0, f"{s.get('max_drift')}s")
    # "Voices used" used to hold the compute engine's name (identical to the
    # separate "Engine" row below it) rather than which voice actually spoke.
    check("voices used names the voice, not the compute engine",
          s.get("voices") == "Emma", str(s.get("voices")))
    check("the engine is still reported, just under its own row",
          s.get("engine") in ("Apple GPU (MLX)", "Portable (CPU)"), str(s.get("engine")))
    # The undubbed-time figure used to be labelled "silent stretches", as if a
    # gap in an otherwise-present track were the fault, when the raw dub holds
    # only the spoken lines and everything between them is silence by
    # construction.
    check("undubbed time is reported under its new name",
          "no_line_seconds" in s and "no_line_share" in s, str(sorted(s)))
    # What produced this file, kept with the file. Settings are free to change
    # between one run and the next, so the panel cannot read them off the app.
    used = s.get("settings") or {}
    check("the finished job recorded the settings it ran under",
          used.get("voice") == "bf_emma" and used.get("write_srt") is True
          and used.get("translator") == "ollama", str(used)[:90])
    check("and no API key went into that record",
          not any(k in used for k in Settings.SECRET_KEYS), str(sorted(used)))

    stages = [e[0] for e in events]
    for want in ("download", "transcribe", "translate", "synthesize", "assemble", "finish"):
        check(f"stage reported: {want}", want in stages)
    moving = [e for e in events if e[1] > 0]
    backwards = [(a[0], a[1], b[1]) for a, b in zip(moving, moving[1:]) if b[1] < a[1] - 0.001]
    check("progress only moves forward", not backwards,
          "; ".join(f"{stage} {was:.3f}->{now:.3f}" for stage, was, now in backwards[:3]))
    peak = max(e[1] for e in moving)
    check("progress reaches 100%", peak >= 0.999, f"{peak:.3f}")

    check("subtitle file was saved", out.with_suffix(".srt").exists())

    # Prove the audio actually contains the English we asked for, using the app's
    # own portable recogniser rather than a separate copy of one.
    from app.backends import asr as asr_backend
    check_wav = work / "check.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out),
                    "-ac", "1", "-ar", "16000", str(check_wav)], check=True)
    heard = [s["text"].strip() for s in asr_backend._transcribe_onnx(check_wav)[:8]]
    joined = " ".join(h for h in heard if h).lower()
    print("      heard back:", (joined[:150] + "…") if joined else "(nothing)")
    check("dubbed audio is the English we synthesised",
          any(w in joined for w in ("crochet", "chain", "round", "stitch", "pattern")),
          joined[:80])

    global E2E_JOB_ID
    E2E_JOB_ID = job.id


# ========================================================== 5. resuming a link
E2E_JOB_ID = ""


def test_resume():
    """Re-running a link must reuse the expensive stages, not redo them.

    This is what makes iterating on a long video bearable, and it is easy to
    break without noticing: the job folder is keyed off the link, so anything
    that makes the key vary per run silently disables resuming altogether.
    """
    print("\n[5] Resuming a repeated link")
    from app import pipeline
    from app.config import Settings

    if not E2E_JOB_ID:
        check("end-to-end job ran first", False, "no job id captured")
        return

    settings = Settings()
    settings.translator = "ollama"
    settings.voice = "bf_emma"
    settings.audio_mode = "replace"

    messages: list[str] = []
    listener = lambda p: messages.append(p["message"])       # noqa: E731
    pipeline.runner.subscribe(listener)

    t0 = time.time()
    job = pipeline.runner.submit("https://example.com/fake", settings)
    check("the same link maps to the same job folder", job.id == E2E_JOB_ID,
          f"{job.id} vs {E2E_JOB_ID}")

    while job.status in ("queued", "running") and time.time() - t0 < 600:
        time.sleep(1)
    pipeline.runner.unsubscribe(listener)

    check("second run completed", job.status == "done", f"{job.status}: {job.error}")
    check("transcription was reused, not redone",
          any("Reusing the transcription" in m for m in messages))
    check("translation was reused, not redone",
          any("Reusing the translation" in m for m in messages))

    # Changing the transcription engine must invalidate the cached transcript
    # rather than quietly handing back the other engine's work.
    from app.pipeline import _fingerprint
    parakeet = _fingerprint("parakeet", True, True)
    whisper = _fingerprint("whisper", True, True)
    check("a different ASR engine invalidates the cache", parakeet != whisper)


# ============================== 6. a preset change re-derives the audio
def _tone(hz: float, seconds: float, rate: int) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _dominant_hz(path: Path) -> float:
    data, rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    spec = np.abs(np.fft.rfft(data))
    return float(np.fft.rfftfreq(len(data), 1 / rate)[int(np.argmax(spec))])


def test_preset_change_reseparates():
    """Running a link on Fast and then on Balanced must not reuse the downmix.

    The job folder is keyed off the link, so the second run finds the first
    run's speech16k.wav — a downmix of the *whole* soundtrack — already in
    place. Guarded by existence alone it was skipped, so Demucs ran and was
    paid for, the report said separation had happened, and the transcript was
    rebuilt from the music-contaminated mix anyway.

    The mix is a 200 Hz tone and the stubbed stem is an 800 Hz one, so which
    audio reached the recogniser is a fact rather than an impression.
    """
    print("\n[6] A preset change re-derives the audio it feeds on")
    from app import pipeline
    from app.backends import translate as T
    from app.config import Settings, JOBS

    MIX_HZ, STEM_HZ, RATE = 200.0, 800.0, 44100
    URL = "https://example.com/preset-change"
    # From a clean folder: a link maps to a stable job id, so leftovers from a
    # previous run of this suite would be reused and the test would be asserting
    # against whatever the last run happened to leave.
    shutil_rmtree(JOBS / pipeline._job_id(URL))
    work = WORK / "preset"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"
    if not clip.exists():
        mix = work / "mix.wav"
        sf.write(mix, _tone(MIX_HZ, 8.0, RATE), RATE)
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-i", str(mix), "-map", "0:v", "-map", "1:a", "-t", "8",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)

    fake_probe, fake_download = stub_download(clip, "Preset Change", 8.0)

    def fake_separate(audio, workdir, prefer_gpu=True, progress=None):
        stems = Path(workdir) / "stems" / "stub"
        stems.mkdir(parents=True, exist_ok=True)
        vocals, bed = stems / "vocals.wav", stems / "no_vocals.wav"
        sf.write(vocals, _tone(STEM_HZ, 8.0, RATE), RATE)
        sf.write(bed, _tone(120.0, 8.0, RATE), RATE)
        if progress:
            progress(1.0, "Separated")
        return vocals, bed

    heard: list[int] = []

    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        hz = int(round(_dominant_hz(Path(audio_wav)) / 10.0) * 10)
        heard.append(hz)
        if progress:
            progress(1.0, "Heard 1 line")
        return [{"start": 0.5, "end": 3.0, "text": f"tono de {hz} hercios"}]

    def fake_llm(prompt, model=None, host=None, key=None, **_):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|A tone, and then some more words to say." for i in ids)

    real_probe = pipeline.download.probe
    real_download = pipeline.download.download
    real_separate = pipeline.separate_backend.separate
    real_transcribe = pipeline.asr_backend.transcribe
    real_prune = pipeline.prune_workdir
    pipeline.download.probe = fake_probe
    pipeline.download.download = fake_download
    pipeline.separate_backend.separate = fake_separate
    pipeline.asr_backend.transcribe = fake_transcribe
    # Pruning after a successful job removes the derived audio, which would hide
    # this bug rather than fix it — and only for jobs that succeed. A run that
    # failed or was cancelled keeps everything, and is then exactly the stale
    # folder the next run reads from, so that is the state to test against.
    pipeline.prune_workdir = lambda workdir: 0
    T._call_ollama = fake_llm

    def run(preset: str):
        s = Settings().apply_preset(preset)
        s.translator = "ollama"
        s.voice = "bf_emma"
        job = pipeline.runner.submit(URL, s)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)
        return job

    try:
        fast = run("fast")
        check("the Fast run completed", fast.status == "done",
              f"{fast.status}: {fast.error}")
        check("Fast transcribed the full mix", heard[:1] == [int(MIX_HZ)], str(heard))

        balanced = run("balanced")
        check("the Balanced run completed", balanced.status == "done",
              f"{balanced.status}: {balanced.error}")
        check("the same link reused the same job folder", balanced.id == fast.id)
        check("Balanced reported that it separated",
              balanced.stats.get("separated") is True, str(balanced.stats.get("separated")))
        check("Balanced transcribed the separated stem, not the stale mix",
              len(heard) == 2 and heard[1] == int(STEM_HZ),
              f"heard {heard}, wanted [{int(MIX_HZ)}, {int(STEM_HZ)}]")

        # The two runs must not be sharing derived audio at all.
        derived = sorted((JOBS / fast.id / "derived").iterdir())
        check("each set of settings got its own derived audio folder",
              len(derived) >= 3, f"{len(derived)} folders")
        rates = {_dominant_hz(p): p for p in
                 (JOBS / fast.id / "derived").glob("*/speech16k.wav")}
        check("both the mix and the stem survive as separate files",
              any(abs(hz - MIX_HZ) < 15 for hz in rates) and
              any(abs(hz - STEM_HZ) < 15 for hz in rates),
              str(sorted(round(h) for h in rates)))
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.separate_backend.separate = real_separate
        pipeline.asr_backend.transcribe = real_transcribe
        pipeline.prune_workdir = real_prune


# ================================ 7. one sample rate across the whole track
def test_mixed_sample_rates():
    """A voice that dies mid-job must not leave two rates in one track.

    sample_rate used to be a single variable reassigned per line and handed to
    assemble() once, for all of them. The mid-job fallback swaps a cloning
    engine for the portable one partway through the loop, so lines synthesised
    before the swap were timed against the rate of the engine that replaced
    them — a timing error, which is the one failure the design exists to stop.
    """
    print("\n[7] One sample rate across the whole track")
    from app import pipeline
    from app.backends import translate as T
    from app.backends import tts as tts_backend
    from app.config import Settings
    from app.steps import align

    # Directly: assemble() is told one rate, so a line that declares another is
    # refused rather than placed against the wrong clock.
    sr = 24000
    lines = [{"start": 0.0, "end": 1.0, "samples": _tone(440, 0.5, sr), "rate": 24000},
             {"start": 2.0, "end": 3.0, "samples": _tone(440, 0.5, sr), "rate": 32000}]
    try:
        align.assemble(lines, 6.0, sr)
        check("assemble refuses a line recorded at another rate", False, "it accepted it")
    except ValueError as exc:
        check("assemble refuses a line recorded at another rate", True, str(exc)[:60])

    # And through the pipeline: an engine that reports 32 kHz and then fails,
    # forcing the real mid-job fallback to the 24 kHz portable voice.
    STARTS = [(1.0, 3.0), (5.0, 7.0)]
    URL = "https://example.com/rate-switch"
    # Cached lines from an earlier run of this suite would be read straight back
    # and the engine never called, so the fallback this test exists to trigger
    # would never happen.
    from app.config import JOBS as JOBS_DIR
    shutil_rmtree(JOBS_DIR / pipeline._job_id(URL))
    work = WORK / "rates"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"
    if not clip.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                        "-map", "0:v", "-map", "1:a", "-t", "10",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)

    class RateSwitchingEngine:
        """32 kHz for the first line, then broken — exactly the fallback path."""
        name = "Stub (32 kHz)"
        sample_rate = 32000

        def __init__(self):
            self.calls = 0

        def say(self, text, voice="", speed=1.0, speaker=0):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("the cloning model fell over")
            return _tone(300, 1.2, 32000), 32000

    fake_probe, fake_download = stub_download(clip, "Rate Switch", 10.0)

    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        if progress:
            progress(1.0, "Heard 2 lines")
        return [{"start": a, "end": b, "text": f"linea {n}"}
                for n, (a, b) in enumerate(STARTS)]

    def fake_llm(prompt, model=None, host=None, key=None, **_):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|This is line number {i} speaking." for i in ids)

    real_probe = pipeline.download.probe
    real_download = pipeline.download.download
    real_transcribe = pipeline.asr_backend.transcribe
    real_make = pipeline.JobRunner._make_engine
    real_prune = pipeline.prune_workdir
    pipeline.download.probe = fake_probe
    pipeline.download.download = fake_download
    pipeline.asr_backend.transcribe = fake_transcribe
    pipeline.JobRunner._make_engine = lambda *a, **k: (RateSwitchingEngine(), False)
    # The assembled track is one of the intermediates a successful job drops.
    pipeline.prune_workdir = lambda workdir: 0
    T._call_ollama = fake_llm

    try:
        s = Settings().apply_preset("fast")
        s.translator = "ollama"
        job = pipeline.runner.submit(URL, s)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)

        check("the job survived the mid-job voice failure", job.status == "done",
              f"{job.status}: {job.error}")
        if job.status != "done":
            return
        check("the fallback was reported",
              any("stopped working" in n and "portable" in n
                  for n in job.stats.get("notes", [])),
              str(job.stats.get("notes")))
        check("the reported voice is unchanged by the engine fallback",
              "stopped working" not in str(job.stats.get("voices", "")),
              str(job.stats.get("voices")))
        check("the track was assembled at the engine's rate",
              job.stats.get("sample_rate") == 32000, str(job.stats.get("sample_rate")))
        check("both lines were spoken", job.stats.get("lines_spoken") == 2,
              str(job.stats.get("lines_spoken")))
        check("nothing was pushed off its mark", job.stats.get("max_drift", 99) == 0.0,
              f"{job.stats.get('max_drift')}s")

        # Both lines must actually be where the original speaker was.
        from app.config import JOBS
        track, rate = sf.read(JOBS / job.id / "dubbed.wav", dtype="float32")
        check("the rendered track is at one rate", rate == 32000, str(rate))

        def loud_between(a, b):
            seg = track[int(a * rate):int(b * rate)]
            return float(np.sqrt((seg ** 2).mean())) if seg.size else 0.0

        for n, (a, b) in enumerate(STARTS):
            check(f"line {n} lands in its slot", loud_between(a, b + 1.0) > 0.02,
                  f"rms {loud_between(a, b + 1.0):.3f}")
        check("the gap between the lines stayed quiet",
              loud_between(3.6, 4.8) < 0.01, f"rms {loud_between(3.6, 4.8):.4f}")
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.asr_backend.transcribe = real_transcribe
        pipeline.JobRunner._make_engine = real_make
        pipeline.prune_workdir = real_prune


# ==================================== 8. a finished job drops its bulk
def test_cleanup():
    """Keep what makes a re-run cheap; drop the rest.

    A job folder held the download, the full-band audio, the stems and the
    rendered track — about 1.5 GB for an hour of video, under Application
    Support where nobody would find it.
    """
    print("\n[8] Clearing up after a finished job")
    from app.pipeline import prune_workdir
    from app.storage import dir_size
    from app.config import JOBS

    work = WORK / "cleanup"
    shutil_rmtree(work)
    (work / "derived" / "abc" / "stems").mkdir(parents=True)
    (work / "lines" / "def").mkdir(parents=True)

    big = np.zeros(240000, dtype=np.float32)
    sf.write(work / "dubbed.wav", big, 24000)
    sf.write(work / "derived" / "abc" / "full.wav", big, 24000)
    sf.write(work / "derived" / "abc" / "stems" / "vocals.wav", big, 24000)
    sf.write(work / "lines" / "def" / "00000.wav", big, 24000)
    (work / "derived" / "abc" / "source.mp4").write_bytes(b"x" * 5000)
    (work / "segments.json").write_text('[{"start": 0}]')
    (work / "translated.json").write_text("[]")
    (work / "subtitles.srt").write_text("1\n")

    before = dir_size(work)
    freed = prune_workdir(work)

    check("something was actually reclaimed", freed > 0, f"{freed} bytes")
    check("the transcript is kept", (work / "segments.json").exists())
    check("the translation is kept", (work / "translated.json").exists())
    check("the subtitles are kept", (work / "subtitles.srt").exists())
    check("the rendered lines are kept",
          (work / "lines" / "def" / "00000.wav").exists())
    for gone in ("dubbed.wav", "derived/abc/full.wav", "derived/abc/source.mp4",
                 "derived/abc/stems/vocals.wav"):
        check(f"{gone} is dropped", not (work / gone).exists())
    check("emptied folders go too", not (work / "derived").exists())
    check("the folder is smaller than it was", dir_size(work) < before,
          f"{dir_size(work)} < {before}")

    # And the real job from section 4 kept what a resume needs.
    if E2E_JOB_ID:
        job_dir = JOBS / E2E_JOB_ID
        check("the finished job kept its transcript",
              (job_dir / "segments.json").exists())
        check("the finished job dropped its working audio",
              not (job_dir / "dubbed.wav").exists())


def shutil_rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


# =============================== 9. reshaping and matching the voices
def test_segments_and_voices():
    """Two ideas taken from other dubbing projects, checked against real numbers."""
    print("\n[9] Joining run-on lines, and matching voices by pitch")
    from app.steps.segments import merge_adjacent
    from app.backends.diarize import median_pitch
    from app.config import Settings

    # Fast dialogue: short lines, almost no gap, one speaker.
    rapid = [{"start": t, "end": t + 0.9, "text": f"line {n}", "speaker": 0}
             for n, t in enumerate([0.0, 1.0, 2.0, 3.0])]
    out = merge_adjacent(rapid)
    check("run-on lines are joined", len(out) == 1, f"{len(rapid)} -> {len(out)}")
    check("the joined line spans the original run",
          abs(out[0]["end"] - 3.9) < 1e-6 and out[0]["start"] == 0.0)
    check("the words are all still there",
          out[0]["text"] == "line 0 line 1 line 2 line 3", out[0]["text"])

    # A different speaker interrupting must break the run.
    mixed = [dict(r) for r in rapid]
    mixed[2]["speaker"] = 1
    check("a change of speaker breaks the run", len(merge_adjacent(mixed)) == 3,
          str(len(merge_adjacent(mixed))))

    # Real pauses — the instructional material the app was built for.
    spaced = [{"start": t, "end": t + 1.5, "text": "x", "speaker": 0}
              for t in (0.0, 3.0, 6.0)]
    check("material with real pauses is left alone",
          len(merge_adjacent(spaced)) == 3, str(len(merge_adjacent(spaced))))
    check("ids are renumbered for the translator",
          [s_["i"] for s_ in merge_adjacent(spaced)] == [0, 1, 2])

    # Pitch, on synthesised tones with a known fundamental.
    sr = 16000
    for hz in (110.0, 220.0):
        t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
        # A couple of harmonics, so it is a plausible voice rather than a sine.
        wave = (0.5 * np.sin(2 * np.pi * hz * t)
                + 0.3 * np.sin(4 * np.pi * hz * t)).astype(np.float32)
        got = median_pitch(wave, sr)
        check(f"pitch of a {hz:.0f} Hz tone is found", abs(got - hz) < 12,
              f"{got:.1f} Hz")

    # --- matching a whole speaker to a voice, not one clip of them
    #
    # A speaker's voice must follow the majority of what they said, whether
    # the minority is one long clip or many short ones.
    from app.pipeline import _voice_map

    def tone(hz, seconds, sr=16000):
        t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * hz * t)
                + 0.3 * np.sin(4 * np.pi * hz * t)).astype(np.float32)

    def is_male(name):
        return name.split("_")[0][-1] == "m"

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "speech16k.wav"
        # Speaker 1: uniformly low-pitched. Speaker 2: uniformly high-pitched.
        # Speaker 3: one long low-pitched segment (5s) against six shorter
        # high-pitched ones (12s total) — a single unrepresentative clip must
        # not outweigh the rest. Speaker 4: the reverse shape — one dominant
        # high-pitched segment (25s, 71%) against ten short low-pitched
        # interjections (10s, 29%) — many short clips must not outvote one
        # long one either.
        chunks = [
            tone(110.0, 4.0), tone(115.0, 4.0),
            tone(230.0, 4.0), tone(225.0, 4.0),
            tone(105.0, 5.0), *[tone(220.0, 2.0) for _ in range(6)],
            tone(230.0, 25.0), *[tone(110.0, 1.0) for _ in range(10)],
        ]
        sf.write(wav_path, np.concatenate(chunks), sr)

        segments = [
            {"start": 0.0, "end": 4.0, "speaker": 1},
            {"start": 4.0, "end": 8.0, "speaker": 1},
            {"start": 8.0, "end": 12.0, "speaker": 2},
            {"start": 12.0, "end": 16.0, "speaker": 2},
        ]
        offset = 16.0
        segments.append({"start": offset, "end": offset + 5.0, "speaker": 3})
        offset += 5.0
        for _ in range(6):
            segments.append({"start": offset, "end": offset + 2.0, "speaker": 3})
            offset += 2.0
        segments.append({"start": offset, "end": offset + 25.0, "speaker": 4})
        offset += 25.0
        for _ in range(10):
            segments.append({"start": offset, "end": offset + 1.0, "speaker": 4})
            offset += 1.0

        s_ = Settings()
        s_.voice = "bf_emma"
        voices = _voice_map(s_, segments, [1, 2, 3, 4], wav_path)

        check("a uniformly low-pitched speaker gets a male voice",
              is_male(voices.get(1, "")), voices.get(1))
        check("a uniformly high-pitched speaker gets a female voice",
              not is_male(voices.get(2, "")), voices.get(2))
        check("one unrepresentative low-pitched segment does not override "
              "the high-pitched bulk of what a speaker actually said",
              not is_male(voices.get(3, "")), voices.get(3))
        check("many short low-pitched interjections do not outvote one "
              "dominant high-pitched segment",
              not is_male(voices.get(4, "")), voices.get(4))

    # --- naming the voices actually used, for the report
    #
    # stats["voices"] used to hold the compute engine's name, which read as
    # "Voices used: Apple GPU (MLX)" on every built-in-voice run — identical to
    # the separate "Engine" row and naming no voice at all.
    from app.pipeline import _voice_names

    s_ = Settings()
    s_.voice = "bf_emma"
    check("one speaker is named by their voice, not its raw id",
          _voice_names(s_, [0], {}) == "Emma", _voice_names(s_, [0], {}))
    check("several speakers are named in order",
          _voice_names(s_, [0, 1, 2], {1: "bm_george", 2: "bf_alice"})
          == "Emma, George, Alice",
          _voice_names(s_, [0, 1, 2], {1: "bm_george", 2: "bf_alice"}))
    check("a voice shared by more than one speaker is named once",
          _voice_names(s_, [0, 1, 2, 3],
                       {1: "bm_george", 2: "bf_alice", 3: "bm_george"})
          == "Emma, George, Alice",
          _voice_names(s_, [0, 1, 2, 3],
                       {1: "bm_george", 2: "bf_alice", 3: "bm_george"}))

    # --- a speaker count that cannot be right
    #
    # Asked of the pipeline itself rather than of a copy of the rule kept here: a
    # test that re-derives the threshold agrees with itself whatever the app goes
    # on to say.
    from app.pipeline import _speaker_note

    def note(count, sample=False):
        return _speaker_note(count, sample)

    # Over-segmentation of a single presenter, which is how this step actually
    # fails. The real run that prompted the guard reported 28 speakers for a
    # 10m35s film with about seven characters and said so without comment.
    for count in (3, 4, 5):
        check(f"one presenter split into {count} voices is flagged", bool(note(count)))
    check("28 speakers in a ten-minute film is flagged", bool(note(28)))

    # And the material that is genuinely what it looks like.
    check("a two-person interview is left alone", not note(2))
    check("a single-voice dub is left alone", not note(1))

    # Length is not evidence either way — a 90-minute craft tutorial has one or
    # two speakers and a ten-minute film can have seven — so it is not an input
    # at all, and no video is long enough to talk the guard out of firing.
    #
    # Nor is a stated speaker count: the clustering is never told how many people
    # to find, so the count it reports is evidence rather than an instruction
    # played back. A count the guard was measured against could never be exceeded
    # by one it had itself fixed, which is what made this silent.
    import inspect
    check("the count found is the only evidence the guard weighs",
          list(inspect.signature(_speaker_note).parameters) == ["found", "sample"],
          str(list(inspect.signature(_speaker_note).parameters)))
    check("a 90-minute video does not need dozens of voices to be flagged",
          bool(note(3)))

    # A sample is the cheap moment to fix this, so it says so instead of
    # telling the user to run the whole thing again.
    check("a sample points at the run still to come",
          "before dubbing the whole video" in note(5, sample=True))
    check("a full run asks for a re-run", "run it again" in note(5))
    check("a sample of a two-person interview is still left alone",
          not note(2, sample=True))

    # Register: what was found, why it might be wrong, one thing to do — and the
    # thing to do has to be a control that exists.
    for label, text in (("the note", note(4)), ("the sample note", note(5, sample=True))):
        check(f"{label} is plain, calm copy",
              "!" not in text and "diariz" not in text.lower()
              and "cluster" not in text.lower() and "segment" not in text.lower(),
              text)
        check(f"{label} says what was found and what to do",
              text.split()[0].isdigit() and "Who's speaking" in text, text)
        check(f"{label} does not send the user to a setting that is gone",
              "how many" not in text.lower(), text)

    s_ = Settings()
    s_.voice = "bf_emma"
    male = s_.voice_for(1, male=True)
    female = s_.voice_for(1, male=False)
    check("a low-pitched speaker gets a male voice", male.split("_")[0][-1] == "m", male)
    check("a high-pitched speaker gets a female voice",
          female.split("_")[0][-1] == "f", female)
    check("the choice still avoids the primary voice", female != "bf_emma", female)


# ======================= 10. a 30-second sample before the whole video
def test_preview():
    """A sample must be cheap to take and must not pretend to be the real thing.

    Three things make it worth having, and each is a way it can silently stop
    being worth having: the window has to land on speech rather than on the
    title card, the download it pays for has to survive for the full run behind
    it, and its output must not leak into the finished videos or the history —
    a thirty-second stub filed beside real output is a mess the user cannot
    reason their way out of.
    """
    print("\n[10] A sample before committing to the whole video")
    from app import pipeline
    from app.backends import translate as T
    from app.config import HISTORY_FILE, JOBS as JOBS_DIR, OUTPUT_DIR, Settings

    work = WORK / "preview"
    work.mkdir(parents=True, exist_ok=True)

    # 90 seconds that open on 20 of silence — the title card this feature exists
    # to skip past — and then talk steadily.
    LEAD_IN, TOTAL = 20.0, 90.0
    clip = work / "clip.mp4"
    if not clip.exists():
        rate = 24000
        track = np.concatenate([np.zeros(int(LEAD_IN * rate), dtype=np.float32),
                                _tone(220, TOTAL - LEAD_IN, rate)])
        sf.write(work / "audio.wav", track, rate)
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-i", str(work / "audio.wav"), "-map", "0:v", "-map", "1:a",
                        "-t", f"{TOTAL:g}", "-c:v", "libx264", "-preset", "ultrafast",
                        "-c:a", "aac", str(clip)], check=True)

    # ---------------------------------------------- choosing the window
    found = pipeline._speech_start(clip, TOTAL)
    check("the window skips a silent opening", 18.0 <= found <= 22.0, f"{found:g}s")

    # A video whose speech starts near the end must not be handed a window that
    # runs off it.
    clamped = pipeline._speech_start(clip, 35.0)
    check("the window is clamped to the end of the video", clamped <= 5.01, f"{clamped:g}s")

    # An unknown duration — some hosts report none — falls back rather than
    # dividing or subtracting its way into nonsense.
    check("an unknown duration still gives a usable window",
          pipeline._speech_start(clip, 0.0) >= 0.0)

    # ------------------------------------------- weighting the progress bar
    # A sample shrinks every stage but the download, which still fetches the
    # whole video. Left at its full-run share the bar sat near zero for most of
    # the wait and then sprinted, which is the one thing a progress bar must not
    # do to someone deciding whether to give up.
    balanced = Settings().apply_preset("balanced")
    plain = pipeline.runner._plan(balanced)
    heavy = pipeline.runner._plan(balanced, 6.0)
    check("a sample gives the download a far larger share of the bar",
          heavy["download"][1] > plain["download"][1] * 3,
          f"{plain['download'][1]:.2f} -> {heavy['download'][1]:.2f}")
    check("the stages after it are pushed later to make room",
          heavy["transcribe"][0] > plain["transcribe"][0],
          f"{plain['transcribe'][0]:.2f} -> {heavy['transcribe'][0]:.2f}")
    check("the weights still add up to one",
          abs(sum(w for _, w, _ in heavy.values()) - 1.0) < 1e-6)
    check("the plan names only the stages this preset runs",
          set(pipeline.runner._plan(Settings().apply_preset("fast"))) ==
          {"download", "transcribe", "translate", "synthesize", "assemble", "finish"})

    # ------------------------------------------------- a sample end to end
    URL = "https://example.com/sample-me"
    shutil_rmtree(JOBS_DIR / pipeline._link_id(URL))
    fake_probe, fake_download = stub_download(clip, "Sample Me", TOTAL)

    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        if progress:
            progress(1.0, "Heard 2 lines")
        return [{"start": 1.0, "end": 4.0, "text": "primera linea"},
                {"start": 8.0, "end": 11.0, "text": "segunda linea"}]

    def fake_llm(prompt, model=None, host=None, key=None, **_):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|This is a line of dubbed speech." for i in ids)

    real_probe, real_download = pipeline.download.probe, pipeline.download.download
    real_transcribe = pipeline.asr_backend.transcribe
    pipeline.download.probe, pipeline.download.download = fake_probe, fake_download
    pipeline.asr_backend.transcribe = fake_transcribe
    T._call_ollama = fake_llm

    try:
        s = Settings().apply_preset("fast")
        s.translator = "ollama"

        # Elapsed time is time spent working, not time since the link was
        # pasted. A job that sat behind another and then ran reported the wait
        # as part of its own duration, which made "Took" wrong and the estimate
        # of the time remaining wrong by the same margin.
        waiting = pipeline.Job(id="x", url="u")
        check("a job that has not started reports no elapsed time",
              waiting.public()["elapsed"] == 0)
        waiting.began = time.time() - 30
        waiting.finished = waiting.began + 10
        check("a finished job reports the time it ran, not the time it waited",
              waiting.public()["elapsed"] == 10, str(waiting.public()["elapsed"]))

        check("a sample and the full run are different jobs",
              pipeline._job_id(URL, True) != pipeline._job_id(URL, False))
        check("but they share one work folder",
              pipeline._link_id(URL) == pipeline._job_id(URL, False))

        job = pipeline.runner.submit(URL, s, preview=True)
        # An impatient second click is the same job, not two racing over one
        # folder. The buttons are disabled while a submission is in flight, but
        # that is a courtesy; this is the guarantee.
        check("a second click while it runs is the same job",
              pipeline.runner.submit(URL, s, preview=True) is job)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)
        check("the sample completed", job.status == "done", f"{job.status}: {job.error}")
        if job.status != "done":
            return

        out = Path(job.output)
        length = pipeline.download.media_duration(out)
        check("the sample is about thirty seconds",
              abs(length - pipeline.PREVIEW_SECONDS) < 2.0, f"{length:.1f}s")
        check("it was taken from where the speech starts",
              18.0 <= job.preview_from <= 22.0, f"{job.preview_from:g}s")
        check("the report says it is a sample", job.stats.get("preview") is True)

        # Not a deliverable: not in the videos folder, not in the history.
        check("it did not land in the finished videos folder",
              not list(OUTPUT_DIR.glob("Sample-Me*")),
              str([p.name for p in OUTPUT_DIR.glob("Sample-Me*")]))
        history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
        check("it was not recorded in the history",
              not any(h.get("output") == job.output for h in history))

        # And the download it paid for is still there for the full run behind it.
        workdir = JOBS_DIR / pipeline._link_id(URL)
        sources = list(workdir.glob("derived/*/source.mp4"))
        check("the shared download survived the sample",
              any("preview" not in str(p) and p.stat().st_size > 0 for p in sources)
              and len(sources) >= 2, f"{len(sources)} source files")

        # ------------------------------------- too short to be worth sampling
        SHORT_URL = "https://example.com/too-short"
        shutil_rmtree(JOBS_DIR / pipeline._link_id(SHORT_URL))
        short = work / "short.mp4"
        if not short.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip),
                            "-ss", "25", "-t", "12", "-c:v", "libx264",
                            "-preset", "ultrafast", "-c:a", "aac", str(short)], check=True)
        pipeline.download.probe, pipeline.download.download = stub_download(
            short, "Too Short", 12.0)

        brief = pipeline.runner.submit(SHORT_URL, s, preview=True)
        t0 = time.time()
        while brief.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)
        check("a video shorter than the window completed",
              brief.status == "done", f"{brief.status}: {brief.error}")
        if brief.status == "done":
            check("it was dubbed whole rather than sampled", brief.preview is False)
            check("and it says why",
                  any("only" in note_text(n) and "whole thing" in note_text(n)
                      for n in brief.stats.get("notes", [])),
                  str(brief.stats.get("notes")))
            check("so it did reach the finished videos folder",
                  Path(brief.output).parent == OUTPUT_DIR, brief.output)

        # ------------------------------------------ nothing to hear at all
        # A video with no speech in it — a music video, a silent screencast —
        # should fail plainly and in a minute, which is most of the argument for
        # sampling in the first place.
        SILENT_URL = "https://example.com/all-quiet"
        shutil_rmtree(JOBS_DIR / pipeline._link_id(SILENT_URL))
        silent = work / "silent.mp4"
        if not silent.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error",
                            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                            "-map", "0:v", "-map", "1:a", "-t", "90",
                            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                            str(silent)], check=True)
        check("a silent video starts its window at the beginning",
              pipeline._speech_start(silent, 90.0) == 0.0)

        pipeline.download.probe, pipeline.download.download = stub_download(
            silent, "All Quiet", 90.0)
        pipeline.asr_backend.transcribe = lambda *a, **k: []

        quiet = pipeline.runner.submit(SILENT_URL, s, preview=True)
        t0 = time.time()
        while quiet.status in ("queued", "running") and time.time() - t0 < 600:
            time.sleep(1)
        check("a video with no speech fails rather than producing silence",
              quiet.status == "error", quiet.status)
        check("and says so in plain English",
              "no speech" in quiet.error.lower(), quiet.error)
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.asr_backend.transcribe = real_transcribe


# ============================== 11. knowing where the disk went
def test_storage():
    """Filling the boot disk does not just fail a job — macOS stops working too.

    So the app has to be able to say what it is holding, let it go a piece at a
    time, and refuse a video it has no room for rather than finding out at 80%.
    """
    print("\n[11] Disk space, and getting it back")
    from app import storage as store
    from app.config import JOBS, OUTPUT_DIR

    # --- sizes as the user reads them
    # One formatter for the disk warning and the download alike. The small end
    # matters: a first, cautious try is a short video, and it used to be
    # announced as "about 0.0 GB" of "0 MB".
    check("gigabytes keep one decimal", store.human_size(1_828_000_000) == "1.7 GB")
    check("past 100 MB a decimal is dropped", store.human_size(660_000_000) == "629 MB")
    check("a few megabytes keep theirs", store.human_size(5_000_000) == "4.8 MB")
    check("and a short video is not rounded away to zero",
          store.human_size(617_000) == "603 KB")
    check("nor is an empty one", store.human_size(0) == "1 KB")

    # --- estimating before anything is written
    hour = store.estimate_needed(3600, "720")
    check("an hour at 720p is estimated in gigabytes",
          2 * 1024**3 < hour < 3 * 1024**3, f"{hour / 1024**3:.1f} GB")
    check("best quality is estimated larger than 720p",
          store.estimate_needed(3600, "best") > hour)
    check("a very short video still reserves a floor",
          store.estimate_needed(2, "720") == store.MINIMUM_NEED)
    check("an unknown duration doesn't go negative",
          store.estimate_needed(-5, "720") == store.MINIMUM_NEED)
    check("an unknown quality is treated as the largest",
          store.estimate_needed(600, "??") == store.estimate_needed(600, "best"))

    # The real download size, where the site published one, beats the per-second
    # guess — and the case it rescues is the preview, where the duration passed
    # in is the 30-second sample but the whole video is fetched to cut it out of.
    big = 1_800_000_000                                  # a 1.7 GB source
    check("a known source size is used when it exceeds the guess",
          store.estimate_needed(30, "best", source_bytes=big) > 2 * big,
          "twice the download, for the merge, plus the derived files")
    check("so a preview of a long video no longer reserves only the floor",
          store.estimate_needed(30, "best", source_bytes=big) > store.MINIMUM_NEED * 5,
          f"{store.estimate_needed(30, 'best', source_bytes=big) / 1024**3:.1f} GB")
    check("a source size smaller than the guess doesn't shrink the estimate",
          store.estimate_needed(3600, "720", source_bytes=1024) == hour)
    check("and no size at all leaves the estimate exactly as it was",
          store.estimate_needed(3600, "720", source_bytes=0) == hour)

    # A file already on this disk is read where it lies rather than copied into
    # the job folder, so the fetch comes out of the estimate. The dub does not:
    # mux copies the picture through rather than re-encoding it, so the finished
    # file lands about the size of the one it came from, and a source that isn't
    # already H.264 gets a converted copy beside it before that one takes over.
    # Dropping the source term outright estimated a 12 GB file at 600 MB and
    # sailed past the guard that exists to stop a full disk taking the machine
    # down with it.
    huge = 12 * 1024**3                                  # 20 minutes of 4K
    local = store.estimate_needed(1200, "best", source_bytes=huge, local_source=True)
    check("a local file still reserves room for the dub written back out",
          local > 2 * huge, f"{local / 1024**3:.1f} GB for a {huge / 1024**3:.0f} GB file")
    # Where the two differ is the fetch. At this bitrate they happen to agree —
    # both are then dominated by picture-sized files, a fetched copy in the job
    # folder in one case and the dub written out in the other — so the ordinary
    # video is what shows it: there, the download falls back to its per-second
    # guess for a source it has to hold as well as write.
    ordinary = store.estimate_needed(3600, "best", source_bytes=big, local_source=True)
    check("but is not charged for a download it never makes",
          ordinary < store.estimate_needed(3600, "best", source_bytes=big),
          f"{ordinary / 1024**3:.1f} GB against "
          f"{store.estimate_needed(3600, 'best', source_bytes=big) / 1024**3:.1f} GB")
    # A sample writes a cut of the picture, not a copy of it, and the caller
    # scales what it passes accordingly — so the same file sampled must not
    # reserve the room its full run would.
    sample = store.estimate_needed(30, "best", source_bytes=huge // 40, local_source=True)
    check("and a sample of it reserves a sample's worth",
          sample < local / 10, f"{sample / 1024**3:.2f} GB against "
                               f"{local / 1024**3:.1f} GB")
    # A file that will not say how big it is falls back to the per-second guess
    # rather than to the derived figure alone, which would leave nothing for the
    # dub at all.
    check("a local file of unknown size still leaves room for one",
          store.estimate_needed(3600, "best", source_bytes=0, local_source=True)
          > 3600 * store.DERIVED_PER_SECOND * 2,
          f"{store.estimate_needed(3600, 'best', source_bytes=0, local_source=True) / 1024**3:.1f} GB")
    check("and a very short local file still reserves the floor",
          store.estimate_needed(2, "best", source_bytes=1024,
                                local_source=True) == store.MINIMUM_NEED)

    check("free space is a real number", store.free_bytes() > 0)

    # --- a breakdown, not one figure
    keys = [g["key"] for g in store.groups()]
    check("every place the app writes to is accounted for",
          set(keys) == {"jobs", "models", "previews", "hfmodels", "venv",
                        "ollama", "output"}, str(keys))
    by_key = {g["key"]: g for g in store.groups()}
    check("finished videos are never offered for deletion",
          not by_key["output"]["clearable"])
    check("the translation model is shown but not deletable",
          not by_key["ollama"]["clearable"])
    # Deleting it would remove the interpreter running the request. It is in the
    # list because it is one of the largest things on disk; Uninstall removes it.
    check("the Python environment is shown but not deletable",
          not by_key["venv"]["clearable"])

    # The model cache is shared with anything else on the machine that uses
    # Hugging Face, so only the repositories this app fetches are counted.
    ours = {p.name for p in store.model_cache_dirs()}
    check("only this app's model repositories are counted",
          all(n.startswith(store.OUR_MODEL_REPOS) for n in ours), str(sorted(ours)))
    if store.HF_HUB.is_dir():
        every = {p.name for p in store.HF_HUB.iterdir() if p.is_dir()}
        check("another tool's models are left out of the total",
              ours <= every and all(not n.startswith(store.OUR_MODEL_REPOS)
                                    for n in every - ours),
              str(sorted(every - ours)))

    # --- clearing one job at a time
    a, b = JOBS / "aaaaaaaaaaaa", JOBS / "bbbbbbbbbbbb"
    for folder in (a, b):
        (folder / "derived").mkdir(parents=True, exist_ok=True)
        (folder / "derived" / "big.wav").write_bytes(b"x" * 2_000_000)
    rows = store.job_folders()
    check("each job folder is listed with its size",
          {r["id"] for r in rows} >= {a.name, b.name})
    # A folder holding little more than an error log is not where a disk went.
    tiny = JOBS / "cccccccccccc"
    tiny.mkdir(parents=True, exist_ok=True)
    (tiny / "error.log").write_text("it broke")
    check("a folder with nothing but an error log is left off the list",
          not any(r["id"] == tiny.name for r in store.job_folders()))
    freed = store.clear(f"job:{a.name}")
    check("clearing one job frees roughly its size", 1_500_000 < freed < 3_000_000, str(freed))
    check("that job's folder is gone", not a.exists())
    check("the other job is untouched", b.exists())

    # A job that is still running must survive a clear-all.
    store.clear("jobs", keep={b.name})
    check("a job named as live is kept", b.exists())
    store.clear("jobs")
    check("clearing them all empties the folder",
          not any(p.is_dir() for p in JOBS.iterdir()))

    # Someone short of space may well keep the models on an external drive, and
    # that folder is not this app's to empty. The suite itself symlinks it, so
    # this also stops a test run deleting the real 700 MB.
    if store.MODELS.is_symlink():
        target = store.MODELS.resolve()
        check("a symlinked models folder is left alone", store.clear("models") == 0)
        check("and it is still there", target.is_dir())

    # --- nothing else is reachable
    before = store.dir_size(OUTPUT_DIR) if OUTPUT_DIR.is_dir() else 0
    for bad in ("output", "..", "everything", "models/../.."):
        try:
            store.clear(bad)
            check(f"“{bad}” is refused outright", False, "it was accepted")
        except ValueError:
            check(f"“{bad}” is refused outright", True)
    check("the finished videos folder survived every one of those",
          (store.dir_size(OUTPUT_DIR) if OUTPUT_DIR.is_dir() else 0) == before)

    # A job id is a folder name, not a path. This one is named rather than
    # joined-and-trusted, so it resolves outside JOBS and is dropped.
    check("a job id that climbs out of the folder frees nothing",
          store.clear("job:../../etc") == 0)

    # --- refusing a video there is no room for
    from app import pipeline
    from app.config import Settings
    clip = WORK / "preview" / "clip.mp4"
    real_probe, real_download = pipeline.download.probe, pipeline.download.download
    real_free = store.free_bytes
    pipeline.download.probe, pipeline.download.download = stub_download(
        clip, "No Room", 3600.0)
    store.free_bytes = lambda path=None: 100 * 1024 ** 2          # 100 MB left
    try:
        URL = "https://example.com/no-room"
        workdir = JOBS / pipeline._link_id(URL)
        shutil_rmtree(workdir)
        job = pipeline.runner.submit(URL, Settings().apply_preset("fast"))
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 120:
            time.sleep(0.5)
        check("an hour of video with 100 MB free is refused", job.status == "error",
              f"{job.status}: {job.error}")
        check("the message says what it needs and what there is",
              "room on the disk" in job.error and "100 MB" in job.error, job.error)
        check("and it refused before downloading anything",
              not list(workdir.rglob("*.mp4")) if workdir.exists() else True)
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        store.free_bytes = real_free


# ================================ 12. not letting the Mac doze off mid-job
def test_keep_awake():
    """An hour of video is an hour of work, and a laptop left alone sleeps.

    The job survives it — everything resumes when somebody touches the trackpad
    — but to whoever started it and walked away, a run that stopped at 40% for
    half an hour is indistinguishable from one that hung.
    """
    print("\n[12] Staying awake while there is work")
    import platform
    from app.pipeline import KeepAwake

    def held():
        out = subprocess.run(["pmset", "-g", "assertions"],
                             capture_output=True, text=True).stdout
        return "caffeinate" in out

    check("the display is deliberately left alone", "-d" not in KeepAwake.FLAGS,
          str(KeepAwake.FLAGS))
    check("idle sleep, disk sleep and mains sleep are all covered",
          set(KeepAwake.FLAGS) == {"-i", "-m", "-s"}, str(KeepAwake.FLAGS))

    if platform.system() != "Darwin":
        check("nothing is attempted off macOS", True, "skipped")
        return

    was = held()
    keeper = KeepAwake()
    keeper.start()
    time.sleep(0.5)
    check("an assertion is taken while working", held(), "pmset reports caffeinate")
    check("and it is tied to this process, not left loose",
          "-w" in keeper._proc.args and str(os.getpid()) in keeper._proc.args)
    keeper.start()
    check("starting twice holds one, not two", keeper.active)
    keeper.stop()
    time.sleep(0.5)
    check("it is given back when the queue empties", held() == was)
    keeper.stop()
    check("stopping twice is harmless", not keeper.active)

    # The pill in the running panel is a switch, and a switch that only takes
    # effect on the next job would be lying for the hour it matters most.
    from app.pipeline import JobRunner
    r = JobRunner()
    check("switching it on with nothing running holds nothing",
          r.sync_keep_awake(True) is False)
    r.awake.start()
    time.sleep(0.4)
    released = r.sync_keep_awake(False) is False
    # Against the baseline rather than against False, and after a moment: the
    # assertion is dropped when caffeinate exits, which is not the instant it is
    # signalled, and anything else on the Mac may be holding one of its own —
    # Time Machine and a Homebrew build both do. Asserting "nothing at all is
    # awake" made a passing app look broken.
    time.sleep(0.5)
    check("switching it off releases the one being held right now",
          released and held() == was, f"before={was} after={held()}")

    # A machine without caffeinate must not take the app down with it.
    import subprocess as _sp
    real = _sp.Popen
    try:
        _sp.Popen = lambda *a, **k: (_ for _ in ()).throw(OSError("no caffeinate"))
        k = KeepAwake()
        k.start()
        check("a Mac without caffeinate is simply not held awake", not k.active)
        k.stop()
    finally:
        _sp.Popen = real


# ============== 13. catching a translation that went wrong before it is spoken
def test_translation_qc():
    """A real 52-minute dub read out "id: 63" and then several minutes of Spanish.

    Nothing downstream could tell: the audio check heard speech, the frame check
    saw the picture survive, and the file was the right length. The failure is
    upstream of everything that looks at the finished article.
    """
    print("\n[13] Catching a translation that went wrong")
    from app.backends.translate import _looks_untranslated, _parse, _strip_echo
    from app.steps import qc

    SRC = "Blusa de verano facil en todas las tallas paso a paso"

    # Scaffolding the model echoes back into its own answer.
    # The slot marker goes out as "[2.0s]" but comes back with the opening
    # bracket dropped often enough to matter: 25 of 83 lines in a real run, each
    # one spoken aloud as "two point oh ess" in front of its sentence.
    for raw, want in [("id: 63 Now we chain three.", "Now we chain three."),
                      ("[2.0s] Now we chain three.", "Now we chain three."),
                      ("2.0s] Now we chain three.", "Now we chain three."),
                      ("[2.0s] id 63. Now we chain three.", "Now we chain three."),
                      ("#63 Now we chain three.", "Now we chain three."),
                      ("<id>|Now we chain three.", "Now we chain three."),
                      ("Now we chain three.", "Now we chain three.")]:
        got = _strip_echo(raw)
        check(f"stripped: {raw[:26]!r}", got == want, got)

    # The reply format spelled back at us instead of filled in. Found by
    # translating the same batches with a stronger model and diffing: 72 of 448
    # lines in a real 98-minute dub came back like this, 16% of the video, and
    # every check passed them — a line of pure template resembles neither its
    # Portuguese source nor an empty string.
    # Checked through the parser rather than on the helper, because rejection is
    # the behaviour that matters: a miss is what makes the halving retry ask
    # again, and asking again is the only thing that can turn a template into an
    # actual translation. Some of these empty out and some stay recognisably
    # template — either way none may reach the audio.
    from app.backends.translate import _is_scaffolding
    one = [{"i": 1, "text": "As duas correntinhas para finalizar."}]
    for template in ("<id>|<translation>", "<translation>", "<id>|", "<text>"):
        check(f"template answer is a miss, so the retry fires: {template!r}",
              _parse(f"1|{template}", one)[0] == {})
    for real in ("Now we chain three.", "3. Chain three stitches.",
                 "Row 12) turn your work.", "Work 2 together, <that> is the trick."):
        check(f"and a real line is not: {real[:28]!r}",
              not _is_scaffolding(real) and _parse(f"1|{real}", one)[0] == {1: real})

    # With the batch in hand the slot marker does not have to be guessed at: the
    # value that went out is known, so every mangling of it comes back off, and
    # a sentence that merely opens on the same number does not.
    for raw, slot in [("[5.1s] Why do we do this?", 5.1),
                      ("5.1s] Why do we do this?", 5.1),
                      ("5.1s Why do we do this?", 5.1),
                      ("(5.1 s) Why do we do this?", 5.1),
                      ("2s] Why do we do this?", 2.0)]:
        got = _strip_echo(raw, slot)
        check(f"slot echo removed: {raw[:24]!r}", got == "Why do we do this?", got)
    for keep, slot in [("5 stitches in the round.", 5.0),
                       ("60s is a long time.", 60.0),
                       ("5.1 seconds is the limit.", 5.1),
                       ("2 seconds later, turn.", 2.0)]:
        check(f"slot in hand, sentence intact: {keep[:24]!r}",
              _strip_echo(keep, slot) == keep, _strip_echo(keep, slot))
    spoken = [{"i": 49, "start": 325.3, "end": 330.4, "text": "¿Para qué hacemos esto?"}]
    check("and the parser takes the slot off before anything is spoken",
          _parse("49|5.1s] Why do we do this?", spoken)[0] == {49: "Why do we do this?"},
          _parse("49|5.1s] Why do we do this?", spoken)[0])

    # Only the punctuation that closes the marker is the marker's. What opens a
    # sentence is the sentence's, whatever alphabet it is written in.
    for keep, slot in [("¿Por qué no?", 5.1), ("¡Vamos!", 2.0), ("— then turn.", 5.1)]:
        got = _strip_echo(f"{slot:.1f}s] {keep}", slot)
        check(f"opening punctuation survives the strip: {keep[:18]!r}", got == keep, got)

    # Slots this app really produces, from a clipped syllable to a long run-on.
    for slot in (0.0, 0.1, 2.0, 12.5, 100.0):
        got = _strip_echo(f"[{slot:.1f}s] Chain three.", slot)
        check(f"slot {slot} marker removed", got == "Chain three.", got)
    # Layered, and the shape that leaves nothing behind: an answer that was only
    # ever scaffolding has to read as a miss so the retry asks again.
    check("a doubled marker comes off in one pass",
          _strip_echo("[2.0s] [2.0s] Chain three.", 2.0) == "Chain three.")
    check("a reply that is only the marker empties out",
          _strip_echo("[2.0s]", 2.0) == "")
    check("and the parser counts that as a miss rather than speaking it",
          _parse("1|[2.0s]", [{"i": 1, "start": 0.0, "end": 2.0, "text": "hola"}])[0] == {})
    check("a batch carrying no timings still parses",
          _parse("1|Chain three.", [{"i": 1, "text": "hola"}])[0] == {1: "Chain three."})

    # Numbers that belong to the sentence must survive. These are the shapes a
    # crochet or cookery tutorial actually produces, and eating the front of one
    # would corrupt a good translation silently.
    for keep in ("3 chain stitches, then turn.",
                 "3. Chain three stitches.",
                 "Line 5 of the pattern is a chain.",
                 "Row 12) is worked in the back loop."):
        check(f"kept intact: {keep[:28]!r}", _strip_echo(keep) == keep, _strip_echo(keep))

    check("the source handed back is spotted", _looks_untranslated(SRC, SRC))
    check("a real translation is not", not _looks_untranslated("A summer blouse.", SRC))
    check("a short line unchanged is left alone", not _looks_untranslated("OK", "OK"))

    # And the parser refuses it, which turns a wrong answer into a missing one —
    # so the retry above it gets a go, and the 5% ceiling can fail the job rather
    # than deliver an hour of the original language.
    check("the parser rejects an untranslated line",
          _parse(f"63|{SRC}", [{"i": 63, "text": SRC}])[0] == {})
    check("and repairs an echoed one",
          _parse("63|id: 63 Now we chain three.", [{"i": 63, "text": SRC}])[0]
          == {63: "Now we chain three."})

    # The check itself, which also covers a translation restored from cache.
    segs = [{"text": SRC, "translation": "id: 63 Now we chain three."},
            {"text": SRC, "translation": SRC},
            {"text": "Vale", "translation": "OK"},
            {"text": "Y seguimos", "translation": "And we carry on."},
            {"text": "Nada", "translation": ""}]
    report = qc.check(segs)
    check("the echoed line is repaired, not dropped",
          segs[0]["translation"] == "Now we chain three.")
    check("the untranslated line is left silent", segs[1]["translation"] == "")
    check("the good lines are untouched",
          segs[2]["translation"] == "OK" and segs[3]["translation"] == "And we carry on.")
    check("the counts are right",
          (report["repaired"], report["untranslated"], report["spoken"]) == (1, 1, 3),
          str(report))
    check("and it says so in a sentence", "original language" in qc.summarise(report))
    check("a clean translation says nothing at all",
          qc.summarise(qc.check([{"text": "Hola", "translation": "Hello there."}])) == "")

    # The cached path is the one that matters for a video already translated
    # once: the parser's guards run at translation time, but translated.json is
    # replayed straight past them. Stripping "<id>|" left "<translation>", which
    # is not empty and does not resemble its source — so it was spoken, and
    # counted as a repair, telling the user a fix had happened.
    cached = [{"text": SRC, "translation": "<id>|<translation>"},
              {"text": SRC, "translation": "<id>|Now we chain three."}]
    tmpl = qc.check(cached)
    check("a cached template line is silenced, not spoken",
          cached[0]["translation"] == "", repr(cached[0]["translation"]))
    check("but a real translation behind the scaffolding is kept",
          cached[1]["translation"] == "Now we chain three.")
    check("and it is counted apart from a genuinely untranslated line",
          (tmpl["template"], tmpl["untranslated"], tmpl["repaired"]) == (1, 0, 1),
          str(tmpl))
    check("the sentence does not claim it came back in the original language",
          "reply format" in qc.summarise(tmpl))

    # A batch the model cannot manage whole is halved until it can. Twenty-five
    # missing lines used to fall outside the "small prompts land" rule, so the
    # identical question was asked twice and then given up on.
    from app.backends import translate as _T
    sizes = []

    def flaky(prompt):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        sizes.append(len(ids))
        if len(ids) > 4:                    # echoes its input above four lines
            return "\n".join(f"{i}|id: {i} texto original numero {i} sin traducir"
                              for i in ids)
        return "\n".join(f"{i}|Chain three and turn, number {i}." for i in ids)

    big = [{"i": n, "start": n, "end": n + 2,
            "text": f"texto original numero {n} sin traducir"} for n in range(25)]
    got = _T._translate_chunk(big, [], "English", "", flaky)
    check("a batch the model chokes on is split until it lands",
          len(got) == 25, f"{len(got)} of 25")
    check("and it halves rather than asking the same thing twice",
          sizes[0] == 25 and max(sizes[1:]) <= 13, str(sizes[:6]))

    # --- the codec that made a finished dub unplayable
    from app.steps.mux import WIDELY_PLAYABLE
    check("H.264 counts as playable anywhere", "h264" in WIDELY_PLAYABLE)
    check("AV1 does not", "av1" not in WIDELY_PLAYABLE)

    # Read from the file, not through the module: earlier tests replace
    # download() with a stub and never put it back, so inspect would be reading
    # the fake.
    src = (ROOT / "app" / "steps" / "download.py").read_text()
    # The selector is one documented builder rather than strings assembled in
    # two branches, so every rung can be checked without touching the network.
    from app.steps.download import choose_format, format_selector as fsel

    for q in ("best", "1080", "720", "nonsense", ""):
        chain = fsel(q).split("/")
        check(f"{(q or 'blank'):8s}: ends in a bare best", chain[-1] == "b", fsel(q))
        check(f"{(q or 'blank'):8s}: no empty rung",
              all(r.strip() for r in chain), fsel(q))

    check("H.264 is matched by both names YouTube gives it",
          "^(avc|h264)" in fsel("best"))
    check("the height cap is dropped by the last rung rather than failing",
          fsel("720").split("/")[-1] == "b")
    check("a capped chain offers a combined stream, for sites with no separate audio",
          "b[height<=720]" in fsel("720"))
    check("an unrecognised quality is treated as best, not as an empty cap",
          fsel("nonsense") == fsel("best"))

    # Chosen from the format list the site actually published, rather than from a
    # selector string and hope — so the codec and the size are known before
    # anything is fetched.
    LISTING = {"formats": [
        {"format_id": "137", "vcodec": "avc1.640028", "acodec": "none",
         "height": 1080, "filesize": 1733_000_000},
        {"format_id": "399", "vcodec": "av01.0.08M.08", "acodec": "none",
         "height": 1080, "filesize": 758_000_000},
        {"format_id": "248", "vcodec": "vp9", "acodec": "none",
         "height": 1080, "filesize": 803_000_000},
        {"format_id": "136", "vcodec": "avc1.4d401f", "acodec": "none",
         "height": 720, "filesize": 660_000_000},
        {"format_id": "140", "vcodec": "none", "acodec": "mp4a", "ext": "m4a",
         "tbr": 129, "filesize": 95_000_000},
    ]}
    check("H.264 is chosen over the smaller AV1 and VP9",
          choose_format(LISTING, "best")["spec"] == "137+140",
          str(choose_format(LISTING, "best")))
    check("the height cap is honoured when one is set",
          choose_format(LISTING, "720")["spec"] == "136+140")
    check("the real size comes back, not an estimate",
          choose_format(LISTING, "720")["bytes"] == 660_000_000 + 95_000_000)
    check("a listing with nothing usable falls back to the selector",
          choose_format({"formats": []}, "best") is None)
    # Nothing left is universally playable, so the smallest is taken and the
    # report says so — ranking codecs nobody can play would be false precision.
    no264 = {"formats": [f for f in LISTING["formats"]
                         if "avc" not in (f.get("vcodec") or "")]}
    picked = choose_format(no264, "best")
    check("without H.264 the smallest is taken rather than refusing the video",
          picked["spec"] == "399+140", str(picked))
    above_cap = {"formats": [f for f in LISTING["formats"]
                             if (f.get("height") or 0) != 720]}
    check("a cap nothing satisfies is dropped rather than returning nothing",
          choose_format(above_cap, "720") is not None)

    # The seam that broke: choose_format was correct and never ran, because
    # probe() built a fresh dict and dropped the format list on the way out. It
    # is checked here rather than only through a synthetic listing, since a
    # synthetic listing is exactly what hid it.
    # Bounded by the next definition rather than by a character count, which a
    # comment or a guard added inside probe() silently pushes the return past.
    _p = src.index("def probe(")
    _end = src.index("\ndef ", _p + 1)
    probe_out = src[_p:_end]
    kept = probe_out[probe_out.index("return {"):]
    check("probe returns the format list, rather than only fetching it",
          '"formats"' in kept, kept[:60].replace("\n", " "))

    # "Highest bitrate" on its own picks the 5.1 track where a video has one:
    # three times the bytes for audio that is replaced outright or downmixed to
    # mono for the transcriber.
    SURROUND = {"formats": LISTING["formats"] + [
        {"format_id": "258", "vcodec": "none", "acodec": "mp4a", "ext": "m4a",
         "tbr": 388, "audio_channels": 6, "filesize": 285_000_000},
    ]}
    check("stereo audio is chosen over a louder 5.1 track",
          choose_format(SURROUND, "best")["spec"] == "137+140",
          str(choose_format(SURROUND, "best")))
    check("and the surround track's bytes are not counted against the disk check",
          choose_format(SURROUND, "best")["bytes"] == 1733_000_000 + 95_000_000)

    # Which file the pipeline is handed after the download. This was
    # sorted(glob("source.*"))[0], which is right until an attempt is
    # interrupted: a refused try leaves source.f137.mp4.part behind, the retry
    # writes source.mp4, and the stub sorts first. On a real run that turned a
    # complete 1.7 GB download into "the video is in a format that can't be read".
    from app.steps.download import _finished_file

    def _tiny_media(seconds: float) -> bytes:
        """A few KB of genuinely decodable video and audio, for cases where
        _finished_file() now has to ask ffprobe rather than trust a name — a
        plain run of repeated bytes used to stand in for "the video" until
        that check existed, and it fails ffprobe the same way a thumbnail
        does, which is exactly the distinction being tested here.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            dst = Path(tmp.name)
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=32x32:rate=5",
                        "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono",
                        "-map", "0:v", "-map", "1:a", "-t", f"{seconds:g}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(dst)], check=True)
        data = dst.read_bytes()
        dst.unlink()
        return data

    # ffprobe doesn't care what the file is named — it sniffs the container
    # from the bytes, same as it would on a real download — so one small real
    # clip and one larger one, copied under whatever filename each case below
    # needs, cover every "is this actually media" question without a separate
    # ffmpeg encode per container.
    SMALL_MEDIA = _tiny_media(1)
    LARGE_MEDIA = _tiny_media(3)
    check("the two fixture clips are genuinely different sizes",
          len(LARGE_MEDIA) > len(SMALL_MEDIA), (len(SMALL_MEDIA), len(LARGE_MEDIA)))

    shed = SCRATCH / "picking"
    shutil.rmtree(shed, ignore_errors=True)
    shed.mkdir(parents=True)
    (shed / "source.f137.mp4.part").write_bytes(b"x" * 10_000)
    (shed / "source.mp4.ytdl").write_bytes(b"{}")
    (shed / "source.mp4").write_bytes(SMALL_MEDIA)
    check("a leftover .part is never mistaken for the finished download",
          _finished_file(shed).name == "source.mp4", _finished_file(shed).name)
    (shed / "source.mp4").unlink()
    (shed / "source.f137.mp4").write_bytes(LARGE_MEDIA)
    check("an unmerged format is used when there is no merged file",
          _finished_file(shed).name == "source.f137.mp4")
    (shed / "source.webm").write_bytes(SMALL_MEDIA)
    check("but the merged name wins even when it is smaller",
          _finished_file(shed).name == "source.webm")
    for leftover in shed.iterdir():
        leftover.unlink()
    (shed / "source.mp4.part").write_bytes(b"x" * 500)
    check("a folder holding only working files reports nothing finished",
          _finished_file(shed) is None)

    # The container allowlist this replaced only knew five extensions, so a
    # complete download in anything else — a Wikimedia .ogv, an archive.org
    # .mpg, both of which ffmpeg reads without complaint — was invisible to it
    # and thrown away as if the fetch had failed, after every byte had arrived.
    for leftover in shed.iterdir():
        leftover.unlink()
    (shed / "source.ogv").write_bytes(LARGE_MEDIA)
    check("a completed download in a container outside the old mp4/mkv/webm/mov/m4v allowlist is accepted",
          _finished_file(shed) is not None and _finished_file(shed).name == "source.ogv",
          str(_finished_file(shed)))
    (shed / "source.ogv.part").write_bytes(b"x" * 9_700_000)
    check("a .part fragment beside it is still never chosen over the complete file, allowlisted container or not",
          _finished_file(shed).name == "source.ogv")
    for leftover in shed.iterdir():
        leftover.unlink()
    # No standalone check that a lone fragment reports nothing finished: any
    # fragment name ends in -Frag<n>, which never matches a five-extension
    # allowlist either, so that alone would have passed before this fix too.
    # What actually distinguishes the new behaviour is that the real file
    # sitting beside the fragment is still found, in a container the old
    # allowlist would have refused outright regardless of the fragment.
    (shed / "source.f303.mp4.part-Frag12").write_bytes(b"x" * 500_000)
    (shed / "source.mpg").write_bytes(LARGE_MEDIA)
    check("a genuinely finished download is picked over a lingering fragment of an interrupted attempt",
          _finished_file(shed).name == "source.mpg")

    # Excluding scratch state was not enough on its own: the reviewer who
    # caught this demonstrated a workdir with --write-thumbnail set in the
    # user's own yt-dlp config (left readable — see download()) producing
    # source.mp4 (633,710 bytes) and source.webp (23,038 bytes) side by side,
    # neither a .part nor a .ytdl. A sidecar reachable this way is asked about
    # by ffprobe like everything else, not waved through because its name or
    # size happened to fit.
    for leftover in shed.iterdir():
        leftover.unlink()
    (shed / "source.jpg").write_bytes(b"\xff\xd8\xff" + b"not actually a jpeg" * 50)
    check("a workdir holding only a thumbnail sidecar reports nothing finished, not the thumbnail",
          _finished_file(shed) is None)
    (shed / "source.jpg").unlink()
    (shed / "source.info.json").write_bytes(b'{"id": "abc123", "title": "not a video"}')
    check("and a workdir holding only an info.json sidecar reports nothing finished either",
          _finished_file(shed) is None)
    (shed / "source.info.json").unlink()

    big_sidecar = b"\xff\xd8\xff" + b"not actually a jpeg" * 5000
    (shed / "source.jpg").write_bytes(big_sidecar)
    (shed / "source.mp4").write_bytes(SMALL_MEDIA)
    check("a sidecar bigger than the real video still loses to the video, not to its size",
          len(big_sidecar) > len(SMALL_MEDIA) and _finished_file(shed).name == "source.mp4",
          (len(big_sidecar), len(SMALL_MEDIA)))

    # The merge postprocessor's own scratch file: FFmpegMergerPP builds the
    # merged output at source.temp.mp4 before renaming it over source.mp4, so
    # for as long as a mux is running both names exist and only one is finished.
    for leftover in shed.iterdir():
        leftover.unlink()
    (shed / "source.temp.mp4").write_bytes(LARGE_MEDIA)
    check("a merge-in-progress source.temp.mp4 is recognised as scratch, not as the video",
          _finished_file(shed) is None)
    (shed / "source.mp4").write_bytes(SMALL_MEDIA)
    check("and the real file is picked once it exists, even though the temp file is still larger",
          _finished_file(shed).name == "source.mp4")

    # The regression a duration floor reintroduced: a completely genuine,
    # fully downloaded video-only remux can report no duration at all. A
    # container's duration is normally patched in by seeking back once the
    # last frame is known; a mux that was piped to a non-seekable output, or
    # simply never reached a clean close, never gets that pass, even though
    # every frame it wrote decodes correctly. Reproduced here the same way —
    # ffmpeg started with no fixed length, so there is no total to write up
    # front, and killed the moment real video data exists rather than let it
    # reach the shutdown that would patch a duration in — and a duration
    # floor threw this away exactly the way the container allowlist threw
    # away a .ogv: proof was demanded of a file that never had a reason to
    # carry it.
    def _tiny_video_only(unfinalized: bool) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            dst = Path(tmp.name)
        if not unfinalized:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                            "-i", "testsrc=size=32x32:rate=5", "-an", "-t", "1",
                            "-c:v", "libvpx", str(dst)], check=True)
        else:
            with open(dst, "wb") as fh:
                proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                     "-i", "testsrc=size=32x32:rate=5", "-an",
                     "-c:v", "libvpx", "-f", "webm", "-"], stdout=fh)
                deadline = time.time() + 5
                while time.time() < deadline and dst.stat().st_size < 20_000:
                    time.sleep(0.02)
                proc.terminate()
                proc.wait(timeout=5)
        data = dst.read_bytes()
        dst.unlink()
        return data

    for leftover in shed.iterdir():
        leftover.unlink()
    (shed / "source.webm").write_bytes(_tiny_video_only(unfinalized=True))
    check("a video-only remux whose muxer never wrote a duration is accepted, not discarded for lack of proof",
          _finished_file(shed) is not None and _finished_file(shed).name == "source.webm",
          str(_finished_file(shed)))
    for leftover in shed.iterdir():
        leftover.unlink()
    (shed / "source.webm").write_bytes(_tiny_video_only(unfinalized=False))
    check("and a video-only remux that does report a duration is accepted too",
          _finished_file(shed) is not None and _finished_file(shed).name == "source.webm")

    # Real thumbnails, not just the corrupt bytes used above — the discriminator
    # is which demuxer ffprobe read the file through, so it has to hold for a
    # well-formed image too, not merely for one already too broken to decode.
    def _tiny_image(suffix: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            dst = Path(tmp.name)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "color=c=red:s=32x32", "-frames:v", "1",
                        str(dst)], check=True)
        data = dst.read_bytes()
        dst.unlink()
        return data

    for leftover in shed.iterdir():
        leftover.unlink()
    (shed / "source.jpg").write_bytes(_tiny_image(".jpg"))
    check("a real JPEG thumbnail is rejected, not just a corrupt one",
          _finished_file(shed) is None)
    (shed / "source.jpg").unlink()
    (shed / "source.png").write_bytes(_tiny_image(".png"))
    check("and a real PNG thumbnail is rejected the same way",
          _finished_file(shed) is None)

    # A probe that cannot answer is not evidence of anything and must not be
    # treated as one — the earlier duration floor turned "ffprobe timed out"
    # into "throw the completed download away", silently, which is the same
    # mistake as the two checks just above, just with the probe itself failing
    # instead of merely being unable to read a duration.
    import app.steps.download as _dl_module

    class _DeadProbe:
        TimeoutExpired = subprocess.TimeoutExpired

        @staticmethod
        def run(*_a, **_k):
            raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    for leftover in shed.iterdir():
        leftover.unlink()
    (shed / "source.mp4").write_bytes(SMALL_MEDIA)
    real_subprocess = _dl_module.subprocess
    _dl_module.subprocess = _DeadProbe
    try:
        timed_out_result = _finished_file(shed)
    finally:
        _dl_module.subprocess = real_subprocess
    check("an ffprobe that times out does not cause a completed download to be discarded",
          timed_out_result is not None and timed_out_result.name == "source.mp4",
          timed_out_result)

    # The same bug through a new door: ffmpeg reads a one-frame thumbnail and
    # a genuinely animated GIF through the identical "gif" demuxer, so
    # rejecting on format_name alone (as jpeg_pipe/png_pipe correctly can)
    # would throw away a real animated GIF, and yt-dlp's own Imgur extractor
    # really does offer one as a download when a post has no video transcode.
    def _tiny_gif(frames: int) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            dst = Path(tmp.name)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc=size=32x32:rate=5", "-frames:v", str(frames),
                        str(dst)], check=True)
        data = dst.read_bytes()
        dst.unlink()
        return data

    for leftover in shed.iterdir():
        leftover.unlink()
    (shed / "source.gif").write_bytes(_tiny_gif(1))
    check("a genuine single-frame GIF is rejected, same as any other thumbnail",
          _finished_file(shed) is None)
    (shed / "source.gif").unlink()
    (shed / "source.gif").write_bytes(_tiny_gif(10))
    check("but a real multi-frame animated GIF is accepted, not thrown away for sharing a demuxer with one",
          _finished_file(shed) is not None and _finished_file(shed).name == "source.gif",
          str(_finished_file(shed)))

    # _explain_missing_video() only ever runs once download() has already
    # decided to fail, and nothing in it may itself raise — an unhandled
    # exception there would replace the friendly "no video file appeared"
    # message with a raw traceback in the generic handler in app/pipeline.py,
    # which is a worse failure than the one this whole item exists to fix.
    from app.steps.download import _explain_missing_video

    locked = SCRATCH / "locked-workdir"
    shutil.rmtree(locked, ignore_errors=True)
    locked.mkdir(parents=True)
    (locked / "source.mp4").write_bytes(SMALL_MEDIA)
    os.chmod(locked, 0o000)
    try:
        locked_detail = _explain_missing_video(locked)
        locked_ok = isinstance(locked_detail, str) and bool(locked_detail.strip())
    except Exception as exc:                                      # noqa: BLE001
        locked_detail, locked_ok = repr(exc), False
    finally:
        os.chmod(locked, 0o755)                # so cleanup can still delete it
    check("an unreadable working directory does not crash the explanation",
          locked_ok, locked_detail)

    missing = SCRATCH / "workdir-that-was-never-created"
    shutil.rmtree(missing, ignore_errors=True)
    try:
        missing_detail = _explain_missing_video(missing)
        missing_ok = isinstance(missing_detail, str) and bool(missing_detail.strip())
    except Exception as exc:                                      # noqa: BLE001
        missing_detail, missing_ok = repr(exc), False
    check("a missing working directory does not crash the explanation",
          missing_ok, missing_detail)

    workdir_is_a_file = SCRATCH / "workdir-that-is-actually-a-file"
    workdir_is_a_file.write_text("not a directory")
    try:
        file_detail = _explain_missing_video(workdir_is_a_file)
        file_ok = isinstance(file_detail, str) and bool(file_detail.strip())
    except Exception as exc:                                      # noqa: BLE001
        file_detail, file_ok = repr(exc), False
    check("a working directory path that is actually a file does not crash the explanation",
          file_ok, file_detail)

    vanishing = SCRATCH / "vanishing-candidate"
    shutil.rmtree(vanishing, ignore_errors=True)
    vanishing.mkdir(parents=True)
    (vanishing / "source.mp4").write_bytes(SMALL_MEDIA)
    real_looks_like_media = _dl_module._looks_like_media

    def _vanish_then_raise(p):
        p.unlink(missing_ok=True)
        raise FileNotFoundError(p)

    _dl_module._looks_like_media = _vanish_then_raise
    try:
        vanish_detail = _explain_missing_video(vanishing)
        vanish_ok = isinstance(vanish_detail, str) and bool(vanish_detail.strip())
    except Exception as exc:                                      # noqa: BLE001
        vanish_detail, vanish_ok = repr(exc), False
    finally:
        _dl_module._looks_like_media = real_looks_like_media
    check("a candidate that disappears between being listed and being probed does not crash the explanation",
          vanish_ok, vanish_detail)

    # --- retrying and client choice, which are yt-dlp's job and not ours
    # What was here was a ladder of our own: three identical attempts, then two
    # more as different player clients, with the decision to continue taken by
    # substring-matching yt-dlp's error text. It got that decision wrong — the
    # error the fallback clients provoke was not on the "transient" list, so a
    # fallback rung ended the ladder — and every part of it is something yt-dlp
    # already does properly. Now it is asked once, for all the clients at once,
    # with yt-dlp's own exponential backoff.
    from app.steps.download import _RETRY_ARGS
    import app.steps.download as dlmod

    args = " ".join(_RETRY_ARGS)
    check("retries are yt-dlp's, with real exponential backoff",
          "--retry-sleep" in args and "exp=" in args, args)
    check("extraction gets its own retry count, not just the HTTP fetch",
          "--extractor-retries" in args and "extractor:exp=" in args)
    # The seam that broke: the probe is where concrete format ids are chosen, so
    # a download that asked a client the probe never spoke to could be told to
    # fetch ids that client has not got — which is the "Requested format is not
    # available" that used to end the ladder. It used to be held by pinning the
    # same client list on both commands; that list then went stale, and a video
    # which plays fine in a browser was refused with "HTTP Error 403" about
    # 20 MB in because every client it named had been retired. Neither command
    # names one now, so both get yt-dlp's own current defaults: same binary,
    # same defaults, same ids, and the invariant holds by omission.
    #
    # Asserted against the argv actually built, not against the source text — the
    # comment explaining why no client is pinned contains the word
    # "player_client" and would satisfy any grep for it.
    # A clean copy of the module: earlier sections of the suite leave probe() and
    # download() stubbed out, and a stub builds no argv at all — which is how
    # this first went green while checking nothing (both commands reported
    # "<never built>" and passed the "pins no player client" test on a string
    # that was never a command).
    import importlib.util as _ilu
    _argv_spec = _ilu.spec_from_file_location(
        "app.steps._download_argv_check", ROOT / "app" / "steps" / "download.py")
    dlargv = _ilu.module_from_spec(_argv_spec)
    _argv_spec.loader.exec_module(dlargv)

    built = {}

    class _ProbeReply:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"title": "t", "duration": 1.0, "formats": []})

    class _SubprocessShim:
        def __init__(self, real):
            self._real = real

        def run(self, cmd, *a, **k):
            built["probe"] = list(cmd)
            return _ProbeReply()

        def __getattr__(self, name):
            return getattr(self._real, name)

    real_subprocess = dlargv.subprocess
    dlargv.subprocess = _SubprocessShim(real_subprocess)
    try:
        dlargv.probe("https://example.invalid/watch?v=x")
    finally:
        dlargv.subprocess = real_subprocess

    def _capture_stream(cmd, on_line=None, tail_lines=30):
        built["download"] = list(cmd)
        return 0, ""

    real_stream = dlargv.stream
    dlargv.stream = _capture_stream
    try:
        dlargv.download("https://example.invalid/watch?v=x", WORK / "argv-check",
                        "best", None, {"title": "t", "duration": 1.0, "formats": []})
    except Exception:                                             # noqa: BLE001
        pass                       # no file appears, which is not what is under test
    finally:
        dlargv.stream = real_stream

    check("both yt-dlp commands were actually built, so the rest is a real check",
          sorted(built) == ["download", "probe"], str(sorted(built)))
    for which in ("probe", "download"):
        argv = " ".join(built.get(which, []))
        check(f"the {which} command pins no player client",
              bool(argv) and "player_client" not in argv, argv[:160] or "<never built>")
    check("the probe and the download run the very same yt-dlp",
          built.get("probe", [])[:3] == built.get("download", [])[:3]
          == dlargv._ytdlp_cmd(), str(built.get("probe", [])[:3]))
    # The copy inside .venv, reached through the interpreter already running —
    # not whichever yt-dlp PATH resolves. The launcher activates .venv, so a
    # bare "yt-dlp" found the pip copy there while the installer kept a
    # different, newer one on PATH freshened; the app ran the stale one.
    check("and it is this environment's own yt-dlp, not one found on PATH",
          dlargv._ytdlp_cmd()[0] == sys.executable
          and dlargv._ytdlp_cmd()[1:] == ["-m", "yt_dlp"], str(dlargv._ytdlp_cmd()))
    # Demonstrated on a real job through the real UI: a --write-thumbnail left
    # in a user's yt-dlp config turned an ordinary download into source.mp4
    # *and* source.webp sitting in the same working directory _finished_file()
    # scans. Checked on both commands built here, not just the download, since
    # a probe that reads the same config could just as easily have its listing
    # reshaped by a format or extractor option it was never asked for.
    from app.steps.download import _NO_CONFIG_ARGS
    check("neither yt-dlp invocation reads the user's own configuration",
          "--ignore-config" in _NO_CONFIG_ARGS
          and src.count("_NO_CONFIG_ARGS") >= 2 and "+ _NO_CONFIG_ARGS" in src)
    check("and nothing decides what to retry by reading yt-dlp's error text",
          "_looks_transient" not in src and "_TRANSIENT" not in src
          and "_looks_terminal" not in src)
    # Seen in the field before this was fixed: a real 403 logged with
    # "detail": "DUBPROG|64512|3..." — the progress template had filled the tail
    # stream() keeps, so the pasteable detail held no error at all. A clean copy
    # of the module, because earlier sections leave download() stubbed.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "app.steps._download_under_test", ROOT / "app" / "steps" / "download.py")
    dlfresh = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(dlfresh)

    def failing_download():
        """Run the real download() against a yt-dlp that talks then dies."""
        def fake_stream(cmd, on_line=None, tail_lines=30):
            for n in range(40):                 # a large download's worth
                on_line(f"{dlfresh._PROGRESS_PREFIX}{n * 1024}|41943040\n")
            on_line("ERROR: unable to download video data: HTTP Error 403: Forbidden\n")
            # what stream() would hand back: its tail is all progress
            return 1, "".join(f"{dlfresh._PROGRESS_PREFIX}{n}|1\n" for n in range(25))
        dlfresh.stream = fake_stream
        dlfresh.probe = lambda url, cookies_from="": {
            "title": "t", "duration": 1.0, "formats": [], "is_live": False}
        pen = SCRATCH / "detail"
        shutil.rmtree(pen, ignore_errors=True)
        pen.mkdir(parents=True)
        try:
            dlfresh.download("https://youtu.be/x", pen)
        except Exception as exc:                                 # noqa: BLE001
            return exc
        return None

    err = failing_download()
    detail = str(getattr(err, "detail", ""))
    check("the pasteable detail carries yt-dlp's error, not the progress records",
          "403" in detail and dlfresh._PROGRESS_PREFIX not in detail, detail[:70])
    check("and the message the user reads is the plain-English one",
          "refused to send it" in str(err), str(err)[:60])

    # The other way a download can fail to hand back a video: yt-dlp exits 0
    # — it thinks it succeeded — but _finished_file() finds nothing usable in
    # the working directory. This used to raise a bare RuntimeError, the one
    # failure in this file with nothing for the UI's "what the downloader
    # actually said" pane or Copy details to show. It has to fail the same way
    # every other download failure does.
    def vanished_download():
        """Run the real download() against a yt-dlp that reports success but leaves nothing behind."""
        def fake_stream(cmd, on_line=None, tail_lines=30):
            on_line(f"{dlfresh._PROGRESS_PREFIX}1024|1024\n")
            return 0, ""
        dlfresh.stream = fake_stream
        dlfresh.probe = lambda url, cookies_from="": {
            "title": "t", "duration": 1.0, "formats": [], "is_live": False}
        pen = SCRATCH / "vanished"
        shutil.rmtree(pen, ignore_errors=True)
        pen.mkdir(parents=True)
        try:
            dlfresh.download("https://youtu.be/x", pen)
        except Exception as exc:                                 # noqa: BLE001
            return exc
        return None

    err2 = vanished_download()
    check("a download yt-dlp reported as successful but left nothing playable behind "
          "fails the same way every other download failure does",
          isinstance(err2, dlfresh.DownloadError), repr(err2))
    check("the user-facing message stays plain English",
          str(err2) == "The download finished but no video file appeared.", str(err2))
    check("and the detail is non-empty, so Copy details has something an engineer can act on",
          bool(getattr(err2, "detail", "").strip()), repr(getattr(err2, "detail", "")))
    check("specifically, what was actually sitting in the working directory",
          "vanished" in getattr(err2, "detail", ""), getattr(err2, "detail", ""))

    # The point of every _explain_missing_video() "does not crash" check
    # above: none of that matters if the crash it used to produce simply
    # moved one call up the stack. This runs the real download() end to end
    # against an unreadable working directory and confirms what actually
    # reaches the caller is still the plain-English DownloadError, not a
    # PermissionError escaping into app/pipeline.py's generic handler.
    def locked_workdir_download():
        def fake_stream(cmd, on_line=None, tail_lines=30):
            return 0, ""
        dlfresh.stream = fake_stream
        dlfresh.probe = lambda url, cookies_from="": {
            "title": "t", "duration": 1.0, "formats": [], "is_live": False}
        pen = SCRATCH / "locked-workdir"
        shutil.rmtree(pen, ignore_errors=True)
        pen.mkdir(parents=True)
        (pen / "source.mp4").write_bytes(b"x" * 1000)
        os.chmod(pen, 0o000)
        try:
            try:
                dlfresh.download("https://youtu.be/x", pen)
            except Exception as exc:                             # noqa: BLE001
                return exc
            return None
        finally:
            os.chmod(pen, 0o755)                # so cleanup can still delete it

    err3 = locked_workdir_download()
    check("an unreadable working directory still fails as a DownloadError, not a raw crash",
          isinstance(err3, dlfresh.DownloadError), repr(err3))
    check("with the plain-English message intact",
          str(err3) == "The download finished but no video file appeared.", str(err3))
    check("and a non-empty detail, rather than the explanation itself dying silently",
          bool(getattr(err3, "detail", "").strip()), repr(getattr(err3, "detail", "")))

    check("the plain-English message for a format refusal names no CLI flags",
          "--list-formats" not in dlmod._friendly(
              "ERROR: [youtube] x: Requested format is not available. "
              "Use --list-formats for a list of available formats")
          and "Settings" in dlmod._friendly(
              "ERROR: [youtube] x: Requested format is not available."))

    # The progress counter cannot be trusted alone across a retry: yt-dlp answers
    # a resumed attempt with "Resuming download at byte N" and what it counts
    # from there depends on the format. The file on disk has no such opinion.
    check("the bytes on disk are used as a floor under the reported count",
          "def fetched()" in src and "max(done[\"before\"] + got_b, fetched())" in src)

    # Picture and sound are two downloads that each count from zero. Summed, so
    # the bar cannot fill up and then start again from nothing near the end —
    # which reads as the job restarting itself.
    check("the two streams are summed rather than reported separately",
          'done["before"] += done["last"]' in src)
    # There is nothing to reset between attempts any more: the fetch is one
    # invocation and yt-dlp retries inside it, resuming rather than restarting,
    # so the counter never goes back to zero behind our back.
    check("the video is fetched in a single invocation",
          src.count("stream(cmd, show") == 1, str(src.count("stream(cmd, show")))

    # An age-restricted video refuses at the lookup, before the download that
    # knows about the user's cookies is ever reached — so the 403 message's
    # advice to name a browser in Settings has to work on the probe too.
    check("the lookup can use the browser cookies as well as the download",
          "def probe(url: str, cookies_from" in src)

    # H.264 first is now stated once, in the selector chain and in choose_format's
    # ranking, rather than restated by a fallback rung that rewrote the format
    # spec on its way past. Both are checked above.

    # When the site offers nothing playable the picture is converted rather than
    # shipped with a warning the user cannot act on.
    from app.steps.mux import transcode_h264, _H264_ENCODERS
    check("a hardware encoder is tried before the software one",
          _H264_ENCODERS[0][0] == "h264_videotoolbox")
    check("and there is a software fallback for a build without it",
          _H264_ENCODERS[-1][0] == "libx264")
    mux_src = (ROOT / "app" / "steps" / "mux.py").read_text()
    check("10-bit is converted down, since H.264 can carry it and little can play it",
          "yuv420p" in mux_src)
    check("audio and subtitles are copied, not re-encoded a second time",
          '"-map", "0", "-c", "copy", "-c:v"' in mux_src)

    # Progress is read from a template we define, not scraped out of text that
    # yt-dlp writes for a terminal and is free to change.
    check("no percentage is scraped from human-readable output", "_PCT" not in src)
    check("a progress template is asked for", "--progress-template" in src)
    check("and it carries the byte counts, not just a percentage",
          "downloaded_bytes" in src and "total_bytes" in src)

    # Whisper on the Apple GPU is one blocking call that swallows the whole
    # file. It reported 5% and then nothing until it was done — on a measured
    # 52-minute video that was 5m50s of frozen bar, under a label still reading
    # "Loading Whisper (first run downloads about 3 GB)", which it printed
    # whether or not anything was being downloaded. It does publish its
    # position, to a tqdm bar rather than to a callback, so the meter is
    # borrowed. Exercised against a stand-in module rather than by transcribing
    # something, which would put a 3 GB model and a minute of GPU in the way of
    # a unit test.
    from app.backends import asr as _asr
    import types as _types
    stand_in = _types.ModuleType("mlx_whisper.transcribe")
    stand_in.tqdm = _types.SimpleNamespace(tqdm=object)
    was_there = sys.modules.get("mlx_whisper.transcribe")
    sys.modules["mlx_whisper.transcribe"] = stand_in
    try:
        seen = []
        with _asr._mlx_whisper_progress(lambda f, m: seen.append((f, m)), 0.05, 0.98):
            # Built and driven exactly as mlx_whisper's decode loop does it:
            # a context manager over a frame total, updated by however far the
            # read head moved. disable=True is passed because it passes it —
            # and is ignored, which is the point. A tqdm subclass could not be
            # used here: its update() returns early when disabled, so the
            # counter would never move.
            with stand_in.tqdm.tqdm(total=100, unit="frames", disable=True) as bar:
                bar.update(25)
                bar.update(25)
                bar.update(50)
        check("whisper's own decode position is reported as progress",
              [round(f, 4) for f, _ in seen] == [0.2825, 0.515, 0.98], str(seen))
        check("and the stage says it is transcribing, not still loading",
              seen and all("Transcribing" in m for _, m in seen), str(seen))
        check("the borrowed meter is handed back afterwards",
              stand_in.tqdm.tqdm is object)
        # It reaches into another package's internals, so the failure that
        # matters is the one where those internals have moved: it must lose the
        # progress reporting and nothing else.
        del stand_in.tqdm
        ran = []
        with _asr._mlx_whisper_progress(lambda f, m: ran.append(f), 0.05, 0.98):
            ran.append("body ran")
        check("a mlx_whisper without a tqdm to borrow still transcribes",
              ran == ["body ran"], str(ran))
        sys.modules.pop("mlx_whisper.transcribe")
        ran2 = []
        with _asr._mlx_whisper_progress(lambda f, m: ran2.append(f), 0.05, 0.98):
            ran2.append("body ran")
        check("and so does one that is not imported at all", ran2 == ["body ran"])
    finally:
        if was_there is None:
            sys.modules.pop("mlx_whisper.transcribe", None)
        else:
            sys.modules["mlx_whisper.transcribe"] = was_there

    from app.config import can_decode_av1, mac_generation, AV1_FROM_GENERATION
    check("AV1 is gated on the generation that can decode it",
          AV1_FROM_GENERATION == 3)
    check("this machine reports a generation", mac_generation() >= 0)
    check("and the two agree",
          can_decode_av1() == (mac_generation() >= AV1_FROM_GENERATION))

    # Batches are capped by size as well as count: long joined lines crowd the
    # context window, which is exactly when a small model starts repeating.
    from app.backends.translate import _batches, BATCH, BATCH_CHARS
    short = _batches([{"text": "x" * 40} for _ in range(60)])
    check("short lines batch by count", max(len(b) for b in short) == BATCH,
          str([len(b) for b in short]))
    long_ = _batches([{"text": "y" * 400} for _ in range(20)])
    check("long ones batch by size instead",
          max(len(b) for b in long_) < BATCH
          and all(sum(len(s["text"]) for s in b) <= BATCH_CHARS for b in long_),
          str([len(b) for b in long_]))
    check("and a single over-long line is still carried, not dropped",
          [len(b) for b in _batches([{"text": "z" * 9000}])] == [1])

    # From the file: earlier tests replace _call_ollama with a stub and never
    # put it back, so inspect would be reading the fake.
    tsrc = (ROOT / "app" / "backends" / "translate.py").read_text()
    check("the local model is asked not to repeat itself",
          "repeat_penalty" in tsrc and "min_p" in tsrc)
    check("and told outright not to copy the source back",
          "NEVER COPY" in tsrc)


# ============================ 14. something to paste when it goes wrong
def test_observability():
    """The whole point is a non-technical user having one thing to send.

    Before this the app wrote nothing anyone could use: the log named in the
    README held twenty-two lines of the same deprecation warning, and the
    installer's held pip's "Requirement already satisfied" 136 times.
    """
    print("\n[14] Something to paste when it goes wrong")
    import importlib
    import logging
    import logging.handlers
    from app.config import Settings, detect_machine

    log_dir = SCRATCH / "logs"
    shutil.rmtree(log_dir, ignore_errors=True)
    os.environ["DUBBING_STUDIO_LOGS"] = str(log_dir)
    from app import logs as logs_mod
    importlib.reload(logs_mod)
    logs_mod.setup()

    # --- one file, JSON, one object per line, tagged with the job
    log = logs_mod.get("job-42")
    log.info("stage started", extra={"stage": "translate"})
    log.warning("translation batch incomplete, halving",
                extra={"asked": 25, "missing": 25})
    logs_mod.get().info("app start", extra={"engine": "mlx"})

    lines = [l for l in logs_mod.LOG_FILE.read_text().splitlines() if l.strip()]
    check("the log is one JSON object per line",
          all(isinstance(json.loads(l), dict) for l in lines), f"{len(lines)} lines")
    first = json.loads(lines[0])
    check("records carry a time, a level and the event",
          {"time", "level", "event"} <= set(first), str(sorted(first)))
    check("a record made during a job knows which job",
          first.get("job_id") == "job-42")
    check("and the fields passed at the call site survive",
          first.get("stage") == "translate", str(first))
    check("a record made outside a job carries no job id",
          "job_id" not in json.loads(lines[2]))

    # The translation retry and the quality check sit several layers below the
    # pipeline and are handed no job id. They are also the two places whose
    # records matter most, so the job rides on the thread instead.
    logs_mod.current_job.set("job-99")
    logs_mod.get().warning("translation batch incomplete, halving",
                           extra={"asked": 25})
    tagged = json.loads([l for l in logs_mod.LOG_FILE.read_text().splitlines() if l][-1])
    check("a backend that knows nothing about jobs still tags its records",
          tagged.get("job_id") == "job-99", str(tagged))
    logs_mod.current_job.set("")

    # A field named after one of LogRecord's own attributes used to raise, which
    # turned a failed job into a second failure inside the call reporting it.
    logs_mod.get("j").error("job failed", extra={"message": "boom", "name": "x"})
    last = json.loads([l for l in logs_mod.LOG_FILE.read_text().splitlines() if l][-1])
    check("a field clashing with LogRecord's own names is renamed, not fatal",
          last.get("message_") == "boom" and last.get("name") == "app", str(last))

    # --- rotation keeps the file bounded without losing the recent past
    check("a whole job's records fit in one file before it rotates",
          logs_mod.MAX_BYTES >= 5 * 1024 ** 2,
          f"{logs_mod.MAX_BYTES // 1024 ** 2} MB x {logs_mod.BACKUPS}")
    handler = logging.getLogger("app").handlers[0] if logging.getLogger("app").handlers \
        else logging.getLogger().handlers[0]
    check("rotation is the stdlib's, not something hand-rolled",
          isinstance(handler, logging.handlers.RotatingFileHandler),
          type(handler).__name__)
    check("recent() reads the log back as records",
          len(logs_mod.recent(3)) == 3)

    # --- the report
    from app import diagnostics
    settings = Settings.load()
    settings.anthropic_key = "sk-ant-LEAK-CANARY-1111"
    settings.openai_key = "sk-proj-LEAK-CANARY-2222"
    settings.save()
    text = diagnostics.report(limit=20)

    # The one check that must never fail. /api/state returns these in full, which
    # is tolerable over localhost and is not tolerable on a clipboard.
    check("no API key ever reaches the clipboard",
          "LEAK-CANARY" not in text, "both keys were set")
    check("but whether one is set is still reported",
          "anthropic_key: set" in text and "openai_key: set" in text)
    # Everything except the secrets, rather than a hand-kept list of the
    # interesting ones — an allowlist quietly omits each setting added after it,
    # and the missing one is always the one that explains the failure.
    from dataclasses import asdict as _asdict
    block = text.split("Settings\n")[1].split("\n\n")[0].splitlines()
    check("every setting reaches the report, not a chosen few",
          len(block) == len(_asdict(settings)), f"{len(block)} rows")
    check("and each is one scannable line, whatever is pasted into it",
          all(l.startswith("  ") and "\n" not in l for l in block))
    for section in ("This Mac", "Versions", "Setup check", "Settings", "Recent activity"):
        check(f"the report has a {section} section", section in text)
    check("it names the machine's memory", f"{detect_machine().ram_gb} GB" in text)
    check("it carries the recent log entries",
          "translation batch incomplete" in text)
    check("it is text, not JSON — this gets pasted into a chat window",
          not text.lstrip().startswith("{"))
    settings.anthropic_key = settings.openai_key = ""
    settings.save()

    # Whatever dies below Python never reaches the logger, so the bundle keeps
    # stderr — but in its own file. Appending it to the rotating log gave two
    # writers one path, and the redirect holds the old inode the moment the
    # handler rotates, so each quietly overwrites the other's work.
    launcher = (ROOT / "packaging" / "build_app.sh").read_text()
    check("stderr is not appended to the file the log handler rotates",
          "DubbingStudio-crash.log" in launcher
          and '2>> "\\$HOME/Library/Logs/DubbingStudio.log"' not in launcher)
    (logs_mod.LOG_DIR / "DubbingStudio-crash.log").write_text("libc++abi: terminating\n")
    check("and a native crash still reaches the report",
          "libc++abi: terminating" in diagnostics.report(limit=2))
    uninstall = (ROOT / "Uninstall.command").read_text()
    check("uninstall clears the rotated logs too, not just the live one",
          'DubbingStudio.log"*' in uninstall and "crash.log" in uninstall)

    # --- stage timing
    from app.pipeline import Job, JobRunner
    r, job = JobRunner(), Job(id="t", url="u")
    job.stage, job.stage_began = "download", time.time() - 2.0
    r._close_stage(job)
    check("a finished stage records how long it took",
          1.5 <= job.stage_times.get("download", 0) <= 3.0, str(job.stage_times))
    check("and the clock is stopped, not left running", job.stage_began == 0.0)
    job.stage, job.stage_began = "download", time.time() - 1.0
    r._close_stage(job)
    check("a stage entered twice adds up rather than overwriting",
          job.stage_times["download"] >= 2.5, str(job.stage_times))
    job.stage = ""
    r._close_stage(job)
    check("closing when nothing is running is harmless",
          list(job.stage_times) == ["download"])
    check("the running stage's elapsed is worked out on this machine's clock",
          "stage_elapsed" in Job(id="x", url="u").public())

    # The silence warning must not cry wolf. It decides on the share of the
    # track that is quiet, and that share is also just what a time-fitted dub
    # looks like: lines start on their original timestamps and English is
    # shorter than most of what it replaces. Measured on a real 98-minute run —
    # 92% speech in the original, 38% in the dub, all 448 lines spoken — and the
    # report told the user it was "what a half-failed run looks like".
    psrc = (ROOT / "app" / "pipeline.py").read_text()
    check("a share-of-silence warning is gated on something actually failing",
          "everything_spoken" in psrc and "checked.get(\"untranslated\")" in psrc)
    check("but a completely silent track is always reported",
          'stats.get("audio_present") is False' in psrc)

    # --- the installer
    inst = (ROOT / "Install.command").read_text()
    check("every line the user is told also reaches the install log",
          'note "$*"' in inst and 'note "WARN $*"' in inst)
    check("a failure puts the details on the clipboard, not a file path",
          "pbcopy" in inst)
    check("there is a fallback when the clipboard refuses",
          "Details are in this file" in inst)
    check("an unexpected death is caught too, not just the checked failures",
          "trap 'code=$?" in inst and "finish_badly" in inst)
    check("and the ending cannot fire twice", "HANDLED" in inst)
    inst_sh = (ROOT / "install.sh").read_text()
    check("the installer will not replace a folder that is not an install",
          "is_install" in inst_sh)
    check("an empty destination is not mistaken for occupied over a lone .DS_Store",
          ".DS_Store" in inst_sh)
    unin = (ROOT / "Uninstall.command").read_text()
    check("removal goes to the Bin rather than straight off the disk",
          "bin_it" in unin and "rm -rf" not in unin)

    # install.sh has no app to ask, and Uninstall.command's own question to it
    # can fail, so both carry a literal copy of the rule config.py computes via
    # platformdirs. Checked against platformdirs directly rather than against
    # app.config, since this process's DUBBING_STUDIO_HOME is the scratch
    # override above, not the real default the literals are meant to match.
    from platformdirs import user_cache_dir, user_data_dir
    _home = str(Path.home())
    _real_base = user_data_dir("DubbingStudio", appauthor=False).replace(_home, "$HOME")
    _real_cache = user_cache_dir("DubbingStudio", appauthor=False).replace(_home, "$HOME")
    check("install.sh's default install path still matches what platformdirs resolves",
          f"{_real_base}/program" in inst_sh)
    check("Uninstall.command's fallback data root still matches what platformdirs resolves",
          _real_base in unin)
    check("Uninstall.command's fallback cache root still matches what platformdirs resolves",
          _real_cache in unin)
    for script in ("Install.command", "install.sh", "Update.command",
                   "Uninstall.command"):
        ok = subprocess.run(["bash", "-n", str(ROOT / script)],
                            capture_output=True).returncode == 0
        check(f"{script} parses", ok)

    # --- the UI
    # The frontend is now split into one Web Component per panel rather than
    # one file, so each of these lives in the component that owns it.
    header_js = (ROOT / "app" / "static" / "js" / "components" / "app-header.js").read_text()
    doctor_js = (ROOT / "app" / "static" / "js" / "components" / "doctor-panel.js").read_text()
    failed_js = (ROOT / "app" / "static" / "js" / "components" / "failed-panel.js").read_text()
    diag_js = (ROOT / "app" / "static" / "js" / "components" / "diagnostics-panel.js").read_text()
    active_js = (ROOT / "app" / "static" / "js" / "components" / "active-panel.js").read_text()
    format_js = (ROOT / "app" / "static" / "js" / "format.js").read_text()
    settings_js = (ROOT / "app" / "static" / "js" / "components" / "settings-panel.js").read_text()
    main_js = (ROOT / "app" / "static" / "js" / "main.js").read_text()
    api_js = (ROOT / "app" / "static" / "js" / "api.js").read_text()
    check("a failed job can hand over the details",
          'diagnostics-panel")?.open()' in failed_js, "failed panel")
    # "It finished but the voice is wrong" is a report worth sending too, so the
    # details have to be reachable when nothing has visibly failed. They live in
    # Settings now rather than in the header.
    check("and they are reachable when nothing has failed, from Settings",
          'diagnostics-panel")?.open()' in settings_js
          and "copyBtn" not in header_js, "settings panel")
    check("the shipped defaults reach the panel that marks what differs from them",
          "settings_defaults: initial.settings_defaults" in main_js
          and "settings_defaults" in settings_js)
    check("a reset is asked about before it happens", "confirm(" in settings_js)
    check("and it goes through main.js and api.js like a save does",
          '"reset-settings"' in main_js and "resetSettings" in api_js
          and "/api/settings/reset" in api_js)
    # A blanket reset already spares the keys server-side; naming one in a
    # per-tab list would clear it by the back door.
    tabs_block = settings_js.split("const TABS = ", 1)[1].split("];", 1)[0]
    check("no tab's reset list names an API key",
          "anthropic_key" not in tabs_block and "openai_key" not in tabs_block)
    from app.config import Settings as _S
    check("the three settings a preset owns say so beside the field",
          settings_js.count("${PRESET_TAG}") == len(_S.PRESET_KEYS)
          and all(f'"{k}"' in settings_js for k in _S.PRESET_KEYS))
    # A creator set Original audio to keep the original quietly underneath,
    # meaning to keep the music bed, and got his original speech back too —
    # established from outside by holding everything else constant and
    # flipping only that one select. The panel used to just make the "mix it
    # back" control disappear when that happens, which told him nothing; it
    # has to say so in Settings, at the moment of choice, not only in a report
    # row read after the run.
    check("the panel explains itself instead of just disappearing when the "
          "music-and-effects mix-back is overridden",
          "keepMusicTag" in settings_js and "keepMusicHint" in settings_js
          and "keep_music_applies" in settings_js)
    check("the explanation points at how to actually get the bed under the "
          "new voices without the original speech",
          "Set Original audio " in settings_js
          and "to Replace completely for the music and effects" in settings_js)
    # A disabled control drops out of the tab order outright: tabbing from
    # "Music and effects" landed straight on "Speaking speed" and a keyboard
    # user never reached this field, its tag or its hint at all — the one
    # control this item exists to make legible was the one control a keyboard
    # user could not reach. The reviewer sided against disabling it for
    # exactly that reason, so the fix has to stay operable while inert rather
    # than frozen.
    check("the control stays enabled while overridden — dimmed, not disabled, "
          "so a keyboard user still reaches it",
          '.disabled = overridden' not in settings_js
          and '$("#keep_music").classList.toggle("dim", overridden)' in settings_js)
    # Sighted-only was the same complaint one layer up: a tag and a hint that
    # only ever sat on screen told a screen reader nothing when it landed on
    # the control they are about. Both ids ride on the one attribute assistive
    # tech already reads for every other hint in this panel.
    check("the tag and the hint are both reachable from the control itself, "
          "not merely present on screen",
          'aria-describedby="keepMusicTag keepMusicHint"' in settings_js)
    # "changed" and "Not in force" side by side contradicted each other, and
    # the dialog's own total counted a setting it had just said was inert.
    # shown() already keeps a hidden field out of that accounting; overridden()
    # is the same idea for a field that is on screen but not acting on
    # anything, and markChanged() has to consult it or the contradiction is
    # back the moment the control is visible rather than hidden.
    check("an overridden field is excluded from the changed count and flag "
          "the same way a hidden one already is",
          "&& this.shown(key) && !this.overridden(key)" in settings_js)
    check("overridden() is read off the tag applyConditions() already shows, "
          "not a second copy of keep_music_applies()'s own condition",
          "data-override=" in settings_js
          and "overridden(key){" in settings_js)

    # That mirror is close to structurally necessary — the panel has to react
    # to unsaved changes without a server round trip — but nothing checked the
    # two conditions actually agree, so a third condition added to one side
    # could silently desync from the other. Extracted verbatim from the
    # component rather than reimplemented, and run against the real
    # keep_music_applies() across every meaningful combination.
    mirror = "\n".join(l.strip() for l in settings_js.splitlines()
                       if l.strip().startswith(("const replacing", "const separating",
                                                 "const overridden")))
    combos = [(sep, mode) for sep in (True, False)
              for mode in ("replace", "duck:-12", "dual")]
    script = "\n".join(
        f'{{ const $ = sel => ({{value: sel === "#audio_mode" ? "{mode}" : '
        f'"{str(sep).lower()}"}}); {mirror.replace("this.$", "$")} '
        f'console.log(JSON.stringify({{sep: {str(sep).lower()}, mode: "{mode}", '
        f'overridden}})); }}'
        for sep, mode in combos
    )
    rows = [json.loads(l) for l in subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    ).stdout.splitlines()]
    for row in rows:
        applies = Settings(separate_audio=row["sep"],
                           audio_mode=row["mode"].split(":")[0]).keep_music_applies()
        check(f"the JS mirror agrees with the real keep_music_applies() for "
              f"separate_audio={row['sep']} audio_mode={row['mode']}",
              (row["sep"] and not row["overridden"]) == applies, row)

    # The truncation fix was giving the control room, not shorter words — this
    # is the property that actually did that, and until now nothing asserted
    # it, so the rejected "shorten the labels" fix could have silently come
    # back in its place.
    grid_full_block = settings_js.split('class="grid-full"', 1)[1][:200]
    check("Original audio spans the full grid row rather than sharing a half "
          "column, which is what actually fixed the truncation",
          'for="audio_mode"' in grid_full_block)
    # Duck mode ducks the whole original track, its own speech included, not
    # only the music riding along in it — the exact misreading that put a
    # creator's original speech under his dub when all he asked for was the
    # crowd noise. Saying so inside the option label pushed the dB figure —
    # the only thing telling the three duck levels apart — past what a closed
    # <select> shows, trading one unreadable thing for another; the labels
    # stay at their shipped length and the disambiguation lives in a hint
    # under the control instead, shown whenever there is a whole original in
    # the running to talk about.
    check("the Original-audio option labels stay short enough that the "
          "chosen one — duck level included — is still readable closed",
          "Keep quietly underneath — quiet (-12 dB)" in settings_js
          and "Keep as a second track" in settings_js
          and "Keep the whole original" not in settings_js)
    check("what keeping the original actually keeps is said once, in a hint, "
          "not folded into every option label",
          "audioModeHint" in settings_js
          and "not just its music" in settings_js)
    # Settings and the Quality report used to name this same option two
    # different ways once the Settings label grew a qualifier the report's own
    # copy never got — one control with two names in one product is exactly
    # the kind of thing that let the original bug hide. Reusing the shipped
    # option wording keeps them in step without the report needing its own
    # copy of the rule.
    report_js = (ROOT / "app" / "static" / "js" / "run-report.js").read_text()
    check("the Quality report names Original audio the same way Settings does",
          "Keep quietly underneath" in report_js
          and "Keep as a second track" in report_js)
    # "Voice" (which built-in voice speaks) and "Voices" (built-in versus
    # cloned) were a tester's own reported confusion in the audit that
    # produced this whole item, one letter apart and easy to mistake even
    # while a row apart. Making Original audio span the full grid row to fix
    # the truncation above closed that gap and put them side by side, which
    # would have sharpened the exact confusion already on file rather than
    # waiting for it to be renamed separately. "Cloning" is reused from
    # wording the app already uses for this same choice elsewhere, so nothing
    # new is coined and nothing shares a stem with "Voice" left to misread.
    check("the built-in-versus-cloned field is named apart from \"Voice\", "
          "not \"Voices\"",
          'label for="voice_mode">Cloning<' in settings_js
          and 'label for="voice_mode">Voices<' not in settings_js)
    check("and the Quality report calls it the same thing",
          'add("voice_mode", "Cloning"' in report_js)
    check("the engine is stated where the machine is described, not in the header",
          "engine" in doctor_js and "ram_gb" in doctor_js
          and 'class="engine"' not in header_js, "setup check")
    # Nine green-or-not rows never answered "is my machine the problem?" in
    # words. /api/doctor already computed the answer; the panel just had
    # nowhere to say it.
    check("Setup check states the verdict `ready` already computed, not just "
          "nine dots",
          "s.doctor.ready" in doctor_js, "doctor-panel.js")
    check("the not-ready wording still points at what to do, not just that "
          "something's wrong",
          "needs fixing" in doctor_js, "doctor-panel.js")
    # A green row and the finished run's own report used to disagree about the
    # same substituted model; the row can now say what happened without
    # turning the dot itself amber or red.
    check("a substituted model's explanation is shown even on an ok row",
          "c.note" in doctor_js, "doctor-panel.js")
    check("the text is shown before it is sent, not just copied",
          'id="diagText"' in diag_js)
    check("a refused clipboard tells the user what to do instead",
          "⌘C" in diag_js)
    check("the chips show how long each stage took",
          "stage_times" in active_js and "stage_elapsed" in active_js)
    check("durations reuse the existing formatter rather than a new one",
          "fmtShort" not in format_js and "fmtShort" not in active_js)

    # A run stays "current" — the success card keeps naming it — for as long as
    # nothing else has taken its place there, which can be indefinitely if
    # nobody starts another job. Filtering the dubbed-videos list on that same
    # flag meant a finished run waited for someone to move on before it showed
    # up below its own success message; a reload "fixed" it only because
    # reloading forgets what was current.
    hist_js = (ROOT / "app" / "static" / "js" / "components" / "history-list.js").read_text()
    check("a finished run is listed the moment it's done, not once something "
          "else takes its place as the current job",
          "s.current" not in hist_js)
    check("the list still leaves samples out, done or not",
          '"done" && !j.preview' in hist_js)
    check("the refresh reaches it through the store every panel already "
          "listens to, not main.js reaching into the history component",
          "history-list" not in main_js)
    settled_block = main_js.split("const settled = ", 1)[1].split(";", 1)[0]
    check("a run that stopped by cancelling or by failing leaves the same "
          "kind of workdir behind as one that finished, so the storage panel "
          "is walked again after all three, not just a success",
          '"cancelled"' in settled_block and '"error"' in settled_block)

    # A failed run used to have nowhere to be after a reload: /api/events
    # replays every known job, but main.js only ever took the view for one that
    # was running, already current, or queued with nothing else current — so an
    # errored job arrived over the wire and was never shown. Reachable after a
    # reload or a restart means not depending on "current" at all, the same way
    # the dubbed-videos list already does not.
    manage_js = (ROOT / "app" / "static" / "js" / "components" / "manage-panel.js").read_text()
    failed_list_js = (ROOT / "app" / "static" / "js" / "components" / "failed-list.js").read_text()
    # A past bug force-selected a tab from doctor data and walked over a
    # hand-picked one; the verdict line lives inside doctor-panel and must not
    # bring that back.
    check("the readiness verdict still leaves tab selection to manage-panel's "
          "own guard",
          "_userPicked" in manage_js and "revealDoctor" not in manage_js
          and "revealDoctor" not in doctor_js)
    check("a failed run is read straight from the job store, not from whichever "
          "job happens to be current",
          "s.current" not in failed_list_js)
    check("it is told apart from a dubbed video by its own status",
          'status !== "error"' in failed_list_js or 'status === "error"' in failed_list_js)
    check("it can still hand over the details and offer another attempt",
          'diagnostics-panel")?.open()' in failed_list_js
          and '"start-job"' in failed_list_js, "failed list")
    # The disclosure used to name the downloader by name, which stopped being
    # true the moment a translation failure started filling the same pane.
    check("the failed run's disclosure names no particular stage",
          "downloader" not in failed_js and "downloader" not in failed_list_js)
    check("both places that disclosure appears say it the same way",
          "Error details" in failed_js and "Error details" in failed_list_js)
    check("a failed run's disclosure says what settings it ran under, since a "
          "retry runs under whatever Settings holds now instead",
          "job.preset" in failed_list_js and "presetLabel" in failed_list_js)
    # A third tab would need its own logic to decide when to jump to it, and a
    # commit already had to undo exactly that kind of code walking over a
    # tab the user picked by hand. Sitting inside History instead means the
    # existing tab switch is the only thing that can show or hide it.
    check("a failed run sits inside the History tab rather than a tab of its "
          "own that would need its own switch-to logic",
          '"failed-list"' not in manage_js.split("const TABS = ", 1)[1].split("];", 1)[0])
    check("and shares that tab's visibility switch outright",
          '$("failed-list").hidden = key !== "history"' in manage_js)
    # update() is what auto-picks a tab, and it must never learn about jobs —
    # that is exactly the door a failure walking in and stealing the tab would
    # open back up.
    update_block = manage_js.split("update(s){", 1)[1].split("\n  }", 1)[0]
    check("a job failing never drives the auto-picked tab",
          "s.jobs" not in update_block, update_block.strip()[:200])

    # A finished sample has no list of its own the way a dubbed video or a
    # failure does, so reloading used to lose it outright: /api/events replays
    # every known job, but main.js only ever took the view for one that was
    # running, already current, or queued with nothing else current — a
    # finished sample matched none of those and sat in the store unseen, the
    # player, the report and "Dub it" gone.
    check("a finished sample can reclaim the view on reload, not just a job "
          "that is running or already queued",
          "function latestSample(jobs)" in main_js
          and "latestSample(jobs)" in main_js.split("async function boot", 1)[1])
    # A finished sample lives only in the view, so it is shown again only
    # while it is the most recent thing that happened, of any kind — anything
    # newer means the user has moved on to that instead.
    check("relevance is decided across every job, not just other samples, so "
          "a full run or a failure since outranks an old one",
          "activity(b) > activity(a)" in main_js)
    check("and only a finished sample ever reclaims the view this way — a "
          "finished full run already has its own card in the dubbed-videos "
          "list, and showing it here too would be two things claiming the "
          "same run",
          'newest.status === "done" && newest.preview' in main_js)
    boot_block = main_js.split("async function boot", 1)[1].split("\ndocument.addEventListener", 1)[0]
    check("the sample fallback only runs once nothing is actually live to "
          "steal the view",
          "if(live) store.setCurrent(live.id);\n  else {" in boot_block
          and "latestSample(jobs)" in boot_block)

    # The "Won't sleep" / "May sleep" pill read as a status badge rather than a
    # control: a tester clicked it just to see what it was and silently gave up
    # the thing keeping her Mac awake through a long dub. An info-tip beside it,
    # the same device Settings already uses to explain a control before it's
    # changed, says what it is and what turning it off means before that click.
    check("the awake pill has an info-tip beside it, the same device Settings "
          "uses to explain a control before it's changed",
          'id="awakeTip"' in header_js and "info-tip" in header_js)
    check("the tip explains both what being kept awake means and what "
          "clicking it off costs, not just repeating the pill's own label",
          "kept awake while a video is dubbing" in header_js
          and "pauses" in header_js and "until you wake it" in header_js)

    # A history row said e.g. "August 19 · 52s" beside the title — the time the
    # dub took, read next to a title as if it were the video's own length.
    check("a history row's duration says it is how long the dub took, not the "
          "video's own length",
          '`Took ${fmt(j.elapsed)}`' in hist_js)

    # The sample screen's note explaining what a 30-second sample can and
    # cannot promise is good, honest writing — styling it in the same amber
    # box as a real problem made a successful sample read as a second error.
    # pipeline.py now tags that note (and every other explanation-or-good-news
    # note) "info" at the source, rather than the panel guessing which one it
    # was by popping the last entry off the list.
    done_js = (ROOT / "app" / "static" / "js" / "components" / "done-panel.js").read_text()
    check("an info-tagged note is read by its kind, not by its position in "
          "the list",
          "notes.pop()" not in done_js
          and "const isInfo = n => n && typeof n === \"object\" && n.kind === \"info\";"
          in done_js)
    check("an info note is shown as plain text instead of in the warning box",
          'notes.filter(isInfo).map(n=>`<p class="hint" style="margin:0 0 12px">'
          in done_js)
    check("real notes — actual fallbacks and warnings, and any note recorded "
          "before this tagging existed — still get the warning box",
          'class="banner info" style="margin:0 0 12px">`\n          + warnings.map'
          in done_js)

    # Proven against the panel's own classifier, extracted verbatim, rather
    # than reimplemented here: a stored job or a history.json written before
    # this change holds notes as bare strings, and one of those must still
    # come out a warning instead of crashing the panel or vanishing.
    classifier = "\n".join(l.strip() for l in done_js.splitlines()
                           if l.strip().startswith(("const isInfo", "const text")))
    script = classifier + """
const oldStyle = "Speech and music could not be separated, so the original "
  + "soundtrack was replaced rather than kept.";
const tagged = {kind: "info", text: "This is a 30-second sample..."};
console.log(JSON.stringify({
  oldStyleIsInfo: isInfo(oldStyle), oldStyleText: text(oldStyle),
  taggedIsInfo: isInfo(tagged), taggedText: text(tagged),
}));
"""
    result = json.loads(subprocess.run(["node", "-e", script], capture_output=True,
                                       text=True, check=True).stdout)
    check("an old-style plain-string note is classified as a warning, its "
          "text unchanged",
          result["oldStyleIsInfo"] is False
          and result["oldStyleText"].startswith("Speech and music"), str(result))
    check("a new info-tagged note is classified as info, its text read from "
          "the tag rather than the object itself",
          result["taggedIsInfo"] is True
          and result["taggedText"] == "This is a 30-second sample...", str(result))

    # A finished sample can sit on screen after a reload while its working
    # files are gone underneath it — Storage -> Clear, or ordinary tidying.
    # "Dub it" promotes a sample without downloading again only while those
    # files are still there, so the button must not be offered once the panel
    # is already telling the user, via SAMPLE_GONE, that they are not — and
    # it must read that from `here`, the flag the panel computes once for
    # that message, not from a second check of its own.
    check("Dub it is gated on the same `here` the panel uses for the "
          "gone-away message, not on the sample flag alone",
          'sample\n        ? (here ? `<button class="primary" id="dEscalate">'
          in done_js)

    start = done_js.index("actions.innerHTML = (sample")
    end = done_js.index("`;", start) + 2
    actions_expr = done_js[start:end].replace(
        "actions.innerHTML = (sample", "const html = (sample")
    combos = [(True, True), (True, False), (False, True), (False, False)]
    script = "\n".join(
        f'{{ const sample={str(s).lower()}, here={str(h).lower()}; {actions_expr} '
        f'console.log(JSON.stringify({{sample, here, html}})); }}'
        for s, h in combos
    )
    rows = [json.loads(l) for l in subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    ).stdout.splitlines()]
    for row in rows:
        has_escalate = "dEscalate" in row["html"]
        has_reveal = "dReveal" in row["html"]
        wants_escalate = row["sample"] and row["here"]
        wants_reveal = (not row["sample"]) and row["here"]
        check(f"actions for sample={row['sample']} here={row['here']} offer "
              "exactly the control that matches, proven against the panel's "
              "own expression rather than reimplemented here",
              has_escalate == wants_escalate and has_reveal == wants_reveal,
              row["html"])
        check(f"Dub another is offered regardless (sample={row['sample']} "
              f"here={row['here']}) — resetting never depends on a file",
              "dReset" in row["html"])


# ============================ 15. terminology lifted from the video itself
def test_terminology():
    """The glossary the app ships is Spanish; the videos need not be.

    Measured on a real Portuguese job: not one built-in term matched, and the
    model still got 87-100% of the craft vocabulary right on its own. Where it
    did not know a word — "amêndoa" — it called the stitch "amaranth" 18 times
    and "shell" 15 times across one video. Asked directly, with the transcript
    in front of it, the same model names it correctly and identically every run.
    """
    print("\n[15] Terminology lifted from the video")
    from app.backends import translate as T

    TX = ("ponto baixo e ponto alto, com correntinha e ponto amêndoa. "
          "iniciar com dois pontos altos e finalizar a carreira.")

    def runs(*dicts):
        return list(dicts)

    good = {"ponto baixo": "single crochet", "ponto amêndoa": "shell stitch"}
    kept = T._agreed(runs(good, dict(good)), TX)
    check("terms both runs agree on are kept", kept == good, str(kept))

    check("a term only one run returned is dropped",
          "carreira" not in T._agreed(
              runs({**good, "carreira": "row"}, dict(good)), TX))
    check("a term the runs disagree on is dropped",
          "ponto amêndoa" not in T._agreed(
              runs({"ponto amêndoa": "shell stitch"},
                   {"ponto amêndoa": "cluster stitch"}), TX))

    # The dominant noise is not wrong terms, it is compositional phrases built
    # around them, which vary run to run and are not terms at all.
    phrase = {"iniciar com dois pontos altos": "start with two doubles"}
    check("a phrase built around a term is dropped",
          T._agreed(runs(phrase, dict(phrase)), TX) == {})

    # The first real run came back with "3 -> Today's lesson is about..." for
    # every line, because the model was enumerating rather than extracting.
    numbered = {"3": "Today's lesson is about making the second piece"}
    check("an enumerated transcript line is dropped",
          T._agreed(runs(numbered, dict(numbered)), TX) == {})
    check("a sentence-length rendering is dropped",
          T._agreed(runs({"ponto": "a stitch worked into the top of the previous row"},
                         {"ponto": "a stitch worked into the top of the previous row"}),
                    TX) == {})
    check("a term the video never says is dropped",
          T._agreed(runs({"tricô": "knitting"}, {"tricô": "knitting"}), TX) == {})

    # Known-good beats guessed, settled here rather than left to the model.
    merged = T._merge_glossaries("ponto baixo -> single crochet",
                                 "ponto baixo -> low stitch\nponto amêndoa -> shell stitch")
    check("a built-in term wins a collision with an extracted one",
          "low stitch" not in merged and "single crochet" in merged, merged)
    check("and the extracted term that adds something survives",
          "shell stitch" in merged)
    check("no extraction leaves the glossary untouched",
          T._merge_glossaries("a -> b", "") == "a -> b")

    # The terminology pass asks the same model a different question, so the
    # backends must carry their own system prompt. They did not: the first real
    # run inherited the translation prompt — "one line per input line,
    # id|translation" — and dutifully numbered 106 transcript lines.
    seen = {}

    def spy(prompt, model=None, host=None, key=None, system=None, **_):
        seen["system"] = system
        return "ponto amêndoa | shell stitch"

    T._call_ollama = spy
    from app.config import Settings
    st = Settings(); st.translator = "ollama"
    # Long enough to clear the two-batch floor, below which extraction is
    # skipped because a single-batch video has no cross-batch drift.
    long_enough = [{"text": TX}] * 60
    out = T.extract_terms(long_enough, st, 16)
    check("extraction asks with its own system prompt, not the translator's",
          seen.get("system") == T.EXTRACT_SYSTEM and seen["system"] != T.SYSTEM)
    check("and its result comes back as glossary lines",
          "ponto amêndoa -> shell stitch" in out, out)

    # A glossary is a nicety. A job that dies because the nicety failed is not.
    def broken(*a, **k):
        raise RuntimeError("ollama fell over")

    T._call_ollama = broken
    check("a failed terminology pass never fails the job",
          T.extract_terms(long_enough, st, 16) == "")

    check("a video too short to drift is not asked at all",
          T.extract_terms([{"text": TX}] * 3, st, 16) == "")

    check("a video too short to drift is not asked at all",
          T.extract_terms([{"text": TX}] * 3, st, 16) == "")

    psrc = (ROOT / "app" / "pipeline.py").read_text()
    check("the extracted terms are part of the translation fingerprint",
          "settings.glossary_text(),\n                                       found)" in psrc)
    check("and are cached rather than re-asked every run",
          'terminology.txt' in psrc and '"terminology", gprint' in psrc)



# ===================== 16. what a finished run keeps about itself
def test_job_record():
    """A finished run has to answer for a file the app cannot watch.

    Dubbed videos are moved and deleted in the Finder, usually with the app
    closed, and the settings that produced one are free to change the moment it
    is done — so the history has to carry the settings and work out the file
    afresh, and it has to do both for entries written before either existed.
    """
    print("\n[16] What a finished run records")
    from app import pipeline
    from app.config import HISTORY_FILE, OUTPUT_DIR, Settings

    # --- the settings a run was made under
    s = Settings()
    s.voice, s.speed, s.keep_video_quality = "bm_george", 1.2, "720"
    s.anthropic_key, s.openai_key = "sk-ant-SNAP-CANARY", "sk-oai-SNAP-CANARY"
    s.custom_glossary = "punto -> stitch"
    s.translator, s.anthropic_model = "anthropic", "claude-sonnet-5"
    s.youtube_cookies, s.keep_awake = "safari", False
    snap = s.run_snapshot()

    check("the record carries what changed the result",
          snap["voice"] == "bm_george" and snap["speed"] == 1.2
          and snap["keep_video_quality"] == "720"
          and snap["max_stretch"] == s.max_stretch, str(snap)[:70])
    check("including which translator and model did the words",
          snap["translator"] == "anthropic"
          and snap["translator_model"] == "claude-sonnet-5",
          str(snap.get("translator_model")))
    # A key in the history file is a key in a file nobody thinks of as holding
    # one, and history is copied around far more freely than settings.json is.
    check("no API key reaches the record",
          "CANARY" not in json.dumps(snap)
          and not any(k in snap for k in Settings.SECRET_KEYS), str(sorted(snap)))
    # Free text of no fixed size, which would be copied whole into every entry
    # it applied to.
    check("a custom glossary is recorded as a fact, not copied in",
          snap["has_custom_glossary"] is True and "punto" not in json.dumps(snap))
    # Preferences about this Mac, not about the video. Cookies decide whether
    # the video can be fetched at all; what actually arrived is measured off the
    # finished file.
    for skip in ("keep_awake", "youtube_cookies"):
        check(f"{skip} is left out of the record", skip not in snap)

    # --- which fields get recorded, asserted as a rule and not as a sample
    #
    # A hand-kept list of them drifts in the unsafe direction: add a setting that
    # changes the output, forget the list, and it is silently absent from every
    # entry written from then on with nothing anywhere to raise. Naming four of
    # them here would not catch that either — a hand-written expectation only
    # ever agrees with itself. So the record is the complement of a short list of
    # exclusions, and that is what is checked.
    import dataclasses as _dc
    declared = {f.name for f in _dc.fields(Settings)}
    missing = declared - set(Settings.recorded_keys()) - set(Settings.UNRECORDED_KEYS)
    check("every setting is recorded unless it is named as an exception",
          not missing, str(sorted(missing)))
    check("and no exception names a field that does not exist",
          set(Settings.UNRECORDED_KEYS) <= declared,
          str(sorted(set(Settings.UNRECORDED_KEYS) - declared)))
    check("the exceptions are the short list, not most of the dataclass",
          len(Settings.UNRECORDED_KEYS) < len(Settings.recorded_keys()),
          f"{len(Settings.UNRECORDED_KEYS)} of {len(declared)}")

    # The safe direction, demonstrated rather than asserted: a setting the app
    # does not yet have is in the record the moment it is declared.
    @_dc.dataclass
    class _NextRelease(Settings):
        pitch_shift: float = 0.0
    check("a setting added later is recorded without anyone remembering to",
          _NextRelease().run_snapshot().get("pitch_shift") == 0.0,
          str(sorted(_NextRelease().run_snapshot()))[:70])

    # --- a setting that had no bearing on the run is left out of the record
    #
    # Whether the separated music was mixed back is only a question when there was
    # separated music and the original audio was being replaced. Answered by
    # whether the key is there at all, so a panel that shows the row does not need
    # its own copy of the rule to decide.
    applied = Settings()
    applied.separate_audio, applied.audio_mode = True, "replace"
    applied.keep_music = False
    check("the choice is recorded when it applied, including when it said drop it",
          applied.run_snapshot().get("keep_music") is False)
    applied.keep_music = True
    check("and when it said keep it", applied.run_snapshot().get("keep_music") is True)
    for label, separate, mode in (("separation never ran", False, "replace"),
                                  ("the original was ducked instead", True, "duck"),
                                  ("the original was kept as a second track", True, "dual")):
        idle = Settings()
        idle.separate_audio, idle.audio_mode = separate, mode
        check(f"and left out when {label}",
              "keep_music" not in idle.run_snapshot(),
              str(sorted(idle.run_snapshot()))[:70])

    # --- whether the file is still there, asked every time it is asked for
    backup = HISTORY_FILE.read_text() if HISTORY_FILE.exists() else None
    here = OUTPUT_DIR / "record-still-here.mp4"
    gone = OUTPUT_DIR / "record-taken-away.mp4"
    try:
        for path in (here, gone):
            path.write_bytes(b"x" * 64)
        HISTORY_FILE.write_text(json.dumps([
            {"id": "rec-here", "title": "Still Here", "output": str(here),
             "status": "done", "started": 1, "finished": 2, "elapsed": 1,
             "stats": {"preset": "balanced"}},
            {"id": "rec-gone", "title": "Taken Away", "output": str(gone),
             "status": "done", "started": 3, "finished": 4, "elapsed": 1,
             "stats": {"preset": "balanced"}},
        ]))
        rows = {j["id"]: j for j in pipeline.runner.public_jobs()}
        check("a finished run whose file is there says so",
              rows["rec-here"]["output_exists"] is True)

        gone.unlink()
        rows = {j["id"]: j for j in pipeline.runner.public_jobs()}
        # Listed, because it happened; flagged, because it must not be offered
        # as something to open.
        check("one whose file has been deleted is still listed",
              "rec-gone" in rows, str(sorted(rows))[:70])
        check("and says plainly that the file is not there",
              rows["rec-gone"]["output_exists"] is False)
        check("nothing was rewritten into the history to work that out",
              not any("output_exists" in h
                      for h in json.loads(HISTORY_FILE.read_text())))
        # The stored answer would have been "yes" for both of these.
        here.unlink()
        rows = {j["id"]: j for j in pipeline.runner.public_jobs()}
        check("a file removed with the app closed flips the flag on the next read",
              rows["rec-here"]["output_exists"] is False)

        # --- entries written before any of this existed
        check("an old entry with no settings block is served without complaint",
              (rows["rec-here"].get("stats") or {}).get("settings") is None)
        # A history file outlives the build that wrote it, and these entries name
        # a setting the app no longer has. Snapshots are carried through as they
        # were found rather than validated against today's fields, so a run
        # recorded under a setting since removed still describes itself.
        HISTORY_FILE.write_text(json.dumps(
            [{"id": "rec-legacy", "output": str(here), "started": 7, "finished": 8,
              "stats": {"preset": "balanced",
                        "settings": {"preset": "balanced", "diarize": True,
                                     "expected_speakers": 4, "voice": "bf_emma"}}}]))
        rows = {j["id"]: j for j in pipeline.runner.public_jobs()}
        check("an entry recorded under a setting since removed is served intact",
              rows["rec-legacy"]["stats"]["settings"]["expected_speakers"] == 4,
              str(rows["rec-legacy"]["stats"]["settings"]))
        HISTORY_FILE.write_text(json.dumps(
            [{"id": "rec-ancient", "output": str(here), "started": 5, "finished": 6}]))
        rows = {j["id"]: j for j in pipeline.runner.public_jobs()}
        check("and one carrying nothing but a filename survives too",
              rows["rec-ancient"]["output_exists"] is False, str(rows["rec-ancient"]))

        # --- and what the writer itself puts on disk
        HISTORY_FILE.write_text("[]")
        job = pipeline.Job(id="rec-written", url="https://example.com/rec")
        job.status, job.output, job.finished = "done", str(here), time.time()
        job.stats = {"preset": "fast", "settings": Settings().run_snapshot()}
        check("a live job answers for its own file too",
              job.public()["output_exists"] is False)
        pipeline._record_history(job)
        stored = json.loads(HISTORY_FILE.read_text())
        check("a recorded run keeps the settings it used",
              stored[0]["stats"]["settings"]["voice"] == Settings().voice,
              str(stored[0]["stats"]["settings"])[:60])
        check("but never a claim about whether its file still exists",
              "output_exists" not in stored[0], str(sorted(stored[0]))[:80])

        # --- a failed run has to survive exactly as far as a finished one does
        #
        # The four-session audit this shipped to fix found the same complaint
        # from two testers who never spoke to each other: reload the page after
        # a failure and it is gone — no error, no Try again, no Copy details,
        # and nothing in the history panel, even though jobs/*/error.log sat on
        # disk the whole time. A fresh JobRunner stands in for a restart: its
        # self.jobs is empty exactly the way the real one is on every launch.
        #
        # Every access below goes through .get() rather than [] — a regression
        # here has to be reported by check() and let the rest of the suite keep
        # running, not abort the whole function on the first KeyError or
        # IndexError it happens to hit.
        HISTORY_FILE.write_text("[]")
        r = pipeline.JobRunner()
        failed = pipeline.Job(id="rec-failed", url="https://example.com/broke",
                              title="Broke It")
        failed.began = time.time() - 4
        failed.status, failed.finished = "error", time.time()
        failed.error = "That link isn't one yt-dlp recognises."
        failed.error_detail = "raw tool words"
        pipeline._record_history(failed)
        stored = json.loads(HISTORY_FILE.read_text())
        first = stored[0] if stored else {}
        check("a failed run is recorded the same way a finished one is",
              first.get("status") == "error" and first.get("error") == failed.error,
              str(first)[:100])
        check("carrying the detail a Copy-details style pane would show",
              first.get("error_detail") == "raw tool words")
        check("and the job it came from, so a live copy can be told from a "
              "recorded one that has no file to be told apart by",
              first.get("job_id") == "rec-failed")

        # While the job is still in memory the live copy answers for it, the
        # same as a job that just finished — the recorded entry must not
        # repeat the row.
        r.jobs["rec-failed"] = failed
        rows = [j.get("id") for j in r.public_jobs()]
        check("a failure still in memory is listed once, not twice",
              rows.count("rec-failed") == 1
              and not any(str(x).startswith("rec-failed-") for x in rows), str(rows))

        # Closing and reopening the app rebuilds self.jobs from nothing — the
        # in-memory copy is gone, and the record is what is left to read back.
        r.jobs.clear()
        rows = {j.get("id"): j for j in r.public_jobs()}
        recovered = [j for j in rows.values() if j.get("job_id") == "rec-failed"]
        first_recovered = recovered[0] if recovered else {}
        check("a restart still finds the failure, from the record alone",
              len(recovered) == 1, str(sorted(rows))[:100])
        check("with what was being dubbed, when, and why, all still readable",
              first_recovered.get("title") == "Broke It"
              and first_recovered.get("url") == "https://example.com/broke"
              and first_recovered.get("error") == failed.error,
              str(first_recovered)[:150])
        check("and Try again has what it would need — the same url and preview "
              "flag a fresh submission takes",
              "url" in first_recovered and "preview" in first_recovered)

        # --- a failure and a success are not the same kind of claim
        #
        # A repeat failure of the same link used to pile up as its own row —
        # three genuine failures of one link left three duplicate rows sitting
        # in the panel after a real restart. A failure is "what happened, most
        # recently", not "what happened, once" — so a later one replaces the
        # earlier rather than sitting beside it, and a later success replaces it
        # too: the link did finish in the end, and a "didn't finish" row next to
        # a "dubbed video" row for the very same run is two contradictory claims
        # about one thing, not two things that both happened.
        again = pipeline.Job(id="rec-failed", url="https://example.com/broke")
        again.began, again.finished = failed.finished + 1, failed.finished + 5
        again.status, again.error = "error", "A different failure this time."
        pipeline._record_history(again)
        stored = json.loads(HISTORY_FILE.read_text())
        matching = [h for h in stored if h.get("job_id") == "rec-failed"]
        check("a repeat failure of the same link replaces the earlier one, "
              "rather than sitting beside it",
              len(matching) == 1 and matching[0].get("error") == again.error,
              str(matching)[:150])

        succeeded = pipeline.Job(id="rec-failed", url="https://example.com/broke")
        succeeded.status = "done"
        succeeded.output = str(here)
        succeeded.finished = again.finished + 5
        pipeline._record_history(succeeded)
        stored = json.loads(HISTORY_FILE.read_text())
        matching = [h for h in stored if h.get("job_id") == "rec-failed"]
        check("and a success clears the recorded failure for that link outright",
              len(matching) == 1 and matching[0].get("status") == "done",
              str(matching)[:150])

        # A failure of the same link *after* the success must not be able to
        # take the success's row back — the file the success produced is still
        # on disk, and a failed retry does not un-produce it.
        failed_again = pipeline.Job(id="rec-failed", url="https://example.com/broke")
        failed_again.status, failed_again.error = "error", "failed on retry"
        failed_again.finished = succeeded.finished + 5
        pipeline._record_history(failed_again)
        stored = json.loads(HISTORY_FILE.read_text())
        matching = [h for h in stored if h.get("job_id") == "rec-failed"]
        check("a failure after a success does not erase the success",
              len(matching) == 2
              and {m.get("status") for m in matching} == {"done", "error"},
              str(matching)[:200])

        # Two real successes of the same link — the ordinary "-2" case — must
        # both still keep their row: successes are deduped by output, never by
        # job_id, so this fix cannot start collapsing them the way it collapses
        # failures.
        HISTORY_FILE.write_text("[]")
        for suffix in ("", "-2"):
            done = pipeline.Job(id="rec-twice", url="https://example.com/twice")
            done.status = "done"
            done.output = str(here) + suffix
            done.finished = time.time() + (1 if suffix else 0)
            pipeline._record_history(done)
        stored = json.loads(HISTORY_FILE.read_text())
        check("two successful runs of the same link both keep their row",
              len({h.get("output") for h in stored}) == 2, str(stored)[:200])

        # --- a live job only speaks for a recorded failure while it is itself
        #     the same failure
        #
        # The suppression in public_jobs() used to fire for a live job in *any*
        # state at that id. Retrying a link and then cancelling the retry made
        # the earlier, still-true failure disappear the instant the cancel
        # landed — as if the link had never failed — and it only came back after
        # the app was restarted. Nothing about the earlier failure changed by
        # being retried, so nothing but another live failure at that id should
        # be able to hide it.
        HISTORY_FILE.write_text("[]")
        r2 = pipeline.JobRunner()
        stale = pipeline.Job(id="rec-stale", url="https://example.com/stale")
        stale.status, stale.error, stale.finished = "error", "first failure", time.time()
        pipeline._record_history(stale)

        def recorded_failure_visible():
            return any(j.get("job_id") == "rec-stale" and j.get("status") == "error"
                      and j.get("id") != "rec-stale"
                      for j in r2.public_jobs())

        for live_status in ("running", "queued", "cancelled", "done"):
            live_job = pipeline.Job(id="rec-stale", url="https://example.com/stale")
            live_job.status = live_status
            if live_status == "done":
                live_job.output = str(gone)
            r2.jobs["rec-stale"] = live_job
            check(f"a live {live_status} job at that id does not silence a "
                  "true recorded failure",
                  recorded_failure_visible())

        live_job = pipeline.Job(id="rec-stale", url="https://example.com/stale")
        live_job.status = "error"
        r2.jobs["rec-stale"] = live_job
        check("a live job that is itself the same failure does suppress the "
              "recorded duplicate",
              not recorded_failure_visible())

        # --- the two kinds do not compete for the same shelf space
        #
        # Before failures were recorded at all, HISTORY_LIMIT was 50 finished
        # videos, full stop. Recording failures into the same flat, truncated
        # list would let a long enough losing streak — a stale yt-dlp, a host
        # that keeps refusing this machine, the exact shape of the audit this
        # item answers to — silently push every one of those videos off the
        # end, which is backwards: a losing streak is exactly when the history
        # panel, and the finished file it points at, matters most.
        HISTORY_FILE.write_text("[]")
        for i in range(3):
            done = pipeline.Job(id=f"rec-succ-{i}", url=f"https://example.com/ok{i}")
            done.status = "done"
            done.output = str(here) + f".{i}"
            done.finished = 1000.0 + i
            pipeline._record_history(done)
        for i in range(pipeline.FAILED_HISTORY_LIMIT + 10):
            broke = pipeline.Job(id=f"rec-fail-{i}", url=f"https://example.com/bad{i}")
            broke.status, broke.error = "error", "boom"
            broke.finished = 2000.0 + i
            pipeline._record_history(broke)
        stored = json.loads(HISTORY_FILE.read_text())
        kept_succ = [h for h in stored if h.get("output")]
        kept_fail = [h for h in stored if not h.get("output")]
        check("a run of failures well past the failure cap leaves every "
              "recorded success untouched",
              len(kept_succ) == 3, str(len(kept_succ)))
        check("and is itself capped at its own limit, not HISTORY_LIMIT",
              len(kept_fail) == pipeline.FAILED_HISTORY_LIMIT, str(len(kept_fail)))
        check("keeping the most recent failures, not the oldest",
              {h.get("job_id") for h in kept_fail} ==
              {f"rec-fail-{i}" for i in range(10, pipeline.FAILED_HISTORY_LIMIT + 10)},
              str(sorted(h.get("job_id") for h in kept_fail))[:120])
        check("the file itself stays newest-last, whichever kind trimmed it",
              all(stored[i].get("finished", 0) <= stored[i + 1].get("finished", 0)
                  for i in range(len(stored) - 1)))

        # And the other way round: a long run of successes must not be able to
        # starve the failure shelf either — the two caps are independent, not
        # one budget split at read time.
        HISTORY_FILE.write_text("[]")
        for i in range(2):
            broke = pipeline.Job(id=f"rec-fail2-{i}", url=f"https://example.com/bad2-{i}")
            broke.status, broke.error = "error", "boom"
            broke.finished = 1000.0 + i
            pipeline._record_history(broke)
        for i in range(pipeline.HISTORY_LIMIT + 5):
            done = pipeline.Job(id=f"rec-succ2-{i}", url=f"https://example.com/ok2-{i}")
            done.status = "done"
            done.output = str(gone) + f".{i}"
            done.finished = 2000.0 + i
            pipeline._record_history(done)
        stored = json.loads(HISTORY_FILE.read_text())
        kept_succ = [h for h in stored if h.get("output")]
        kept_fail = [h for h in stored if not h.get("output")]
        check("a run of successes well past HISTORY_LIMIT leaves both recorded "
              "failures untouched",
              len(kept_fail) == 2, str(len(kept_fail)))
        check("and is itself capped at HISTORY_LIMIT",
              len(kept_succ) == pipeline.HISTORY_LIMIT, str(len(kept_succ)))
    finally:
        here.unlink(missing_ok=True)
        gone.unlink(missing_ok=True)
        if backup is None:
            HISTORY_FILE.unlink(missing_ok=True)
        else:
            HISTORY_FILE.write_text(backup)


# =================================== 17. a run that dubbed almost nothing
def test_near_empty_dub():
    """A run can succeed completely and still have found almost nothing to say.

    The app already speaks up, unprompted, about a speaker count that is
    probably wrong and about a translation model that is probably too weak.
    A run where the whole file carries one line and no dubbing anywhere else
    deserves the same rather than a plain "Done" — real audit case: a 1m59s
    clip came back with one line spoken, 97% of it with no dubbed line over
    it, and nothing in the report said so.
    """
    print("\n[17] A run that dubbed almost nothing")
    from app import pipeline
    from app.backends import translate as T
    from app.config import Settings

    work = WORK / "near-empty"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"
    DURATION = 40.0
    if not clip.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                        "-map", "0:v", "-map", "1:a", "-t", f"{DURATION:g}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)

    URL = "https://example.com/near-empty"
    from app.config import JOBS as JOBS_DIR
    shutil_rmtree(JOBS_DIR / pipeline._job_id(URL))

    class TinyEngine:
        name = "Test Engine"
        sample_rate = 24000

        def say(self, text, voice="", speed=1.0, speaker=0):
            return _tone(300, 1.0, 24000), 24000

    fake_probe, fake_download = stub_download(clip, "Almost Nothing Spoken", DURATION)

    # One short line in the middle of a much longer video — the shape of a
    # largely-sung short with a single spoken line in it.
    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        if progress:
            progress(1.0, "Heard 1 line")
        return [{"start": 18.0, "end": 19.0, "text": "one line"}]

    def fake_llm(prompt, model=None, host=None, key=None, **_):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|That is the only line spoken." for i in ids)

    real_probe = pipeline.download.probe
    real_download = pipeline.download.download
    real_transcribe = pipeline.asr_backend.transcribe
    real_make = pipeline.JobRunner._make_engine
    real_prune = pipeline.prune_workdir
    pipeline.download.probe = fake_probe
    pipeline.download.download = fake_download
    pipeline.asr_backend.transcribe = fake_transcribe
    pipeline.JobRunner._make_engine = lambda *a, **k: (TinyEngine(), False)
    pipeline.prune_workdir = lambda workdir: 0
    T._call_ollama = fake_llm

    try:
        s = Settings().apply_preset("fast")
        s.translator = "ollama"
        job = pipeline.runner.submit(URL, s)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 300:
            time.sleep(1)

        check("the job still succeeds — one real line, correctly dubbed, "
              "is not a failure", job.status == "done", f"{job.status}: {job.error}")
        if job.status != "done":
            return

        s_ = job.stats
        check("only the one line was spoken", s_.get("lines_spoken") == 1,
              str(s_.get("lines_spoken")))
        check("almost all of the video carries no dubbed line",
              s_.get("no_line_share", 0) >= 0.9, str(s_.get("no_line_share")))
        check("a run that dubbed almost nothing says so, plainly",
              any("1 line" in note_text(n) and "dubbed" in note_text(n)
                  for n in s_.get("notes", [])),
              str(s_.get("notes")))
        check("the note does not claim the run failed",
              not any("fail" in note_text(n).lower() for n in s_.get("notes", [])),
              str(s_.get("notes")))
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.asr_backend.transcribe = real_transcribe
        pipeline.JobRunner._make_engine = real_make
        pipeline.prune_workdir = real_prune


# ================== 18. a translation failure leaves finished lines behind
def test_translate_resume():
    """A translation that dies part way must not throw the finished lines away.

    Before this, translate() built its results in an in-memory dict that was
    only ever handed to the caller on a clean return — so a TranslationError
    three quarters of the way through a long video (a rate limit, a key that
    got revoked mid-run, a provider outage) discarded every line already
    translated, and pressing "try again" paid for the whole stage twice.
    """
    print("\n[18] Resuming a translation that failed part way")
    from app import pipeline
    from app.backends import translate as T
    from app.config import Settings, JOBS as JOBS_DIR

    class TinyEngine:
        name = "Test Engine"
        sample_rate = 24000

        def say(self, text, voice="", speed=1.0, speaker=0):
            return _tone(300, 0.3, 24000), 24000

    # Spaced well past merge_adjacent's gap threshold, so a change to
    # merge_lines (used below to invalidate the cache) changes the fingerprint
    # without changing which lines it applies to — the two settings this test
    # cares about are kept independent of each other.
    N = 26
    DURATION = 42.0

    def make_segments():
        return [{"start": n * 1.5, "end": n * 1.5 + 1.0, "text": f"linea {n}"}
                for n in range(N)]

    work = WORK / "translate-resume"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"
    if not clip.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                        "-map", "0:v", "-map", "1:a", "-t", f"{DURATION:g}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)

    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        if progress:
            progress(1.0, "Heard the lines")
        return make_segments()

    real_probe = pipeline.download.probe
    real_download = pipeline.download.download
    real_transcribe = pipeline.asr_backend.transcribe
    real_make = pipeline.JobRunner._make_engine
    real_prune = pipeline.prune_workdir
    real_ollama = T._call_ollama
    pipeline.asr_backend.transcribe = fake_transcribe
    pipeline.JobRunner._make_engine = lambda *a, **k: (TinyEngine(), False)
    pipeline.prune_workdir = lambda workdir: 0

    def run(url, settings, call, timeout=300):
        fake_probe, fake_download = stub_download(clip, "Translate Resume", DURATION)
        pipeline.download.probe = fake_probe
        pipeline.download.download = fake_download
        T._call_ollama = call
        job = pipeline.runner.submit(url, settings)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < timeout:
            time.sleep(0.5)
        return job

    def ids_in(prompt):
        return [int(l.split("|")[0]) for l in prompt.splitlines()
                if l and l[0].isdigit() and "|" in l]

    try:
        # --- A. fails after the first batch (25 of 26 lines) -----------
        URL_A = "https://example.com/translate-resume-a"
        shutil_rmtree(JOBS_DIR / pipeline._job_id(URL_A))

        calls = {"n": 0}

        def dies_after_first_batch(prompt, model, host="x", **_):
            calls["n"] += 1
            ids = ids_in(prompt)
            if calls["n"] > 1:
                raise T.TranslationError("simulated outage partway through")
            return "\n".join(f"{i}|English line {i}" for i in ids)

        settings_a = Settings().apply_preset("fast")
        settings_a.translator = "ollama"
        job_a = run(URL_A, settings_a, dies_after_first_batch)

        check("the job fails rather than silently losing the batch",
              job_a.status == "error", f"{job_a.status}: {job_a.error}")
        check("the failure itself is the one that was raised",
              "simulated outage" in job_a.error, job_a.error)

        pcache = JOBS_DIR / job_a.id / "translated.partial.json"
        tcache = JOBS_DIR / job_a.id / "translated.json"
        check("the completed batch was written to disk before the failure",
              pcache.exists())
        partial = json.loads(pcache.read_text()) if pcache.exists() else {}
        check("exactly the finished lines are recoverable, no more and no fewer",
              sorted(int(k) for k in partial) == list(range(25)),
              str(sorted(int(k) for k in partial)))
        check("no completed translation exists to be mistaken for the real thing",
              not tcache.exists())

        # --- B. retrying with the same settings resumes, not restarts ---
        requested: list[int] = []

        def records_and_answers(prompt, model, host="x", **_):
            ids = ids_in(prompt)
            requested.extend(ids)
            return "\n".join(f"{i}|English line {i}" for i in ids)

        settings_b = Settings().apply_preset("fast")
        settings_b.translator = "ollama"
        job_b = run(URL_A, settings_b, records_and_answers)

        check("the retry completes", job_b.status == "done",
              f"{job_b.status}: {job_b.error}")
        if job_b.status == "done":
            check("only the line that was still missing was ever asked for",
                  requested == [25], str(requested))
            check("the resume was reported",
                  any("already translated" in note_text(n)
                      for n in job_b.stats.get("notes", [])),
                  str(job_b.stats.get("notes")))
            check("every line ended up spoken, not just the resumed one",
                  job_b.stats.get("lines_spoken") == N,
                  str(job_b.stats.get("lines_spoken")))
        check("the partial file is cleared once the translation is complete",
              not pcache.exists())

        # --- C. a setting that changes the fingerprint gets no head start ---
        URL_C = "https://example.com/translate-resume-c"
        shutil_rmtree(JOBS_DIR / pipeline._job_id(URL_C))

        calls["n"] = 0     # dies_after_first_batch is reused; give it a clean count
        settings_c1 = Settings().apply_preset("fast")
        settings_c1.translator = "ollama"
        settings_c1.merge_lines = True
        job_c1 = run(URL_C, settings_c1, dies_after_first_batch)
        check("the first attempt on link C fails, leaving a partial behind",
              job_c1.status == "error", f"{job_c1.status}: {job_c1.error}")

        requested_c: list[int] = []

        def records_all(prompt, model, host="x", **_):
            ids = ids_in(prompt)
            requested_c.extend(ids)
            return "\n".join(f"{i}|English line {i}" for i in ids)

        # merge_lines is folded into the same fingerprint as everything else
        # translation depends on; the segments themselves are unaffected,
        # since they're spaced well past merge_adjacent's gap threshold.
        settings_c2 = Settings().apply_preset("fast")
        settings_c2.translator = "ollama"
        settings_c2.merge_lines = False
        job_c2 = run(URL_C, settings_c2, records_all)

        check("the retry under different settings still completes",
              job_c2.status == "done", f"{job_c2.status}: {job_c2.error}")
        if job_c2.status == "done":
            check("a settings change discards the stale partial instead of "
                  "resuming from it — every line is asked for again",
                  sorted(set(requested_c)) == list(range(N)), str(sorted(set(requested_c))))
            check("no resume was reported, since none happened",
                  not any("already translated" in note_text(n)
                          for n in job_c2.stats.get("notes", [])),
                  str(job_c2.stats.get("notes")))
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.asr_backend.transcribe = real_transcribe
        pipeline.JobRunner._make_engine = real_make
        pipeline.prune_workdir = real_prune
        T._call_ollama = real_ollama


def test_translate_resume_respects_settings_change():
    """A partial left by one run's settings must never be adopted by another's.

    The cache-meta stamp used to be written the moment a translation started,
    not when its content actually changed on disk. A run under new settings
    that stamped its own fingerprint and then died before its first batch
    landed left the *previous* settings' lines on disk claiming the *new*
    fingerprint — so the next attempt under the new settings resumed from
    them and would have quietly mixed two runs' translations into one dub.
    """
    print("\n[19] A settings change must not adopt another run's stale partial")
    from app import pipeline
    from app.backends import translate as T
    from app.config import Settings, JOBS as JOBS_DIR

    class TinyEngine:
        name = "Test Engine"
        sample_rate = 24000

        def say(self, text, voice="", speed=1.0, speaker=0):
            return _tone(300, 0.3, 24000), 24000

    N = 26
    DURATION = 42.0

    def make_segments():
        return [{"start": n * 1.5, "end": n * 1.5 + 1.0, "text": f"linea {n}"}
                for n in range(N)]

    work = WORK / "translate-resume-settings"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"
    if not clip.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                        "-map", "0:v", "-map", "1:a", "-t", f"{DURATION:g}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)

    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        if progress:
            progress(1.0, "Heard the lines")
        return make_segments()

    real_probe = pipeline.download.probe
    real_download = pipeline.download.download
    real_transcribe = pipeline.asr_backend.transcribe
    real_make = pipeline.JobRunner._make_engine
    real_prune = pipeline.prune_workdir
    real_ollama = T._call_ollama
    pipeline.asr_backend.transcribe = fake_transcribe
    pipeline.JobRunner._make_engine = lambda *a, **k: (TinyEngine(), False)
    pipeline.prune_workdir = lambda workdir: 0

    def run(url, settings, call, timeout=300):
        fake_probe, fake_download = stub_download(clip, "Translate Resume Settings", DURATION)
        pipeline.download.probe = fake_probe
        pipeline.download.download = fake_download
        T._call_ollama = call
        job = pipeline.runner.submit(url, settings)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < timeout:
            time.sleep(0.5)
        return job

    def ids_in(prompt):
        return [int(l.split("|")[0]) for l in prompt.splitlines()
                if l and l[0].isdigit() and "|" in l]

    try:
        URL = "https://example.com/translate-resume-settings"
        shutil_rmtree(JOBS_DIR / pipeline._job_id(URL))

        # --- A completes one batch (25 of 26 lines), then dies ------------
        calls_a = {"n": 0}

        def a_dies_after_first_batch(prompt, model, host="x", **_):
            calls_a["n"] += 1
            ids = ids_in(prompt)
            if calls_a["n"] > 1:
                raise T.TranslationError("simulated outage partway through")
            return "\n".join(f"{i}|EN-A {i}" for i in ids)

        settings_a = Settings().apply_preset("fast")
        settings_a.translator = "ollama"
        settings_a.merge_lines = True
        job_a = run(URL, settings_a, a_dies_after_first_batch)
        check("A fails, leaving a partial behind",
              job_a.status == "error", f"{job_a.status}: {job_a.error}")

        pcache = JOBS_DIR / job_a.id / "translated.partial.json"
        check("A's partial is on disk", pcache.exists())

        # --- settings change to B; B dies before its first batch lands ----
        def b_dies_immediately(prompt, model, host="x", **_):
            raise T.TranslationError("simulated outage before any batch landed")

        settings_b = Settings().apply_preset("fast")
        settings_b.translator = "ollama"
        settings_b.merge_lines = False
        job_b1 = run(URL, settings_b, b_dies_immediately)
        check("B's first attempt fails before writing anything",
              job_b1.status == "error", f"{job_b1.status}: {job_b1.error}")

        # --- the next attempt under B must not resume from A's lines ------
        requested_b: list[int] = []

        def b_records_and_answers(prompt, model, host="x", **_):
            ids = ids_in(prompt)
            requested_b.extend(ids)
            return "\n".join(f"{i}|EN-B {i}" for i in ids)

        job_b2 = run(URL, settings_b, b_records_and_answers)
        check("B's retry completes", job_b2.status == "done",
              f"{job_b2.status}: {job_b2.error}")
        check("every line was asked for again under B, none adopted from A",
              sorted(set(requested_b)) == list(range(N)), str(sorted(set(requested_b))))
        check("no resume was reported, since A's lines belong to a different run",
              not any("already translated" in note_text(n)
                      for n in job_b2.stats.get("notes", [])),
              str(job_b2.stats.get("notes")))

        tcache = JOBS_DIR / job_b2.id / "translated.json"
        saved = json.loads(tcache.read_text()) if tcache.exists() else []
        check("none of A's translated text leaked into B's finished dub",
              not any("EN-A" in s.get("translation", "") for s in saved),
              str([s.get("translation") for s in saved]))
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.asr_backend.transcribe = real_transcribe
        pipeline.JobRunner._make_engine = real_make
        pipeline.prune_workdir = real_prune
        T._call_ollama = real_ollama


def test_preview_failure_is_superseded_by_a_later_success():
    """A sample that fails and then succeeds must not stay listed as failed.

    _fail() recorded a history row for a preview job just as it does for a
    full run, but the success path deliberately skips previews (a finished
    sample gets no history row — it is not a thing the user asked to have).
    Nothing later could ever supersede that failure row, so a sample that
    failed once and then worked kept sitting under "Runs that didn't finish"
    even after it worked.
    """
    print("\n[20] A failed sample does not stay listed as failed once it works")
    from app import pipeline
    from app.backends import translate as T
    from app.config import HISTORY_FILE, JOBS as JOBS_DIR, Settings

    work = WORK / "preview-failure"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"
    DURATION = 60.0
    if not clip.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                        "-map", "0:v", "-map", "1:a", "-t", f"{DURATION:g}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)

    def fake_transcribe(audio_wav, use_mlx, model="parakeet", progress=None):
        if progress:
            progress(1.0, "Heard a line")
        return [{"start": 1.0, "end": 4.0, "text": "una linea"}]

    def fails(prompt, model=None, host=None, key=None, **_):
        raise T.TranslationError("simulated translator outage")

    def succeeds(prompt, model=None, host=None, key=None, **_):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|A dubbed line." for i in ids)

    real_probe, real_download = pipeline.download.probe, pipeline.download.download
    real_transcribe = pipeline.asr_backend.transcribe
    real_ollama = T._call_ollama
    pipeline.asr_backend.transcribe = fake_transcribe

    try:
        URL = "https://example.com/preview-fails-then-succeeds"
        shutil_rmtree(JOBS_DIR / pipeline._link_id(URL))
        s = Settings().apply_preset("fast")
        s.translator = "ollama"
        job_id = pipeline._job_id(URL, True)

        pipeline.download.probe, pipeline.download.download = stub_download(
            clip, "Preview Fails Then Succeeds", DURATION)
        T._call_ollama = fails
        job = pipeline.runner.submit(URL, s, preview=True)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 300:
            time.sleep(0.5)
        check("the sample fails", job.status == "error", f"{job.status}: {job.error}")

        def failed_rows():
            history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
            return [h for h in history if h.get("job_id") == job_id and not h.get("output")]

        T._call_ollama = succeeds
        retry = pipeline.runner.submit(URL, s, preview=True)
        t0 = time.time()
        while retry.status in ("queued", "running") and time.time() - t0 < 300:
            time.sleep(0.5)
        check("the retry succeeds", retry.status == "done", f"{retry.status}: {retry.error}")

        check("a sample that later succeeded is not left listed as failed",
              not failed_rows(), str(failed_rows()))
    finally:
        pipeline.download.probe = real_probe
        pipeline.download.download = real_download
        pipeline.asr_backend.transcribe = real_transcribe
        T._call_ollama = real_ollama


# ================================================= N. a video already on this Mac
def test_local_file():
    """Dubbing a file off the disk rather than a link off a site.

    The interesting differences are all invisible from the outside: nothing is
    fetched, nothing is copied, and the job folder has to be keyed on something
    other than a string that can quietly come to mean a different video.
    """
    print("\n[N] Local video files")
    from app import pipeline
    from app.backends import translate as T
    from app.config import JOBS, OUTPUT_DIR, Settings
    from app.steps import download as dl

    work = WORK / "local"
    work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ telling the two apart
    home = str(Path.home())
    cases = [
        ("https://youtu.be/abc", "https://youtu.be/abc", False),
        ("  https://youtu.be/abc  ", "https://youtu.be/abc", False),
        ("~/Movies/a.mp4", f"{home}/Movies/a.mp4", True),
        ("file:///Users/x/My%20Video.mp4", "/Users/x/My Video.mp4", True),
        ('"/Users/x/My Video.mp4"', "/Users/x/My Video.mp4", True),
        # Quotes come off whatever is inside them. Asking about http:// first
        # read this as a filename and reported "no file at .../https:/www…".
        ('"https://www.youtube.com/watch?v=abc"',
         "https://www.youtube.com/watch?v=abc", False),
        ("'https://youtu.be/abc'", "https://youtu.be/abc", False),
        ('""', "", False),
        ("", "", False),
    ]
    for typed, want, local in cases:
        got = dl.normalise_source(typed)
        check(f"{typed!r} reads as {want!r}", got == want, got)
        check(f"and is {'a file' if local else 'not a file'}",
              dl.is_local_source(got) is local)

    # A path is made absolute so that two spellings of one video cannot become
    # two jobs racing over one folder.
    check("a relative path is made absolute",
          Path(dl.normalise_source("clip.mp4")).is_absolute())

    # The one place the guess matters: a link typed without its scheme, which
    # would otherwise be reported as a missing file.
    check("a scheme-less link is spotted", dl.looks_like_bare_link("youtube.com/watch?v=x"))
    check("and so is one with www on the front", dl.looks_like_bare_link("www.vimeo.com/1"))
    check("but a filename with a dot in it is not",
          not dl.looks_like_bare_link("holiday.mp4"))
    check("nor is anything that starts like a path",
          not dl.looks_like_bare_link("/Users/me/a.b.mp4")
          and not dl.looks_like_bare_link("~/a.b.mkv"))

    # ------------------------------------------------------- what ffprobe sees
    clip = work / "local-clip.mp4"
    if not clip.exists():
        audio = speech_wav(work / "speech.wav", seconds=70.0)
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15",
                        "-i", str(audio), "-map", "0:v", "-map", "1:a", "-t", "70",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                        str(clip)], check=True)

    info = dl.probe_local(clip)
    check("a local file describes itself like a link does",
          set(info) >= {"title", "duration", "uploader", "formats", "is_live"},
          str(sorted(info)))
    check("its title is the filename", info["title"] == "local-clip", info["title"])
    check("its duration comes from the file", 69 < info["duration"] < 71,
          str(info["duration"]))
    # The empty list is the honest answer, and the one choose_format() already
    # knows how to read: there is nothing to choose between.
    check("it publishes no formats to choose between", info["formats"] == [])
    check("and choose_format agrees there is nothing to pick",
          dl.choose_format(info, "best") is None)

    silent = work / "silent.mp4"
    if not silent.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc=size=160x120:rate=10", "-t", "2",
                        "-c:v", "libx264", "-preset", "ultrafast", str(silent)],
                       check=True)
    try:
        dl.probe_local(silent)
        check("a file with no sound is refused", False, "it was accepted")
    except dl.DownloadError as exc:
        check("a file with no sound is refused, in plain English",
              "no sound" in str(exc), str(exc))

    soundonly = work / "sound-only.m4a"
    if not soundonly.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440", "-t", "2", str(soundonly)],
                       check=True)
    try:
        dl.probe_local(soundonly)
        check("a file with no picture is refused", False, "it was accepted")
    except dl.DownloadError as exc:
        check("a file with no picture is refused, in plain English",
              "no picture" in str(exc), str(exc))

    try:
        dl.probe_local(work / "not-here.mp4")
        check("a file that isn't there is refused", False, "it was accepted")
    except dl.DownloadError as exc:
        check("a file that isn't there is refused",
              "no file at" in str(exc), str(exc))

    # -------------------------------------------- what names the job folder
    key = pipeline._source_key(str(clip), local=True)
    check("the same file twice is the same job", key == pipeline._source_key(str(clip), True))
    check("and a link is still keyed on the link alone",
          pipeline._source_key("https://x/y") == pipeline._link_id("https://x/y"))

    moved = work / "same-bytes-elsewhere.mp4"
    shutil.copy(clip, moved)
    check("the same footage under another name is another job",
          pipeline._source_key(str(moved), True) != key)

    # The case this exists for: an export written again over the top of the last
    # one. Every cached stage in that folder was made from the old footage.
    rewritten = work / "rewritten.mp4"
    shutil.copy(clip, rewritten)
    before = pipeline._source_key(str(rewritten), True)
    time.sleep(1.1)                       # the key holds mtime to the second
    shutil.copy(silent, rewritten)
    check("replacing the file starts a fresh job rather than resuming the old one",
          pipeline._source_key(str(rewritten), True) != before)

    # ------------------------------------------------------- the stage plan
    settings = Settings()
    settings.separate_audio = False
    settings.diarize = False
    plan = pipeline.runner._plan(settings, local=True)
    check("the fetching step is renamed for a file that is already here",
          plan["download"][2] == "Opening the video file", plan["download"][2])
    check("and it is worth a fraction of a real download",
          plan["download"][1] < pipeline.runner._plan(settings)["download"][1],
          f"{plan['download'][1]:.3f}")

    # ----------------------------------------------------- the run itself
    PHRASES = ["Right, let's carry on with the next round.",
               "We chain three and turn the work.",
               "Now single crochet into the same stitch.",
               "Keep the tension loose so it drapes nicely."]

    def fake_llm(prompt, model=None, host=None, key=None, **_):
        ids = [int(l.split("|")[0]) for l in prompt.splitlines()
               if l and l[0].isdigit() and "|" in l]
        return "\n".join(f"{i}|{PHRASES[i % len(PHRASES)]}" for i in ids)

    real_ollama = T._call_ollama
    T._call_ollama = fake_llm
    settings.translator = "ollama"
    settings.voice = "bf_emma"
    settings.audio_mode = "replace"

    stamp = (clip.stat().st_size, clip.stat().st_mtime_ns)

    def run(source, preview=False):
        job = pipeline.runner.submit(source, settings, preview=preview)
        print("      running…", end="", flush=True)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < 900:
            time.sleep(2)
            print(".", end="", flush=True)
        print()
        return job

    try:
        # "Try 30 seconds" is the first button on the panel, so it is the first
        # thing to work. Deliberately submitted the way somebody would type it
        # rather than pre-resolved: submit() is where that is settled, and a
        # feature that only works when handed a tidy path is one that only works
        # from the file chooser.
        typed = f"~{str(clip)[len(home):]}" if str(clip).startswith(home) else str(clip)
        sample = run(typed, preview=True)
        check("a sample of a local file dubs", sample.status == "done",
              f"{sample.status}: {sample.error}")
        if sample.status != "done":
            return
        check("it really was a sample, not the whole thing", sample.preview is True)
        check("taken from where the speech starts", sample.preview_from >= 0,
              str(sample.preview_from))
        check("and it plays out of the job folder rather than the finished videos",
              JOBS in Path(sample.output).parents, sample.output)

        job = run(typed)
        check("a local file dubs", job.status == "done", f"{job.status}: {job.error}")
        if job.status != "done":
            return

        # The sample and the full run are two jobs over one folder — the whole
        # value of sampling first is not paying for the same work twice.
        check("the full run picks up the sample's folder", job.key == sample.key,
              f"{job.key} vs {sample.key}")
        check("but is its own job", job.id != sample.id)

        check("the job knows it came off the disk", job.local is True)
        check("its title is the file's name", job.title == "local-clip", job.title)
        out = Path(job.output)
        check("a dubbed video was written", out.is_file(), str(out))
        check("into the finished-videos folder, like any other",
              OUTPUT_DIR in out.parents, str(out))
        check("and it is the whole video, not the sample",
              abs(pipeline.download.media_duration(out) - info["duration"]) < 2.0,
              str(pipeline.download.media_duration(out)))

        # The whole reason nothing is copied: the source is the largest thing a
        # job touches, and this app refuses to start when the disk is tight.
        # Judged by size rather than by absence, because the sample cut its own
        # 30 seconds out into a source.mp4 of its own and that one is earned.
        whole = clip.stat().st_size
        sizes = {p.name: p.stat().st_size for p in (JOBS / job.key).rglob("source.*")}
        check("the video itself was never copied into the job folder",
              all(size < whole for size in sizes.values()),
              ", ".join(f"{n} {v} of {whole}" for n, v in sizes.items()) or "nothing there")
        check("and the user's own file is exactly as it was",
              (clip.stat().st_size, clip.stat().st_mtime_ns) == stamp)

        # Which stream to fetch is a choice only a site offers.
        used = job.stats.get("settings") or {}
        check("the report doesn't claim a video quality nothing chose",
              "keep_video_quality" not in used, str(sorted(used))[:80])
        check("while a link's report still does",
              "keep_video_quality" in Settings().run_snapshot())

        check("the stage that opened the file is named as such",
              job.stage_times.get("download") is not None, str(job.stage_times))
    finally:
        T._call_ollama = real_ollama


if __name__ == "__main__":
    test_align()
    test_translate()
    test_server()
    test_end_to_end()
    test_resume()
    test_preset_change_reseparates()
    test_mixed_sample_rates()
    test_cleanup()
    test_segments_and_voices()
    test_preview()
    test_storage()
    test_keep_awake()
    test_translation_qc()
    test_observability()
    test_terminology()
    test_job_record()
    test_near_empty_dub()
    test_translate_resume()
    test_translate_resume_respects_settings_change()
    test_preview_failure_is_superseded_by_a_later_success()
    test_local_file()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("All checks passed.")
