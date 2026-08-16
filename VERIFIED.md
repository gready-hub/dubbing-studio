# What has actually been run

Every claim below is marked **Run** (executed on this machine, with the evidence
quoted) or **Not run** (asserted from reading the code, or not tested at all).
Where something was only partly exercised it says so and says which part.

Machine: Apple M-series, macOS 26.5.2, arm64, 48 GB. Homebrew present; ffmpeg,
yt-dlp and python@3.12 were *not* installed beforehand, and there was no `.venv`
— so the installer was genuinely exercised from a clean state rather than
re-run over an existing one.

---

## 1. `Install.command` from a clean state — **Run, and it was broken**

Executed three times from clean. It failed twice with real bugs, both now fixed.

**Bug 1 — died on the first missing tool.** `say "  Installing $name…"`: under
`set -u`, bash treats the bytes of the following multi-byte character as part of
an unbraced variable name, so it expanded `name…` rather than `name` and exited
with `name?: unbound variable` at step 3 of 7.

Reproduced on `/bin/bash` 3.2.57 and Homebrew bash 5.3.9, under both
`en_US.UTF-8` and `C.UTF-8`; only the `C` locale is unaffected, so a normal
UTF-8 terminal always hit it. The line sits in the `else` branch of the
ffmpeg / yt-dlp / python@3.12 loop, so it runs *only when one of those is
actually missing* — that is, only on the clean machine the installer exists to
provision. Any re-run on a Mac that already had the three tools took the `if`
branch and never touched it. That is how it shipped described as verified.

**Bug 2 — built the environment from an interpreter that cannot start.**
`command -v python3.12` matched `~/.pyenv/shims/python3.12`, which exists and is
executable but exits non-zero with "command not found" because pyenv's selected
version is `system`. Step 3 therefore reported "python3.12 already installed"
and skipped `brew install python@3.12`; step 4 picked the shim, `python -m venv`
failed into `|| true`, `source .venv/bin/activate` failed with "No such file or
directory", and every later pip call ran against the broken shim before the step
gave up with "Python setup failed".

The existing comment had anticipated exactly this shim problem, but the guard
only changed *which name* was probed, not *how* — `command -v` still finds a
shim that cannot run.

**Third run, after both fixes: completed all steps.** Xcode CLT present →
Homebrew present → installed ffmpeg, yt-dlp, python@3.12 → built the venv on
Python 3.12.14 → installed `requirements-mac.txt` → fetched the spacy
pronunciation model → installed the quality extras (torch, demucs,
chatterbox-tts, faster-whisper) → installed Ollama → built the `.app`.

Two things worth recording from that run:

- The Ollama cask installed, but the GUI app did not answer within the 90-second
  wait, so the installer's fallback of running `ollama serve` directly was
  exercised, and worked.
- The model pull was **interrupted deliberately by me**, not by a bug: the
  installer correctly selected `qwen3:32b` for 48 GB, and 20 GB of download was
  starving the test suite's model fetches. The installer then reported
  "Finished, with some gaps" and listed the failure rather than claiming
  success, which is the behaviour it should have. `qwen3:8b` was pulled
  afterwards and used for the real run in section 9.

## 2. The MLX paths, actually invoked — **Run**

Not merely imported. `detect_machine().fast_path` is true on this machine, so
the full test-suite pipeline runs on MLX; `/api/state` reports engine
`Apple GPU (MLX)`.

- **`parakeet-mlx`** — Run. Signatures match what `HANDOFF.md` assumed:
  `from_pretrained(hf_id_or_path, *, dtype, cache_dir) -> BaseParakeet`, and
  `.transcribe(path, *, chunk_duration, overlap_duration, …) -> AlignedResult`
  with `.sentences`, each `AlignedSentence` carrying `text / start / end`. It
  transcribed the 75-second suite clip, and the end-to-end check transcribed the
  *finished dub* back and recovered the English that was synthesised into it.
- **`mlx-audio` Kokoro** — Run. `load_model(...)` and
  `generate(text=, voice=, speed=, lang_code=)` are as assumed. It spoke all 23
  lines of the end-to-end job.
- **`mlx-whisper`** — **Not run.** `mlx_whisper.transcribe` has exactly the
  assumed signature (`audio, *, path_or_hf_repo, word_timestamps, verbose`), and
  the package imports, but no job in either suite selects Whisper, so the code
  path has never executed. It is only reached by the "Best quality" preset.

## 3. Chatterbox cloning — **Not run**

Signatures check out: `ChatterboxTTS.from_pretrained(device)` and
`generate(text, …, audio_prompt_path=…)` are as `clone.py` assumes, and
`clone.available()` is true. But `sr` is an *instance* attribute, not a class
one, so **the sample rate it reports could not be confirmed without loading the
model**, and I did not load it. That rate is the invariant behind A2.

This is now much less load-bearing than it was: the rate is no longer trusted.
It is read once, everything is resampled to it on the way into the track, and
`align.assemble()` refuses a line that disagrees. So an unexpected Chatterbox
rate can no longer cause the mistimed-lines failure — but "cloning produces a
recognisable copy of the speaker, judged by ear" remains **untested**.

## 4. Demucs on MPS — **Run, and the stems are good**

This is the one with a history of producing silence or NaNs. `test_upgrade.py`
asserts against both, on real separated output:

```
PASS  separation produced both stems
PASS  speech stem contains no NaN or infinity
PASS  speech stem is not silent        [peak 0.3158]
PASS  background stem contains no NaN or infinity
PASS  background stem is not silent    [peak 0.1018]
```

It also ran on the real 10.5-minute video in section 9. No fallback to CPU is
needed and none has been added.

## 5. The `.app` bundle — **Run, with two caveats**

Built by the installer to **`~/Applications/Dubbing Studio.app`** — note that is
the user Applications folder, not `/Applications`; that is deliberate, since it
needs no administrator rights.

- Launches from Finder: **Run.** `open` starts it, `python -m app.desktop` runs,
  and it serves on `127.0.0.1:8765`.
- Icon: **Run.** `icon.icns` is generated (113 KB, `ic12`), `Info.plist` passes
  `plutil -lint`, and Spotlight reports the bundle as
  `com.apple.application-bundle` named "Dubbing Studio".
- Survives a logout: **Not run.** I am not going to log this machine out.
  Nothing in the bundle depends on session state, but that is an assertion.
- A folder path containing a space: **Run.** `build_app.sh` was run from
  `…/dub space test/dubbing studio` — two spaces, one in each of the last two
  components — with `HOME` redirected so the installed bundle was untouched. The
  generated launcher quotes `SRC` correctly, and `cd "$SRC"` from the generated
  stub reaches the folder and finds `.venv` there.

## 6. The pywebview window — **Run, and it is genuinely native**

pywebview 6.2.1. The running process has `WebKit.framework`, `WebCore.framework`,
`JavaScriptCore.framework`, `AppKit.framework` and `Cocoa.framework` mapped, and
`~/Library/Logs/DubbingStudio.log` contains no "Opening … in your browser" line.
It has not silently fallen back to the browser.

## 7. A real download — **Run**

`yt-dlp` 2026.07.04. `download.probe()` against a live YouTube URL returns title,
duration, uploader and licence. A real download of the 10.5-minute video in
section 9 completed through the app's own download step.

One thing to know: the **first** attempt failed with `HTTP Error 403: Forbidden`
during the media fetch, immediately after a successful metadata probe. The
identical format selector succeeded on retry, so it is transient throttling
rather than a broken format string. The failure surfaced correctly — the job
went to `error`, the message reached the interface, and nothing was left in a
half-finished state. Worth knowing that a 403 on a long video means "try again",
not "the link is bad".

## 8. The Docker build — **Run, and it was broken**

`docker compose build` succeeds and produces a 1.04 GB image. The container then
**could not be reached at all**.

`server.main()` bound uvicorn to `127.0.0.1` unconditionally. Inside a container
that is the container's own loopback, and Docker publishes a port by forwarding
to the container's external interface — so the app listened where nothing could
reach it. The container logged "Dubbing Studio is running" and
`http://localhost:8765`, which is exactly what the README tells people to open,
answered nothing. **The Docker build cannot previously have been run.**

Fixed by binding `0.0.0.0` only when in a container. After the fix:

```
engine   : Portable (CPU)
in_docker: True | has_mlx: False
ffmpeg   : True | yt-dlp: True
ollama   : True   <- the host's Ollama, via host.docker.internal
features : separation False, cloning False, whisper False, diarization True
```

The `OLLAMA_HOST=http://host.docker.internal:11434` wiring from
`docker-compose.yml` works — the container reaches the Mac's Ollama. The three
false features are correct and expected: `requirements-portable.txt` deliberately
omits torch, demucs and chatterbox.

**A full video end to end inside the container: Not run.** The container is
CPU-only and roughly five times slower, and the Mac was busy with the real run
in section 9 for the whole window. What is confirmed is that it builds, serves,
detects its environment correctly and can reach the translator.

---

## 9. The quality bar — a real video — **Run, and it did not reach the bar**

**Sprite Fright**, Blender Open Movie, 10m35s, fetched from YouTube by the app
itself, Balanced preset, 720p, translated by a local `qwen3:8b`. Driven through
the installed `.app`, against the real job and output folders.

Structural results — all good:

| | |
|---|---|
| Video frames in / out | **15116 / 15116**, stream copied (still AV1, never re-encoded) |
| Audio vs video length | 629.85s vs 629.83s — **0.02s apart** |
| Lines transcribed / spoken | 113 / 113, none failed |
| Separation | ran, music and effects kept under the dub |
| Finished audio | peak −0.5 dB, mean −17.8 dB, 9.8s silent (1.6%) |
| Subtitles | written and muxed as `mov_text` |
| Working files reclaimed | 318 MB, folder went 390 MB → 13 MB |
| Elapsed | 48 minutes |

Timing results — **well short of the reference run**:

| | this run | reference run |
|---|---|---|
| Lines | 113 | 830 |
| Compressed | **61 (54%)** | 7 (0.8%) |
| Worst squeeze | **1.55× (the cap)** | 1.18× |
| Lines over the cap | **15** | 0 |
| Max drift | **4.70s** | 0 |

**Why, and it is not a regression.** I checked the obvious suspect first and it
is innocent: the translation did not inflate the text, coming out at 0.95× the
source length. The cause is the material. Measured over the 113 lines:

- median slot **2.00s**, and 41 of 113 slots are under 1.5s
- median gap to the next line **0.08s**

That is continuous, fast, overlapping film dialogue with essentially no pauses.
The reference run was a 68-minute crochet tutorial — slow, deliberate,
instructional speech with long gaps, where a synthesised line drops into its
slot with room to spare. Kokoro speaks at a measured pace, so against dialogue
with an 0.08s median gap it almost always has to be compressed, 15 lines could
not fit even at the 1.55× cap, and the overflow accumulated into 4.7s of drift.

So: **the pipeline reproduces the bar on the content it was designed for, and
this run says plainly that it did not on this material.** The quality report
being the thing that surfaced it is the design working, not failing. Two of the
controls added in section C are the levers — `max_stretch`, to allow a harder
squeeze on dialogue like this, and `expected_speakers`, because see below.

**Diarization over-segmented badly.** It reported **28 speakers** for a film
with about seven characters, so lines from one character were handed several
different voices from the pool. This is the clearest possible argument for
having exposed `expected_speakers`: the user knows the answer and the model
plainly does not.

**Not tested:** how it sounds. I cannot judge a dub by ear. Every number above
is measured; "does it sound right" is not among them.

---

## Things worth knowing that are not in the list above

- **`submit()` ran every link at once.** A thread per job, with only the *same*
  link deduplicated, so two links contended for the GPU and each made the other
  slower — while the interface, which can only show one, made it look like the
  second had not started. Now a queue with one worker.
- **A hand-set user agent causes the 403 it looks like it would fix.** Same
  video, same format selector: yt-dlp's default succeeded, a desktop Chrome
  agent produced a reliable 403. YouTube ties a media URL's authorisation to the
  client yt-dlp negotiated as. There is now a comment at that line saying so.
- **`qwen3` was reasoning before every answer.** 92 generated tokens against 5
  for identical output with thinking off. Translation is the slow stage on a
  local model and most of it was the model talking to itself.
- **Auditioning a voice mid-job took the interface down.** Found by doing it:
  rendering a preview loads a second speech model, which put this machine into
  ~16 GB of swap and stopped the server answering for minutes. Cached previews
  still serve; new ones are refused while a job runs.

