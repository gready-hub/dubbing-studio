# Dubbing Studio

Paste a video link. Get the video back speaking English.

Everything runs on your own machine. No account, no upload, no subscription.

---

## Installing on a Mac

1. Put the **Dubbing Studio** folder somewhere permanent in your home folder —
   `/Users/you/Dubbing Studio` is ideal. The app runs from wherever you leave it,
   so don't put it in the bin afterwards.

   **Not Downloads, Desktop, Documents or iCloud Drive.** macOS blocks apps from
   reading those folders, and it refuses silently rather than asking, so the app
   icon would simply never open. The installer warns you if it finds itself in
   one of them.
2. Double-click **Install**.
3. Wait. The first run takes 10–20 minutes because it downloads a lot. You can
   leave it and come back.
4. When it finishes, **Dubbing Studio** appears in your Applications folder.
   Double-click it like any other app, and drag it to your Dock if you want.

The installer asks for your Mac password once, when it installs Homebrew. That is
normal — Homebrew is the standard tool for installing software on a Mac and it
needs permission to create its folder.

### If macOS refuses to open the Install file

Right-click it, choose **Open**, then click **Open** again in the dialog. macOS
does this for anything not bought from the App Store. You only need to do it once.

---

## Using it

1. Open Dubbing Studio.
2. Paste a video link into the box.
3. Press **Dub it**.

Finished videos are saved to **Movies → Dubbed**, and the app shows a **Show in
Finder** button when it's done.

### How long it takes

This depends far more on the preset than on the video, and translation is usually
the slowest part. Measured on an M1 with 16 GB, dubbing a three-minute clip using
the built-in local translation model:

| Preset | Time for 3 minutes of video | Roughly |
|---|---|---|
| Fast | about 3 minutes | the video's own length |
| Balanced | about 15 minutes | five times the video's length |
| Best quality | longer again — see the note on cloning below | |

A newer or larger Mac will beat this comfortably, and switching **Translated by**
to an API key removes the single biggest chunk of time from every preset. The
first video is slower still, because it fetches the speech models (about 700 MB,
one time only).

Start anything long and leave it. A run that fails or is cancelled picks up where
it left off when you paste the same link again.

---

## Quality presets

The three buttons under the link box decide how much work goes in.

**Fast** — one voice, no separation. Right for a single person talking to camera
with no music. Quickest by a distance.

**Balanced** *(default)* — splits the speech away from the music and effects
before dubbing, so replacing the voices doesn't wipe the soundtrack, and detects
multiple speakers so an interview gets two distinct voices instead of one
narrator reading both parts. This is the setting that makes the app work on
general video rather than just talking heads.

**Best quality** — as Balanced, plus Whisper for transcription and a cloned voice
per speaker, so the dub keeps the original speakers' own voices. Considerably
slower: cloning generates speech at around four times slower than real time on an
M1, so it is the preset to choose deliberately for something short, not the one
to leave running on a feature-length video.

### On voice cloning

Cloning carries the speaker's accent across languages. That is usually what you
want — it preserves who they are. But for instructional content, where someone is
following along with numbers or steps, a neutral built-in voice is often easier
to understand than an accented clone. Worth trying both on anything you care
about.

Cloning someone's voice is also not the same act as picking a stock one. Fine for
private use; think about it before publishing.

---

## Settings worth knowing about

**Subject vocabulary** is the one that matters most. Translation models guess at
specialist terms and guess inconsistently — the same stitch, tool or ingredient
comes out three different ways across one video. Picking a vocabulary locks those
terms down. There are built-in lists for crochet (US and UK conventions), cooking
and woodworking, and you can add your own lines under Advanced in the form
`source -> translation`.

**Original audio** decides what happens to the original speaker:

- *Replace completely* — cleanest listen, original is gone.
- *Keep quietly underneath* — the usual documentary treatment; you still hear
  their tone and any background sound.
- *Keep as a second track* — both tracks in the file, switchable in a player like
  VLC or IINA.

**Translated by** picks who does the translating:

- *Local model* — free, private, works offline. Installed for you.
- *Claude or OpenAI API* — noticeably better on specialist material. Costs a few
  pence per video. Paste a key and it's used instead.

**Where an API key is kept.** In plain text, in
`~/Library/Application Support/DubbingStudio/settings.json`, so the app can
translate without asking for it every time. The file is readable only by your
user account, and it is listed in `.gitignore` so it cannot be committed by
accident — but it is not encrypted, and anyone who can log in as you can read it.
The macOS Keychain was the alternative and was not used: it would put a system
authorisation prompt in front of every job. If that trade isn't right for you,
leave the key blank and use the local model.

---

## Running it anywhere else (Docker)

For Windows, Linux, or handing to someone else:

```bash
cd docker
docker compose up --build
```

Then open <http://localhost:8765>. Finished videos land in `docker/output`.

**One caveat.** Docker cannot reach the Mac's GPU, so the container runs on CPU
only and is roughly five times slower than the native app. A local translation
model is impractical at that speed, so use an API key in Settings for the Docker
version.

---

## When something goes wrong

Open **Setup check** in the app. It lists what's missing and the exact command to
fix it.

**"Could not reach Ollama"** — open the Ollama app, wait a few seconds, try again.

**"Translation only returned N of M lines"** — the local model is too small for
the job. Put a bigger one in Settings (`qwen3:14b` if you have 24 GB of memory or
more) or switch to an API key.

**The video downloads but nothing is heard** — the video may have no speech, or
speech the recogniser can't pick out of loud music.

**Something crashed** — the log is at `~/Library/Logs/DubbingStudio.log`, and each
job keeps its own working files under
`~/Library/Application Support/DubbingStudio/jobs`.

Jobs resume. If one fails partway through, running the same link again reuses the
transcription and translation it already finished rather than starting over.

**Disk space.** A job succeeds and then drops its bulky intermediates — the
downloaded video, the separated stems and the full-band audio, which for an hour
of video come to well over a gigabyte. What stays is the transcript, the
translation and the rendered lines, which are the expensive parts to recompute
and a small fraction of the size. A job that *failed* keeps everything, because
that is exactly when you re-run the link. The main window shows the running
total with a **Clear** button beside it; clearing never touches finished videos.

---

## A word on what you download

This tool will fetch whatever link you give it, which is a question of what you're
entitled to download rather than what's technically possible. Downloading from
YouTube generally breaches their terms of service unless it's your own content,
it's offered for download, or the licence permits it. Dubbing someone's video also
creates a derivative work. Your call — worth making deliberately.

---

## What's under the bonnet

| Stage | Tool |
|---|---|
| Download | yt-dlp |
| Speech / music separation | Demucs (htdemucs_ft) |
| Speaker detection | pyannote segmentation 3.0 + 3D-Speaker embeddings |
| Speech recognition | Parakeet TDT 0.6b v3 (25 languages), or Whisper large-v3 |
| Translation | your local Ollama model, or Claude / OpenAI |
| Speech synthesis | Kokoro-82M, or Chatterbox for cloned voices |
| Audio and video | ffmpeg |

On Apple Silicon the two AI models run through **MLX**, Apple's GPU framework.
Everywhere else the same models run on CPU via **ONNX Runtime**, which is what
makes the Docker build possible.

The window is macOS's built-in WebView rather than a bundled copy of Chrome, which
is why the app is a few megabytes instead of a few hundred.

### How the timing works

Each translated line is spoken and then fitted into the slot the original speaker
used. If a line comes out too long for its gap it's compressed with a
pitch-preserving filter, up to 1.55× — beyond that it's allowed to run over and
following pauses absorb it. Lines never start before their original timestamp, so
narration stays locked to what's on screen.

The quality report after each job tells you how much of that had to happen. Few
compressions and near-zero drift means a clean fit.
