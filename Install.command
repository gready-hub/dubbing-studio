#!/bin/bash
# Dubbing Studio — installer.
# Double-click this file in Finder. It sets up everything needed and is safe to
# re-run at any time: anything already present is left alone.

cd "$(dirname "$0")" || exit 1
set -uo pipefail

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; RESET=$'\033[0m'

# Everything verbose goes to a log rather than the screen, so the installer stays
# readable but a failure is still diagnosable after the fact.
LOG="$HOME/Library/Logs/DubbingStudio-install.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

say()  { printf "%s\n" "$*"; }
step() { printf "\n${BOLD}%s${RESET}\n" "$*"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
WARNINGS=()
warn() { WARNINGS+=("$*"); printf "  ${YELLOW}!${RESET} %s\n" "$*"; }
bad()  { printf "  ${RED}✗${RESET} %s\n" "$*"; }

clear
cat <<'BANNER'
  ┌──────────────────────────────────────────┐
  │           Dubbing Studio Setup           │
  └──────────────────────────────────────────┘

  This installs everything the app needs. It can take
  10-20 minutes the first time, mostly downloading.
  You can leave it running and come back.

BANNER

if [[ "$(uname)" != "Darwin" ]]; then
  bad "This installer is for macOS. On Windows or Linux, use the Docker version"
  say "    in the docker/ folder instead."
  read -r -p "Press return to close."
  exit 1
fi

ARCH="$(uname -m)"
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))

# macOS refuses to let an app bundle read Downloads, Desktop or Documents unless
# it has been granted access, and for a locally-built bundle like ours it denies
# outright rather than prompting. The app would install and then fail to open, so
# say so now while the user is still here to act on it.
PROTECTED=""
case "$PWD/" in
  "$HOME"/Downloads/*)                PROTECTED="Downloads" ;;
  "$HOME"/Desktop/*)                  PROTECTED="Desktop" ;;
  "$HOME"/Documents/*)                PROTECTED="Documents" ;;
  "$HOME"/Library/Mobile\ Documents/*) PROTECTED="iCloud Drive" ;;
esac

step "Your Mac"
ok "$( [[ "$ARCH" == "arm64" ]] && echo "Apple Silicon — the fast path is available" \
                                || echo "Intel Mac — will use the portable engine" )"
ok "${RAM_GB} GB memory"

# ---------------------------------------------------------------- 1. Xcode
step "1 of 8  Apple developer tools"
if xcode-select -p >/dev/null 2>&1; then
  ok "Already installed"
else
  warn "Not installed. A system window will open — click Install and wait."
  xcode-select --install 2>/dev/null
  say "  Waiting for that to finish…"
  until xcode-select -p >/dev/null 2>&1; do sleep 10; done
  ok "Installed"
fi

# ------------------------------------------------------------- 2. Homebrew
step "2 of 8  Homebrew (installs the other tools)"

# Pick up an existing Homebrew *before* deciding whether to install one. This
# script runs under bash, so it never sees the PATH that Homebrew adds to
# ~/.zprofile — without this, a Mac that already has Homebrew looks bare and we
# try to install it a second time.
brew_shellenv() {
  for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    [[ -x "$p" ]] && eval "$("$p" shellenv)" && return 0
  done
  return 1
}
brew_shellenv || true

if command -v brew >/dev/null 2>&1; then
  ok "Already installed"
else
  warn "Installing Homebrew. It will ask for your Mac password — this is normal."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
    || { bad "Homebrew install failed. See the message above."; read -r -p "Press return."; exit 1; }
  brew_shellenv || true
fi
command -v brew >/dev/null 2>&1 || { bad "Homebrew still not found."; read -r -p "Press return."; exit 1; }
ok "Homebrew ready"

# ------------------------------------------------------- 3. ffmpeg, yt-dlp
step "3 of 8  Video tools"
# The command each formula is expected to put on PATH. python@3.12 provides
# "python3.12", not "python" — checking for the latter would skip the install on
# any Mac with a pyenv or conda shim, and leave us building the venv from an
# older interpreter than the packages support.
check_cmd_for() {
  case "$1" in
    python@3.12) echo "python3.12" ;;
    *)           echo "$1" ;;
  esac
}

# Being on PATH is not the same as being usable. A pyenv shim for a version
# pyenv has not selected sits on PATH, satisfies `command -v`, and then exits
# non-zero with "command not found" the moment it is run. Testing with
# `command -v` therefore skipped the Homebrew install here and left step 4
# building the environment from an interpreter that cannot start — so for the
# interpreters, ask whether it actually runs.
tool_usable() {
  case "$1" in
    python3.12|python3) "$1" -c 'import sys' >/dev/null 2>&1 ;;
    *)                  command -v "$1" >/dev/null 2>&1 ;;
  esac
}

for tool in ffmpeg yt-dlp python@3.12; do
  name="$(check_cmd_for "$tool")"
  if brew list --formula "$tool" >/dev/null 2>&1 || tool_usable "$name"; then
    ok "$name already installed"
  else
    # Braced deliberately: bash treats the bytes of a following multi-byte
    # character as part of an unbraced name, so "$name…" expands the variable
    # "name…", which set -u then kills the installer over. This line only runs
    # when a tool is actually missing, so it survived every re-run on a machine
    # that already had them.
    say "  Installing ${name}…"
    brew install "$tool" >/dev/null 2>&1 && ok "$name installed" || warn "$name may have failed — check below"
  fi
done
brew upgrade yt-dlp >/dev/null 2>&1 && ok "yt-dlp up to date" || true

# ------------------------------------------------------------- 4. Python
step "4 of 8  Python environment"
# Resolve the interpreter by running it, and try Homebrew's own path before
# whatever PATH happens to resolve first: a pyenv or conda shim earlier in PATH
# shadows a perfectly good python3.12 and then refuses to start.
pick_python() {
  local candidate resolved
  for candidate in "$(brew --prefix 2>/dev/null)/opt/python@3.12/bin/python3.12" \
                   /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 \
                   python3.12 python3; do
    resolved="$(command -v "$candidate" 2>/dev/null)" || continue
    "$resolved" -c 'import sys, venv; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' \
      >/dev/null 2>&1 && { printf "%s\n" "$resolved"; return 0; }
  done
  return 1
}

if ! PY="$(pick_python)"; then
  bad "No usable Python 3.10 or newer was found."
  say "    Install one with:  brew install python@3.12"
  read -r -p "Press return."; exit 1
fi
ok "Using $("$PY" -V 2>&1)"

# A half-built .venv is worse than none: it has the directory layout but no
# working interpreter, so `source activate` succeeds and every later step runs
# against the wrong python. Failing to build one is fatal, not a warning —
# continuing past it is what turned a bad interpreter into four more errors.
if [[ -d .venv && ! -x .venv/bin/python ]]; then
  warn "Removing an incomplete .venv left by an earlier attempt"
  rm -rf .venv
fi
if [[ ! -x .venv/bin/python ]]; then
  if ! "$PY" -m venv .venv >>"$LOG" 2>&1; then
    bad "Could not create the Python environment. See $LOG"
    read -r -p "Press return."; exit 1
  fi
fi
# shellcheck disable=SC1091
if ! source .venv/bin/activate; then
  bad "Could not activate the Python environment."
  read -r -p "Press return."; exit 1
fi
python -m pip install --quiet --upgrade pip wheel
REQ="requirements-portable.txt"
[[ "$ARCH" == "arm64" ]] && REQ="requirements-mac.txt"
say "  Installing Python packages (this is the slow bit)…"
if python -m pip install -r "$REQ" >>"$LOG" 2>&1; then
  ok "Packages installed"
else
  warn "The fast Apple-GPU packages failed; falling back to the portable engine."
  warn "Details: $LOG"
  python -m pip install -r requirements-portable.txt >>"$LOG" 2>&1 \
    || { bad "Python setup failed. See $LOG"; read -r -p "Press return."; exit 1; }
fi

# The Apple-GPU voice phonemises English through spacy and fetches this model the
# first time it speaks. Do it now, so the first dub isn't quietly running a
# download part way through the job.
if python -c "import misaki" 2>/dev/null && ! python -c "import en_core_web_sm" 2>/dev/null; then
  say "  Fetching the pronunciation model…"
  python -m spacy download en_core_web_sm >>"$LOG" 2>&1 \
    && ok "Pronunciation model ready" \
    || warn "Could not fetch the pronunciation model; it will download on first use."
fi

# ---------------------------------------------------- 5. Quality extras
step "5 of 8  Quality extras"
say "  These enable keeping music and effects, telling speakers apart, and"
say "  cloning voices. Around 3 GB — the biggest download here."
if python -c "import demucs" 2>/dev/null && python -c "import chatterbox" 2>/dev/null; then
  ok "Already installed"
elif python -m pip install -r requirements-quality.txt >>"$LOG" 2>&1; then
  ok "Quality extras installed"
else
  warn "Those failed to install. The app still works — you'll be limited to the"
  warn "Fast preset, and can re-run this installer later to try again."
  warn "Details: $LOG"
fi

# --------------------------------------------------------- 6. Speech models
step "6 of 8  Speech models"
say "  Fetching the models the app will need, so the first video doesn't stop"
say "  part way through to download them."
# Not piped through grep: `||` binds to the last command in a pipeline, so the
# warning was keyed on grep's exit status and could never fire.
if ! python -m app.warmup 2>&1 | tee -a "$LOG"; then
  warn "Some speech models could not be fetched; they'll download on first use."
fi

# -------------------------------------------------------------- 7. Ollama
step "7 of 8  Local translation model"
if command -v ollama >/dev/null 2>&1; then
  ok "Ollama already installed"
else
  say "  Installing Ollama…"
  brew install --cask ollama >/dev/null 2>&1 && ok "Ollama installed" \
    || warn "Could not install Ollama. You can add an API key in Settings instead."
fi

if command -v ollama >/dev/null 2>&1; then
  open -a Ollama 2>/dev/null || true
  # Wait for the server to actually answer rather than guessing at a sleep. A
  # first launch has to unpack and start the helper, which takes well over the
  # four seconds this used to allow, and `ollama pull` just errors if it is early.
  ollama_up() { curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; }
  say "  Waiting for Ollama to start (first launch is slow)…"
  for _ in $(seq 1 90); do ollama_up && break; sleep 1; done

  # If the app is stuck behind a first-run window, run the server directly. The
  # command-line server is the same binary and needs no interaction.
  if ! ollama_up; then
    say "  Starting the Ollama server directly…"
    nohup ollama serve >>"$LOG" 2>&1 &
    for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
  fi
  # Largest first, then smaller ones. The top tier is a 20 GB download, which on
  # a home connection is a long time to be exposed to a dropped one — measured
  # here, it died at 6% with "unexpected EOF". A smaller model that arrives beats
  # a better one that doesn't, and the app translates instructional speech well
  # with any of them. Keep the first entry of each tier in step with
  # suggest_ollama_model() in app/config.py.
  if   (( RAM_GB >= 48 )); then LADDER=(qwen3:32b qwen3:14b qwen3:8b)
  elif (( RAM_GB >= 24 )); then LADDER=(qwen3:14b qwen3:8b)
  elif (( RAM_GB >= 16 )); then LADDER=(qwen3:8b qwen3:4b)
  else                          LADDER=(qwen3:4b); fi

  if ollama_up; then
    # Anything on the ladder already there is the answer: re-running this
    # installer must not re-download twenty gigabytes to arrive where it is.
    HAVE=""
    for m in "${LADDER[@]}"; do
      if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$m"; then HAVE="$m"; break; fi
    done

    if [[ -n "$HAVE" ]]; then
      ok "$HAVE already installed"
    else
      total=${#LADDER[@]}; i=0
      for m in "${LADDER[@]}"; do
        i=$((i + 1))
        say "  Downloading $m — chosen to fit your ${RAM_GB} GB. This is a few GB."
        if ollama pull "$m"; then HAVE="$m"; ok "$m ready"; break; fi
        warn "That download didn't finish."
        if (( i < total )); then say "  Trying a smaller model instead…"; fi
      done
      if [[ -z "$HAVE" ]]; then
        warn "No translation model could be downloaded. Open the Ollama app and"
        warn "re-run this installer, or paste an API key in Settings instead."
      fi
    fi
  else
    warn "Ollama isn't responding. Open the Ollama app once, then re-run this installer."
    warn "You can also use a Claude or OpenAI key in Settings instead."
  fi
fi

# ------------------------------------------------------------ 8. Build app
step "8 of 8  Building the app"
python -m pip install --quiet pywebview pillow 2>/dev/null || \
  warn "Native window support unavailable — the app will open in your browser instead."

if [[ -n "$PROTECTED" ]]; then
  warn "This folder is in $PROTECTED, which macOS stops apps from reading."
  warn "The app icon will not open until you move this folder somewhere else —"
  warn "your home folder ($HOME) works. Move it, then run this installer again."
  warn "Meanwhile \"Start Dubbing Studio\" in this folder still works."
fi

chmod +x packaging/build_app.sh "Start Dubbing Studio.command" 2>/dev/null
APP_PATH="$(bash packaging/build_app.sh 2>/dev/null | tail -1)"
if [[ -d "$APP_PATH" ]]; then
  ok "Installed to $APP_PATH"
  open "$HOME/Applications" 2>/dev/null
else
  warn "Could not build the app bundle. Use \"Start Dubbing Studio\" in this folder instead."
fi

if (( ${#WARNINGS[@]} == 0 )); then
  cat <<'DONE'

  ┌──────────────────────────────────────────┐
  │              All finished                │
  └──────────────────────────────────────────┘

  Dubbing Studio is now in your Applications folder.
  Double-click it like any other app. Drag it to your
  Dock if you want it handy.

  Keep this folder where it is — the app runs from here.

  The models for your chosen quality setting are already
  downloaded, so the first video goes straight to work.
  Switching to a different setting later fetches whatever
  that one needs, once.

DONE
else
  # Don't claim success when something was skipped — the app may still open and
  # then fail on the one thing that didn't install, which is far more confusing
  # than being told here.
  cat <<'DONE'

  ┌──────────────────────────────────────────┐
  │        Finished, with some gaps          │
  └──────────────────────────────────────────┘

  Dubbing Studio is installed and will open, but these
  did not complete:

DONE
  for w in "${WARNINGS[@]}"; do printf "    • %s\n" "$w"; done
  cat <<DONE

  Re-running this installer is safe and will retry them.
  Full details: $LOG

DONE
fi
read -r -p "Press return to close this window."
