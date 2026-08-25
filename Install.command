#!/bin/bash
# Dubbing Studio — installer.
# Double-click this file in Finder. It sets up everything needed and is safe to
# re-run at any time: anything already present is left alone.

cd "$(dirname "$0")" || exit 1
set -uo pipefail

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; RESET=$'\033[0m'

# Verbose output goes to a log, not the screen, so the installer stays readable
# but failures are still diagnosable.
LOG="$HOME/Library/Logs/DubbingStudio-install.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

# Timestamped, colour-code-free copy of everything said, so the log shows which
# step was running when something failed (not just raw command output).
note() { printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*" >> "$LOG"; }
say()  { printf "%s\n" "$*";                       note "$*"; }
step() { printf "\n${BOLD}%s${RESET}\n" "$*";      note "== $*"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*";   note "ok   $*"; }
WARNINGS=()
warn() { WARNINGS+=("$*"); printf "  ${YELLOW}!${RESET} %s\n" "$*"; note "WARN $*"; }
bad()  { printf "  ${RED}✗${RESET} %s\n" "$*";     note "FAIL $*"; }

# Copies details to the clipboard instead of naming the log file — Finder hides
# ~/Library/Logs by default, and that's usually where a bug report dies.
copy_details() {
  local summary
  summary="$(
    printf 'Dubbing Studio — install details\n\n'
    printf 'macOS %s on %s, %s GB memory\n' \
      "$(sw_vers -productVersion 2>/dev/null)" "$(uname -m)" "${RAM_GB:-?}"
    printf 'Installed to: %s\n' "$PWD"
    if (( ${#WARNINGS[@]} )); then
      printf '\nDid not complete:\n'
      printf '  - %s\n' "${WARNINGS[@]}"
    fi
    printf '\nLast 40 lines of the log:\n'
    tail -n 40 "$LOG" 2>/dev/null | sed 's/^/  /'
  )"
  # printf, not say(): logging this line would put the previous run's "copied to
  # clipboard" message inside the next run's summary.
  if printf '%s\n' "$summary" | pbcopy 2>/dev/null; then
    printf '  The details are on your clipboard — paste them into a message\n'
    printf '  to whoever helps you with this.\n\n'
  else
    printf '  Details are in this file, which you can attach to a message:\n'
    printf '  %s\n\n' "$LOG"
  fi
}

# Shared exit path for every failure, so the clipboard copy and prompt aren't
# duplicated at each check, and the EXIT trap below doesn't fire a second time.
HANDLED=0
finish_badly() {
  (( HANDLED )) && return 0
  HANDLED=1
  copy_details
  read -r -p "Press return to close this window."
}

# Catches failures with no check nearby, so a mid-step crash doesn't just close
# the window and leave nothing behind.
trap 'code=$?; (( code )) && { bad "Setup stopped unexpectedly (error $code)."; finish_badly; }' EXIT

clear
cat <<'BANNER'
  ┌──────────────────────────────────────────┐
  │           Dubbing Studio Setup           │
  └──────────────────────────────────────────┘

  Installs everything the app needs. 10-20 minutes the
  first time, mostly downloading. Safe to leave running.

BANNER

if [[ "$(uname)" != "Darwin" ]]; then
  bad "This app is for macOS only."
  HANDLED=1  # avoid a second prompt from the EXIT trap
  read -r -p "Press return to close."
  exit 1
fi

# uname -m reports this process's architecture, not the chip — under Rosetta it
# says x86_64 even on Apple Silicon. hw.optional.arm64 asks the kernel about the
# real CPU instead, and stays accurate even when we're the ones being translated
# (proc_translated), which we treat as fatal: continuing would silently install
# the Intel/portable engine, including Ollama.
if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" == "1" ]]; then
  ARCH="arm64"
  if [[ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" == "1" ]]; then
    bad "This is an Apple Silicon Mac, but this Terminal window is running"
    bad "translated through Rosetta — installing from here would quietly give"
    bad "you the slower Intel version of everything, including Ollama."
    say "  To fix it: close this window, select the app that opened it (usually"
    say "  Terminal, in Applications > Utilities) in Finder, press Cmd+I, and"
    say "  uncheck \"Open using Rosetta\" if it's checked. Then re-run this installer."
    HANDLED=1
    read -r -p "Press return to close this window."
    exit 1
  fi
else
  ARCH="$(uname -m)"
fi
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))

# A locally-built app bundle can't get the Downloads/Desktop/Documents access
# macOS gates behind a prompt — it's denied outright instead. Flag it now, while
# the user is still here, rather than after install when the app just won't open.
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

step "1 of 8  Apple developer tools"
if xcode-select -p >/dev/null 2>&1; then
  ok "Already installed"
else
  warn "Not installed. A system window will open — click Install."
  xcode-select --install 2>/dev/null
  say "  Waiting for that to finish…"
  until xcode-select -p >/dev/null 2>&1; do sleep 10; done
  ok "Installed"
fi

step "2 of 8  Homebrew (installs the other tools)"

# This script's bash session never sees the PATH that Homebrew's installer adds
# to ~/.zprofile, so pick up an existing install before assuming there isn't one.
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
  warn "Installing Homebrew. It will ask for your Mac password."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
    || { bad "Homebrew install failed. See the message above."; finish_badly; exit 1; }
  brew_shellenv || true
fi
command -v brew >/dev/null 2>&1 || { bad "Homebrew still not found."; finish_badly; exit 1; }
ok "Homebrew ready"

# Homebrew's prefix is arch-pinned: /usr/local only ever builds x86_64, even
# from an untranslated arm64 shell. /opt/homebrew is always the arm64 one.
if [[ "$ARCH" == "arm64" && "$(brew --prefix 2>/dev/null)" != "/opt/homebrew" ]]; then
  warn "Homebrew is installed at $(brew --prefix 2>/dev/null), which only ever"
  warn "builds Intel packages — even on this Apple Silicon Mac. Ollama and"
  warn "other tools installed below will be slower than they should be."
  warn "Fix: install Homebrew's Apple Silicon build alongside it —"
  warn '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  warn "— then re-run this installer from a fresh Terminal window."
fi

step "3 of 8  Video tools"
# The command each formula puts on PATH — python@3.12 provides "python3.12", not
# "python", so checking for the latter would wrongly skip the install.
check_cmd_for() {
  case "$1" in
    python@3.12) echo "python3.12" ;;
    *)           echo "$1" ;;
  esac
}

# On PATH isn't the same as usable: an unselected pyenv shim passes `command -v`
# but exits "command not found" when run. For interpreters, check it actually runs.
tool_usable() {
  case "$1" in
    python3.12|python3) "$1" -c 'import sys' >/dev/null 2>&1 ;;
    *)                  command -v "$1" >/dev/null 2>&1 ;;
  esac
}

for tool in ffmpeg python@3.12; do
  name="$(check_cmd_for "$tool")"
  if brew list --formula "$tool" >/dev/null 2>&1 || tool_usable "$name"; then
    ok "$name already installed"
  else
    # ${name} braced deliberately: "$name…" would expand the variable "name…"
    # (bash treats the following multi-byte char's bytes as part of the name),
    # which set -u then kills the installer over.
    say "  Installing ${name}…"
    brew install "$tool" >/dev/null 2>&1 && ok "$name installed" || warn "$name may have failed — check below"
  fi
done
# yt-dlp is deliberately not brew-installed here: the app runs the venv's
# `python -m yt_dlp` copy, which .venv/bin on PATH always shadows anyway. It's
# upgraded in section 4, once that venv exists.

step "4 of 8  Python environment"
# Resolve by running it, preferring Homebrew's own path over whatever PATH
# resolves first — a pyenv/conda shim can shadow a working python3.12.
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
  finish_badly; exit 1
fi
ok "Using $("$PY" -V 2>&1)"

# A half-built .venv (dir exists, no working interpreter) lets `activate`
# succeed and every later step run against a broken python — worse than none.
if [[ -d .venv && ! -x .venv/bin/python ]]; then
  warn "Removing an incomplete .venv left by an earlier attempt"
  rm -rf .venv
fi
if [[ ! -x .venv/bin/python ]]; then
  if ! "$PY" -m venv .venv >>"$LOG" 2>&1; then
    bad "Could not create the Python environment. See $LOG"
    finish_badly; exit 1
  fi
fi
# shellcheck disable=SC1091
if ! source .venv/bin/activate; then
  bad "Could not activate the Python environment."
  finish_badly; exit 1
fi
python -m pip install --quiet --upgrade pip wheel
REQ="requirements-portable.txt"
[[ "$ARCH" == "arm64" ]] && REQ="requirements-mac.txt"
say "  Installing Python packages…"
if python -m pip install -r "$REQ" >>"$LOG" 2>&1; then
  ok "Packages installed"
else
  warn "The fast Apple-GPU packages failed; falling back to the portable engine."
  warn "Details: $LOG"
  python -m pip install -r requirements-portable.txt >>"$LOG" 2>&1 \
    || { bad "Python setup failed. See $LOG"; finish_badly; exit 1; }
fi

# Always force-upgraded: YouTube breaks yt-dlp every few weeks, and the ">="
# pin in requirements is satisfied by whatever's already installed, so a
# re-run would otherwise never pick up the fix.
say "  Updating yt-dlp…"
python -m pip install --quiet --upgrade yt-dlp >>"$LOG" 2>&1 \
  && ok "yt-dlp $(python -m yt_dlp --version 2>/dev/null || echo '?')" \
  || warn "Could not update yt-dlp — downloads may fail. Details: $LOG"

# Pre-fetch the pronunciation model the Apple-GPU voice needs, so the first dub
# doesn't stall mid-job downloading it.
if python -c "import misaki" 2>/dev/null && ! python -c "import en_core_web_sm" 2>/dev/null; then
  say "  Fetching the pronunciation model…"
  python -m spacy download en_core_web_sm >>"$LOG" 2>&1 \
    && ok "Pronunciation model ready" \
    || warn "Could not fetch the pronunciation model; it will download on first use."
fi

step "5 of 8  Quality extras"
say "  Music separation, speaker detection and voice cloning. Around 3 GB."
if python -c "import demucs" 2>/dev/null && python -c "import chatterbox" 2>/dev/null; then
  ok "Already installed"
elif python -m pip install -r requirements-quality.txt >>"$LOG" 2>&1; then
  ok "Quality extras installed"
else
  warn "Those failed to install. The app still works, limited to the Fast"
  warn "preset. Re-run this installer to try again."
  warn "Details: $LOG"
fi

step "6 of 8  Speech models"
say "  Fetched now so the first video doesn't stop to download them."
# Not piped through grep: `||` would bind to grep's exit status, which is
# never nonzero here, so the warning could never fire.
if ! python -m app.warmup 2>&1 | tee -a "$LOG"; then
  warn "Some speech models could not be fetched; they'll download on first use."
fi

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
  # Poll rather than sleep a fixed amount: first launch can take well over a
  # few seconds to unpack and start, and `ollama pull` errors if run too early.
  ollama_up() { curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; }
  say "  Waiting for Ollama to start…"
  for _ in $(seq 1 90); do ollama_up && break; sleep 1; done

  # Fall back to the CLI server directly in case the app is stuck behind a
  # first-run window — same binary, no interaction needed.
  if ! ollama_up; then
    say "  Starting the Ollama server directly…"
    nohup ollama serve >>"$LOG" 2>&1 &
    for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
  fi
  # Largest-first per RAM tier, falling back smaller on failure: a 20 GB download
  # dropping mid-transfer is worse than arriving with a smaller model, and all of
  # them translate instructional speech well. Keep tier-1 entries in sync with
  # suggest_ollama_model() in app/config.py.
  if   (( RAM_GB >= 24 )); then LADDER=(qwen3:14b qwen3:8b)
  elif (( RAM_GB >= 16 )); then LADDER=(qwen3:8b qwen3:4b)
  else                          LADDER=(qwen3:4b); fi

  if ollama_up; then
    # Anything already on the ladder is good enough — a re-run must not
    # re-download twenty gigabytes to arrive where it already is.
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
        say "  Downloading $m — chosen to fit your ${RAM_GB} GB. A few GB."
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

step "8 of 8  Building the app"
python -m pip install --quiet pywebview pillow 2>/dev/null || \
  warn "Native window support unavailable — the app will open in your browser instead."

if [[ -n "$PROTECTED" ]]; then
  warn "This folder is in $PROTECTED, which macOS stops apps from reading."
  warn "The app icon will not open until the app lives somewhere else. Re-run"
  warn "the one-line installer from the README and it will move it to"
  warn "Application Support, which macOS never blocks."
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

  Dubbing Studio is in your Applications folder.

  Keep this folder where it is — the app runs from here.

  The models for your chosen quality preset are already
  downloaded. Switching preset later fetches what that
  one needs, once.

DONE
else
  # Don't claim success when something was skipped — better to say so here than
  # have the app open and fail later on whatever didn't install.
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

DONE
  copy_details
fi
read -r -p "Press return to close this window."
