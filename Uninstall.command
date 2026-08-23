#!/bin/bash
# Dubbing Studio — remove it.
#
# Double-click this file, or run it from Terminal. It shows what it will remove
# and asks before doing anything.
#
# The distinction that matters here is between what belongs to this app alone and
# what it happens to use. Homebrew, ffmpeg, yt-dlp and Ollama are installed
# system-wide and other software on this Mac may be relying on them, so they are
# listed with their sizes and the command to remove them, and left alone.
#
# Everything this script does remove goes to the Bin, which is where macOS puts
# things and what makes an uninstall reversible for the ten minutes afterwards
# when someone realises they wanted a file out of it. Emptying the Bin is what
# actually returns the space.

cd "$(dirname "$0")" || exit 1
set -uo pipefail

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; RESET=$'\033[0m'
say()  { printf "%s\n" "$*"; }
step() { printf "\n${BOLD}%s${RESET}\n" "$*"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "  ${YELLOW}!${RESET} %s\n" "$*"; }

# Removal means the Bin, not unlink. /usr/bin/trash arrived in macOS 14, so on 12
# and 13 the move is made by hand; ~/.Trash is on the same volume as everything
# below, which makes it a rename. The hand-rolled move loses the Put Back that
# Finder offers, which is why it is the fallback and not the method. Anything
# that will not go is left exactly where it is and named at the end — an
# uninstall that misses a folder is a nuisance, one that erases the wrong folder
# is not.
HAVE_TRASH=""
[[ -x /usr/bin/trash ]] && HAVE_TRASH="yes"

STRANDED=()
bin_it() {
  local failed=0 path dest base n existing=()
  for path in "$@"; do [[ -e "$path" ]] && existing+=("$path"); done
  (( ${#existing[@]} )) || return 0

  # One call for everything that still exists, rather than one process per
  # path — trash(1) moves what it can even when the batch as a whole reports
  # failure, so a non-zero exit here just means the loop below has fewer
  # paths left to pick up than it started with.
  if [[ -n "$HAVE_TRASH" ]] && /usr/bin/trash "${existing[@]}" 2>/dev/null; then
    return 0
  fi

  for path in "${existing[@]}"; do
    [[ -e "$path" ]] || continue
    if [[ -n "$HAVE_TRASH" ]] && /usr/bin/trash "$path" 2>/dev/null; then
      continue
    fi
    base="$(basename "$path")"; dest="$HOME/.Trash/$base"; n=1
    while [[ -e "$dest" ]]; do n=$((n + 1)); dest="$HOME/.Trash/$base $n"; done
    mv "$path" "$dest" 2>/dev/null || { STRANDED+=("$path"); failed=1; }
  done
  return $failed
}

APP_DIR="$PWD"

# The app's own Python already knows these two roots — asking it is the one
# source of truth for the rule, rather than a second copy of it in bash that
# a future change to config.py has no way to reach. Falls back to the same
# computation platformdirs makes only when the venv itself can't answer,
# which is exactly when this script tends to get run.
_paths=""
if [[ -x "$APP_DIR/.venv/bin/python3" ]]; then
  _paths="$(cd "$APP_DIR" && "$APP_DIR/.venv/bin/python3" -c '
from app.config import BASE, CACHE
print(BASE)
print(CACHE)
' 2>/dev/null)"
fi
if [[ -n "$_paths" ]]; then
  SUPPORT="$(sed -n 1p <<< "$_paths")"
  CACHE="$(sed -n 2p <<< "$_paths")"
else
  SUPPORT="${DUBBING_STUDIO_HOME:-$HOME/Library/Application Support/DubbingStudio}"
  if [[ -n "${DUBBING_STUDIO_CACHE:-}" ]]; then
    CACHE="$DUBBING_STUDIO_CACHE"
  elif [[ -n "${DUBBING_STUDIO_HOME:-}" ]]; then
    CACHE="$SUPPORT/cache"
  else
    CACHE="$HOME/Library/Caches/DubbingStudio"
  fi
fi
# Trimmed either way: a trailing slash on DUBBING_STUDIO_HOME would otherwise
# survive into the NESTED match below and make it miss.
SUPPORT="${SUPPORT%/}"
CACHE="${CACHE%/}"
BUNDLE="$HOME/Applications/Dubbing Studio.app"
OUTPUT="${DUBBING_STUDIO_OUTPUT:-$HOME/Movies/Dubbed}"
HF="$HOME/.cache/huggingface/hub"

# Where the code lives decides how this ends. Installed under Application
# Support it sits inside the support folder, so one move takes the app, the
# settings and the history together; an install from the older layout keeps them
# in separate places and both have to be named.
case "$APP_DIR/" in
  "$SUPPORT"/*) NESTED="yes" ;;
  *)            NESTED="" ;;
esac

# Only the repositories this app fetches. Anything else in that cache belongs to
# another tool and is not ours to remove.
HF_OURS=()
if [[ -d "$HF" ]]; then
  while IFS= read -r line; do [[ -n "$line" ]] && HF_OURS+=("$line"); done < <(
    find "$HF" -maxdepth 1 -type d \( \
      -name 'models--mlx-community--parakeet-tdt*' -o \
      -name 'models--mlx-community--Kokoro-82M*'   -o \
      -name 'models--mlx-community--whisper*'      -o \
      -name 'models--prince-canuma--Kokoro-82M*'   -o \
      -name 'models--adefossez--HTDemucs*'         -o \
      -name 'models--ResembleAI--chatterbox*' \) 2>/dev/null)
fi

size_of() { [[ -e "$1" ]] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"; }

clear
cat <<'BANNER'
  ┌──────────────────────────────────────────┐
  │        Remove Dubbing Studio             │
  └──────────────────────────────────────────┘
BANNER

step "This will be moved to the Bin"
say "  $(printf '%-6s' "$(size_of "$CACHE")")  Working files, speech models and voice samples"
say "            ${DIM}$CACHE${RESET}"
if [[ -n "$NESTED" ]]; then
  # One row rather than two: the app folder is inside this one, and listing both
  # would count the Python environment twice.
  say "  $(printf '%-6s' "$(size_of "$SUPPORT")")  The app, its Python environment, settings and history"
  say "            ${DIM}$SUPPORT${RESET}"
else
  say "  $(printf '%-6s' "$(size_of "$SUPPORT")")  Settings and history"
  say "            ${DIM}$SUPPORT${RESET}"
  say "  $(printf '%-6s' "$(size_of "$APP_DIR")")  The app and its Python environment"
  say "            ${DIM}$APP_DIR${RESET}"
fi
say "  $(printf '%-6s' "$(size_of "$BUNDLE")")  The Applications shortcut"
if (( ${#HF_OURS[@]} )); then
  total=$(du -sch "${HF_OURS[@]}" 2>/dev/null | tail -1 | cut -f1)
  say "  $(printf '%-6s' "$total")  Downloaded AI models (${#HF_OURS[@]} of them)"
  say "            ${DIM}$HF${RESET}"
fi
say "  -       Logs"

step "This will be left alone"
say "  $(printf '%-6s' "$(size_of "$OUTPUT")")  Dubbed videos"
say "            ${DIM}$OUTPUT${RESET}"

step "Shared with the rest of your Mac — not touched"
say "  Other software may be using these. To remove them yourself:"
say ""
if command -v ollama >/dev/null 2>&1; then
  say "    Ollama and its models   $(size_of "$HOME/.ollama")"
  # trash(1) is only guaranteed from macOS 14; typed advice gets no fallback
  # of its own the way bin_it does, so it has to pick a command that will
  # actually run on whatever's in front of it.
  if [[ -n "$HAVE_TRASH" ]]; then
    say "      ${DIM}brew uninstall --cask ollama && trash ~/.ollama${RESET}"
  else
    say "      ${DIM}brew uninstall --cask ollama && mv ~/.ollama ~/.Trash/${RESET}"
  fi
fi
if command -v brew >/dev/null 2>&1; then
  say "    ffmpeg and yt-dlp"
  say "      ${DIM}brew uninstall ffmpeg yt-dlp${RESET}"
  say "    Homebrew itself"
  say "      ${DIM}see https://github.com/homebrew/install#uninstall-homebrew${RESET}"
fi

printf "\n${BOLD}Type ${RED}remove${RESET}${BOLD} to go ahead, or press return to cancel: ${RESET}"
read -r answer
if [[ "$answer" != "remove" ]]; then
  say ""
  ok "Nothing was moved."
  read -r -p "Press return to close this window."
  exit 0
fi

step "Removing"
# The app serves on loopback and may still be running; a live process holding the
# folder open makes the rest of this messier than it needs to be.
pkill -f "app.desktop" 2>/dev/null && ok "Closed the running app"
pkill -f "app.server" 2>/dev/null

bin_it "$CACHE"  && ok "Working files, speech models and voice samples"
bin_it "$BUNDLE" && ok "Applications shortcut"
# Glob rather than a list: the rotating handler leaves DubbingStudio.log.1 and
# friends behind, and naming each file by hand is how one gets left on the disk.
bin_it "$HOME/Library/Logs/DubbingStudio.log"* \
       "$HOME/Library/Logs/DubbingStudio-install.log" \
       "$HOME/Library/Logs/DubbingStudio-crash.log" && ok "Logs"
if (( ${#HF_OURS[@]} )); then
  bin_it "${HF_OURS[@]}" && ok "Downloaded AI models"
fi
# Kept back when the app folder is inside it, so the folder holding this
# running script is always the one handled last, detached below.
if [[ -z "$NESTED" ]]; then
  bin_it "$SUPPORT" && ok "Settings and history"
fi

if (( ${#STRANDED[@]} )); then
  say ""
  warn "These could not be moved to the Bin and are still on the disk:"
  for path in "${STRANDED[@]}"; do say "      $path"; done
fi

say ""
ok "Dubbed videos are still in $OUTPUT"

# Last, because this script is running from inside it — detached, so the shell
# is not moving the ground out from under itself mid-run. "remove" already was
# the confirmation; this does not ask a second time for what the banner above
# already said would go.
if [[ -n "$NESTED" ]]; then
  ( sleep 1; bin_it "$SUPPORT" ) >/dev/null 2>&1 &
  ok "The app, its settings and its history — moving to the Bin"
else
  ( sleep 1; bin_it "$APP_DIR" ) >/dev/null 2>&1 &
  ok "The app folder — moving to the Bin"
fi

say ""
say "  ${BOLD}Done.${RESET}"
read -r -p "  Press return to close this window."
