#!/bin/bash
# Dubbing Studio — remove it.
#
# Double-click this file, or run it from Terminal. It shows what it will delete
# and asks before doing anything.
#
# The distinction that matters here is between what belongs to this app alone and
# what it happens to use. Homebrew, ffmpeg, yt-dlp and Ollama are installed
# system-wide and other software on this Mac may be relying on them, so they are
# listed with their sizes and the command to remove them, and left alone.

cd "$(dirname "$0")" || exit 1
set -uo pipefail

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; RESET=$'\033[0m'
say()  { printf "%s\n" "$*"; }
step() { printf "\n${BOLD}%s${RESET}\n" "$*"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "  ${YELLOW}!${RESET} %s\n" "$*"; }

APP_DIR="$PWD"
SUPPORT="$HOME/Library/Application Support/DubbingStudio"
BUNDLE="$HOME/Applications/Dubbing Studio.app"
OUTPUT="${DUBBING_STUDIO_OUTPUT:-$HOME/Movies/Dubbed}"
HF="$HOME/.cache/huggingface/hub"

# Only the repositories this app fetches. Anything else in that cache belongs to
# another tool and is not ours to delete.
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

step "This will be deleted"
say "  $(printf '%-6s' "$(size_of "$SUPPORT")")  Settings, history and working files"
say "            ${DIM}$SUPPORT${RESET}"
say "  $(printf '%-6s' "$(size_of "$APP_DIR")")  The app and its Python environment"
say "            ${DIM}$APP_DIR${RESET}"
say "  $(printf '%-6s' "$(size_of "$BUNDLE")")  The Applications shortcut"
if (( ${#HF_OURS[@]} )); then
  total=$(du -sch "${HF_OURS[@]}" 2>/dev/null | tail -1 | cut -f1)
  say "  $(printf '%-6s' "$total")  Downloaded AI models (${#HF_OURS[@]} of them)"
  say "            ${DIM}$HF${RESET}"
fi
say "  -       Logs"

step "This will be kept"
say "  $(printf '%-6s' "$(size_of "$OUTPUT")")  Your dubbed videos"
say "            ${DIM}$OUTPUT${RESET}"

step "Shared with the rest of your Mac — not touched"
say "  Other software may be using these. To remove them yourself:"
say ""
if command -v ollama >/dev/null 2>&1; then
  say "    Ollama and its models   $(size_of "$HOME/.ollama")"
  say "      ${DIM}brew uninstall --cask ollama && rm -rf ~/.ollama${RESET}"
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
  ok "Nothing was deleted."
  read -r -p "Press return to close this window."
  exit 0
fi

step "Removing"
# The app serves on loopback and may still be running; a live process holding the
# folder open makes the rest of this messier than it needs to be.
pkill -f "app.desktop" 2>/dev/null && ok "Closed the running app"
pkill -f "app.server" 2>/dev/null

rm -rf "$SUPPORT"      && ok "Settings, history and working files"
rm -rf "$BUNDLE"       && ok "Applications shortcut"
# Glob rather than a list: the rotating handler leaves DubbingStudio.log.1 and
# friends behind, and naming each file by hand is how one gets left on the disk.
rm -f  "$HOME/Library/Logs/DubbingStudio.log"* \
       "$HOME/Library/Logs/DubbingStudio-install.log" \
       "$HOME/Library/Logs/DubbingStudio-crash.log" && ok "Logs"
if (( ${#HF_OURS[@]} )); then
  rm -rf "${HF_OURS[@]}" && ok "Downloaded AI models"
fi

say ""
ok "Your dubbed videos are still in $OUTPUT"

# Last, because it is the folder this script is running from. Detached, so the
# shell is not deleting the file underneath itself mid-run.
step "One thing left"
say "  The app folder itself:"
say "    ${DIM}$APP_DIR${RESET}"
printf "\n  Delete it too? [y/N] "
read -r last
if [[ "$last" == "y" || "$last" == "Y" ]]; then
  ( sleep 1; rm -rf "$APP_DIR" ) >/dev/null 2>&1 &
  ok "It will be gone in a moment."
else
  say "  Left in place. Drag it to the bin whenever you like."
fi

say ""
say "  ${BOLD}Done.${RESET}"
read -r -p "  Press return to close this window."
