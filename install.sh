#!/bin/bash
# Dubbing Studio — one-line install.
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/gready-hub/dubbing-studio/main/install.sh)"
#
# The zip-download route fails quietly: macOS refuses to let a locally-built app
# read Downloads/Desktop/Documents/iCloud Drive, so an app left in the folder it
# unzipped into never opens. Browser downloads are tagged com.apple.quarantine
# (the "unidentified developer" dialog); curl doesn't tag them, so this route has
# no dialog and needs no Apple Developer account — the app is built on the
# machine it runs on.
#
# Safe to run again at any time: it replaces the code and keeps the Python
# environment, which makes it the updater too.

set -uo pipefail

REPO="gready-hub/dubbing-studio"
BRANCH="${DUBBING_STUDIO_BRANCH:-main}"

# Whatever is at DEST is replaced wholesale (moved aside, then deleted), so it
# must be a folder this app owns — DUBBING_STUDIO_DIR pointed at Documents or a
# home folder would take that folder down with it.
is_install() { [[ -f "$1/Install.command" && -f "$1/app/pipeline.py" ]]; }
# .DS_Store doesn't count as occupied: Finder writes one just from a folder
# being looked at, before anything is put in it.
is_empty() {
  [[ -d "$1" ]] || return 1
  [[ -z "$(ls -A "$1" 2>/dev/null | grep -v '^\.DS_Store$')" ]]
}

# Piped from curl there's no script on disk, so DEST defaults to Application
# Support — not blocked for a locally-built app, unlike Documents/Desktop/
# Downloads/iCloud Drive. Run from inside an existing install (what the app's
# own "Update now" does), DEST is that install instead, wherever it lives.
# settings.json and history.json live in the folder above; nesting is deliberate
# since an update replaces this folder wholesale.
DEST="${DUBBING_STUDIO_DIR:-}"
if [[ -z "$DEST" ]]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)" || HERE=""
  if [[ -n "$HERE" ]] && is_install "$HERE"; then
    DEST="$HERE"
  else
    DEST="$HOME/Library/Application Support/DubbingStudio/program"
  fi
fi
TARBALL="https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz"

BOLD=$'\033[1m'; GREEN=$'\033[32m'; RED=$'\033[31m'; RESET=$'\033[0m'
say()  { printf "%s\n" "$*"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
bad()  { printf "  ${RED}✗${RESET} %s\n" "$*"; }

printf "\n${BOLD}Dubbing Studio${RESET}\n\n"

if [[ "$(uname)" != "Darwin" ]]; then
  bad "This is for macOS only."
  exit 1
fi

if [[ -e "$DEST" ]] && ! is_install "$DEST" && ! is_empty "$DEST"; then
  bad "That path holds something that is not a Dubbing Studio install:"
  say "      $DEST"
  say ""
  say "  Installing would replace it, so this stops here instead. Move that"
  say "  folder yourself if it should go, or point DUBBING_STUDIO_DIR somewhere"
  say "  else."
  exit 1
fi

# Under $HOME on purpose: moving the finished tree into place is then a rename
# on the same volume, not a copy — /tmp is a different filesystem.
TMP="$(mktemp -d "$HOME/.dubbing-studio-setup.XXXXXX")" || {
  bad "Couldn't create a temporary folder in your home folder."; exit 1; }
trap 'rm -rf "$TMP"' EXIT

say "  Fetching the latest version…"
if ! curl -fsSL "$TARBALL" | tar xz -C "$TMP" 2>/dev/null; then
  bad "Couldn't download it. Check your internet connection and try again."
  say "    Tried: $TARBALL"
  exit 1
fi

SRC="$TMP/$(basename "$REPO")-$BRANCH"
[[ -f "$SRC/Install.command" ]] || {
  bad "The download didn't contain what was expected."; exit 1; }
ok "Downloaded"

# Set aside and restored at the exact same path: a virtualenv records its own
# location, so it survives a round trip but not a relocation.
if [[ -d "$DEST/.venv" ]]; then
  mv "$DEST/.venv" "$TMP/venv-keep" 2>/dev/null && ok "Keeping the existing Python setup"
fi

MOVED=""
if [[ -e "$DEST" ]]; then
  rm -rf "$TMP/old" && mv "$DEST" "$TMP/old" || {
    bad "Couldn't replace the existing folder at: $DEST"; exit 1; }
  MOVED="yes"
fi
mkdir -p "$(dirname "$DEST")"
if ! mv "$SRC" "$DEST"; then
  # The exit trap is about to delete the temp folder holding the old install —
  # restoring it now is the difference between a failed update and no app at all.
  bad "Couldn't put the app in: $DEST"
  if [[ -n "$MOVED" ]] && mv "$TMP/old" "$DEST" 2>/dev/null; then
    [[ -d "$TMP/venv-keep" ]] && mv "$TMP/venv-keep" "$DEST/.venv"
    say "  Your existing version has been left exactly as it was."
  fi
  exit 1
fi
[[ -d "$TMP/venv-keep" ]] && mv "$TMP/venv-keep" "$DEST/.venv"

# Records what was installed so the app can notice it's behind. Best effort —
# a missing file just means no update is ever offered.
curl -fsSL "https://api.github.com/repos/$REPO/commits/$BRANCH" 2>/dev/null \
  | sed -n 's/.*"sha": *"\([0-9a-f]\{40\}\)".*/\1/p' | head -1 > "$DEST/.version" || true
[[ -s "$DEST/.version" ]] || rm -f "$DEST/.version"

# Belt and braces: a folder left over from an earlier zip download might carry
# the quarantine flag, and one quarantined file is enough to trigger the dialog.
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
chmod +x "$DEST/Install.command" "$DEST/Update.command" \
         "$DEST/Uninstall.command" "$DEST/install.sh" 2>/dev/null || true
ok "Installed to $DEST"
# Hidden in Finder, so the path alone is not much use to anyone.
say "    Open that folder later with:  open \"$DEST\""

# Used by the test suite; also useful standalone to update the code without
# the full setup.
if [[ -n "${DUBBING_STUDIO_FETCH_ONLY:-}" ]]; then
  say ""
  ok "Code updated. Skipping setup, as asked."
  exit 0
fi

# Removed here, not left to the exit trap: exec below replaces this process,
# so the trap never fires and the old version would otherwise never be cleaned up.
rm -rf "$TMP"
trap - EXIT

say ""
exec "$DEST/Install.command"
