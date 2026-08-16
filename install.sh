#!/bin/bash
# Dubbing Studio — one-line install.
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/gready-hub/dubbing-studio/main/install.sh)"
#
# Why this exists. The other route is: download the zip, find it in Downloads,
# unzip it, move the folder somewhere that isn't Downloads, right-click
# Install.command, choose Open, then click Open again in a warning dialog. Seven
# steps, and two of them go wrong quietly — macOS refuses to let a locally-built
# app read Downloads, Desktop, Documents or iCloud Drive, so leaving the folder
# where it landed installs an app that then simply never opens.
#
# Everything a browser downloads is tagged com.apple.quarantine, which is what
# produces the "unidentified developer" dialog. Nothing fetched by curl is. So
# this route has no dialog to talk anyone through and no Apple Developer
# account behind it — the code is fetched, not opened, and the app is built on
# the machine it will run on.
#
# Safe to run again at any time: it replaces the code and keeps the Python
# environment, which makes it the updater too.

set -uo pipefail

REPO="gready-hub/dubbing-studio"
BRANCH="${DUBBING_STUDIO_BRANCH:-main}"
# Where to install. Piped from curl there is no script on disk, so this is the
# default location; run from inside an install — which is what the app's own
# "Update now" does — it is that install, wherever the user happens to keep it.
DEST="${DUBBING_STUDIO_DIR:-}"
if [[ -z "$DEST" ]]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)" || HERE=""
  if [[ -n "$HERE" && -f "$HERE/app/pipeline.py" && -f "$HERE/Install.command" ]]; then
    DEST="$HERE"
  else
    DEST="$HOME/Dubbing Studio"
  fi
fi
TARBALL="https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz"

BOLD=$'\033[1m'; GREEN=$'\033[32m'; RED=$'\033[31m'; RESET=$'\033[0m'
say()  { printf "%s\n" "$*"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
bad()  { printf "  ${RED}✗${RESET} %s\n" "$*"; }

printf "\n${BOLD}Dubbing Studio${RESET}\n\n"

if [[ "$(uname)" != "Darwin" ]]; then
  bad "This is for macOS. On Windows or Linux use the Docker version instead."
  exit 1
fi

# Under $HOME on purpose, so moving the finished tree into place is a rename on
# the same volume rather than a copy — the Python environment being preserved
# below is a couple of gigabytes, and /tmp is a different filesystem.
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

# The Python environment lives inside the folder and is the expensive part of an
# install — several minutes and a couple of gigabytes. Set aside and put back at
# exactly the same path, which is what lets it keep working: a virtualenv records
# its own location, so it survives a round trip but not a relocation.
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
  # The old install is sitting in a temporary folder that the exit trap is about
  # to delete. Putting it back is the difference between a failed update and no
  # app at all.
  bad "Couldn't put the app in: $DEST"
  if [[ -n "$MOVED" ]] && mv "$TMP/old" "$DEST" 2>/dev/null; then
    [[ -d "$TMP/venv-keep" ]] && mv "$TMP/venv-keep" "$DEST/.venv"
    say "  Your existing version has been left exactly as it was."
  fi
  exit 1
fi
[[ -d "$TMP/venv-keep" ]] && mv "$TMP/venv-keep" "$DEST/.venv"

# What was installed, so the app can notice when it is behind. Best effort — a
# missing file just means no update is ever offered, which is the right failure.
curl -fsSL "https://api.github.com/repos/$REPO/commits/$BRANCH" 2>/dev/null \
  | sed -n 's/.*"sha": *"\([0-9a-f]\{40\}\)".*/\1/p' | head -1 > "$DEST/.version" || true
[[ -s "$DEST/.version" ]] || rm -f "$DEST/.version"

# Belt and braces. Nothing fetched by curl carries it, but a folder left over
# from an earlier zip download might, and one quarantined file in the tree is
# enough to produce the dialog this route exists to avoid.
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
chmod +x "$DEST/Install.command" "$DEST/Update.command" \
         "$DEST/Uninstall.command" "$DEST/install.sh" 2>/dev/null || true
ok "Installed to $DEST"

# Used by the test suite, and genuinely useful on its own: update the code
# without sitting through the full setup again.
if [[ -n "${DUBBING_STUDIO_FETCH_ONLY:-}" ]]; then
  say ""
  ok "Code updated. Skipping setup, as asked."
  exit 0
fi

# Removed here rather than left to the exit trap, which never fires: exec
# replaces this process. On an update the folder holds the whole previous
# version, so without this every update leaves a copy of it in the home folder.
rm -rf "$TMP"
trap - EXIT

say ""
exec "$DEST/Install.command"
