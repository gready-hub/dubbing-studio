#!/bin/bash
# Double-click to launch Dubbing Studio. Closing this window stops the app.

cd "$(dirname "$0")" || exit 1

for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do
  [[ -x "$p" ]] && eval "$("$p" shellenv)"
done

if [[ ! -d .venv ]]; then
  echo
  echo "  Dubbing Studio isn't set up yet."
  echo "  Double-click \"Install\" first, then try again."
  echo
  read -r -p "Press return to close."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

command -v ollama >/dev/null 2>&1 && ! pgrep -qx ollama && open -a Ollama 2>/dev/null

PORT=8765
while lsof -i ":$PORT" >/dev/null 2>&1; do PORT=$((PORT+1)); done

clear
echo
echo "  Dubbing Studio is starting…"
echo "  Your browser will open automatically."
echo
echo "  Keep this window open while you use the app."
echo "  Close it when you're finished."
echo
exec python -m app.server "$PORT"
