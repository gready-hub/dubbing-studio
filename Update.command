#!/bin/bash
# Dubbing Studio — update in place.
#
# Fetches the current version, keeps the Python environment and everything under
# Application Support, then runs the setup again for anything new. Double-click
# it, or press "Update now" in the app.
cd "$(dirname "$0")" || exit 1
if [[ ! -f install.sh ]]; then
  echo "  install.sh is missing from this folder — reinstall from the README instead."
  read -r -p "  Press return to close this window."
  exit 1
fi
chmod +x install.sh 2>/dev/null
exec /bin/bash ./install.sh
