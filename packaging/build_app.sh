#!/bin/bash
# Builds "Dubbing Studio.app" into ~/Applications.
#
# Assembled locally rather than shipped pre-built: Gatekeeper doesn't flag
# files created on this Mac, so the app opens with a normal double-click
# instead of the "unidentified developer" warning a downloaded .app would get.

set -uo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
APPS="$HOME/Applications"
APP="$APPS/Dubbing Studio.app"
CONTENTS="$APP/Contents"

mkdir -p "$APPS"
rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>Dubbing Studio</string>
  <key>CFBundleDisplayName</key>       <string>Dubbing Studio</string>
  <key>CFBundleIdentifier</key>        <string>local.dubbingstudio.app</string>
  <key>CFBundleVersion</key>           <string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleExecutable</key>        <string>DubbingStudio</string>
  <key>CFBundleIconFile</key>          <string>icon</string>
  <key>NSHighResolutionCapable</key>   <true/>
  <key>LSMinimumSystemVersion</key>    <string>12.0</string>
  <key>NSAppTransportSecurity</key>
  <dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict>
</plist>
PLIST

cat > "$CONTENTS/MacOS/DubbingStudio" <<LAUNCHER
#!/bin/bash
# Stub only — the real code lives in the folder you installed from.
SRC="$SRC"

for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do
  [[ -x "\$p" ]] && eval "\$("\$p" shellenv)"
done

cd "\$SRC" || {
  osascript -e 'display alert "Dubbing Studio" message "The app folder has moved. Re-run the installer from its new location."'
  exit 1
}

if [[ ! -d .venv ]]; then
  osascript -e 'display alert "Dubbing Studio" message "Setup is not finished. The installer will open in a Terminal window — let it run, then start Dubbing Studio again."'
  open -a Terminal "\$SRC/Install.command"
  exit 1
fi

source .venv/bin/activate 2>/dev/null

# macOS silently denies (rather than prompting) a bundle like this one reading
# Downloads/Desktop/Documents, so surface the fix instead of failing silently.
if ! command -v python >/dev/null 2>&1; then
  osascript -e 'display alert "Dubbing Studio" message "macOS is blocking the app from reading its own folder.\n\nThis happens when the app is kept in Downloads, Desktop or Documents. Re-run the one-line installer from the README — it installs to Application Support, which macOS allows.\n\nUntil then you can still use \"Start Dubbing Studio\" inside the folder." as critical'
  exit 1
fi

command -v ollama >/dev/null 2>&1 && ! pgrep -qx ollama && open -a Ollama 2>/dev/null

# Separate file from DubbingStudio.log: that one rotates via its own handler,
# and a second writer appending to the same path would hold the old inode
# across a rotation, silently clobbering it.
exec python -m app.desktop 2>> "\$HOME/Library/Logs/DubbingStudio-crash.log"
LAUNCHER
chmod +x "$CONTENTS/MacOS/DubbingStudio"

if [[ -f "$SRC/.venv/bin/python" ]]; then
  ICONSET="$(mktemp -d)/icon.iconset"
  mkdir -p "$ICONSET"
  if "$SRC/.venv/bin/python" "$SRC/packaging/make_icon.py" "$ICONSET/../icon.png" >/dev/null 2>&1; then
    BASE="$ICONSET/../icon.png"
    for s in 16 32 64 128 256 512; do
      sips -z $s $s     "$BASE" --out "$ICONSET/icon_${s}x${s}.png"      >/dev/null 2>&1
      sips -z $((s*2)) $((s*2)) "$BASE" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null 2>&1
    done
    iconutil -c icns "$ICONSET" -o "$CONTENTS/Resources/icon.icns" >/dev/null 2>&1
  fi
fi

# Nudge Finder/Launchpad to notice the new bundle.
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
  -f "$APP" >/dev/null 2>&1 || true

echo "$APP"
