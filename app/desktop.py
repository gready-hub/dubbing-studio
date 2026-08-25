"""Native desktop window.

Runs the web server on a background thread and shows it via macOS's built-in
WebView (no Chromium bundled). Falls back to the default browser if pywebview
isn't importable, so the app still works if that dependency failed to install.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.request


def _free_port(start: int = 8765) -> int:
    port = start
    while port < start + 60:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start


def _serve(port: int) -> None:
    import uvicorn
    from .server import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


def _wait_until_up(url: str, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:                                        # noqa: BLE001
            time.sleep(0.25)
    return False


def main() -> None:
    # First, since this is the app-icon entry point: a crash before the window
    # appears is the one failure with no interface left to report it.
    from . import logs
    logs.setup()
    logs.get().info("app start", extra={"launched_by": "app bundle"})

    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    threading.Thread(target=_serve, args=(port,), daemon=True).start()
    if not _wait_until_up(url):
        print("The app failed to start.", file=sys.stderr)
        sys.exit(1)

    try:
        import webview
    except ImportError:
        import webbrowser
        print(f"Opening {url} in your browser.")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return
        return

    webview.create_window(
        "Dubbing Studio",
        url,
        width=880,
        height=940,
        min_size=(620, 600),
        confirm_close=False,
    )
    webview.start()          # blocks until the window is closed


if __name__ == "__main__":
    main()
