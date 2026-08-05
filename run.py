#!/usr/bin/env python3
# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# One-Click Application Launcher
# =============================================================================
"""
Launches the AI Phishing Shield Streamlit app in the default browser.

Starts the Streamlit server headlessly, waits for it to become ready, then
opens the default browser automatically. All Streamlit log output is written
to ``logs/app_streamlit.log`` so the console only shows clean status messages.

Usage:
    python run.py                 # start the app and open the browser
    python run.py --port 8502     # use a different preferred port
    python run.py --no-browser    # start the server without opening the browser

Double-clicking ``start.bat`` on Windows is equivalent to ``python run.py``.
"""

from __future__ import annotations

import argparse
import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_SCRIPT = PROJECT_ROOT / "app" / "app.py"
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_PORT = 8501
STARTUP_TIMEOUT = 90.0
HEALTH_PATH = "/_stcore/health"


def _port_in_use(port: int) -> bool:
    """Return True when ``port`` is already bound on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _pick_port(preferred: int) -> int:
    """Return the first free port at or after ``preferred``."""
    for port in range(preferred, preferred + 50):
        if not _port_in_use(port):
            return port
    raise RuntimeError(f"No free port found between {preferred} and {preferred + 49}.")


def _server_ready(url: str) -> bool:
    """Return True when the Streamlit health endpoint responds with 200."""
    try:
        with urllib.request.urlopen(url + HEALTH_PATH, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_until_ready(url: str, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Poll the health endpoint until the server responds or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_ready(url):
            return True
        time.sleep(0.3)
    return False


def _open_browser(url: str) -> None:
    """Open ``url`` in the system default browser."""
    try:
        if webbrowser.open(url):
            return
    except Exception:
        pass
    try:
        os.startfile(url)  # type: ignore[attr-defined]  # Windows fallback
    except Exception:
        print("  Could not open the browser automatically.")
        print(f"  Please open this address manually: {url}")


def main() -> int:
    # Make sure status messages appear even when stdout is redirected or piped.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Launch the AI Phishing Shield application.")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Preferred server port (default: {DEFAULT_PORT})."
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    args = parser.parse_args()

    if not APP_SCRIPT.exists():
        print(f"[ERROR] Application file not found: {APP_SCRIPT}")
        return 1

    preferred_url = f"http://127.0.0.1:{args.port}"

    print("=" * 56)
    print("  AI Phishing Shield")
    print("=" * 56)

    # If an instance of the app is already running, just bring it to the front.
    if _server_ready(preferred_url):
        print("  Application is already running.")
        if not args.no_browser:
            _open_browser(preferred_url)
        return 0

    port = _pick_port(args.port)
    url = f"http://127.0.0.1:{port}"
    if port != args.port:
        print(f"  Port {args.port} is busy; using port {port} instead.")

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / "app_streamlit.log"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_SCRIPT),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,
    )

    def cleanup() -> None:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()

    atexit.register(cleanup)

    print("  Starting the application server...")

    if not _wait_until_ready(url):
        print(f"  [ERROR] Server did not start. Details: {log_path}")
        cleanup()
        return 1

    # Brief settle so the first render has a head start before the tab opens.
    time.sleep(1.0)
    print(f"  Application ready at {url}")

    if not args.no_browser:
        print("  Opening your default browser...")
        _open_browser(url)

    print("-" * 56)
    print("  The application is now running.")
    print("  Close this window or press Ctrl+C to stop it.")
    print("-" * 56)

    try:
        return process.wait()
    except KeyboardInterrupt:
        print()
        print("  Stopping the application...")
        return 0
    finally:
        log_file.close()


if __name__ == "__main__":
    sys.exit(main())
