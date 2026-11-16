# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Frame capture from a page in headless Chrome (or Edge, or Chromium), through
the Chrome DevTools Protocol over a small standard-library WebSocket client:
no Python dependency beyond the browser itself.

The page exposes ``window.__st.frame(i)``, which draws frame ``i`` at a fixed
time step and resolves when the pixels are ready, so capture is
deterministic. WebGL runs on SwiftShader (CPU); with psutil installed, every
browser process is pinned to ``threads`` CPUs.
"""
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np

_CANDIDATES = {
    "win32": [
        r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe",
        r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe",
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
        r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe",
        r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe",
    ],
    "darwin": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ],
}
_NAMES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
          "chrome", "msedge", "microsoft-edge"]


class BrowserUnavailable(RuntimeError):
    """No Chrome, Edge or Chromium executable was found."""


def find_browser(path: Optional[str] = None) -> str:
    """The browser executable: ``path``, ``$SPECTRAL_TRUST_BROWSER``, the
    usual install locations, or the first Chrome-like name on PATH."""
    for p in (path, os.environ.get("SPECTRAL_TRUST_BROWSER")):
        if p:
            if Path(p).is_file():
                return str(p)
            raise BrowserUnavailable(f"browser executable {p!r} does not exist")
    for p in _CANDIDATES.get(sys.platform, []):
        p = os.path.expandvars(p)
        if "%" not in p and Path(p).is_file():
            return p
    for n in _NAMES:
        w = shutil.which(n)
        if w:
            return w
    raise BrowserUnavailable(
        "rendering this view to video needs Chrome, Edge or Chromium (headless). Install "
        "one, or set SPECTRAL_TRUST_BROWSER to its executable, or pass engine='matplotlib' "
        "for the simpler fallback")


class _WebSocket:
    """A minimal RFC 6455 client: text frames, fragmentation, ping and close."""

    def __init__(self, url: str, timeout: float = 120.0):
        assert url.startswith("ws://")
        rest = url[5:]
        hostport, _, path = rest.partition("/")
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port or 80)), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("the browser closed the DevTools connection")
            head += chunk
        status, _, extra = head.partition(b"\r\n\r\n")
        if b" 101 " not in status.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"WebSocket handshake failed: {status[:80]!r}")
        self._buf = bytearray(extra)

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self._buf)))
            if not chunk:
                raise ConnectionError("the browser closed the DevTools connection")
            self._buf += chunk
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def _send_frame(self, opcode: int, payload: bytes):
        head = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        mask = os.urandom(4)
        head += mask
        data = np.frombuffer(payload, dtype=np.uint8)
        m = np.frombuffer(mask * (n // 4 + 1), dtype=np.uint8)[:n]
        self.sock.sendall(bytes(head) + (data ^ m).tobytes())

    def send(self, text: str):
        self._send_frame(0x1, text.encode("utf-8"))

    def recv(self) -> str:
        parts: List[bytes] = []
        while True:
            b0, b1 = self._read(2)
            fin, opcode, n = b0 & 0x80, b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            data = self._read(n)
            if mask:
                m = np.frombuffer(mask * (n // 4 + 1), dtype=np.uint8)[:n]
                data = (np.frombuffer(data, dtype=np.uint8) ^ m).tobytes()
            if opcode == 0x9:                      # ping
                self._send_frame(0xA, data)
                continue
            if opcode == 0x8:
                raise ConnectionError("the browser closed the DevTools connection")
            if opcode in (0x0, 0x1, 0x2):
                parts.append(data)
                if fin:
                    return b"".join(parts).decode("utf-8")

    def close(self):
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        self.sock.close()


#: Environment variable that forces ``--no-sandbox`` on ("1", "true", "yes")
#: or off ("0", "false", "no"); unset, it is on only when running as root.
NO_SANDBOX_ENV = "SPECTRAL_TRUST_BROWSER_NO_SANDBOX"


def _running_as_root() -> bool:
    geteuid = getattr(os, "geteuid", None)          # absent on Windows
    return bool(geteuid is not None and geteuid() == 0)


def needs_no_sandbox(no_sandbox: Optional[bool] = None) -> bool:
    """Whether to pass ``--no-sandbox``: Chrome refuses to start its sandbox
    as root (Colab, most containers). An explicit ``no_sandbox`` wins, then
    the environment variable :data:`NO_SANDBOX_ENV`, then running as root."""
    if no_sandbox is not None:
        return bool(no_sandbox)
    env = os.environ.get(NO_SANDBOX_ENV, "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return _running_as_root()


def browser_args(exe: str, width: int, height: int, threads: int, profile: str,
                 no_sandbox: Optional[bool] = None) -> List[str]:
    """The command line of the headless browser."""
    # Headless Chrome stalls on screenshots of very small windows; the
    # window keeps a usual size and the page is emulated at the exact one.
    win_w, win_h = max(int(width), 640), max(int(height), 480)
    args = [exe, "--headless=new", "--remote-debugging-port=0",
            f"--user-data-dir={profile}", "--no-first-run",
            "--no-default-browser-check", "--disable-extensions", "--mute-audio",
            "--disable-background-networking", "--disable-sync", "--hide-scrollbars",
            "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            "--force-device-scale-factor=1", "--renderer-process-limit=1",
            f"--num-raster-threads={max(1, int(threads))}",
            "--allow-file-access-from-files", f"--window-size={win_w},{win_h}"]
    if needs_no_sandbox(no_sandbox):
        args.append("--no-sandbox")
    return args + ["about:blank"]


class Browser:
    """A headless browser with one page, driven through DevTools.
    ``no_sandbox`` as in :func:`needs_no_sandbox`."""

    def __init__(self, width: int, height: int, threads: int = 2,
                 executable: Optional[str] = None, timeout: float = 60.0,
                 no_sandbox: Optional[bool] = None):
        exe = find_browser(executable)
        self.width, self.height = int(width), int(height)
        self.profile = tempfile.mkdtemp(prefix="st-browser-")
        args = browser_args(exe, self.width, self.height, threads, self.profile, no_sandbox)
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.threads = max(1, int(threads))
        self._pin()
        port_file = Path(self.profile) / "DevToolsActivePort"
        t0 = time.time()
        port = None
        while port is None:
            try:                              # the browser may still be writing it
                text = port_file.read_text().split()
                port = int(text[0]) if text else None
            except (OSError, ValueError):
                port = None
            if port is None:
                if self.proc.poll() is not None or time.time() - t0 > timeout:
                    self.close()
                    raise BrowserUnavailable(f"{exe} did not start a DevTools endpoint")
                time.sleep(0.05)
        pages = []
        while not pages:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as r:
                pages = [t for t in json.loads(r.read()) if t.get("type") == "page"]
            if not pages:
                time.sleep(0.05)
        self.ws = _WebSocket(pages[0]["webSocketDebuggerUrl"])
        self._id = 0
        self.call("Page.enable")
        self.call("Runtime.enable")
        self.call("Emulation.setDeviceMetricsOverride",
                  {"width": int(width), "height": int(height), "deviceScaleFactor": 1,
                   "mobile": False})
        self._pin()

    def _pin(self):
        """Pin the browser and its children to the first ``threads`` CPUs."""
        try:
            import psutil
        except ImportError:
            return
        try:
            cpus = list(range(min(self.threads, os.cpu_count() or 1)))
            root = psutil.Process(self.proc.pid)
            for p in [root] + root.children(recursive=True):
                try:
                    p.cpu_affinity(cpus)
                except (psutil.Error, AttributeError, OSError):
                    pass
        except Exception:
            pass

    def call(self, method: str, params: Optional[dict] = None) -> dict:
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            try:
                msg = json.loads(self.ws.recv())
            except socket.timeout:
                raise RuntimeError(f"the browser did not answer {method} in time") from None
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error'].get('message')}")
                return msg.get("result", {})
            if msg.get("method") == "Runtime.exceptionThrown":
                d = msg["params"]["exceptionDetails"]
                exc = d.get("exception", {}).get("description") or d.get("text")
                raise RuntimeError(f"error in the page: {exc}")

    def evaluate(self, expr: str, await_promise: bool = True):
        r = self.call("Runtime.evaluate", {"expression": expr, "awaitPromise": await_promise,
                                           "returnByValue": True})
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError("error in the page: "
                               + (d.get("exception", {}).get("description") or d.get("text")))
        return r.get("result", {}).get("value")

    def open(self, path: Path, ready: str = "window.__st && window.__st.ready"):
        self.call("Page.navigate", {"url": Path(path).resolve().as_uri()})
        t0 = time.time()
        while True:
            try:
                if self.evaluate(f"!!({ready})", await_promise=False):
                    break
            except RuntimeError:
                pass
            if time.time() - t0 > 60:
                raise RuntimeError("the page did not become ready")
            time.sleep(0.05)
        self.evaluate(ready)                     # a promise: resolved when built
        self._pin()

    def screenshot(self) -> np.ndarray:
        from io import BytesIO

        from PIL import Image

        r = self.call("Page.captureScreenshot", {
            "format": "png", "fromSurface": True,
            "clip": {"x": 0, "y": 0, "width": self.width, "height": self.height, "scale": 1}})
        img = np.asarray(Image.open(BytesIO(base64.b64decode(r["data"]))).convert("RGB"))
        return np.ascontiguousarray(img[:self.height, :self.width])

    def close(self):
        try:
            if getattr(self, "ws", None) is not None:
                try:
                    self.call("Browser.close")
                except Exception:
                    pass
                self.ws.close()
        finally:
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
            shutil.rmtree(self.profile, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
