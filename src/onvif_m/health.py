"""Single-stream health + an optional ``/healthz`` endpoint.

The pipeline records each processed frame into a ``HealthRegistry``; the stream
is "healthy" while it has produced a frame within ``stale_after`` seconds (a
stalled capture surfaces as staleness, since it stops yielding). ``serve_health``
exposes the snapshot over HTTP for liveness probes.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


@dataclass
class HealthRegistry:
    stale_after: float = 30.0
    frames: int = 0
    last_frame_monotonic: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_frame(self, now: float | None = None) -> None:
        ts = time.monotonic() if now is None else now
        with self._lock:
            self.frames += 1
            self.last_frame_monotonic = ts

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        ts = time.monotonic() if now is None else now
        with self._lock:
            if self.last_frame_monotonic is None:
                return {"healthy": False, "frames": self.frames, "seconds_since_frame": None}
            age = round(ts - self.last_frame_monotonic, 1)
            return {
                "healthy": age <= self.stale_after,
                "frames": self.frames,
                "seconds_since_frame": age,
            }


def serve_health(registry: HealthRegistry, port: int) -> HTTPServer:
    """Start a background HTTP server: GET /healthz → snapshot (200 healthy / 503)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") not in ("/healthz", ""):
                self.send_response(404)
                self.end_headers()
                return
            snap = registry.snapshot()
            body = json.dumps(snap).encode()
            self.send_response(200 if snap["healthy"] else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:  # silence access logs
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, name="health", daemon=True).start()
    return server
