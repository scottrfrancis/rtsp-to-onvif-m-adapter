"""Frame capture sources.

A ``CaptureSource`` yields ``CapturedFrame`` (wall-clock UTC + HxWxC uint8 RGB
``np.ndarray``) at a configured cadence. ``RtspCaptureSource`` grabs one frame
per cycle via ffmpeg (robust and dependency-light; a failed grab just reconnects
with exponential backoff) — a low-dependency approach that avoids the
async-GStreamer tax. ``MockCaptureSource`` drives the pipeline in tests without
a stream.
"""

from __future__ import annotations

import io
import logging
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class CapturedFrame:
    timestamp: datetime          # wall-clock UTC of capture
    image: Any                   # np.ndarray, HxWxC uint8 RGB


class CaptureSource(Protocol):
    def frames(self) -> Iterator[CapturedFrame]: ...
    def close(self) -> None: ...


def _now() -> datetime:
    return datetime.now(UTC)


class MockCaptureSource:
    """Yields a fixed image ``count`` times (or once per supplied timestamp)."""

    def __init__(self, image: Any, count: int = 1, timestamps: list[datetime] | None = None):
        self._image = image
        self._count = len(timestamps) if timestamps else count
        self._timestamps = timestamps

    def frames(self) -> Iterator[CapturedFrame]:
        for i in range(self._count):
            ts = self._timestamps[i] if self._timestamps else _now()
            yield CapturedFrame(ts, self._image)

    def close(self) -> None:
        pass


def build_grab_args(url: str, transport: str = "tcp", timeout_s: int = 10) -> list[str]:
    """ffmpeg command to grab a single JPEG frame from an RTSP URL to stdout."""
    args = ["ffmpeg", "-rtsp_transport", transport]
    if timeout_s > 0:
        # ffmpeg socket I/O timeout is in microseconds; a stalled stream errors
        # out instead of hanging, so the supervisor can reconnect.
        args += ["-timeout", str(timeout_s * 1_000_000)]
    args += [
        "-i", url,
        "-frames:v", "1",
        "-f", "image2", "-vcodec", "mjpeg",
        "-hide_banner", "-loglevel", "error",
        "-",
    ]
    return args


class RtspCaptureSource:
    """Per-camera RTSP capture: one frame per cycle via ffmpeg, with reconnect."""

    def __init__(
        self,
        url: str,
        fps: float = 1.0,
        transport: str = "tcp",
        timeout_s: int = 10,
        max_backoff: float = 30.0,
    ):
        self._url = url
        self._interval = 1.0 / fps if fps > 0 else 0.0
        self._transport = transport
        self._timeout_s = timeout_s
        self._max_backoff = max_backoff
        self._stop = threading.Event()

    def _grab(self) -> bytes:
        proc = subprocess.run(
            build_grab_args(self._url, self._transport, self._timeout_s),
            capture_output=True,
        )
        if proc.returncode != 0 or not proc.stdout:
            msg = proc.stderr.decode(errors="replace").strip()[:200] or "empty frame"
            raise RuntimeError(msg)
        return proc.stdout

    @staticmethod
    def _decode(data: bytes) -> Any:
        import numpy as np
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

    def frames(self) -> Iterator[CapturedFrame]:
        backoff = 1.0
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                image = self._decode(self._grab())
            except Exception as exc:
                logger.warning("capture %s failed: %s; retry in %.0fs", self._url, exc, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self._max_backoff)
                continue
            backoff = 1.0
            yield CapturedFrame(_now(), image)
            elapsed = time.monotonic() - start
            if self._interval > elapsed:
                self._stop.wait(self._interval - elapsed)

    def close(self) -> None:
        self._stop.set()
