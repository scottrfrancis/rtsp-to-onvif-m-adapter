"""Single-stream health registry + /healthz endpoint."""

import json
import time
import urllib.request

from onvif_m.health import HealthRegistry, serve_health


class TestRegistry:
    def test_healthy_within_window(self):
        h = HealthRegistry(stale_after=30)
        h.record_frame(now=100.0)
        snap = h.snapshot(now=110.0)  # 10s later
        assert snap == {"healthy": True, "frames": 1, "seconds_since_frame": 10.0}

    def test_stale_is_unhealthy(self):
        h = HealthRegistry(stale_after=5)
        h.record_frame(now=100.0)
        snap = h.snapshot(now=120.0)  # 20s > 5s
        assert snap["healthy"] is False

    def test_no_frame_yet_is_not_healthy(self):
        snap = HealthRegistry().snapshot()
        assert snap["healthy"] is False
        assert snap["frames"] == 0
        assert snap["seconds_since_frame"] is None

    def test_counts_frames(self):
        h = HealthRegistry()
        for _ in range(3):
            h.record_frame(now=1.0)
        assert h.snapshot(now=1.0)["frames"] == 3


def test_healthz_http_endpoint():
    h = HealthRegistry(stale_after=30)
    h.record_frame(now=time.monotonic())
    server = serve_health(h, 0)  # port 0 → ephemeral
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=3) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            assert body["healthy"] is True
            assert body["frames"] == 1
    finally:
        server.shutdown()
