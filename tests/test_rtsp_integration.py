"""RTSP capture integration: read from a real RTSP server through the pipeline.

Requires a server at rtsp://localhost:8554/test — start one with
    bash tests/rtsp/run-rtsp.sh
Self-skips if none is reachable (see tests/rtsp/README.md).
"""

import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from onvif_m.capture import RtspCaptureSource  # noqa: E402
from onvif_m.detect import MockDetector  # noqa: E402
from onvif_m.onvif_xml import to_xml_string  # noqa: E402
from onvif_m.pipeline import Camera, run_camera  # noqa: E402
from onvif_m.publish import FilePublisher  # noqa: E402

HOST, PORT = "localhost", 8554
URL = f"rtsp://{HOST}:{PORT}/test"


def _server_up() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(),
    reason="no RTSP server on localhost:8554 (tests/rtsp/run-rtsp.sh)",
)


def test_captures_a_real_frame():
    src = RtspCaptureSource(URL, fps=1.0)
    try:
        frame = next(src.frames())
    finally:
        src.close()
    assert frame.image.ndim == 3 and frame.image.shape[2] == 3
    assert frame.image.shape[0] > 0 and frame.image.shape[1] > 0


def test_pipeline_from_rtsp_writes_conformant_sidecars(onvif_schema, json_schema, tmp_path: Path):
    # MockDetector keeps it model-free; the point is real capture → conformant output.
    src = RtspCaptureSource(URL, fps=2.0)
    pub = FilePublisher(output_root=tmp_path)
    try:
        n = run_camera(Camera("rtsp-cam", profile_token="S1"), src, MockDetector(objects=[]),
                       pub, module="mock", max_frames=2)
    finally:
        src.close()

    assert n == 2
    sidecars = list((tmp_path / "rtsp-cam").glob("*.meta.json"))
    assert sidecars
    payload = json.loads(sidecars[0].read_text())
    json_schema.validate(payload)
    onvif_schema.validate(to_xml_string(payload))
