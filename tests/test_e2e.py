"""End-to-end: detections → build → publish → the on-disk artifact is conformant.

Exercises the full produce-and-publish path through the default FilePublisher and
asserts the sidecar that lands on disk round-trips to schema-valid ONVIF XML
(uses the official-schema fixture from conftest).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from onvif_m.metadata import build_frame, build_payload
from onvif_m.model import ClassCandidate, DetectedObject, from_pixel_bbox
from onvif_m.publish import FilePublisher, FrameRef


def _detections_for_a_frame():
    # what a detector would hand us: pixel boxes on a 640x480 frame
    return [
        DetectedObject(0, from_pixel_bbox(64, 48, 192, 240, 640, 480),
                       [ClassCandidate("Human", 0.94)]),
        DetectedObject(1, from_pixel_bbox(300, 100, 360, 280, 640, 480),
                       [ClassCandidate("Human", 0.81), ClassCandidate("Animal", 0.05)]),
    ]


def test_capture_to_sidecar_is_conformant(onvif_schema, json_schema, tmp_path: Path):
    ts = datetime(2026, 6, 22, 14, 32, 11, 450000, tzinfo=UTC)
    frame_jpg = tmp_path / "cam-7" / "2026-06-22" / "T143211.450.jpg"
    frame_jpg.parent.mkdir(parents=True)
    frame_jpg.write_bytes(b"jpeg-bytes")

    # produce
    payload = build_payload([build_frame(ts, "cam-7", _detections_for_a_frame())])

    # publish (default sidecar publisher, next to the frame)
    pub = FilePublisher()
    pub.publish(payload, FrameRef("cam-7", ts, frame_path=frame_jpg))
    pub.close()

    # the artifact that landed on disk
    sidecar = frame_jpg.with_suffix(".meta.json")
    assert sidecar.exists()
    on_disk = json.loads(sidecar.read_text())
    assert on_disk == payload                      # survived the JSON round-trip
    assert len(on_disk["Frame"][0]["Object"]) == 2

    # conformant: onvif-mj JSON Schema + ONVIF XSD (authoritative)
    json_schema.validate(on_disk)
    from onvif_m.onvif_xml import to_xml_string
    onvif_schema.validate(to_xml_string(on_disk))


def test_liveness_only_sidecar_is_conformant(onvif_schema, json_schema, tmp_path: Path):
    ts = datetime(2026, 6, 22, 14, 32, 12, 0, tzinfo=UTC)
    payload = build_payload([build_frame(ts, "cam-7", [])])  # nobody in frame

    FilePublisher(output_root=tmp_path).publish(payload, FrameRef("cam-7", ts))

    sidecar = next((tmp_path / "cam-7").glob("*.meta.json"))
    on_disk = json.loads(sidecar.read_text())
    assert "Object" not in on_disk["Frame"][0]     # liveness: bare frame

    json_schema.validate(on_disk)
    from onvif_m.onvif_xml import to_xml_string
    onvif_schema.validate(to_xml_string(on_disk))
