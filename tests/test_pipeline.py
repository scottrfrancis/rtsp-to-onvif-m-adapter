"""Pipeline (mock source): capture → detect → [post-process] → build → publish.

Drives the full pipeline with a MockCaptureSource + MockDetector + FilePublisher
(no RTSP, no model) and asserts the sidecars that land are valid ONVIF, and that
the PostProcessor hook runs between detect and build.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from onvif_m.capture import CapturedFrame, MockCaptureSource
from onvif_m.detect import MockDetector
from onvif_m.model import BoundingBox, ClassCandidate, DetectedObject
from onvif_m.onvif_xml import to_xml_string
from onvif_m.pipeline import Camera, process_frame, run_camera
from onvif_m.publish import FilePublisher, StdoutPublisher


def _detector():
    return MockDetector(objects=[
        DetectedObject(0, BoundingBox(-0.5, 0.5, -0.3, 0.2), [ClassCandidate("Human", 0.9)]),
    ])


def test_mock_pipeline_writes_conformant_sidecars(onvif_schema, json_schema, tmp_path: Path):
    cam = Camera("cam-1", profile_token="S1")
    # distinct per-frame timestamps (at 1 fps they never collide; the mock yields
    # instantly, so supply them explicitly to avoid same-millisecond filenames).
    ts = [datetime(2026, 6, 22, 12, 0, s, tzinfo=UTC) for s in (0, 1, 2)]
    src = MockCaptureSource(np.zeros((48, 64, 3), dtype=np.uint8), timestamps=ts)
    pub = FilePublisher(output_root=tmp_path)

    n = run_camera(cam, src, _detector(), pub, module="mock")
    assert n == 3

    sidecars = sorted((tmp_path / "cam-1").glob("*.meta.json"))
    assert len(sidecars) == 3
    for sc in sidecars:
        payload = json.loads(sc.read_text())
        cls = payload["Frame"][0]["Object"][0]["Appearance"]["Class"]["Type"][0]
        assert cls["#text"] == "Human"
        json_schema.validate(payload)                  # onvif-mj JSON Schema
        onvif_schema.validate(to_xml_string(payload))  # ONVIF XSD (authoritative)


def test_process_frame_returns_payload(json_schema):
    frame = CapturedFrame(datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC),
                          np.zeros((8, 8, 3), dtype=np.uint8))
    payload = process_frame(Camera("c"), frame, _detector(), StdoutPublisher())
    assert payload["Frame"][0]["@Source"] == "c"
    assert payload["Frame"][0]["Object"][0]["@ObjectId"] == 0
    json_schema.validate(payload)


def test_max_frames_bounds_the_run():
    cam = Camera("c")
    src = MockCaptureSource(np.zeros((8, 8, 3), dtype=np.uint8), count=10)
    assert run_camera(cam, src, _detector(), StdoutPublisher(), max_frames=3) == 3


def test_post_processor_runs_between_detect_and_build():
    """A PostProcessor can rewrite the object list before metadata is built."""
    seen = {}

    class Relabel:
        def process(self, objects, frame):
            seen["frame"] = frame
            # re-id: assign a stable object id, swap the class label
            return [DetectedObject(99, o.bbox, [ClassCandidate("Vehicle", 0.5)])
                    for o in objects]

    frame = CapturedFrame(datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC),
                          np.zeros((8, 8, 3), dtype=np.uint8))
    payload = process_frame(Camera("c"), frame, _detector(), StdoutPublisher(),
                            processors=[Relabel()])
    obj = payload["Frame"][0]["Object"][0]
    assert obj["@ObjectId"] == 99
    assert obj["Appearance"]["Class"]["Type"][0]["#text"] == "Vehicle"
    assert seen["frame"] is frame  # processor receives the source frame


def test_processors_chain_in_order():
    calls = []

    class Tag:
        def __init__(self, name):
            self.name = name

        def process(self, objects, frame):
            calls.append(self.name)
            return objects

    frame = CapturedFrame(datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC),
                          np.zeros((8, 8, 3), dtype=np.uint8))
    process_frame(Camera("c"), frame, _detector(), StdoutPublisher(),
                  processors=[Tag("a"), Tag("b")])
    assert calls == ["a", "b"]
