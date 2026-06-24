"""ONVIF Profile-M conformance: the onvif-mj payload round-trips to
tt:MetadataStream XML that validates against the OFFICIAL ONVIF schema.

This is the authoritative compliance evidence for integrators — not "matches our
own example," but "validates against ONVIF's published metadatastream.xsd."
Skipped if xmlschema or the schema closure is unavailable (see conftest).
"""

from datetime import UTC, datetime

from onvif_m.metadata import build_frame, build_payload
from onvif_m.model import BoundingBox, ClassCandidate, DetectedObject, from_pixel_bbox
from onvif_m.onvif_xml import to_xml_string


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def test_populated_frame_is_schema_valid(onvif_schema):
    obj = DetectedObject(
        object_id=15,
        bbox=BoundingBox(-0.9375, -0.6667, -0.6875, -0.875),
        classes=[ClassCandidate("Human", 0.8)],
        center_of_gravity=(-0.8125, -0.7917),
    )
    payload = build_payload([build_frame(_utc("2021-10-05T15:13:27.321"), "MyClassifier", [obj])])
    onvif_schema.validate(to_xml_string(payload))  # raises if invalid


def test_object_without_explicit_cog_is_valid(onvif_schema):
    # CoG defaulted from the box midpoint — must still satisfy the mandatory field.
    obj = DetectedObject(object_id=1, bbox=from_pixel_bbox(64, 48, 192, 240, 640, 480),
                         classes=[ClassCandidate("Vehicle", 0.6)])
    payload = build_payload([build_frame(_utc("2021-10-05T15:13:28.000"), "c", [obj])])
    onvif_schema.validate(to_xml_string(payload))


def test_empty_liveness_frame_is_schema_valid(onvif_schema):
    payload = build_payload([build_frame(_utc("2021-10-05T15:13:29.000"), "c", [])])
    onvif_schema.validate(to_xml_string(payload))


def test_multi_frame_multi_object_is_valid(onvif_schema):
    frames = [
        build_frame(_utc("2021-10-05T15:13:30.000"), "c", [
            DetectedObject(0, BoundingBox(-0.5, 0.5, -0.3, 0.2), [ClassCandidate("Human", 0.91)]),
            DetectedObject(1, BoundingBox(0.1, 0.4, 0.3, -0.1), [ClassCandidate("Human", 0.77)]),
        ]),
        build_frame(_utc("2021-10-05T15:13:31.000"), "c", []),
    ]
    onvif_schema.validate(to_xml_string(build_payload(frames)))


def test_shipped_example_is_schema_valid(onvif_schema):
    # The committed canonical example must itself be conformant.
    import json
    from pathlib import Path

    path = Path(__file__).parent.parent / "schema/onvif-mj.example.json"
    example = json.loads(path.read_text())
    onvif_schema.validate(to_xml_string(example))


def test_fuzz_random_payloads_all_validate(onvif_schema):
    # Property check: arbitrary well-formed detections always round-trip to
    # schema-valid ONVIF XML. Seeded for deterministic CI.
    import random

    rng = random.Random(20260622)
    for _ in range(60):
        frames = []
        for f in range(rng.randint(1, 4)):
            objs = []
            for oid in range(rng.randint(0, 5)):  # 0 => liveness frame
                box = sorted(round(rng.uniform(-1, 1), 4) for _ in range(4))
                objs.append(DetectedObject(
                    object_id=oid,
                    bbox=BoundingBox(box[0], box[3], box[1], box[2]),
                    classes=[ClassCandidate(rng.choice(["Human", "Vehicle", "Animal"]),
                                            round(rng.uniform(0, 1), 3))
                             for _ in range(rng.randint(1, 3))],
                ))
            frames.append(build_frame(_utc(f"2021-10-05T15:13:{f:02d}.000"), "fuzz", objs))
        onvif_schema.validate(to_xml_string(build_payload(frames)))
