"""The non-normative onvif-mj JSON Schema: validity, real output, extensibility.

This schema is an inference from the ONVIF XSD + this implementation, not an
official ONVIF artifact (the XSD remains authoritative). These tests pin three
things: it accepts what the builder emits, it stays OPEN to user extensions
(e.g. ReID), and it still rejects malformed payloads.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from onvif_m.metadata import build_frame, build_payload  # noqa: E402
from onvif_m.model import BoundingBox, ClassCandidate, DetectedObject  # noqa: E402

_SCHEMA_PATH = Path(__file__).parent.parent / "schema/onvif-mj.schema.json"
_EXAMPLE_PATH = Path(__file__).parent.parent / "schema/onvif-mj.example.json"


@pytest.fixture(scope="module")
def schema():
    return json.loads(_SCHEMA_PATH.read_text())


def test_schema_is_itself_valid(schema):
    jsonschema.validators.validator_for(schema).check_schema(schema)


def test_canonical_example_validates(schema):
    jsonschema.validate(json.loads(_EXAMPLE_PATH.read_text()), schema)


def test_builder_output_validates(schema):
    obj = DetectedObject(0, BoundingBox(-0.5, 0.5, -0.3, 0.2), [ClassCandidate("Human", 0.9)])
    payload = build_payload([build_frame(datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC), "c", [obj])])
    jsonschema.validate(payload, schema)


def test_liveness_frame_validates(schema):
    # a frame with no detections omits Object entirely
    payload = build_payload([build_frame(datetime(2026, 6, 22, 12, 0, 1, tzinfo=UTC), "c", [])])
    assert "Object" not in payload["Frame"][0]
    jsonschema.validate(payload, schema)


def test_user_extension_is_allowed(schema):
    """A user-added optional property (e.g. ReID) must NOT break validation —
    the schema is open by design; users formalize extensions in their own schema."""
    obj = DetectedObject(0, BoundingBox(-0.5, 0.5, -0.3, 0.2), [ClassCandidate("Human", 0.9)])
    payload = build_payload([build_frame(datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC), "c", [obj])])
    payload["Frame"][0]["Object"][0]["ReID"] = {"Embedding": [0.11, 0.42, -0.3]}
    jsonschema.validate(payload, schema)  # extra property accepted


@pytest.mark.parametrize("mutate", [
    lambda p: p["Frame"][0].pop("@UtcTime"),                       # missing required attr
    lambda p: p["Frame"][0]["Object"][0].pop("Appearance"),       # missing required child
    lambda p: p["Frame"][0]["Object"][0]["Appearance"]["Shape"]["BoundingBox"].pop("@left"),
    lambda p: p.__setitem__("Frame", "not-an-array"),             # wrong root type
])
def test_malformed_payloads_are_rejected(schema, mutate):
    obj = DetectedObject(0, BoundingBox(-0.5, 0.5, -0.3, 0.2), [ClassCandidate("Human", 0.9)])
    payload = build_payload([build_frame(datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC), "c", [obj])])
    mutate(payload)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)
