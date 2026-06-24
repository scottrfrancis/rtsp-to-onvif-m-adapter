"""The builder must emit the canonical ONVIF onvif-mj JSON binding.

Shapes asserted here mirror the worked example in the ONVIF Analytics Service
Specification §5.4.3 exactly (key names, @-attributes, #text, arrays).
"""

from datetime import UTC, datetime

from onvif_m.metadata import build_frame, build_object, build_payload
from onvif_m.model import BoundingBox, ClassCandidate, DetectedObject, from_pixel_bbox


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


class TestBuildObject:
    def test_matches_onvif_object_shape(self):
        obj = DetectedObject(
            object_id=15,
            bbox=BoundingBox(left=-0.9375, top=-0.6667, right=-0.6875, bottom=-0.875),
            classes=[ClassCandidate("Human", 0.8)],
            center_of_gravity=(-0.8125, -0.7917),
        )
        out = build_object(obj)

        assert out["@ObjectId"] == 15
        assert out["Appearance"]["Shape"]["BoundingBox"] == {
            "@left": -0.9375, "@top": -0.6667, "@right": -0.6875, "@bottom": -0.875,
        }
        assert out["Appearance"]["Shape"]["CenterOfGravity"] == {"@x": -0.8125, "@y": -0.7917}
        # Class.Type is an ARRAY of {@Likelihood, #text}; value is "Human", not "Person"
        assert out["Appearance"]["Class"]["Type"] == [{"@Likelihood": 0.8, "#text": "Human"}]

    def test_object_without_class_omits_class_but_keeps_required_cog(self):
        obj = DetectedObject(object_id=1, bbox=BoundingBox(0.0, 0.2, 0.4, -0.2))
        out = build_object(obj)
        assert "Class" not in out["Appearance"]
        # CenterOfGravity is mandatory in ShapeDescriptor — defaulted to the
        # box midpoint when not supplied.
        assert out["Appearance"]["Shape"]["CenterOfGravity"] == {"@x": 0.2, "@y": 0.0}

    def test_multiple_class_candidates_become_type_array(self):
        obj = DetectedObject(
            object_id=2,
            bbox=BoundingBox(-0.1, 0.1, 0.1, -0.1),
            classes=[ClassCandidate("Human", 0.7), ClassCandidate("Animal", 0.2)],
        )
        types = build_object(obj)["Appearance"]["Class"]["Type"]
        assert types == [
            {"@Likelihood": 0.7, "#text": "Human"},
            {"@Likelihood": 0.2, "#text": "Animal"},
        ]


class TestUtcFormat:
    def test_microseconds_truncated_to_milliseconds(self):
        frame = build_frame(_utc("2021-10-05T15:13:27.321987"), "c", [])
        assert frame["@UtcTime"] == "2021-10-05T15:13:27.321Z"

    def test_zero_subsecond_keeps_three_digits(self):
        frame = build_frame(_utc("2021-10-05T15:13:27"), "c", [])
        assert frame["@UtcTime"] == "2021-10-05T15:13:27.000Z"


class TestBuildFrame:
    def test_multiple_objects_keep_distinct_ids(self):
        objs = [
            DetectedObject(0, BoundingBox(-0.5, 0.5, -0.3, 0.2)),
            DetectedObject(1, BoundingBox(0.1, 0.4, 0.3, -0.1)),
        ]
        frame = build_frame(_utc("2021-10-05T15:13:27.000"), "c", objs)
        assert [o["@ObjectId"] for o in frame["Object"]] == [0, 1]

    def test_frame_attributes_and_object_array(self):
        obj = DetectedObject(object_id=15, bbox=BoundingBox(-0.9, -0.6, -0.6, -0.8),
                             classes=[ClassCandidate("Human", 0.8)])
        frame = build_frame(_utc("2021-10-05T15:13:27.321"), "MyClassifier", [obj])

        assert frame["@UtcTime"] == "2021-10-05T15:13:27.321Z"
        assert frame["@Source"] == "MyClassifier"
        assert isinstance(frame["Object"], list) and len(frame["Object"]) == 1

    def test_empty_frame_omits_object_key(self):
        # liveness: the bare frame is emitted, with no Object children
        frame = build_frame(_utc("2021-10-05T15:13:28.000"), "MyClassifier", [])
        assert frame["@UtcTime"] == "2021-10-05T15:13:28.000Z"
        assert "Object" not in frame


class TestBuildPayload:
    def test_payload_root_is_frame_array(self, json_schema):
        f0 = build_frame(_utc("2021-10-05T15:13:27.000"), "c", [])
        f1 = build_frame(_utc("2021-10-05T15:13:28.000"), "c", [])
        payload = build_payload([f0, f1])
        # Root is exactly {"Frame": [...]} — no MetadataStream/VideoAnalytics in the body
        assert list(payload.keys()) == ["Frame"]
        assert payload["Frame"] == [f0, f1]
        json_schema.validate(payload)

    def test_payload_with_objects_is_schema_valid(self, json_schema):
        obj = DetectedObject(15, BoundingBox(-0.9, -0.6, -0.6, -0.8),
                             classes=[ClassCandidate("Human", 0.8)])
        payload = build_payload([build_frame(_utc("2021-10-05T15:13:27.321"), "c", [obj])])
        json_schema.validate(payload)


class TestPixelToOnvifCoords:
    def test_top_left_pixel_maps_to_y_up_normalized(self):
        # 320x240 image; box pixels (20,80)-(100,160), top-left origin
        bb = from_pixel_bbox(20, 80, 100, 160, width=320, height=240)
        assert bb.left == 20 / 320 * 2 - 1      # -0.875
        assert bb.right == 100 / 320 * 2 - 1    # -0.375
        # y flips: pixel-top (smaller y) -> larger ONVIF value (y-up)
        assert bb.top == 1 - 80 / 240 * 2       # 0.3333...
        assert bb.bottom == 1 - 160 / 240 * 2   # -0.3333...
        assert bb.top > bb.bottom               # ONVIF invariant
