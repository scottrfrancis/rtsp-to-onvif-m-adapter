"""Unit tests for the data model and the pixel→ONVIF coordinate conversion."""

import pytest

from onvif_m.model import BoundingBox, ClassCandidate, DetectedObject, from_pixel_bbox


class TestFromPixelBbox:
    def test_full_frame_maps_to_corners(self):
        # whole 640x480 frame -> ONVIF [-1,1], y-up
        bb = from_pixel_bbox(0, 0, 640, 480, 640, 480)
        assert (bb.left, bb.top, bb.right, bb.bottom) == (-1.0, 1.0, 1.0, -1.0)

    def test_center_pixel_maps_to_origin(self):
        bb = from_pixel_bbox(320, 240, 320, 240, 640, 480)
        assert (bb.left, bb.top) == (0.0, 0.0)

    def test_y_is_up(self):
        # a box in the upper-left pixel quadrant: smaller pixel-y -> larger ONVIF y
        bb = from_pixel_bbox(0, 0, 320, 240, 640, 480)
        assert bb.top > bb.bottom
        assert bb.top == 1.0 and bb.bottom == 0.0

    def test_does_not_clamp(self):
        # conversion is linear; out-of-frame pixels are allowed to exceed [-1,1]
        # (callers clamp if they need to).
        bb = from_pixel_bbox(-64, -48, 704, 528, 640, 480)
        assert bb.left == pytest.approx(-1.2)
        assert bb.right == pytest.approx(1.2)


class TestDataclasses:
    def test_bounding_box_is_frozen_and_value_equal(self):
        a = BoundingBox(-0.5, 0.5, 0.5, -0.5)
        b = BoundingBox(-0.5, 0.5, 0.5, -0.5)
        assert a == b
        with pytest.raises(AttributeError):
            a.left = 0.0  # frozen

    def test_class_candidate_fields(self):
        c = ClassCandidate("Human", 0.91)
        assert (c.type, c.likelihood) == ("Human", 0.91)

    def test_detected_object_defaults(self):
        o = DetectedObject(object_id=3, bbox=BoundingBox(0, 0, 0.1, -0.1))
        assert o.classes == []
        assert o.center_of_gravity is None
