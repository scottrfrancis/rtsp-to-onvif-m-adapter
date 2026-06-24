"""Detector accuracy regression — real detector vs published COCO ground truth.

Runs the default torchvision detector on a single-person COCO reference image and
asserts it localizes the person with IoU ≥ 0.5 against the COCO GT box. Guards
against regressions in the model wiring, preprocessing, and the pixel→ONVIF
coordinate conversion.

Skipped unless torch/torchvision/PIL are installed AND the fixture exists
(run tests/fixtures/fetch_samples.sh).
"""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

from onvif_m.model import BoundingBox, from_pixel_bbox  # noqa: E402

_IMG = Path(__file__).parent / "fixtures" / "coco" / "000000000785.jpg"

# COCO val2017, image_id 785, 640×425. Ground-truth `person` bbox (pixel xyxy)
# from instances_val2017. The image is a single person (skier).
_GT_XYXY = (280.79, 44.73, 499.49, 391.41)
_GT_W, _GT_H = 640, 425

pytestmark = pytest.mark.skipif(
    not _IMG.exists(),
    reason="fixture missing — run tests/fixtures/fetch_samples.sh",
)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    # ONVIF coords are y-up (top > bottom); use min/max so orientation is moot.
    inter_w = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    inter_h = max(0.0, min(a.top, b.top) - max(a.bottom, b.bottom))
    inter = inter_w * inter_h
    area_a = (a.right - a.left) * (a.top - a.bottom)
    area_b = (b.right - b.left) * (b.top - b.bottom)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def test_detects_person_with_high_iou_vs_coco_gt():
    from onvif_m.detect import create_detector

    img = np.asarray(Image.open(_IMG).convert("RGB"))
    detector = create_detector(backend="torchvision", conf=0.3)

    humans = [o for o in detector.detect(img) if o.classes[0].type == "Human"]
    assert humans, "expected at least one Human detection"

    gt = from_pixel_bbox(*_GT_XYXY, _GT_W, _GT_H)
    best = max(_iou(o.bbox, gt) for o in humans)
    assert best >= 0.5, f"best Human IoU {best:.3f} < 0.5 vs COCO GT"
