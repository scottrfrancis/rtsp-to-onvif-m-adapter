"""Detector unit tests — no torch/ultralytics (mapping uses injected fakes)."""

import pytest

from onvif_m.detect import (
    MockDetector,
    TorchvisionDetector,
    Yolov8Detector,
    create_detector,
    onvif_class,
    resolve_device,
    torchvision_to_objects,
    yolo_to_objects,
)
from onvif_m.model import BoundingBox, DetectedObject

# torchvision weights.meta["categories"]: index 0 background, 1 person, ...
COCO_CATS = ["__background__", "person", "bicycle", "car"]


class TestResolveDevice:
    def test_explicit_wins(self):
        assert resolve_device("cpu", cuda=True, mps=True) == "cpu"

    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        assert resolve_device("auto", cuda=True, mps=True) == "cuda"
        assert resolve_device("auto", cuda=False, mps=True) == "mps"
        assert resolve_device("auto", cuda=False, mps=False) == "cpu"


class TestOnvifClassMapping:
    def test_person_becomes_human(self):
        assert onvif_class("person") == "Human"

    def test_vehicles_and_animals_grouped(self):
        assert onvif_class("car") == "Vehicle"
        assert onvif_class("dog") == "Animal"

    def test_unmapped_is_title_cased(self):
        assert onvif_class("backpack") == "Backpack"


class TestTorchvisionMapping:
    def test_person_box_maps_to_onvif_object(self):
        out = {"boxes": [[64.0, 48.0, 192.0, 240.0]], "labels": [1], "scores": [0.93]}
        objs = torchvision_to_objects(out, width=640, height=480, categories=COCO_CATS)

        assert len(objs) == 1
        o = objs[0]
        assert o.object_id == 0
        assert o.classes[0].type == "Human"
        assert o.classes[0].likelihood == pytest.approx(0.93)
        # ONVIF [-1,1], y-up
        assert o.bbox.left == pytest.approx(-0.8)
        assert o.bbox.top == pytest.approx(0.8)
        assert o.bbox.right == pytest.approx(-0.4)
        assert o.bbox.bottom == pytest.approx(0.0)
        assert o.bbox.top > o.bbox.bottom

    def test_conf_floor_filters_and_reindexes(self):
        out = {
            "boxes": [[10, 10, 20, 20], [30, 30, 40, 40]],
            "labels": [1, 3],
            "scores": [0.10, 0.80],
        }
        objs = torchvision_to_objects(out, 100, 100, COCO_CATS, conf_floor=0.25)
        assert len(objs) == 1
        assert objs[0].object_id == 0          # surviving object re-indexed from 0
        assert objs[0].classes[0].type == "Vehicle"  # "car"

    def test_class_map_override(self):
        out = {"boxes": [[10, 10, 20, 20]], "labels": [1], "scores": [0.9]}
        objs = torchvision_to_objects(out, 100, 100, COCO_CATS, class_map={"person": "Caregiver"})
        assert objs[0].classes[0].type == "Caregiver"

    def test_keep_classes_person_only(self):
        # person + car + bicycle in; keep_classes={"person"} keeps only the person.
        out = {
            "boxes": [[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]],
            "labels": [1, 3, 2],  # person, car, bicycle
            "scores": [0.9, 0.9, 0.9],
        }
        objs = torchvision_to_objects(out, 100, 100, COCO_CATS, keep_classes={"person"})
        assert len(objs) == 1
        assert objs[0].object_id == 0            # re-indexed from 0
        assert objs[0].classes[0].type == "Human"  # the kept person

    def test_keep_classes_none_keeps_all(self):
        out = {"boxes": [[10, 10, 20, 20], [30, 30, 40, 40]],
               "labels": [1, 3], "scores": [0.9, 0.9]}
        objs = torchvision_to_objects(out, 100, 100, COCO_CATS, keep_classes=None)
        assert len(objs) == 2


class _FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [xyxy]
        self.conf = [conf]
        self.cls = [cls]


class _FakeResult:
    def __init__(self, orig_shape, names, boxes):
        self.orig_shape = orig_shape
        self.names = names
        self.boxes = boxes


class _FakeCompiled:
    """Stands in for an OpenVINO CompiledModel: callable → [boxes, labels, scores]."""
    def __init__(self, boxes, labels, scores):
        self._out = [boxes, labels, scores]

    def __call__(self, inputs):
        return self._out


class TestYoloMapping:
    def test_maps_person(self):
        result = _FakeResult((480, 640), {0: "person"},
                             [_FakeBox([64.0, 48.0, 192.0, 240.0], 0.93, 0.0)])
        objs = yolo_to_objects(result)
        assert objs[0].classes[0].type == "Human"
        assert objs[0].bbox.left == pytest.approx(-0.8)


class TestMockAndFactory:
    def test_mock_returns_objects_and_suppress_biometrics(self):
        objs = [DetectedObject(0, BoundingBox(-0.5, 0.5, -0.3, 0.2))]
        det = MockDetector(objects=objs, suppress_biometrics=True)
        assert det.detect(object()) == objs
        assert det.suppress_biometrics is True

    def test_factory_mock(self):
        assert isinstance(create_detector(backend="mock"), MockDetector)

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown detector backend"):
            create_detector(backend="frcnn-typo")

    def test_yolov8_detector_with_injected_model(self):
        result = _FakeResult((480, 640), {0: "person"},
                             [_FakeBox([64.0, 48.0, 192.0, 240.0], 0.9, 0.0)])

        class FakeModel:
            def __call__(self, frame, **kw):
                return [result]

        det = Yolov8Detector(_model=FakeModel(), suppress_biometrics=False)
        out = det.detect(object())
        assert out[0].classes[0].type == "Human"
        assert det.suppress_biometrics is False


class TestMinMaxPassthrough:
    """v0.2.0: TorchvisionDetector accepts min_size/max_size (the resolution knob)."""

    def test_torchvision_stores_min_max(self):
        det = TorchvisionDetector(_model=object(), _categories=COCO_CATS,
                                  min_size=800, max_size=1333)
        assert det.min_size == 800
        assert det.max_size == 1333

    def test_torchvision_min_max_default_none(self):
        det = TorchvisionDetector(_model=object(), _categories=COCO_CATS)
        assert det.min_size is None
        assert det.max_size is None


class TestOpenVINODetector:
    """v0.2.0: OpenVINO-FP32 backend. Unit tests use an injected CompiledModel so
    they need neither torch nor openvino — only numpy for the frame tensor prep."""

    def test_maps_via_injected_compiled(self):
        np = pytest.importorskip("numpy")
        from onvif_m.detect import OpenVINODetector

        fake = _FakeCompiled(boxes=[[64.0, 48.0, 192.0, 240.0]], labels=[1], scores=[0.93])
        det = OpenVINODetector(_compiled=fake, _categories=COCO_CATS, suppress_biometrics=True)
        objs = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(objs) == 1
        assert objs[0].classes[0].type == "Human"
        assert objs[0].classes[0].likelihood == pytest.approx(0.93)
        assert det.suppress_biometrics is True

    def test_conf_floor_filters(self):
        np = pytest.importorskip("numpy")
        from onvif_m.detect import OpenVINODetector

        fake = _FakeCompiled(boxes=[[10, 10, 20, 20], [30, 30, 40, 40]],
                             labels=[1, 3], scores=[0.10, 0.80])
        det = OpenVINODetector(_compiled=fake, _categories=COCO_CATS, conf=0.25)
        objs = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(objs) == 1
        assert objs[0].classes[0].type == "Vehicle"

    def test_stores_min_max(self):
        from onvif_m.detect import OpenVINODetector

        det = OpenVINODetector(_compiled=_FakeCompiled([], [], []), _categories=COCO_CATS,
                               min_size=800, max_size=1333)
        assert det.min_size == 800
        assert det.max_size == 1333

    def test_num_threads_default_zero(self):
        from onvif_m.detect import OpenVINODetector

        det = OpenVINODetector(_compiled=_FakeCompiled([], [], []), _categories=COCO_CATS)
        assert det.num_threads == 0

    def test_stores_num_threads(self):
        # The multi-camera thread cap: N detectors pack the box's cores.
        from onvif_m.detect import OpenVINODetector

        det = OpenVINODetector(_compiled=_FakeCompiled([], [], []), _categories=COCO_CATS,
                               num_threads=1)
        assert det.num_threads == 1


class _FakeOrtSession:
    """Stand-in ORT session: ignores the input, returns a crafted YOLOX head."""
    def __init__(self, out, size=64):
        self._out = out
        self._size = size
    def get_inputs(self):
        size = self._size
        class _I:
            name = "images"
            shape = [1, 3, size, size]
        return [_I()]
    def run(self, _names, _feed):
        import numpy as np
        return [np.asarray(self._out)[None]]  # [1, N, 85]


def _yolox_head(size=64, cls_index=0, obj=0.9, cls=0.9):
    """One high-scoring detection in the first (stride-8, grid 0,0) anchor."""
    np = pytest.importorskip("numpy")
    ng = (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2
    out = np.zeros((ng, 85), dtype=np.float32)
    out[:, 4] = 0.001                               # obj low everywhere
    out[:, 5:] = 0.001                              # cls low everywhere
    out[0, :2] = 0.5                                # center in first cell
    out[0, 2:4] = 0.0                               # wh = stride
    out[0, 4] = obj
    out[0, 5 + cls_index] = cls
    return out


class TestYoloxDecode:
    def test_decode_person_maps_to_human(self):
        pytest.importorskip("numpy")
        from onvif_m.detect import _COCO80, yolox_decode
        objs = yolox_decode(_yolox_head(cls_index=0), 64, 1.0, 64, 64, 0.25,
                            _COCO80, keep_classes={"person"})
        assert len(objs) == 1
        assert objs[0].classes[0].type == "Human"
        assert objs[0].classes[0].likelihood == pytest.approx(0.81, abs=1e-3)

    def test_keep_classes_drops_non_person(self):
        pytest.importorskip("numpy")
        from onvif_m.detect import _COCO80, yolox_decode
        # bicycle (index 1) high, but person-only allowlist drops it
        objs = yolox_decode(_yolox_head(cls_index=1), 64, 1.0, 64, 64, 0.25,
                            _COCO80, keep_classes={"person"})
        assert len(objs) == 0

    def test_conf_floor_filters(self):
        pytest.importorskip("numpy")
        from onvif_m.detect import _COCO80, yolox_decode
        objs = yolox_decode(_yolox_head(obj=0.2, cls=0.2), 64, 1.0, 64, 64, 0.25, _COCO80)
        assert len(objs) == 0  # score 0.04 < 0.25


class TestYoloxDetector:
    def test_detect_via_injected_session(self):
        np = pytest.importorskip("numpy")
        from onvif_m.detect import YoloxDetector
        det = YoloxDetector(_session=_FakeOrtSession(_yolox_head(), size=64),
                            keep_classes={"person"})
        assert det.input_size == 64
        objs = det.detect(np.zeros((64, 64, 3), dtype=np.uint8))
        assert len(objs) == 1 and objs[0].classes[0].type == "Human"
        assert det.suppress_biometrics is True
