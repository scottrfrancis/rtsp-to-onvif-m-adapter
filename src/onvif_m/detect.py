"""Pluggable object detectors.

A ``Detector`` turns a frame (HxWxC uint8 RGB ``np.ndarray``) into a list of
``DetectedObject`` already in ONVIF coordinates and ONVIF object classes
(COCO ``person`` → ``Human``, vehicles/animals grouped). Backends:

- ``MockDetector``        — deterministic, for tests/wiring.
- ``TorchvisionDetector`` — default; permissive **BSD-3** torchvision detectors.
- ``Yolov8Detector``      — opt-in; Ultralytics is **AGPL-3.0**, so it is never
                            the default and is imported only on demand.

Heavy deps (torch / ultralytics) are imported lazily, so the core and the unit
suite need neither — the pixel→ONVIF mapping is exercised with injected fakes.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from . import plugins
from .model import DetectedObject, from_pixel_bbox

logger = logging.getLogger(__name__)

# Detector backends shipped in core; third-party backends register via the
# ``onvif_m.detectors`` entry-point group (see ``plugins``).
BUILTIN_DETECTORS = ["mock", "torchvision", "yolov8"]

# COCO class label → ONVIF ObjectClass. Unmapped labels are Title-cased.
COCO_TO_ONVIF: dict[str, str] = {
    "person": "Human",
    "bicycle": "Vehicle", "car": "Vehicle", "motorcycle": "Vehicle",
    "airplane": "Vehicle", "bus": "Vehicle", "train": "Vehicle",
    "truck": "Vehicle", "boat": "Vehicle",
    "bird": "Animal", "cat": "Animal", "dog": "Animal", "horse": "Animal",
    "sheep": "Animal", "cow": "Animal", "elephant": "Animal", "bear": "Animal",
    "zebra": "Animal", "giraffe": "Animal",
}


def onvif_class(label: str) -> str:
    """Map a detector class label to an ONVIF ObjectClass string."""
    return COCO_TO_ONVIF.get(label.lower(), label.title())


def resolve_device(requested: str, *, cuda: bool, mps: bool) -> str:
    """Pick a torch device: an explicit value, or ``auto`` → cuda > mps > cpu.
    Lets the same detector run on NVIDIA (CUDA), Apple Silicon (MPS), or CPU."""
    if requested != "auto":
        return requested
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: Any) -> list[DetectedObject]:  # frame: np.ndarray HxWxC
        ...

    @property
    def suppress_biometrics(self) -> bool:
        # When True, the detector loads no face/body submodels and emits no
        # biometric (HumanFace / HumanBody) fields. A declared capability, not a
        # post-hoc output filter: the safety is in not computing the data.
        ...


def _object(
    object_id: int,
    label: str,
    confidence: float,
    x1: float, y1: float, x2: float, y2: float,
    width: int, height: int,
    class_map: dict[str, str] | None,
) -> DetectedObject:
    from .model import ClassCandidate

    name = (class_map or {}).get(label.lower()) or onvif_class(label)
    return DetectedObject(
        object_id=object_id,
        bbox=from_pixel_bbox(x1, y1, x2, y2, width, height),
        classes=[ClassCandidate(name, confidence)],
    )


def torchvision_to_objects(
    output: dict[str, Any],
    width: int,
    height: int,
    categories: list[str],
    conf_floor: float = 0.0,
    class_map: dict[str, str] | None = None,
) -> list[DetectedObject]:
    """Project a torchvision detection ``output`` (parallel boxes/labels/scores,
    pixel xyxy; ``categories`` indexed by label id) into ONVIF objects."""
    objects: list[DetectedObject] = []
    oid = 0
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"], strict=False):
        conf = float(score)
        if conf < conf_floor:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        objects.append(_object(oid, str(categories[int(label)]), conf,
                               x1, y1, x2, y2, width, height, class_map))
        oid += 1
    return objects


def yolo_to_objects(
    result: Any,
    conf_floor: float = 0.0,
    class_map: dict[str, str] | None = None,
) -> list[DetectedObject]:
    """Project one ultralytics ``Results`` (per-box ``xyxy``/``conf``/``cls``;
    ``result.names`` id→label; ``result.orig_shape`` = (H, W)) into ONVIF objects."""
    height, width = result.orig_shape
    names = result.names
    objects: list[DetectedObject] = []
    oid = 0
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_floor:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        objects.append(_object(oid, str(names[int(box.cls[0])]), conf,
                               x1, y1, x2, y2, width, height, class_map))
        oid += 1
    return objects


class MockDetector:
    """Deterministic detector for tests and dry-run wiring."""

    def __init__(
        self,
        objects: list[DetectedObject] | None = None,
        suppress_biometrics: bool = True,
    ):
        self._objects = objects or []
        self._suppress_biometrics = suppress_biometrics

    def detect(self, frame: Any) -> list[DetectedObject]:
        return self._objects

    @property
    def suppress_biometrics(self) -> bool:
        return self._suppress_biometrics


class TorchvisionDetector:
    """Default detector — torchvision COCO models (BSD-3, no GPU required)."""

    def __init__(
        self,
        model_name: str = "ssdlite320_mobilenet_v3_large",
        conf: float = 0.25,
        suppress_biometrics: bool = True,
        class_map: dict[str, str] | None = None,
        device: str = "auto",
        _model: Any | None = None,
        _categories: list[str] | None = None,
    ):
        self._conf = conf
        self._suppress_biometrics = suppress_biometrics
        self._class_map = class_map
        if _model is not None:
            self._model = _model
            self._categories = _categories or []
            self._device = "cpu"
            return
        try:
            import torch
            import torchvision
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("torchvision is required: pip install '.[detect]'") from exc
        self._device = resolve_device(
            device, cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available()
        )
        weights = torchvision.models.get_model_weights(model_name).DEFAULT
        logger.info("loading torchvision %s on device=%s (conf>=%.2f)",
                    model_name, self._device, conf)
        self._model = torchvision.models.get_model(model_name, weights=weights)
        self._model.eval().to(self._device)
        self._categories = list(weights.meta["categories"])

    @property
    def device(self) -> str:
        return self._device

    def detect(self, frame: Any) -> list[DetectedObject]:
        import torch

        height, width = frame.shape[:2]
        tensor = (torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0).to(self._device)
        with torch.no_grad():
            out = self._model([tensor])[0]
        plain = {
            "boxes": out["boxes"].cpu().tolist(),
            "labels": out["labels"].cpu().tolist(),
            "scores": out["scores"].cpu().tolist(),
        }
        return torchvision_to_objects(
            plain, width, height, self._categories, self._conf, self._class_map
        )

    @property
    def suppress_biometrics(self) -> bool:
        return self._suppress_biometrics


class Yolov8Detector:
    """Opt-in detector — Ultralytics YOLOv8. AGPL-3.0; install separately."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        suppress_biometrics: bool = True,
        class_map: dict[str, str] | None = None,
        _model: Any | None = None,
    ):
        self._conf = conf
        self._suppress_biometrics = suppress_biometrics
        self._class_map = class_map
        if _model is not None:
            self._model = _model
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "ultralytics is AGPL-3.0 and not bundled; install it yourself: "
                "pip install ultralytics"
            ) from exc
        self._model = YOLO(model_path)

    def detect(self, frame: Any) -> list[DetectedObject]:
        result = self._model(frame, verbose=False)[0]
        return yolo_to_objects(result, self._conf, self._class_map)

    @property
    def suppress_biometrics(self) -> bool:
        return self._suppress_biometrics


def create_detector(
    backend: str = "torchvision",
    model: str = "ssdlite320_mobilenet_v3_large",
    conf: float = 0.25,
    suppress_biometrics: bool = True,
    class_map: dict[str, str] | None = None,
    device: str = "auto",
) -> Detector:
    """Factory. ``backend`` ∈ {``mock``, ``torchvision`` (default), ``yolov8``}.
    ``device`` ∈ {``auto``, ``cpu``, ``cuda``, ``mps``} (torchvision backend)."""
    if backend == "mock":
        return MockDetector(suppress_biometrics=suppress_biometrics)
    if backend == "torchvision":
        return TorchvisionDetector(
            model_name=model,
            conf=conf,
            suppress_biometrics=suppress_biometrics,
            class_map=class_map,
            device=device,
        )
    if backend == "yolov8":
        return Yolov8Detector(
            model_path=model,
            conf=conf,
            suppress_biometrics=suppress_biometrics,
            class_map=class_map,
        )
    factory = plugins.load_plugin(plugins.DETECTORS, backend)
    if factory is not None:
        return factory()  # type: ignore[no-any-return]
    raise ValueError(f"Unknown detector backend: {backend!r}")
