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
import threading
from typing import Any, Protocol, runtime_checkable

from . import plugins
from .model import DetectedObject, from_pixel_bbox

logger = logging.getLogger(__name__)

# torch.jit.trace / ov.convert_model are NOT thread-safe: two detectors compiling
# their IR at the same instant corrupt each other's trace (OV's two-pass compare
# then throws "dtype float64 != float32"). Serialize compilation process-wide so a
# multi-camera host can start N detectors concurrently. Inference on an already-
# compiled model is thread-safe and stays parallel.
_COMPILE_LOCK = threading.Lock()

# Detector backends shipped in core; third-party backends register via the
# ``onvif_m.detectors`` entry-point group (see ``plugins``).
BUILTIN_DETECTORS = ["mock", "torchvision", "yolov8", "openvino"]

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


def _size_kwargs(min_size: int | None, max_size: int | None) -> dict[str, int]:
    """torchvision ``get_model`` kwargs for the FPN/R-CNN input-resize knob.
    Only emitted when set, so default behavior and fixed-size models are untouched."""
    kw: dict[str, int] = {}
    if min_size is not None:
        kw["min_size"] = min_size
    if max_size is not None:
        kw["max_size"] = max_size
    return kw


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
    keep_classes: set[str] | None = None,
) -> list[DetectedObject]:
    """Project a torchvision detection ``output`` (parallel boxes/labels/scores,
    pixel xyxy; ``categories`` indexed by label id) into ONVIF objects.
    ``keep_classes`` (lowercased raw labels, e.g. ``{"person"}``) drops everything
    else — an allowlist applied before mapping to ONVIF classes."""
    objects: list[DetectedObject] = []
    oid = 0
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"], strict=False):
        conf = float(score)
        if conf < conf_floor:
            continue
        raw = str(categories[int(label)])
        if keep_classes is not None and raw.lower() not in keep_classes:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        objects.append(_object(oid, raw, conf,
                               x1, y1, x2, y2, width, height, class_map))
        oid += 1
    return objects


def yolo_to_objects(
    result: Any,
    conf_floor: float = 0.0,
    class_map: dict[str, str] | None = None,
    keep_classes: set[str] | None = None,
) -> list[DetectedObject]:
    """Project one ultralytics ``Results`` (per-box ``xyxy``/``conf``/``cls``;
    ``result.names`` id→label; ``result.orig_shape`` = (H, W)) into ONVIF objects.
    ``keep_classes`` (lowercased raw labels) is an allowlist; others are dropped."""
    height, width = result.orig_shape
    names = result.names
    objects: list[DetectedObject] = []
    oid = 0
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_floor:
            continue
        raw = str(names[int(box.cls[0])])
        if keep_classes is not None and raw.lower() not in keep_classes:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        objects.append(_object(oid, raw, conf,
                               x1, y1, x2, y2, width, height, class_map))
        oid += 1
    return objects


class MockDetector:
    """Deterministic detector for tests and dry-run wiring."""

    def __init__(
        self,
        objects: list[DetectedObject] | None = None,
        suppress_biometrics: bool = True,
        keep_classes: set[str] | None = None,
    ):
        self._objects = objects or []
        self._suppress_biometrics = suppress_biometrics
        self._keep_classes = keep_classes  # accepted for API parity (unused by mock)

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
        min_size: int | None = None,
        max_size: int | None = None,
        keep_classes: set[str] | None = None,
        _model: Any | None = None,
        _categories: list[str] | None = None,
    ):
        self._conf = conf
        self._suppress_biometrics = suppress_biometrics
        self._class_map = class_map
        self._keep_classes = keep_classes
        # Optional resolution knob for FPN/R-CNN-style detectors: overrides the
        # model's internal GeneralizedRCNNTransform min/max (the real cost lever).
        # Left as None for fixed-size models (e.g. ssd/ssdlite).
        self._min_size = min_size
        self._max_size = max_size
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
        logger.info("loading torchvision %s on device=%s (conf>=%.2f, min/max=%s/%s)",
                    model_name, self._device, conf, min_size, max_size)
        self._model = torchvision.models.get_model(
            model_name, weights=weights, **_size_kwargs(min_size, max_size)
        )
        self._model.eval().to(self._device)
        self._categories = list(weights.meta["categories"])

    @property
    def device(self) -> str:
        return self._device

    @property
    def min_size(self) -> int | None:
        return self._min_size

    @property
    def max_size(self) -> int | None:
        return self._max_size

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
            plain, width, height, self._categories, self._conf, self._class_map,
            self._keep_classes,
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
        keep_classes: set[str] | None = None,
        _model: Any | None = None,
    ):
        self._conf = conf
        self._suppress_biometrics = suppress_biometrics
        self._class_map = class_map
        self._keep_classes = keep_classes
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
        return yolo_to_objects(result, self._conf, self._class_map, self._keep_classes)

    @property
    def suppress_biometrics(self) -> bool:
        return self._suppress_biometrics


# COCO 80 contiguous class names (YOLOX / ultralytics ordering; person == index 0).
_COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _nms(boxes: Any, scores: Any, iou_thr: float = 0.5) -> list[int]:
    """Greedy class-agnostic NMS on xyxy numpy boxes; returns kept indices."""
    import numpy as np
    idx = scores.argsort()[::-1]
    keep: list[int] = []
    while len(idx):
        i = int(idx[0])
        keep.append(i)
        if len(idx) == 1:
            break
        rest = idx[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ar = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        idx = rest[inter / (ai + ar - inter + 1e-9) < iou_thr]
    return keep


def yolox_decode(
    output: Any, input_size: int, ratio: float, width: int, height: int,
    conf_floor: float, categories: list[str],
    class_map: dict[str, str] | None = None, keep_classes: set[str] | None = None,
    iou_thr: float = 0.5,
) -> list[DetectedObject]:
    """Decode a raw YOLOX head ``[N, 5+num_classes]`` (grid strides 8/16/32, letterboxed
    to ``input_size`` with scale ``ratio``, obj+cls already sigmoid'd) into ONVIF objects
    in the original ``width``×``height`` pixel space. ``keep_classes`` allowlists labels."""
    import numpy as np
    out = np.asarray(output, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    grids = []
    strides = []
    for s in (8, 16, 32):
        gg = input_size // s
        xv, yv = np.meshgrid(np.arange(gg), np.arange(gg))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((gg * gg, 1), s))
    g = np.concatenate(grids, 0)
    st = np.concatenate(strides, 0)
    xy = (out[:, :2] + g) * st
    wh = np.exp(out[:, 2:4]) * st
    cls = out[:, 5:]
    cls_id = cls.argmax(1)
    scores = out[:, 4] * cls[np.arange(len(cls)), cls_id]
    m = scores >= conf_floor
    xy, wh, scores, cls_id = xy[m], wh[m], scores[m], cls_id[m]
    if not len(xy):
        return []
    xyxy = np.stack([xy[:, 0] - wh[:, 0] / 2, xy[:, 1] - wh[:, 1] / 2,
                     xy[:, 0] + wh[:, 0] / 2, xy[:, 1] + wh[:, 1] / 2], 1) / ratio
    objects: list[DetectedObject] = []
    oid = 0
    for k in _nms(xyxy, scores, iou_thr):
        cid = int(cls_id[k])
        raw = categories[cid] if cid < len(categories) else str(cid)
        if keep_classes is not None and raw.lower() not in keep_classes:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[k])
        objects.append(
            _object(oid, raw, float(scores[k]), x1, y1, x2, y2, width, height, class_map))
        oid += 1
    return objects


class YoloxDetector:
    """Opt-in YOLOX backend (Apache-2.0), single-stage, via ONNX Runtime (CPU).

    Pure ONNX Runtime + numpy (no torch/torchvision). Single-stage → immune to the
    two-stage ``roi_heads`` reshape crash that rules out traced Faster R-CNN at reduced
    resolution. ``model`` is a path to a YOLOX ``.onnx`` with a fixed square input
    (416 for nano/tiny, 640 for s/m/l/x). ``num_threads`` (>0) caps ORT intra-op threads
    so multiple detectors pack onto a CPU box. Needs ``.[yolox]``."""

    def __init__(
        self,
        model_path: str = "yolox_s.onnx",
        conf: float = 0.25,
        suppress_biometrics: bool = True,
        class_map: dict[str, str] | None = None,
        keep_classes: set[str] | None = None,
        categories: list[str] | None = None,
        num_threads: int = 0,
        iou: float = 0.5,
        _session: Any | None = None,
    ):
        self._conf = conf
        self._suppress_biometrics = suppress_biometrics
        self._class_map = class_map
        self._keep_classes = keep_classes
        self._categories = categories or _COCO80
        self._iou = iou
        if _session is not None:
            self._session = _session
        else:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "YOLOX backend needs onnxruntime: pip install '.[yolox]'"
                ) from exc
            so = ort.SessionOptions()
            if num_threads and num_threads > 0:
                so.intra_op_num_threads = num_threads
            self._session = ort.InferenceSession(
                model_path, so, providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._size = inp.shape[2] if isinstance(inp.shape[2], int) else 640

    def _preprocess(self, frame: Any) -> tuple[Any, float]:
        import numpy as np
        from PIL import Image
        h, w = frame.shape[:2]
        r = min(self._size / h, self._size / w)
        nh, nw = int(h * r), int(w * r)
        im = np.asarray(Image.fromarray(frame).resize((nw, nh)))
        pad = np.ones((self._size, self._size, 3), np.float32) * 114.0
        pad[:nh, :nw] = im
        return pad.transpose(2, 0, 1)[None].astype(np.float32), r

    def detect(self, frame: Any) -> list[DetectedObject]:
        height, width = frame.shape[:2]
        x, ratio = self._preprocess(frame)
        out = self._session.run(None, {self._input_name: x})[0]
        return yolox_decode(out, self._size, ratio, width, height, self._conf,
                            self._categories, self._class_map, self._keep_classes, self._iou)

    @property
    def suppress_biometrics(self) -> bool:
        return self._suppress_biometrics

    @property
    def input_size(self) -> int:
        return self._size


class OpenVINODetector:
    """Opt-in OpenVINO-FP32 backend for torchvision COCO detectors.

    Converts the torchvision model to OpenVINO IR at load and runs it on the
    OpenVINO ``CPU`` device (``LATENCY`` hint) — ~1.7–2.2× faster than eager
    torch on AVX2 CPUs. Output mapping reuses :func:`torchvision_to_objects`
    (same COCO categories, pixel xyxy). The IR is compiled lazily on the first
    frame, at that frame's resolution. Needs ``.[openvino]``. INT8 is not offered
    — it is slower on CPUs without VNNI.
    """

    def __init__(
        self,
        model_name: str = "fasterrcnn_mobilenet_v3_large_fpn",
        conf: float = 0.25,
        suppress_biometrics: bool = True,
        class_map: dict[str, str] | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        num_threads: int = 0,
        keep_classes: set[str] | None = None,
        _compiled: Any | None = None,
        _categories: list[str] | None = None,
    ):
        self._conf = conf
        self._suppress_biometrics = suppress_biometrics
        self._class_map = class_map
        self._keep_classes = keep_classes
        self._min_size = min_size
        self._max_size = max_size
        self._num_threads = num_threads
        self._compiled: Any = _compiled
        self._core: Any = None
        self._wrapped: Any = None
        if _compiled is not None:
            self._categories = _categories or []
            return
        try:
            import openvino as ov
            import torch
            import torchvision
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OpenVINO backend needs torch + torchvision + openvino: "
                "pip install '.[openvino]'"
            ) from exc

        weights = torchvision.models.get_model_weights(model_name).DEFAULT
        model = torchvision.models.get_model(
            model_name, weights=weights, **_size_kwargs(min_size, max_size)
        ).eval()
        self._categories = list(weights.meta["categories"])

        class _DetWrap(torch.nn.Module):
            # Single 4D tensor in → plain (boxes, labels, scores) tensors out:
            # normalizes the list[Tensor]→list[dict] signature that trips the tracer.
            def __init__(self, m: Any) -> None:
                super().__init__()
                self.m = m

            def forward(self, x: Any) -> Any:  # x: (1,3,H,W)
                o = self.m([x[0]])[0]
                return o["boxes"], o["labels"], o["scores"]

        self._wrapped = _DetWrap(model).eval()
        self._core = ov.Core()
        logger.info("openvino backend ready for %s (min/max=%s/%s); IR compiled on first frame",
                    model_name, min_size, max_size)

    def _ensure_compiled(self, height: int, width: int) -> None:
        if self._compiled is not None:
            return
        import openvino as ov
        import torch

        # Serialize the trace+convert across all detectors (see _COMPILE_LOCK).
        # Double-check inside the lock so only the first waiter compiles.
        with _COMPILE_LOCK:
            if self._compiled is not None:
                return
            example = torch.zeros(1, 3, height, width)
            ov_model = ov.convert_model(self._wrapped, example_input=example)
            # Default LATENCY spreads ONE inference across all cores — best for a
            # single camera, but N such detectors then fight for every core. With
            # num_threads set, cap threads + a single stream so N independent
            # detectors PACK onto the cores (each ~num_threads) — required for a
            # multi-camera box.
            cfg: dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
            if self._num_threads and self._num_threads > 0:
                cfg = {"INFERENCE_NUM_THREADS": str(self._num_threads), "NUM_STREAMS": "1"}
            self._compiled = self._core.compile_model(ov_model, "CPU", cfg)

    def detect(self, frame: Any) -> list[DetectedObject]:
        import numpy as np

        height, width = frame.shape[:2]
        self._ensure_compiled(height, width)
        x = (frame.astype("float32") / 255.0).transpose(2, 0, 1)[None]  # (1,3,H,W)
        res = self._compiled([x])
        plain = {
            "boxes": np.asarray(res[0]).tolist(),
            "labels": np.asarray(res[1]).tolist(),
            "scores": np.asarray(res[2]).tolist(),
        }
        return torchvision_to_objects(
            plain, width, height, self._categories, self._conf, self._class_map,
            self._keep_classes,
        )

    @property
    def suppress_biometrics(self) -> bool:
        return self._suppress_biometrics

    @property
    def min_size(self) -> int | None:
        return self._min_size

    @property
    def max_size(self) -> int | None:
        return self._max_size

    @property
    def num_threads(self) -> int:
        return self._num_threads


def create_detector(
    backend: str = "torchvision",
    model: str = "ssdlite320_mobilenet_v3_large",
    conf: float = 0.25,
    suppress_biometrics: bool = True,
    class_map: dict[str, str] | None = None,
    device: str = "auto",
    min_size: int | None = None,
    max_size: int | None = None,
    num_threads: int = 0,
    keep_classes: list[str] | None = None,
) -> Detector:
    """Factory. ``backend`` ∈ {``mock``, ``torchvision`` (default), ``yolov8``,
    ``yolox`` (ONNX+ORT, Apache-2.0, single-stage), ``openvino``}.
    ``device`` ∈ {``auto``, ``cpu``, ``cuda``, ``mps``} (torchvision).
    ``min_size``/``max_size`` set the model's input-resize resolution for the
    torchvision and openvino backends (the cost/precision lever).
    ``num_threads`` (openvino only) caps CPU threads per detector so multiple
    detectors pack onto a multi-camera box instead of each grabbing every core;
    0 (default) keeps the LATENCY hint (best single-camera).
    ``keep_classes`` is an allowlist of raw model labels (e.g. ``["person"]``);
    detections of any other class are dropped. ``None`` keeps all classes."""
    keep = {c.lower() for c in keep_classes} if keep_classes else None
    if backend == "mock":
        return MockDetector(suppress_biometrics=suppress_biometrics, keep_classes=keep)
    if backend == "torchvision":
        return TorchvisionDetector(
            model_name=model,
            conf=conf,
            suppress_biometrics=suppress_biometrics,
            class_map=class_map,
            device=device,
            min_size=min_size,
            max_size=max_size,
            keep_classes=keep,
        )
    if backend == "yolov8":
        return Yolov8Detector(
            model_path=model,
            conf=conf,
            suppress_biometrics=suppress_biometrics,
            class_map=class_map,
            keep_classes=keep,
        )
    if backend == "yolox":
        return YoloxDetector(
            model_path=model,
            conf=conf,
            suppress_biometrics=suppress_biometrics,
            class_map=class_map,
            keep_classes=keep,
            num_threads=num_threads,
        )
    if backend == "openvino":
        return OpenVINODetector(
            model_name=model,
            conf=conf,
            suppress_biometrics=suppress_biometrics,
            class_map=class_map,
            min_size=min_size,
            max_size=max_size,
            num_threads=num_threads,
            keep_classes=keep,
        )
    factory = plugins.load_plugin(plugins.DETECTORS, backend)
    if factory is not None:
        return factory()  # type: ignore[no-any-return]
    raise ValueError(f"Unknown detector backend: {backend!r}")
