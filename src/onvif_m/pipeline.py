"""Wire capture → detect → post-process → build → publish for one stream.

``process_frame`` is the unit of work: detect on a captured frame, run any
``PostProcessor`` hooks, build the ONVIF ``onvif-mj`` payload (one Frame), and
publish it. ``run_camera`` drives a capture source through that for the life of
the stream (bounded by ``max_frames`` in tests).

Extension point — ``PostProcessor``: a user hook that runs after detection and
before the metadata is built. It receives the detected objects plus the source
frame and returns the object list to publish. This is where ReID / tracking
(reassign a stable ``object_id``), per-object histogram or attribute tagging,
face blurring, or filtering live. Processors run in order; each sees the previous
one's output. None ship here — they are entirely the user's to provide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .capture import CapturedFrame, CaptureSource
from .detect import Detector
from .metadata import build_frame, build_payload
from .model import DetectedObject
from .publish import FrameRef, Publisher


@dataclass
class Camera:
    """Identity for the single stream: ``name`` is the ONVIF ``@Source`` and the
    output subdirectory; ``profile_token`` is the ONVIF media profile token used
    in the MQTT topic."""

    name: str
    profile_token: str = "0"


@runtime_checkable
class PostProcessor(Protocol):
    """User extension hook, applied after detect and before build.

    Return the object list to publish (re-identified, enriched, filtered, …).
    ``object_id`` reassigned here flows straight to ONVIF ``@ObjectId``. Note:
    arbitrary descriptors (e.g. ReID embeddings) have no ONVIF metadata field
    today, so emitting those would need a schema/model extension.
    """

    def process(
        self, objects: list[DetectedObject], frame: CapturedFrame
    ) -> list[DetectedObject]: ...


def process_frame(
    camera: Camera,
    frame: CapturedFrame,
    detector: Detector,
    publisher: Publisher,
    module: str = "",
    processors: Sequence[PostProcessor] = (),
) -> dict[str, Any]:
    """Detect → post-process → build the onvif-mj payload → publish. Returns it."""
    objects = detector.detect(frame.image)
    for proc in processors:
        objects = proc.process(objects, frame)
    payload = build_payload([build_frame(frame.timestamp, camera.name, objects)])
    publisher.publish(
        payload,
        FrameRef(
            camera_id=camera.name,
            timestamp=frame.timestamp,
            profile_token=camera.profile_token or "0",
            module_name=module,
        ),
    )
    return payload


def run_camera(
    camera: Camera,
    source: CaptureSource,
    detector: Detector,
    publisher: Publisher,
    module: str = "",
    max_frames: int | None = None,
    health: Any = None,
    processors: Sequence[PostProcessor] = (),
) -> int:
    """Drive ``source`` through detect+publish. Returns frames processed.
    Records each frame into ``health`` (a HealthRegistry) when provided."""
    n = 0
    for frame in source.frames():
        process_frame(camera, frame, detector, publisher, module, processors)
        n += 1
        if health is not None:
            health.record_frame()
        if max_frames is not None and n >= max_frames:
            break
    return n
