"""Build the ONVIF ``onvif-mj`` JSON metadata payload.

The JSON binding maps the ONVIF XML model:

- XML attributes  → ``@``-prefixed keys (``@UtcTime``, ``@left``, ``@ObjectId``)
- element text    → ``#text`` (e.g. ``Class/Type``)
- repeatable elements → JSON arrays (``Frame``, ``Object``, ``Type``)
- payload root    → ``{"Frame": [ ... ]}`` (the ``tt:MetadataStream`` /
  ``tt:VideoAnalytics`` wrapper is added on XML serialization, see ``onvif_xml.py``)

One ``Frame`` per analyzed video frame. A frame with no detections still emits
(with no ``Object`` key) so the timeline stays continuous. The payload serializes
to schema-valid ``tt:MetadataStream`` XML (see the compliance tests).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .model import BoundingBox, DetectedObject


def _utc(ts: datetime) -> str:
    """xs:dateTime in UTC, millisecond precision (e.g. 2021-10-05T15:13:27.321Z)."""
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def _bounding_box(b: BoundingBox) -> dict[str, float]:
    return {"@left": b.left, "@top": b.top, "@right": b.right, "@bottom": b.bottom}


def build_object(obj: DetectedObject) -> dict[str, Any]:
    """One ``tt:Object`` → JSON.

    ONVIF ``ShapeDescriptor`` requires BOTH ``BoundingBox`` and
    ``CenterOfGravity`` (minOccurs=1), so CoG is always emitted — defaulting to
    the box midpoint when the detector doesn't supply one.
    """
    cx, cy = obj.center_of_gravity or (
        (obj.bbox.left + obj.bbox.right) / 2.0,
        (obj.bbox.top + obj.bbox.bottom) / 2.0,
    )
    shape: dict[str, Any] = {
        "BoundingBox": _bounding_box(obj.bbox),
        "CenterOfGravity": {"@x": cx, "@y": cy},
    }

    appearance: dict[str, Any] = {"Shape": shape}
    if obj.classes:
        appearance["Class"] = {
            "Type": [
                {"@Likelihood": c.likelihood, "#text": c.type} for c in obj.classes
            ]
        }

    return {"@ObjectId": obj.object_id, "Appearance": appearance}


def build_frame(
    utc_time: datetime,
    source: str,
    objects: list[DetectedObject],
) -> dict[str, Any]:
    """One ``tt:Frame`` → JSON. ``Object`` omitted when there are no detections
    (the bare frame is the liveness signal)."""
    frame: dict[str, Any] = {"@UtcTime": _utc(utc_time), "@Source": source}
    if objects:
        frame["Object"] = [build_object(o) for o in objects]
    return frame


def build_payload(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``onvif-mj`` payload root: ``{"Frame": [ ... ]}``."""
    return {"Frame": frames}
