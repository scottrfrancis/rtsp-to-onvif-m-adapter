"""Producer-side data model.

Coordinates follow the ONVIF analytics default: a normalized frame of
``[-1, 1]`` on both axes, origin at the **center**, **y-up** (so a box's ``top``
is greater than its ``bottom``). This is the convention an ONVIF-aware consumer
assumes when no ``Transformation`` is supplied (ONVIF Analytics Service Spec
§5.x; coordinate ambiguity discussed in onvif/specs#409).

Detectors emit top-left pixel boxes; ``from_pixel_bbox`` converts them into this
ONVIF space, so the builder can stay convention-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ONVIF tt: schema namespace (used by the XML serializer / XSD validation).
TT_NAMESPACE = "http://www.onvif.org/ver10/schema"


@dataclass(frozen=True)
class BoundingBox:
    """ONVIF normalized box: [-1, 1], origin center, y-up (top > bottom)."""

    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class ClassCandidate:
    """One ONVIF ``tt:Class/tt:Type`` candidate.

    ``type`` is an ONVIF ObjectClass string — e.g. ``"Human"``, ``"Vehicle"``,
    ``"Animal"``, ``"LicensePlate"``, ``"Face"``, ``"Bike"``, ``"Other"``.
    (Note: ONVIF uses ``"Human"``, not ``"Person"``.)
    """

    type: str
    likelihood: float


@dataclass
class DetectedObject:
    object_id: int
    bbox: BoundingBox
    classes: list[ClassCandidate] = field(default_factory=list)
    center_of_gravity: tuple[float, float] | None = None


def from_pixel_bbox(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> BoundingBox:
    """Convert a top-left pixel box ``(x1,y1,x2,y2)`` to ONVIF normalized space.

    x: ``[0,width]`` → ``[-1,1]`` (left→right).
    y: ``[0,height]`` (top→bottom, pixel) → ``[1,-1]`` (top→bottom, ONVIF y-up),
    so the resulting ``top`` is greater than ``bottom``.
    """
    def nx(px: float) -> float:
        return px / width * 2.0 - 1.0

    def ny(py: float) -> float:
        return 1.0 - py / height * 2.0

    return BoundingBox(left=nx(x1), top=ny(y1), right=nx(x2), bottom=ny(y2))
