"""RTSP → object detection → ONVIF Profile-M metadata producer.

Public API for building and serializing conformant ONVIF ``onvif-mj`` metadata.
The output round-trips to ``tt:MetadataStream`` XML that validates against the
official ONVIF ``metadatastream.xsd`` (see tests/test_compliance.py).
"""

__version__ = "0.1.0"

from .metadata import build_frame, build_object, build_payload
from .model import (
    TT_NAMESPACE,
    BoundingBox,
    ClassCandidate,
    DetectedObject,
    from_pixel_bbox,
)
from .onvif_xml import to_metadata_stream_element, to_xml_string

__all__ = [
    "BoundingBox",
    "ClassCandidate",
    "DetectedObject",
    "from_pixel_bbox",
    "TT_NAMESPACE",
    "build_object",
    "build_frame",
    "build_payload",
    "to_metadata_stream_element",
    "to_xml_string",
]
