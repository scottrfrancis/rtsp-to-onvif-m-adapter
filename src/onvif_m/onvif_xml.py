"""Serialize the ``onvif-mj`` JSON payload to ``tt:MetadataStream`` XML.

Reverses the JSON binding (``@`` keys → attributes, ``#text`` → element text,
arrays → repeated elements) and wraps the payload in the ``MetadataStream`` /
``VideoAnalytics`` envelope. The output validates against the official ONVIF
``metadatastream.xsd`` (see tests/test_compliance.py).

Child elements are emitted in JSON insertion order; the builder inserts them in
ONVIF schema-sequence order (Shape before Class, BoundingBox before
CenterOfGravity), so the output is sequence-valid.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from .model import TT_NAMESPACE


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _append(name: str, value: Any, parent: ET.Element, ns: str) -> ET.Element:
    el = ET.SubElement(parent, f"{{{ns}}}{name}")
    if isinstance(value, dict):
        for key, val in value.items():
            if key.startswith("@"):
                el.set(key[1:], _fmt(val))
            elif key == "#text":
                el.text = _fmt(val)
            elif isinstance(val, list):
                for item in val:
                    _append(key, item, el, ns)
            elif isinstance(val, dict):
                _append(key, val, el, ns)
            else:
                child = ET.SubElement(el, f"{{{ns}}}{key}")
                child.text = _fmt(val)
    else:
        el.text = _fmt(value)
    return el


def to_metadata_stream_element(payload: dict[str, Any], ns: str = TT_NAMESPACE) -> ET.Element:
    """Wrap an ``onvif-mj`` payload in ``tt:MetadataStream/tt:VideoAnalytics``."""
    root = ET.Element(f"{{{ns}}}MetadataStream")
    video_analytics = ET.SubElement(root, f"{{{ns}}}VideoAnalytics")
    for frame in payload.get("Frame", []):
        _append("Frame", frame, video_analytics, ns)
    return root


def to_xml_string(payload: dict[str, Any], ns: str = TT_NAMESPACE) -> str:
    ET.register_namespace("tt", ns)
    return ET.tostring(to_metadata_stream_element(payload, ns), encoding="unicode")
