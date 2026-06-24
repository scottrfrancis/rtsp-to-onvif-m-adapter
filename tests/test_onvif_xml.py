"""Structural round-trip: onvif-mj JSON → tt:MetadataStream XML.

No XSD needed — asserts the serializer reconstructs the ONVIF envelope and maps
@-attrs / #text / arrays back to elements/attributes in schema-sequence order.
"""

import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from onvif_m.metadata import build_frame, build_payload
from onvif_m.model import TT_NAMESPACE, BoundingBox, ClassCandidate, DetectedObject
from onvif_m.onvif_xml import to_metadata_stream_element, to_xml_string

TT = f"{{{TT_NAMESPACE}}}"


def _payload():
    obj = DetectedObject(
        object_id=15,
        bbox=BoundingBox(-0.9375, -0.6667, -0.6875, -0.875),
        classes=[ClassCandidate("Human", 0.8)],
        center_of_gravity=(-0.8125, -0.7917),
    )
    frame = build_frame(datetime(2021, 10, 5, 15, 13, 27, 321000, tzinfo=UTC),
                        "MyClassifier", [obj])
    return build_payload([frame])


def test_envelope_and_attribute_mapping():
    root = to_metadata_stream_element(_payload())

    assert root.tag == f"{TT}MetadataStream"
    va = root.find(f"{TT}VideoAnalytics")
    assert va is not None
    frame = va.find(f"{TT}Frame")
    assert frame.get("UtcTime") == "2021-10-05T15:13:27.321Z"
    assert frame.get("Source") == "MyClassifier"

    obj = frame.find(f"{TT}Object")
    assert obj.get("ObjectId") == "15"

    bbox = obj.find(f"{TT}Appearance/{TT}Shape/{TT}BoundingBox")
    assert bbox.get("left") == "-0.9375"
    assert bbox.get("bottom") == "-0.875"

    type_el = obj.find(f"{TT}Appearance/{TT}Class/{TT}Type")
    assert type_el.get("Likelihood") == "0.8"
    assert type_el.text == "Human"


def test_appearance_child_order_is_schema_valid():
    # Appearance sequence: ... Shape ... Class ... — Shape must precede Class.
    root = to_metadata_stream_element(_payload())
    appearance = root.find(f"{TT}VideoAnalytics/{TT}Frame/{TT}Object/{TT}Appearance")
    children = [c.tag for c in appearance]
    assert children.index(f"{TT}Shape") < children.index(f"{TT}Class")


def test_to_xml_string_parses():
    xml = to_xml_string(_payload())
    # parses and carries the namespace
    root = ET.fromstring(xml)
    assert root.tag == f"{TT}MetadataStream"
