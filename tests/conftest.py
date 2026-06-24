"""Shared fixtures — the official ONVIF XSD and the (non-normative) JSON Schema."""

import json
from pathlib import Path
from typing import Any

import pytest

# Vendored ONVIF tt: schema closure (metadatastream.xsd + common.xsd + onvif.xsd
# + humanface/humanbody). The OASIS wsn/b-2 import it references is fetched
# remotely on first build; tests skip if it can't be resolved (offline).
_LOCAL_XSD = (
    Path(__file__).parent.parent
    / "schema/onvif/wsdl/ver10/schema/metadatastream.xsd"
)

# Non-normative onvif-mj JSON Schema (inferred from the XSD + this implementation).
_JSON_SCHEMA = Path(__file__).parent.parent / "schema/onvif-mj.schema.json"


class _NoOpValidator:
    """Used when `jsonschema` is not installed: keeps payload-producing tests
    running (and their other assertions intact) instead of skipping them."""

    def validate(self, instance: Any) -> None:  # pragma: no cover - trivial
        return None

    def is_valid(self, instance: Any) -> bool:  # pragma: no cover - trivial
        return True


@pytest.fixture(scope="session")
def json_schema() -> Any:
    """A validator for the non-normative onvif-mj JSON Schema. Every test that
    produces an onvif-mj payload validates it against this. Falls back to a no-op
    if `jsonschema` is absent (install the `dev` extra to enable real validation)."""
    try:
        import jsonschema
    except ImportError:
        return _NoOpValidator()
    schema = json.loads(_JSON_SCHEMA.read_text())
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


@pytest.fixture(scope="session")
def onvif_schema():
    xmlschema = pytest.importorskip("xmlschema")
    if not _LOCAL_XSD.exists():
        pytest.skip("vendored ONVIF schema missing; run schema fetch")
    try:
        # ONVIF schemas use XSD 1.1 constructs (trailing xs:any after optional
        # elements), so XSD 1.0 validation rejects the schema itself.
        return xmlschema.XMLSchema11(str(_LOCAL_XSD))
    except Exception as exc:  # offline (wsn import unresolved), etc.
        pytest.skip(f"could not build ONVIF schema: {exc}")
