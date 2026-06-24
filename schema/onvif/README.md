# Vendored ONVIF schemas

Official ONVIF XSDs, vendored for the compliance tests
(`tests/test_compliance.py`) so the conformance proof runs without depending on
network availability for the ONVIF files.

```
wsdl/ver10/schema/metadatastream.xsd   ← the Scene Description metadata types
wsdl/ver10/schema/common.xsd
wsdl/ver10/schema/onvif.xsd            ← common tt: types
wsdl/ver20/analytics/humanface.xsd
wsdl/ver20/analytics/humanbody.xsd
```

Source: `github.com/onvif/specs` (`development` branch), `wsdl/` tree. The
directory layout is preserved so the schemas' relative `<xs:import>` paths
resolve locally.

Notes:
- **XSD 1.1 required.** ONVIF's `AppearanceType` ends an optional-element
  sequence with `<xs:any>`, which violates XSD-1.0 Unique Particle Attribution.
  Validate with `xmlschema.XMLSchema11`, not `XMLSchema`.
- **One remaining remote import.** `metadatastream.xsd` imports the OASIS
  `http://docs.oasis-open.org/wsn/b-2.xsd`, fetched on first schema build. The
  compliance tests self-skip if it can't be resolved (offline). Vendoring the
  full OASIS/W3C closure for fully-offline CI is a tracked follow-up.

These files are upstream ONVIF artifacts; their terms are ONVIF's, not this
project's license.
