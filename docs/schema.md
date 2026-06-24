# Output format & consumer guide

The producer emits the ONVIF **`onvif-mj`** metadata binding (the JSON form of an
ONVIF Scene Description). It round-trips to `tt:MetadataStream` XML that validates
against the official ONVIF `metadatastream.xsd` (`tests/test_compliance.py`).
Canonical example:
[`../schema/onvif-mj.example.json`](../schema/onvif-mj.example.json).

## JSON Schema (non-normative)

[`../schema/onvif-mj.schema.json`](../schema/onvif-mj.schema.json) is a JSON
Schema for the payload, provided as a convenience for JSON consumers. It is
**non-normative**: an inference from the ONVIF XSD **and** this implementation,
**not** an official ONVIF artifact. The XSD remains the authoritative source of
conformance; the schema and live builder output are both checked against it in
the test suite.

The schema is intentionally **open** — `additionalProperties` is allowed at every
level. So if you add an optional field (for example a `ReID` descriptor on an
`Object`), validation still passes; that field is simply not described here. To
formalize such an extension, **define your own schema** — extend this one via
`allOf`/`$ref`, or supply your own — and validate against that. (Note: the ONVIF
model has no field for arbitrary descriptors like ReID embeddings, so those live
outside the standard regardless.)

## Payload shape

One payload carries a `Frame` array (one entry per analyzed video frame):

```json
{ "Frame": [ {
  "@UtcTime": "2021-10-05T15:13:27.321Z",
  "@Source": "cam-7",
  "Object": [ {
    "@ObjectId": 0,
    "Appearance": {
      "Shape": {
        "BoundingBox": {"@left": -0.94, "@top": -0.67, "@right": -0.69, "@bottom": -0.88},
        "CenterOfGravity": {"@x": -0.81, "@y": -0.79}
      },
      "Class": {"Type": [ {"@Likelihood": 0.8, "#text": "Human"} ]}
    }
  } ]
} ] }
```

Mapping rules (mechanical, from the XML): attributes → `@`-keys; element text →
`#text`; repeatable elements (`Frame`, `Object`, `Type`) → arrays.

### Field reference

| Path | Meaning |
|---|---|
| `Frame[]` | one entry per analyzed frame, in time order |
| `Frame[].@UtcTime` | frame capture time, UTC (`xs:dateTime`, ms) |
| `Frame[].@Source` | producer source id (camera name) |
| `Frame[].Object[]` | detections in the frame; **absent when none** (the bare frame = liveness) |
| `Object[].@ObjectId` | **within-frame ordinal only — NOT stable across frames** |
| `Appearance.Shape.BoundingBox` | `{@left,@top,@right,@bottom}`, normalized `[-1,1]`, origin center, **y-up** (top > bottom) |
| `Appearance.Shape.CenterOfGravity` | `{@x,@y}` (mandatory in ONVIF `ShapeDescriptor`) |
| `Appearance.Class.Type[]` | `[{@Likelihood, #text}]`; `#text` is an ONVIF ObjectClass (`Human`, `Vehicle`, `Animal`, …) sorted desc |

## Consuming the metadata

**Find.** Two publishers ship:
- **File** (`FilePublisher`): a `.meta.json` sidecar per frame under
  `<output_root>/<camera>/` (or next to the captured frame if a path is given).
- **MQTT** (`MqttPublisher`): topic
  `<prefix>/onvif-mj/VideoAnalytics/<ProfileToken>[/<Module>]` (ONVIF §5.4.2),
  one message per frame.

**Read.** Parse `Frame[]` in time order. `Object: []`/absent means "frame
analyzed, nothing detected" (liveness) — distinct from a gap between frames at
the sample cadence. Coordinates are normalized `[-1,1]`, origin center, y-up;
multiply by the decoded frame's pixel `W×H` (after mapping `[-1,1]→[0,1]`) to get
pixels.

**Reconcile to video.** `@UtcTime` is the wall-clock time of the frame — align it
to your recordings by timestamp. The producer ships **no cross-frame identity**
(`@ObjectId` is per-frame only) and **no tracking** in core: associate objects
across frames yourself (bbox proximity, embeddings, or a tracker), or add a
post-processor that assigns stable `@ObjectId`s (see `examples/processors.py`).

**Biometric suppression.** With `suppress_biometrics` (the default), the detector
loads no face/body submodels and the payload carries no `HumanFace`/`HumanBody`
fields. It is metadata-scope only: it does **not** blur or alter the image, and
this tool never writes or modifies frame imagery at all. Example use: surveying a
field for non-people objects while ensuring no biometric data is computed or
cascaded downstream.
