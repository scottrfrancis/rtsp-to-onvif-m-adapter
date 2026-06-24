# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com); versions
follow [SemVer](https://semver.org).

## [0.1.0] — 2026-06-24

First public release: a vendor-neutral, single-stream **ONVIF Profile-M metadata
producer** — one RTSP stream → object detection → ONVIF metadata.

### Capture & detect
- One RTSP stream per process (ffmpeg, reconnect with backoff); `--name` /
  `--profile-token` for identity. Scale by running one process per stream.
- Pluggable detectors (`Detector` protocol): torchvision (BSD-3, default;
  CPU/MPS/CUDA via `--device`), mock, and opt-in YOLOv8 (AGPL-3.0, never default).
- `PostProcessor` hook runs after detection, before metadata is built — for
  ReID/tracking, tagging, blurring, or filtering.

### Metadata & output
- ONVIF `onvif-mj` metadata builder; output round-trips to `tt:MetadataStream`
  XML that validates against the official ONVIF `metadatastream.xsd` (XSD 1.1).
- Non-normative JSON Schema (`schema/onvif-mj.schema.json`), intentionally open to
  user extensions; every payload the test suite produces is validated against it.
- Output is format (`json`/`xml`) × sink (`file`/`stdout`/`mqtt`), each
  multi-select via CLI or environment; sinks run simultaneously.

### Extensibility & ops
- Entry-point plugins: third-party publishers, detectors, and processors register
  via the `onvif_m.{publishers,detectors,processors}` groups and are usable by
  name in `--sink` / `--detector` / `--processor`.
- Optional `/healthz` liveness endpoint; latency benchmark (`python -m onvif_m.bench`).
- Docker image and a self-contained `docker compose` demo.

### Quality
- Unit / integration / e2e tests, ONVIF XSD compliance, JSON-Schema validation,
  and a COCO-GT accuracy check; `ruff` + `mypy --strict` clean; CI on Python 3.11 / 3.12.

### Notes
- Coordinates use the ONVIF default normalized frame: `[-1, 1]`, origin center, y-up.
- `suppress_biometrics` (default on) is metadata-scope only — it loads no face/body
  submodels and emits no `HumanFace`/`HumanBody` fields, and never alters the image.
- The core is Apache-2.0 and AGPL-free; YOLOv8 is opt-in only.
