# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com); versions
follow [SemVer](https://semver.org).

## [Unreleased]

### Changed
- **Single stream per process.** Dropped the CSV camera roster and the
  thread-per-camera fan-out; the CLI now takes one RTSP URL (positional or
  `ONVIF_M_URL`), with `--name` / `--profile-token` for identity. Scaling to
  many streams is the user's concern — run one process per stream.
- **Output is now format × sink, both multi-select.** `--format {json,xml}` and
  `--sink {file,stdout,mqtt}` are each repeatable and settable via
  `ONVIF_M_FORMAT` / `ONVIF_M_SINK`. Sinks run simultaneously (`MultiPublisher`);
  the file sink writes `.meta.json` and/or `.meta.xml`.
- `/healthz` is single-stream (`{healthy, frames, seconds_since_frame}`).
- Docs describe current behavior only; removed roster/multi-host narration.

### Added
- **Post-processor hook** (`PostProcessor`): runs after detection, before the
  metadata is built, chainable. The supported extension point for ReID /
  tracking, tagging, blurring, or filtering. Wire from the library or
  `--processor module:factory` (`ONVIF_M_PROCESSORS`). See `examples/processors.py`.
- **Entry-point plugins** (`onvif_m.plugins`): third-party packages register
  publishers, detectors, and post-processors via the `onvif_m.publishers` /
  `onvif_m.detectors` / `onvif_m.processors` entry-point groups, usable by name
  in `--sink` / `--detector` / `--processor` after install.
- **Non-normative JSON Schema** (`schema/onvif-mj.schema.json`) for the onvif-mj
  payload — inferred from the XSD + implementation, intentionally open to user
  extensions (e.g. ReID). Every payload the test suite produces is validated
  against it (`jsonschema`, in the `dev`/`compliance` extras).
- Documented that `suppress_biometrics` is metadata-scope only — it never blurs
  or alters the image; this tool writes no image at all.

### Removed
- `cams.csv` roster, `onvif_m.config`, `examples/cams.demo.csv`.

## [0.1.0] — 2026-06-23

First release: a vendor-neutral **ONVIF Profile-M metadata producer**
(RTSP → object detection → ONVIF metadata).

### Added
- ONVIF `onvif-mj` metadata builder; **conformance proven** by round-tripping
  JSON → `tt:MetadataStream` XML and validating against the official ONVIF
  `metadatastream.xsd` (XSD 1.1).
- Pluggable detectors (`Detector` protocol): torchvision (BSD-3, default;
  CPU/MPS/CUDA via `--device`), mock, and opt-in YOLOv8 (AGPL-3.0, never default).
- Pluggable publishers (`Publisher` protocol): file (JSON sidecar), stdout, and
  MQTT (ONVIF-conformant topic).
- RTSP capture (ffmpeg, reconnect with backoff), CSV camera roster, and a
  thread-per-camera CLI (`python -m onvif_m`).
- `/healthz` liveness endpoint and a latency benchmark (`python -m onvif_m.bench`).
- Docker image and a self-contained `docker compose` demo.
- Unit / integration / e2e tests incl. a COCO-GT accuracy check; `ruff` +
  `mypy --strict` clean; CI on Python 3.11 / 3.12.

### Notes
- Coordinates use the ONVIF default normalized frame: `[-1,1]`, origin center, y-up.
- The core is Apache-2.0 and AGPL-free; YOLOv8 is opt-in only.
