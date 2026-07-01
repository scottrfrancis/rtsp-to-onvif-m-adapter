# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com); versions
follow [SemVer](https://semver.org).

## [0.2.2]

### Fixed
- **Thread-safe model compilation.** `torch.jit.trace` / `ov.convert_model` are not
  thread-safe; two `OpenVINODetector`s compiling their IR concurrently (a multi-camera
  host starting N detectors at once) corrupted each other's trace and raised
  `The values for attribute 'dtype' do not match: torch.float64 != torch.float32`,
  killing the detector threads. Compilation is now serialized process-wide with a
  module lock (double-checked); inference on an already-compiled model stays parallel.

## [0.2.1]

### Added
- **`num_threads` knob** on `create_detector` and `OpenVINODetector` (openvino
  backend only). When `> 0`, compiles the IR with
  `{INFERENCE_NUM_THREADS: n, NUM_STREAMS: 1}` instead of the `LATENCY` hint,
  capping CPU threads per detector so **multiple detectors pack onto a
  multi-camera box** instead of each spreading one inference across every core.
  Default `0` keeps the `LATENCY` hint (best single-camera latency) — behavior
  unchanged for existing callers.

## [0.2.0]

### Added
- **`min_size` / `max_size` passthrough** on `create_detector` and
  `TorchvisionDetector` — sets the FPN/R-CNN input-resize resolution (the
  precision/latency lever). Default behavior unchanged; fixed-size models
  (ssd/ssdlite) ignore it.
- **`OpenVINODetector` backend** (`backend="openvino"`, opt-in like YOLOv8) —
  converts a torchvision COCO detector to OpenVINO IR at load and runs it on the
  OpenVINO CPU runtime (`LATENCY` hint). Measured ~1.7–2.2× faster than eager
  torch on AVX2 CPUs; IR compiled lazily at the first frame's resolution. New
  optional extra `[openvino]`. INT8 intentionally not offered (slower on CPUs
  without VNNI).

### Notes
- Output mapping and coordinates are unchanged; the OpenVINO backend reuses
  `torchvision_to_objects`, so it agrees with the eager-torch backend
  detection-for-detection.

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
