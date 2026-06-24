# Requirements

## Functional

### F1. RTSP ingest

- Accept one RTSP stream URL as input (one stream per process).
- Capture frames at a configurable cadence (default: 1 frame per second).
- Tolerate intermittent stream failures: retry with backoff, log, do not crash the producer.
- Use ffmpeg for capture.
- Scaling to many streams is the user's concern — run one process per stream.

### F2. Pluggable object detection

- Accept a detector via a documented plugin interface.
- Ship a permissively-licensed detector as the default (torchvision, BSD-3); YOLOv8 (AGPL) opt-in only.
- The detector plugin contract: `detect(frame: np.ndarray) -> list[DetectedObject]`, where each object carries an `object_id`, an ONVIF-normalized `bbox`, and a list of `ClassCandidate` (`type`, `likelihood`).
- Detector plugins MUST honor the `suppress_biometrics` flag (see F8).
- Third-party detectors register via the `onvif_m.detectors` entry-point group and are then usable by name in `--detector` (see F11).

### F3. Metadata emission — ONVIF Profile M shape

- Emit one metadata payload per processed frame.
- Payload follows the ONVIF Profile M `MetadataStream`→`VideoAnalytics`→`Frame`→`Object[]`→`Appearance` hierarchy.
- Serialize as `onvif-mj` JSON and/or `tt:MetadataStream` XML (see F4). The XML validates against the official ONVIF `metadatastream.xsd`.
- See `schema/onvif-mj.example.json` for the concrete shape.
- `objects[].appearance.class.candidates[]` MUST be a list (not a single label) — each candidate has `type` (string, extensible) and `likelihood` (0.0–1.0).
- `objects[].object_id` is per-frame and need not be stable across frames. A post-processor (F10) may assign stable ids.

### F4. Output: format and sink selection

- **Format** (repeatable, also via `ONVIF_M_FORMAT`): `json` and/or `xml`.
- **Sink** (repeatable, also via `ONVIF_M_SINK`): `file`, `stdout`, and/or `mqtt`.
- Every selected sink emits every selected format.
- **Multiple sinks run simultaneously** — e.g. `--sink file --sink mqtt` writes sidecars and publishes to the broker for the same payload. `MultiPublisher` fans one payload out to all selected sinks.
- Publisher plugin contract: `publish(payload: dict, frame_ref: FrameRef) -> None`; `close() -> None`. Implementations own serialization, transport, and error handling.
- Third-party sinks register via the `onvif_m.publishers` entry-point group and are then usable by name in `--sink` (see F11).

### F5. File sink (default)

- Write one atomic sidecar per payload per format under `<output-root>/<name>/`: `<timestamp>.meta.json` and/or `<timestamp>.meta.xml`.
- Atomic write (write to `.tmp`, rename) so partial writes are never observed.
- The producer does not write or modify image frames — only metadata sidecars.

### F6. Stream configuration

- A single RTSP URL, supplied as a CLI positional argument or `ONVIF_M_URL`.
- Stream identity via `--name` (ONVIF `@Source` and output subdir) and `--profile-token` (MQTT topic), each with an env equivalent.
- No multi-stream roster / CSV.

### F7. Latency instrumentation (backlog)

- Goal: each payload carries per-stage timestamps (`captured_at`, `detection_started_at`, `detection_completed_at`, `published_at`) and a `profile` utility reports p50/p95/p99 from published sidecars.
- Not yet implemented — tracked in [`TODO.md`](TODO.md). `onvif_m.bench` measures live detection latency in the meantime.

### F8. Biometric suppression (`suppress_biometrics`)

- Detector-side flag, **default `True`**.
- When `True`: detector plugins MUST NOT load face/body submodels, and the payload MUST NOT include `HumanFace`/`HumanBody` fields. The guarantee is "do not compute," not "compute then strip."
- It is **metadata-scope only**: it does not blur, mask, or alter the image. The producer never writes or modifies frame imagery; image redaction is a downstream concern.

### F9. Health / liveness

- Optional HTTP `/healthz` endpoint (configurable port, off by default) reporting the stream's frame count, seconds-since-last-frame, and a healthy flag (frame seen within a staleness window).
- Log to stdout.

### F10. Post-processor extension hook

- A documented hook (`PostProcessor`) runs after detection and before metadata is built: `process(objects, frame) -> objects`.
- Processors chain in order; each sees the previous one's output.
- This is the supported extension point for ReID / tracking, histogram or attribute tagging, face blurring, and filtering. None ship in core — they are the user's to provide.
- Wired from the library (pass instances), the CLI (`--processor module:factory`, also `ONVIF_M_PROCESSORS`), or a registered `onvif_m.processors` plugin name (see F11).
- Note: arbitrary descriptors (e.g. ReID embeddings) have no ONVIF metadata field; emitting those requires a schema extension.

### F11. Plugin discovery (entry points)

- Publishers, detectors, and post-processors are discoverable via `importlib.metadata` entry-point groups: `onvif_m.publishers`, `onvif_m.detectors`, `onvif_m.processors`.
- Each entry point is a zero-argument factory (a class works) returning the corresponding object, which self-configures (e.g. from environment).
- A separately-installed plugin package's registered names become usable in `--sink` / `--detector` / `--processor`. Built-in names take precedence; plugins are additive.

## Non-functional

### NF1. Single-stream, single-process

- One process handles one stream. No distributed coordination.
- Stateless beyond the configured output directory.

### NF2. Performance budget (rough)

- Per-frame capture should consume <50 ms of wall clock (RTSP frame fetch).
- Detection budget depends on the model and device; see the benchmark table in the README.
- Profile, don't over-engineer.

### NF3. Resource hygiene

- Bounded memory growth (no unbounded queues).
- Bounded disk growth (the file sink owns its directory; rotation/cleanup is the operator's responsibility, with a documented pattern).
- Graceful shutdown on SIGINT/SIGTERM (close the RTSP session, flush the publisher).

### NF4. Target platforms

- Tested on recent Ubuntu LTS, macOS (Apple Silicon), and Windows 11 (NVIDIA CUDA). Other platforms best-effort.
- Requires `ffmpeg` on `PATH` for RTSP capture.

### NF5. License

- **Apache 2.0** (see [`LICENSE`](LICENSE)). Plugins that pull heavier-licensed dependencies (e.g. AGPL YOLOv8) are kept opt-in only so the core stays permissive.

### NF6. Documentation

- README, ARCHITECTURE, schema/consumer guide, and contributing docs.
- Quickstart: a single stream into file output in a few minutes.

### NF7. Test coverage

- Unit tests for metadata emission (deterministic fake detector → assertions on JSON/XML shape).
- Integration tests for the file sink (write, atomic rename) and the post-processor hook.
- End-to-end test against a canned RTSP source.
- ONVIF XSD compliance test on the XML output.

## Out of scope (and why)

| Item | Why excluded |
|---|---|
| Multi-stream / camera roster | One stream per process; the user runs N processes to scale out. |
| Cross-frame tracking / ReID in core | Per-frame producer; the post-processor hook (F10) is the supported place to add it. |
| Image redaction / blur of frames | The producer never writes frame imagery; redaction is a downstream consumer's job. `suppress_biometrics` controls metadata only. |
| Cloud publishers (GCS, S3, Pub/Sub) | Downstream consumers' job. May be community plugins later. |
| WS-Discovery, ONVIF Device service | This is not a camera; it does not announce or configure itself. |
| SOAP / WS-Notification publisher | Low demand; add later if requested. |
| Video archival | The project is not a VMS; existing tools handle archival. |
| Multi-host orchestration | Single-stream, single-process; scaling is the user's concern. |
