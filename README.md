# rtsp-to-onvif-m-adapter

Reads one RTSP stream, runs a pluggable object detector on captured frames, and
emits per-frame detections as ONVIF Profile-M metadata — as `onvif-mj` JSON or
`tt:MetadataStream` XML. One stream per process; run several to scale out.

## Conformance

The XML output validates against the official ONVIF `metadatastream.xsd` (XSD
1.1) in the test suite (`tests/test_compliance.py`). Canonical output shape:
[`schema/onvif-mj.example.json`](schema/onvif-mj.example.json).

A **non-normative** JSON Schema for the payload is provided at
[`schema/onvif-mj.schema.json`](schema/onvif-mj.schema.json) — an inference from
the XSD and this implementation, not an official ONVIF artifact (the XSD stays
authoritative). It is deliberately open: you can add optional fields (e.g. a
`ReID` descriptor) without breaking validation, and define your own schema to
constrain them. Every payload the test suite produces is validated against it.

```python
from datetime import datetime, timezone
from onvif_m import (BoundingBox, ClassCandidate, DetectedObject,
                     build_frame, build_payload, to_xml_string, from_pixel_bbox)

obj = DetectedObject(
    object_id=0,
    bbox=from_pixel_bbox(64, 48, 192, 240, width=640, height=480),  # pixels -> ONVIF coords
    classes=[ClassCandidate("Human", 0.94)],
)
frame = build_frame(datetime.now(timezone.utc), source="cam-7", objects=[obj])
payload = build_payload([frame])     # onvif-mj JSON  -> {"Frame": [...]}
xml = to_xml_string(payload)         # tt:MetadataStream XML
```

Coordinates use the ONVIF normalized frame: `[-1, 1]`, origin center, y-up.
`from_pixel_bbox` converts top-left pixel boxes.

## Install

Requires Python 3.11+ and `ffmpeg` on `PATH`. Create a virtualenv, then install.

**Linux / macOS**

```bash
python -m venv venv && . venv/bin/activate
pip install -e ".[capture,detect,mqtt]"
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[capture,mqtt]"
# .[detect] installs CPU-only torch; for an NVIDIA GPU install CUDA wheels instead:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

`.[detect]` pulls **CPU-only** torch from PyPI on every platform. For GPU
acceleration, install the matching CUDA wheels as shown (cu121 for NVIDIA
40-series; see [`docs/development.md`](docs/development.md) for other GPUs).

## Run

```bash
# one RTSP URL at a time
python -m onvif_m rtsp://host/stream --detector torchvision --sink file --output-root ./out
python -m onvif_m rtsp://host/stream --detector mock --sink stdout          # dry-run wiring
python -m onvif_m rtsp://host/stream --sink mqtt --mqtt-host broker.local
```

`--device {auto,cpu,cuda,mps}` selects the accelerator. `--health-port N` serves
`/healthz`. SIGINT/SIGTERM stops cleanly. Scaling to many streams is left to the
user — run one process per stream.

## Output

Two independent axes, each repeatable and also settable via environment variable:

| Option | Values | Env |
|---|---|---|
| `--format` | `json`, `xml` | `ONVIF_M_FORMAT` |
| `--sink` | `file`, `stdout`, `mqtt` | `ONVIF_M_SINK` |

Every selected sink writes every selected format. Examples:

```bash
python -m onvif_m rtsp://host/stream --format json --format xml --sink file --sink mqtt
ONVIF_M_FORMAT=json,xml ONVIF_M_SINK=stdout python -m onvif_m rtsp://host/stream
```

- `file`: atomic sidecars under `<output-root>/<name>/` (`.meta.json` / `.meta.xml`).
- `stdout`: one line per payload per format.
- `mqtt`: topic `<prefix>/<onvif-mj|onvif-xml>/VideoAnalytics/<ProfileToken>[/<Module>]`.

## Extending

Three things are pluggable: **post-processors**, **publishers (sinks)**, and
**detectors**. There are three ways to plug in, depending on how you install.

### Post-processors

A post-processor runs after detection and before the metadata is built:

```python
class PostProcessor:                                 # onvif_m.pipeline.PostProcessor
    def process(self, objects, frame): ...           # -> list[DetectedObject]
```

This is the hook for ReID / tracking (reassign a stable `object_id`, which flows
to ONVIF `@ObjectId`), histogram or attribute tagging, face blurring, or
filtering. From the CLI, point `--processor` at any importable `module:factory` —
your own file on the path works from a PyPI install:

```bash
# my_hooks.py in the working directory (or any installed module)
python -m onvif_m rtsp://host/stream --processor my_hooks:HumansOnly
```

[`examples/processors.py`](examples/processors.py) is a copy-paste template (it
ships in the source tree, not the wheel, so reference your own module when
installed from PyPI).

### Plugins from PyPI (entry points)

A separately-installed package can register publishers, detectors, or
post-processors so they work from the CLI **by name**. Declare entry points in
the plugin package's `pyproject.toml`:

```toml
[project.entry-points."onvif_m.publishers"]
s3 = "my_pkg:S3Publisher"          # -> --sink s3

[project.entry-points."onvif_m.detectors"]
yolo11 = "my_pkg:Yolo11Detector"   # -> --detector yolo11

[project.entry-points."onvif_m.processors"]
reid = "my_pkg:ReID"               # -> --processor reid
```

Each entry point is a zero-arg factory (a class works) returning a `Publisher` /
`Detector` / `PostProcessor`. After `pip install my-pkg`, the names appear in the
CLI:

```bash
pip install my-onvif-plugins
python -m onvif_m rtsp://host/stream --detector yolo11 --sink s3 --sink file
```

### Library API (full control)

For custom wiring, drive the pipeline directly — this plugs in *anything*,
including custom publishers/detectors without packaging them:

```python
from onvif_m.capture import RtspCaptureSource
from onvif_m.pipeline import Camera, run_camera
from onvif_m.detect import create_detector
from onvif_m.publish import FilePublisher, MultiPublisher

run_camera(
    Camera("cam"),
    RtspCaptureSource("rtsp://host/stream"),
    create_detector(backend="torchvision"),
    MultiPublisher([FilePublisher("./out"), MyWebhookPublisher()]),
    processors=[MyReID()],
)
```

Note: arbitrary descriptors (e.g. ReID embeddings) have no ONVIF metadata field,
so emitting those needs a schema extension.

## Biometric suppression

`suppress_biometrics` (default on) is a detector-side flag: the detector loads no
face/body submodels and emits no `HumanFace`/`HumanBody` metadata. It does **not**
blur, mask, or alter the image — this tool never writes or modifies the source
frame at all. Image redaction, if needed, is a downstream concern.

Example use: surveying a field for non-people objects (vehicles, animals,
equipment) while ensuring no biometric data is computed or cascaded to downstream
systems.

## Design

- [`docs/schema.md`](docs/schema.md) — output format and consumer guide.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline, detector/publisher plugins,
  the post-processor hook, biometric suppression.
- [`REQUIREMENTS.md`](REQUIREMENTS.md); open backlog in [`TODO.md`](TODO.md).
- [`schema/onvif/README.md`](schema/onvif/README.md) — vendored ONVIF XSDs.

## Develop

```bash
pip install -e ".[dev]"
pytest -q          # incl. the ONVIF XSD compliance suite (self-skips offline)
ruff check . && mypy
```

Test matrix, scripted test servers (MQTT, RTSP), and benchmarking are in
[`docs/development.md`](docs/development.md); see
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/releasing.md`](docs/releasing.md).

The default detector is torchvision (BSD-3). Ultralytics YOLOv8 is AGPL-3.0 and
opt-in only.

## Device & performance

The detector runs on CPU, Apple Silicon (MPS), or NVIDIA (CUDA) — `--device
{auto,cpu,cuda,mps}` (auto = cuda > mps > cpu). Benchmark a combination:

```bash
python -m onvif_m.bench --model ssdlite320_mobilenet_v3_large --device auto
```

Per-frame latency at 640×480 (Apple figures measured on an Apple M2 Max):

| Model | CPU (M2 Max) | MPS (M2 Max) | CPU (x86) | CUDA (RTX PRO 4500) |
|---|---|---|---|---|
| `ssdlite320_mobilenet_v3_large` (default, light) | 48 ms | 64 ms | 28 ms | 13 ms (77 fps) |
| `retinanet_resnet50_fpn` (heavy, accurate) | 970 ms | 95 ms | 536 ms | 17 ms (59 fps) |

Measured on a Windows 11 reference box (Intel CPU + NVIDIA RTX 4070 Laptop, 8 GB,
50 W; torch 2.5.1+cu121), 640×480, 50–200 runs:

| Model | CPU | CUDA (RTX 4070) |
|---|---|---|
| `ssdlite320_mobilenet_v3_large` (light) | 83 ms (12 fps) | 135 ms (7.4 fps) |
| `retinanet_resnet50_fpn` (heavy) | 1584 ms (0.6 fps) | 82 ms (12 fps) |

Guidance: the light default runs in real time on CPU. For heavy/accurate models
use a GPU — CUDA is ~19× faster than CPU on the RTX 4070 here, MPS ~10× on the M2
Max. A tiny model is launch/transfer-bound on a GPU (notably on a power-limited
laptop GPU under Windows/WDDM, where the light default is actually *faster* on
CPU), so pair the light default with CPU and reserve the GPU for heavy models.

Windows 11 + NVIDIA CUDA is a tested platform (`pip install torch torchvision
--index-url https://download.pytorch.org/whl/cu121` for a CUDA build; `ffmpeg`
must be on `PATH`).

## License

[Apache-2.0](LICENSE). Vendored ONVIF XSDs under `schema/onvif/` are upstream
ONVIF artifacts under ONVIF's terms.
