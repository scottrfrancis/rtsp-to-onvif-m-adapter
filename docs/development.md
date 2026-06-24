# Development

How to set up, test, lint, and benchmark the producer.

## Setup

Create a virtualenv, then install editable with the `dev` extra. Dependencies are
declared as extras in `pyproject.toml` (no `requirements.txt`).

**Linux / macOS**

```bash
python -m venv venv && . venv/bin/activate
pip install -e ".[dev]"                        # tests, ruff, mypy, xmlschema, paho
pip install -e ".[detect]"                     # + torch/torchvision (CPU wheels from PyPI)
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                    # may need: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -e ".[dev]"
# For an NVIDIA GPU (e.g. RTX 40-series, driver CUDA 12.x), install CUDA wheels
# explicitly instead of the CPU `detect` extra:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

`pip install -e ".[detect]"` pulls a **CPU-only** torch from PyPI; for GPU install
the matching CUDA wheels as above (cu121 for Ada/40-series; cu128 for Blackwell).
`ffmpeg` must be on `PATH` for RTSP capture.

Extras: `capture` (numpy/pillow), `detect` (torchvision detector), `mqtt` (paho),
`compliance` (xmlschema), `dev` (everything for development).

## Tests

```bash
pytest -q                  # everything; integration tests self-skip if their server/fixture is absent
ruff check . && mypy
```

The suite is layered:

| Layer | Files | Needs |
|---|---|---|
| **Unit** | `test_model`, `test_metadata`, `test_detect`, `test_publish`, `test_capture`, `test_cli`, `test_bench` | nothing (mocks / injected fakes) |
| **Integration — ONVIF XSD** | `test_compliance`, `test_onvif_xml` | `xmlschema`; vendored schema (+ one remote OASIS import on first build) |
| **JSON Schema** | `test_json_schema` | `jsonschema` (in the `dev`/`compliance` extras) |
| **Integration — MQTT** | `test_mqtt_integration` | a broker (below) |
| **Integration — RTSP** | `test_rtsp_integration` | an RTSP server (below) |
| **Integration — accuracy** | `test_detect_accuracy` | torch + the COCO fixture (below) |
| **E2E** | `test_e2e`, `test_pipeline` | the official-schema fixture |

Integration tests `skipif` when their dependency is missing, so the default
`pytest` run stays green offline. To run everything with nothing skipped, bring
up the broker + RTSP server and fetch the fixtures first.

### Scripted test servers (not committed)

Brokers/servers are spun up on demand — only the scripts are committed, never the
binaries.

```bash
bash tests/mqtt/run-broker.sh     # MQTT (Docker eclipse-mosquitto, or local mosquitto)
bash tests/rtsp/run-rtsp.sh       # RTSP  (Docker mediamtx + ffmpeg synthetic stream)
# ... run tests ...
bash tests/mqtt/run-broker.sh stop
bash tests/rtsp/run-rtsp.sh stop
```

Both prefer Docker when the daemon is reachable and fall back / error clearly
otherwise. See `tests/mqtt/README.md` and `tests/rtsp/README.md`.

### Accuracy fixture (COCO)

```bash
bash tests/fixtures/fetch_samples.sh          # COCO val image 785 (gitignored)
pytest tests/test_detect_accuracy.py -v       # asserts IoU >= 0.5 vs published COCO GT
```

The detector localizes the reference person at IoU ~0.89 — a real ground-truth
check, not a golden box.

### ONVIF conformance

`test_compliance.py` serializes each payload to `tt:MetadataStream` XML and
validates it against the **official** ONVIF `metadatastream.xsd` (vendored under
`schema/onvif/`, **XSD 1.1** — ONVIF uses a trailing `xs:any` that XSD-1.0
rejects). One OASIS `wsn` import is fetched remotely on first schema build; the
suite self-skips offline. Full closure vendoring for fully-offline CI is tracked.

## Devices & benchmarking

The torchvision detector runs on **CPU**, **Apple Silicon (MPS)**, or **NVIDIA
(CUDA)** via `--device {auto,cpu,cuda,mps}` (auto = cuda > mps > cpu).

```bash
python -m onvif_m.bench --model ssdlite320_mobilenet_v3_large --device auto
python -m onvif_m.bench --model retinanet_resnet50_fpn --device cuda --runs 50
```

See the **Device & performance** section of the [README](../README.md#device--performance)
for measured per-frame latency (Apple M2 Max, x86 CPU, RTX PRO 4500, and a
Windows 11 + RTX 4070 reference box) and the CPU-vs-GPU guidance.

### Setting up an NVIDIA box (CUDA)

Newer GPUs (e.g. Blackwell / sm_120) need the matching CUDA wheels. Example on
Ubuntu 24.04 with a Blackwell card (CUDA 13 driver):

```bash
# venv prerequisites (Ubuntu splits these out)
sudo apt-get install -y python3-pip python3.12-venv      # provides ensurepip

cd ~/workspace && python3 -m venv onvif-venv
./onvif-venv/bin/pip install numpy pillow
# Blackwell (sm_120) requires cu128 wheels:
./onvif-venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

PYTHONPATH=src ./onvif-venv/bin/python -m onvif_m.bench --device cuda
```

Verify the GPU actually executes (not just `cuda.is_available()`): a successful
`bench --device cuda` run with sane latency confirms real kernel execution.
