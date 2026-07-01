#!/usr/bin/env bash
# Container e2e — build the shipped image and run the producer against a synthetic
# RTSP source, asserting it writes ONVIF-M JSON sidecars. Validates the REAL
# ffmpeg RTSP grab → detect → file publish IN THE IMAGE, which unit tests
# (arg construction) and test_rtsp_integration (self-skips in CI without a
# server) don't cover automatically.
#
#   bash tests/container_e2e.sh          # needs Docker + ffmpeg + network
set -euo pipefail
cd "$(dirname "$0")/.."
WORK="$(mktemp -d)"; IMG="onvif-m:e2e"; NAME="onvif-m-e2e"
cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  bash tests/rtsp/run-rtsp.sh stop >/dev/null 2>&1 || true
  rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

command -v ffmpeg >/dev/null 2>&1 || { echo "SKIP: need ffmpeg"; exit 0; }
docker info >/dev/null 2>&1 || { echo "SKIP: need Docker"; exit 0; }

echo "== 1/4 synthetic RTSP (rtsp://localhost:8554/test) =="
bash tests/rtsp/run-rtsp.sh >/dev/null
for i in $(seq 1 15); do ffprobe -rtsp_transport tcp -i rtsp://localhost:8554/test \
  -show_entries stream=codec_name -of csv=p=0 >/dev/null 2>&1 && break; sleep 1; done

echo "== 2/4 build image =="
docker build -q -t "$IMG" . >/dev/null

echo "== 3/4 run producer (mock detector, file sink, bounded to 3 frames) =="
# Default image ships ffmpeg + capture (no torch); mock detector keeps the e2e
# light — the point is the real ffmpeg RTSP grab → metadata → file publish IN
# the image. The torchvision/openvino detectors are covered by the unit +
# integration suites.
mkdir -p "$WORK/out"
docker run --rm --name "$NAME" --network host -v "$WORK/out:/out" "$IMG" \
  python -m onvif_m rtsp://localhost:8554/test --detector mock \
  --format json --sink file --output-root /out --fps 2 --max-frames 3 \
  --log-level INFO 2>&1 | tail -5

echo "== 4/4 assert sidecars =="
N=$(find "$WORK/out" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "sidecar json files: $N"
[ "$N" -ge 1 ] && echo "ADAPTER_CONTAINER_E2E: PASS" \
  || { echo "ADAPTER_CONTAINER_E2E: FAIL (no sidecars — real ffmpeg grab/detect/publish path)"; exit 1; }
