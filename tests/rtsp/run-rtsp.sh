#!/usr/bin/env bash
# Start a canned RTSP server for the capture integration tests.
#
# NOT vendored — spins up mediamtx (Docker) and publishes a synthetic looping
# test stream into it, so RtspCaptureSource has a real rtsp:// to read.
#
#   bash tests/rtsp/run-rtsp.sh          # serve rtsp://localhost:8554/test
#   bash tests/rtsp/run-rtsp.sh stop
#
# Then:  pytest tests/test_rtsp_integration.py
set -euo pipefail

NAME="onvif-m-rtsp"
PORT="${RTSP_PORT:-8554}"
URL="rtsp://localhost:${PORT}/test"

if [[ "${1:-start}" == "stop" ]]; then
  docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "no $NAME running"
  pkill -f "rtsp://localhost:${PORT}/test" 2>/dev/null || true
  exit 0
fi

if ! docker info >/dev/null 2>&1; then
  echo "Need Docker (mediamtx). Start colima/Docker Desktop." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Need ffmpeg to publish the synthetic stream." >&2
  exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm -d --name "$NAME" -p "${PORT}:8554" bluenviron/mediamtx:latest >/dev/null
sleep 2

# Publish a synthetic, looping test pattern into the server (background).
ffmpeg -re -stream_loop -1 -f lavfi -i "testsrc=size=320x240:rate=10" \
  -c:v libx264 -pix_fmt yuv420p -g 10 -f rtsp -rtsp_transport tcp \
  "$URL" >/dev/null 2>&1 &

echo "mediamtx on :${PORT}, publishing ${URL} (publisher pid $!)  —  stop: bash $0 stop"
