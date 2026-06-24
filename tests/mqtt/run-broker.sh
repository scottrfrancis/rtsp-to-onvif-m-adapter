#!/usr/bin/env bash
# Start a local MQTT broker for the MQTT integration tests.
#
# The broker is NOT vendored into the repo — this script spins one up on demand
# (anonymous, localhost only). Prefers Docker (eclipse-mosquitto); falls back to
# a locally-installed mosquitto.
#
#   bash tests/mqtt/run-broker.sh          # start on :1883
#   bash tests/mqtt/run-broker.sh stop     # stop the docker broker
#
# Then:  pytest tests/test_mqtt_integration.py
set -euo pipefail

PORT="${MQTT_PORT:-1883}"
NAME="onvif-m-mqtt"

if [[ "${1:-start}" == "stop" ]]; then
  docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "no $NAME running"
  exit 0
fi

if docker info >/dev/null 2>&1; then
  # Write the anonymous config INSIDE the container (eclipse-mosquitto 2.x needs
  # an explicit listener). Avoids host bind-mounts, which fail through VM-backed
  # Docker (colima/Lima) and vary across CI.
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run --rm -d --name "$NAME" -p "${PORT}:${PORT}" eclipse-mosquitto:2 \
    sh -c "printf 'listener ${PORT} 0.0.0.0\nallow_anonymous true\n' > /mosquitto/config/mosquitto.conf && exec mosquitto -c /mosquitto/config/mosquitto.conf" >/dev/null
  echo "mosquitto (docker) listening on :${PORT}  —  stop: bash $0 stop"
elif command -v mosquitto >/dev/null 2>&1; then
  # mosquitto 2.x denies anonymous by default — give it an explicit listener.
  CONF="$(mktemp -t mosquitto.XXXXXX.conf)"
  printf 'listener %s\nallow_anonymous true\n' "$PORT" > "$CONF"
  mosquitto -c "$CONF" >/dev/null 2>&1 &
  echo "mosquitto (local) listening on :${PORT}  pid $!  —  stop: kill $!"
else
  echo "Need Docker or mosquitto. Install Docker, or: brew install mosquitto" >&2
  exit 1
fi
