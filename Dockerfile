# ONVIF Profile-M metadata producer.
# Core + MQTT + ffmpeg (RTSP capture). The torchvision detector is an extra —
# build with `--build-arg EXTRAS=detect,mqtt` (and a CPU torch index) for real
# detection; the default image is light and uses --detector mock.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

ARG EXTRAS=mqtt,capture
RUN pip install --no-cache-dir ".[${EXTRAS}]"

# Override `command:` to run the producer, e.g.
#   python -m onvif_m rtsp://host/stream --sink mqtt --mqtt-host mqtt
CMD ["python", "-m", "onvif_m", "--help"]
