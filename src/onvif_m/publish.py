"""Pluggable publishers for ONVIF metadata payloads.

A publisher takes an ``onvif-mj`` payload (``{"Frame": [...]}``) plus a
``FrameRef`` and emits it. Output has two independent axes, both multi-select:

- **format** — ``json`` and/or ``xml`` (the ``tt:MetadataStream`` projection).
- **sink**   — where it goes: ``FilePublisher`` (sidecar files), ``StdoutPublisher``
  (line-delimited), ``MqttPublisher`` (broker topic). ``MultiPublisher`` fans a
  payload out to several sinks at once.

Each sink serializes every selected format. ``paho-mqtt`` is imported lazily so
the core has no broker dependency.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .onvif_xml import to_xml_string

# Topic components must not contain MQTT separators.
_FORBIDDEN = str.maketrans({"/": "", "#": "", "+": ""})

# format → file extension and MQTT payload-prefix.
_EXTENSION = {"json": ".meta.json", "xml": ".meta.xml"}
_PAYLOAD_PREFIX = {"json": "onvif-mj", "xml": "onvif-xml"}


def serialize(payload: dict[str, Any], fmt: str, *, pretty: bool = False) -> str:
    """Render a payload as ``json`` or ``xml``."""
    if fmt == "xml":
        return to_xml_string(payload)
    if fmt == "json":
        if pretty:
            return json.dumps(payload, indent=2)
        return json.dumps(payload, separators=(",", ":"))
    raise ValueError(f"unknown format: {fmt!r}")


@dataclass
class FrameRef:
    """Co-reference context for a payload."""

    camera_id: str
    timestamp: datetime
    frame_path: Path | None = None     # file sink: write next to this path
    profile_token: str = "0"           # MQTT topic: media profile token
    module_name: str = ""              # MQTT topic: analytics module name


class Publisher(Protocol):
    def publish(self, payload: dict[str, Any], frame_ref: FrameRef) -> None: ...
    def close(self) -> None: ...


def build_mqtt_topic(
    topic_prefix: str,
    profile_token: str,
    module_name: str = "",
    payload_prefix: str = "onvif-mj",
) -> str:
    """ONVIF metadata MQTT topic:
    ``TopicPrefix/PayloadPrefix/VideoAnalytics/ProfileToken[/Module]``."""
    producer = profile_token.translate(_FORBIDDEN)
    module = module_name.translate(_FORBIDDEN)
    topic = f"{topic_prefix}/{payload_prefix}/VideoAnalytics/{producer}"
    if module:
        topic += f"/{module}"
    return topic


class StdoutPublisher:
    """Line-delimited payloads to stdout, one line per selected format."""

    def __init__(self, formats: Sequence[str] = ("json",)):
        self.formats = list(formats)

    def publish(self, payload: dict[str, Any], frame_ref: FrameRef) -> None:
        for fmt in self.formats:
            print(serialize(payload, fmt), flush=True)

    def close(self) -> None:
        pass


class FilePublisher:
    """Atomic sidecar files next to the captured frame (``frame_path``), or under
    ``output_root/<camera>/``. Writes one file per selected format
    (``.meta.json`` / ``.meta.xml``)."""

    def __init__(self, output_root: str | Path = ".", formats: Sequence[str] = ("json",)):
        self._output_root = Path(output_root)
        self.formats = list(formats)

    def publish(self, payload: dict[str, Any], frame_ref: FrameRef) -> None:
        if frame_ref.frame_path is not None:
            base = Path(frame_ref.frame_path).with_suffix("")
        else:
            stamp = frame_ref.timestamp.strftime("%Y%m%dT%H%M%S.%f")[:-3]
            base = self._output_root / frame_ref.camera_id / stamp
        base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in self.formats:
            target = base.with_name(base.name + _EXTENSION[fmt])
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(serialize(payload, fmt, pretty=True))
            os.replace(tmp, target)  # atomic

    def close(self) -> None:
        pass


class MqttPublisher:
    """Publish to an MQTT broker, one message per selected format on its own
    payload-prefix topic. ``client`` may be injected (tests); otherwise a
    paho-mqtt v2 client is built and connected with a background loop."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        topic_prefix: str = "onvif-m",
        qos: int = 0,
        retain: bool = False,
        client_id: str = "onvif-m-producer",
        formats: Sequence[str] = ("json",),
        client: Any | None = None,
    ):
        self._topic_prefix = topic_prefix
        self._qos = qos
        self._retain = retain
        self.formats = list(formats)
        if client is not None:
            self._client = client
            self._owns_loop = False
        else:
            try:
                import paho.mqtt.client as mqtt
                from paho.mqtt.enums import CallbackAPIVersion
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "paho-mqtt is required for MqttPublisher: pip install paho-mqtt"
                ) from exc
            self._client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=client_id,
            )
            self._client.connect(host, port)
            self._client.loop_start()
            self._owns_loop = True

    def publish(self, payload: dict[str, Any], frame_ref: FrameRef) -> None:
        for fmt in self.formats:
            topic = build_mqtt_topic(
                self._topic_prefix, frame_ref.profile_token, frame_ref.module_name,
                payload_prefix=_PAYLOAD_PREFIX[fmt],
            )
            self._client.publish(
                topic, serialize(payload, fmt), qos=self._qos, retain=self._retain
            )

    def close(self) -> None:
        if self._owns_loop:
            self._client.loop_stop()
            self._client.disconnect()


class MultiPublisher:
    """Fan a payload out to several sinks; closes them all."""

    def __init__(self, publishers: Iterable[Publisher]):
        self._publishers = list(publishers)

    def publish(self, payload: dict[str, Any], frame_ref: FrameRef) -> None:
        for pub in self._publishers:
            pub.publish(payload, frame_ref)

    def close(self) -> None:
        for pub in self._publishers:
            pub.close()
