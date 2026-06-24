"""MQTT integration: publish via MqttPublisher → a real broker → subscriber.

Requires a broker on localhost:1883 — start one with:
    bash tests/mqtt/run-broker.sh
The test self-skips if no broker is reachable (see tests/mqtt/README.md).
"""

import json
import queue
import socket
from datetime import UTC, datetime

import pytest

mqtt = pytest.importorskip("paho.mqtt.client")

from onvif_m.publish import FrameRef, MqttPublisher  # noqa: E402  (after importorskip)

HOST, PORT = "localhost", 1883


def _broker_up() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _broker_up(),
    reason="no MQTT broker on localhost:1883 (tests/mqtt/run-broker.sh)",
)


def test_publish_roundtrips_through_broker():
    received: queue.Queue = queue.Queue()

    sub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="onvif-m-test-sub")
    sub.on_message = lambda c, u, msg: received.put((msg.topic, msg.payload))
    sub.connect(HOST, PORT)
    sub.subscribe("itest/onvif-mj/#")
    sub.loop_start()

    payload = {"Frame": [{"@UtcTime": "2021-10-05T15:13:27.321Z", "@Source": "cam-7"}]}
    pub = MqttPublisher(HOST, PORT, topic_prefix="itest", qos=1, client_id="onvif-m-test-pub")
    try:
        pub.publish(payload, FrameRef("cam-7", datetime.now(UTC),
                                      profile_token="1", module_name="yolo"))
        topic, data = received.get(timeout=5)
    finally:
        pub.close()
        sub.loop_stop()
        sub.disconnect()

    assert topic == "itest/onvif-mj/VideoAnalytics/1/yolo"
    assert json.loads(data) == payload
