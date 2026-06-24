# MQTT integration tests

The MQTT publisher is unit-tested with an injected client (no broker) in
`tests/test_publish.py`. The end-to-end test in `tests/test_mqtt_integration.py`
publishes through a **real broker** and asserts a subscriber receives the payload
on the ONVIF-conformant topic.

The broker is **not committed** — spin one up on demand:

```bash
# 1. Start a broker (Docker eclipse-mosquitto, or local mosquitto)
bash tests/mqtt/run-broker.sh

# 2. Run the integration test
pip install -e ".[mqtt,dev]"
pytest tests/test_mqtt_integration.py -v

# 3. Stop the broker
bash tests/mqtt/run-broker.sh stop
```

The test **self-skips** if nothing is listening on `localhost:1883`, so the
default `pytest` run stays green without a broker. Set `MQTT_PORT` to use a
different port (update the test's `PORT` accordingly).

Client: [`paho-mqtt`](https://pypi.org/project/paho-mqtt/) (2.x). Broker:
[eclipse-mosquitto](https://hub.docker.com/_/eclipse-mosquitto) `:2`, started with
a minimal anonymous-localhost config.

Topic structure (ONVIF Analytics Service Spec §5.4.2):
`<TopicPrefix>/onvif-mj/VideoAnalytics/<ProfileToken>[/<AnalyticsModule>]`.
