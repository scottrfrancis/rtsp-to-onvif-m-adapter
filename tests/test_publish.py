"""Publisher unit tests — no broker, no network (MQTT uses an injected client)."""

import json
from datetime import UTC, datetime
from pathlib import Path

from onvif_m.publish import (
    FilePublisher,
    FrameRef,
    MqttPublisher,
    MultiPublisher,
    StdoutPublisher,
    build_mqtt_topic,
)

PAYLOAD = {"Frame": [{"@UtcTime": "2021-10-05T15:13:27.321Z", "@Source": "c"}]}


def _ref(**kw) -> FrameRef:
    base = dict(camera_id="cam-7", timestamp=datetime(2021, 10, 5, 15, 13, 27, 321000, tzinfo=UTC))
    base.update(kw)
    return FrameRef(**base)


class TestMqttTopic:
    def test_onvif_topic_structure(self):
        # Analytics Spec §5.4.2 example: MyDevice/onvif-mj/VideoAnalytics/1/MyClassifier
        assert build_mqtt_topic("MyDevice", "1", "MyClassifier") == \
            "MyDevice/onvif-mj/VideoAnalytics/1/MyClassifier"

    def test_topic_without_module(self):
        assert build_mqtt_topic("dev", "0") == "dev/onvif-mj/VideoAnalytics/0"

    def test_topic_strips_mqtt_separators(self):
        # '/', '#', '+' must not appear in topic components
        assert build_mqtt_topic("dev", "pro/file", "mod#ule+") == \
            "dev/onvif-mj/VideoAnalytics/profile/module"


class TestStdoutPublisher:
    def test_emits_one_json_line(self, capsys):
        StdoutPublisher().publish(PAYLOAD, _ref())
        out = capsys.readouterr().out.strip()
        assert json.loads(out) == PAYLOAD
        assert "\n" not in out  # single line


class TestFilePublisher:
    def test_sidecar_next_to_frame(self, tmp_path: Path, json_schema):
        frame = tmp_path / "cam-7" / "2021-10-05" / "T151327.321.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"jpg")

        FilePublisher().publish(PAYLOAD, _ref(frame_path=frame))

        sidecar = frame.with_suffix(".meta.json")
        on_disk = json.loads(sidecar.read_text())
        assert on_disk == PAYLOAD
        json_schema.validate(on_disk)
        # atomic: no leftover temp file
        assert not list(frame.parent.glob("*.tmp"))

    def test_output_root_when_no_frame_path(self, tmp_path: Path, json_schema):
        FilePublisher(output_root=tmp_path).publish(PAYLOAD, _ref())
        files = list((tmp_path / "cam-7").glob("*.meta.json"))
        assert len(files) == 1
        on_disk = json.loads(files[0].read_text())
        assert on_disk == PAYLOAD
        json_schema.validate(on_disk)


class TestMqttPublisherWithInjectedClient:
    def test_publishes_payload_on_onvif_topic(self):
        calls = []

        class FakeClient:
            def publish(self, topic, payload, qos=0, retain=False):
                calls.append((topic, payload, qos, retain))

        pub = MqttPublisher(topic_prefix="MyDevice", qos=1, retain=True, client=FakeClient())
        pub.publish(PAYLOAD, _ref(profile_token="1", module_name="MyClassifier"))
        pub.close()  # injected client: no-op loop

        assert len(calls) == 1
        topic, payload, qos, retain = calls[0]
        assert topic == "MyDevice/onvif-mj/VideoAnalytics/1/MyClassifier"
        assert json.loads(payload) == PAYLOAD
        assert qos == 1 and retain is True

    def test_both_formats_publish_on_distinct_topics(self):
        calls = []

        class FakeClient:
            def publish(self, topic, payload, qos=0, retain=False):
                calls.append((topic, payload))

        pub = MqttPublisher(topic_prefix="dev", formats=["json", "xml"], client=FakeClient())
        pub.publish(PAYLOAD, _ref(profile_token="1"))
        pub.close()

        topics = [t for t, _ in calls]
        assert "dev/onvif-mj/VideoAnalytics/1" in topics
        assert "dev/onvif-xml/VideoAnalytics/1" in topics
        xml_payload = next(p for t, p in calls if "onvif-xml" in t)
        assert xml_payload.startswith("<")  # XML, not JSON


class TestFormats:
    def test_file_writes_both_json_and_xml(self, tmp_path: Path, json_schema):
        FilePublisher(output_root=tmp_path, formats=["json", "xml"]).publish(PAYLOAD, _ref())
        d = tmp_path / "cam-7"
        jsons = list(d.glob("*.meta.json"))
        assert len(jsons) == 1
        json_schema.validate(json.loads(jsons[0].read_text()))
        xmls = list(d.glob("*.meta.xml"))
        assert len(xmls) == 1
        assert xmls[0].read_text().startswith("<")

    def test_stdout_xml_only(self, capsys):
        StdoutPublisher(formats=["xml"]).publish(PAYLOAD, _ref())
        out = capsys.readouterr().out.strip()
        assert out.startswith("<") and "Frame" in out


class TestMultiPublisher:
    def test_fans_out_to_all_and_closes_all(self, tmp_path: Path, capsys):
        closed = []

        class Spy:
            def publish(self, payload, frame_ref):
                closed.append("published")

            def close(self):
                closed.append("closed")

        multi = MultiPublisher([FilePublisher(output_root=tmp_path), Spy()])
        multi.publish(PAYLOAD, _ref())
        multi.close()

        assert list((tmp_path / "cam-7").glob("*.meta.json"))  # file sink ran
        assert "published" in closed and "closed" in closed  # spy sink ran + closed
