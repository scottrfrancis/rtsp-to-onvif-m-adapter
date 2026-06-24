"""CLI parsing + publisher selection (no run)."""

from pathlib import Path

import pytest

from onvif_m.__main__ import SINKS, _build_parser, make_publisher, resolve_multi
from onvif_m.publish import FilePublisher, MultiPublisher, StdoutPublisher


def _args(*extra: str):
    return _build_parser().parse_args(["rtsp://host/stream", *extra])


def test_parser_defaults():
    args = _args()
    assert args.url == "rtsp://host/stream"
    assert args.name == "cam"
    assert args.detector == "torchvision"
    assert args.fps == 1.0
    assert args.transport == "tcp"
    # format/sink default to None on the namespace; resolved later
    assert resolve_multi(args.format, "ONVIF_M_FORMAT", ["json"], {"json", "xml"}) == ["json"]
    assert resolve_multi(args.sink, "ONVIF_M_SINK", ["file"], SINKS) == ["file"]


class TestResolveMulti:
    def test_cli_repeatable(self):
        assert resolve_multi(["json", "xml"], "X", ["json"], {"json", "xml"}) == ["json", "xml"]

    def test_cli_comma_separated(self):
        assert resolve_multi(["json,xml"], "X", ["json"], {"json", "xml"}) == ["json", "xml"]

    def test_dedupes_preserving_order(self):
        assert resolve_multi(["xml", "xml", "json"], "X", ["j"], {"json", "xml"}) == ["xml", "json"]

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("ONVIF_M_FORMAT", "xml,json")
        assert resolve_multi(None, "ONVIF_M_FORMAT", ["json"], {"json", "xml"}) == ["xml", "json"]

    def test_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ONVIF_M_FORMAT", "xml")
        assert resolve_multi(["json"], "ONVIF_M_FORMAT", ["json"], {"json", "xml"}) == ["json"]

    def test_default_when_unset(self):
        assert resolve_multi(None, "ONVIF_M_FORMAT", ["json"], {"json", "xml"}) == ["json"]

    def test_invalid_value_rejected(self):
        with pytest.raises(SystemExit):
            resolve_multi(["yaml"], "X", ["json"], {"json", "xml"})


class TestMakePublisher:
    def test_single_file_sink(self, tmp_path: Path):
        args = _args("--output-root", str(tmp_path))
        pub = make_publisher(["file"], ["json"], args)
        assert isinstance(pub, FilePublisher)

    def test_single_stdout_sink(self):
        pub = make_publisher(["stdout"], ["json"], _args())
        assert isinstance(pub, StdoutPublisher)

    def test_multiple_sinks_wrapped(self, tmp_path: Path):
        args = _args("--output-root", str(tmp_path))
        pub = make_publisher(["file", "stdout"], ["json"], args)
        assert isinstance(pub, MultiPublisher)

    def test_formats_threaded_to_sink(self, tmp_path: Path):
        args = _args("--output-root", str(tmp_path))
        pub = make_publisher(["file"], ["json", "xml"], args)
        assert isinstance(pub, FilePublisher)
        assert list(pub.formats) == ["json", "xml"]


def test_url_from_env_when_positional_omitted(monkeypatch):
    monkeypatch.setenv("ONVIF_M_URL", "rtsp://env/stream")
    args = _build_parser().parse_args([])
    assert (args.url or None) is None  # positional omitted
    # __main__.main resolves the env fallback; the parser allows omission
    from onvif_m.__main__ import resolve_url
    assert resolve_url(args.url) == "rtsp://env/stream"


def test_mqtt_publisher_selected(monkeypatch):
    # stub MqttPublisher so selection is tested without a live broker
    built = {}

    def fake_mqtt(host, port, topic_prefix="onvif-m", formats=("json",)):
        built["formats"] = list(formats)
        return "MQTT"

    monkeypatch.setattr("onvif_m.__main__.MqttPublisher", fake_mqtt)
    pub = make_publisher(["mqtt"], ["json", "xml"], _args("--mqtt-host", "localhost"))
    assert pub == "MQTT"
    assert built["formats"] == ["json", "xml"]


class TestLoadProcessors:
    def test_none_yields_empty(self):
        from onvif_m.__main__ import load_processors
        assert load_processors(None) == []

    def test_loads_factory_by_dotted_path(self, tmp_path, monkeypatch):
        # a user module on sys.path, referenced as module:factory (cwd-independent)
        (tmp_path / "onvif_userproc.py").write_text(
            "class P:\n"
            "    def process(self, objects, frame):\n"
            "        return objects\n"
            "def make():\n"
            "    return P()\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        from onvif_m.__main__ import load_processors
        procs = load_processors(["onvif_userproc:make"])
        assert len(procs) == 1
        assert hasattr(procs[0], "process")

    def test_bad_spec_errors(self):
        from onvif_m.__main__ import load_processors
        with pytest.raises(SystemExit):
            load_processors(["no_colon_here"])
