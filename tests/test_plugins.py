"""Entry-point plugin discovery + CLI integration.

Third-party packages register publishers / detectors / processors via
`importlib.metadata` entry points; the CLI then accepts them by name. These tests
fake the entry points (no real package install needed).
"""

import pytest

from onvif_m import plugins


class _EP:
    """Minimal stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def _fake_entry_points(mapping):
    def inner(group=None):
        return list(mapping.get(group, []))
    return inner


def test_discover_lists_registered_names(monkeypatch):
    monkeypatch.setattr(plugins, "entry_points",
                        _fake_entry_points({plugins.PUBLISHERS: [_EP("s3", object)]}))
    assert set(plugins.discover(plugins.PUBLISHERS)) == {"s3"}


def test_load_plugin_returns_factory(monkeypatch):
    def factory():
        return "PUB"
    monkeypatch.setattr(plugins, "entry_points",
                        _fake_entry_points({plugins.PUBLISHERS: [_EP("s3", factory)]}))
    assert plugins.load_plugin(plugins.PUBLISHERS, "s3") is factory


def test_load_plugin_missing_returns_none(monkeypatch):
    monkeypatch.setattr(plugins, "entry_points", _fake_entry_points({}))
    assert plugins.load_plugin(plugins.PUBLISHERS, "nope") is None


class TestCliIntegration:
    def _args(self, *extra):
        from onvif_m.__main__ import _build_parser
        return _build_parser().parse_args(["rtsp://host/stream", *extra])

    def test_sink_choices_include_plugins(self, monkeypatch):
        from onvif_m.__main__ import sink_choices
        monkeypatch.setattr(plugins, "entry_points",
                            _fake_entry_points({plugins.PUBLISHERS: [_EP("s3", object)]}))
        choices = sink_choices()
        assert "s3" in choices and {"file", "stdout", "mqtt"} <= choices

    def test_make_publisher_uses_plugin_sink(self, monkeypatch):
        class FakePub:
            def publish(self, payload, frame_ref): ...
            def close(self): ...
        monkeypatch.setattr(plugins, "entry_points",
                            _fake_entry_points({plugins.PUBLISHERS: [_EP("s3", FakePub)]}))
        from onvif_m.__main__ import make_publisher
        pub = make_publisher(["s3"], ["json"], self._args())
        assert isinstance(pub, FakePub)

    def test_make_publisher_unknown_sink_errors(self, monkeypatch):
        monkeypatch.setattr(plugins, "entry_points", _fake_entry_points({}))
        from onvif_m.__main__ import make_publisher
        with pytest.raises(SystemExit):
            make_publisher(["nope"], ["json"], self._args())

    def test_processor_by_registered_name(self, monkeypatch):
        class FakeProc:
            def process(self, objects, frame):
                return objects
        monkeypatch.setattr(plugins, "entry_points",
                            _fake_entry_points({plugins.PROCESSORS: [_EP("reid", FakeProc)]}))
        from onvif_m.__main__ import load_processors
        procs = load_processors(["reid"])
        assert len(procs) == 1 and hasattr(procs[0], "process")

    def test_unknown_processor_name_errors(self, monkeypatch):
        monkeypatch.setattr(plugins, "entry_points", _fake_entry_points({}))
        from onvif_m.__main__ import load_processors
        with pytest.raises(SystemExit):
            load_processors(["not_registered"])

    def test_create_detector_plugin_backend(self, monkeypatch):
        class FakeDet:
            def detect(self, frame):
                return []

            @property
            def suppress_biometrics(self):
                return True
        monkeypatch.setattr(plugins, "entry_points",
                            _fake_entry_points({plugins.DETECTORS: [_EP("custom", FakeDet)]}))
        from onvif_m.detect import create_detector
        det = create_detector(backend="custom")
        assert isinstance(det, FakeDet)
