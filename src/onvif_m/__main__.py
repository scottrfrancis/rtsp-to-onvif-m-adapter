"""CLI: run the producer over a single RTSP stream.

    python -m onvif_m rtsp://host/stream --detector torchvision --sink file --output-root ./out
    python -m onvif_m rtsp://host/stream --detector mock --sink stdout --format json,xml
    python -m onvif_m rtsp://host/stream --sink mqtt --mqtt-host broker.local

Capture (1 fps) -> detect -> optional post-processors -> ONVIF onvif-mj payload
-> sink(s). ``--format`` (json/xml) and ``--sink`` (file/stdout/mqtt) are each
repeatable and may also be set via ONVIF_M_FORMAT / ONVIF_M_SINK. SIGINT/SIGTERM
stops cleanly. One stream per process; run several to scale out.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import signal

from . import plugins
from .capture import RtspCaptureSource
from .detect import BUILTIN_DETECTORS, create_detector
from .pipeline import Camera, PostProcessor, run_camera
from .publish import FilePublisher, MqttPublisher, MultiPublisher, Publisher, StdoutPublisher

logger = logging.getLogger(__name__)

FORMATS = {"json", "xml"}
SINKS = {"file", "stdout", "mqtt"}


def sink_choices() -> set[str]:
    """Built-in sinks plus any registered via the ``onvif_m.publishers`` group."""
    return SINKS | set(plugins.discover(plugins.PUBLISHERS))


def detector_choices() -> list[str]:
    """Built-in detectors plus any registered via the ``onvif_m.detectors`` group."""
    return sorted(set(BUILTIN_DETECTORS) | set(plugins.discover(plugins.DETECTORS)))


def _env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="onvif_m", description=__doc__)
    p.add_argument("url", nargs="?", default=None, help="RTSP URL (or ONVIF_M_URL)")
    p.add_argument("--name", default=_env("ONVIF_M_NAME", "cam"),
                   help="stream id: ONVIF @Source and output subdir (default: cam)")
    p.add_argument("--profile-token", default=_env("ONVIF_M_PROFILE_TOKEN", "0"),
                   help="ONVIF media profile token for the MQTT topic (default: 0)")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--transport", choices=["tcp", "udp"], default="tcp")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")

    p.add_argument("--detector", choices=detector_choices(), default="torchvision",
                   help="built-in or a registered onvif_m.detectors plugin")
    p.add_argument("--model", default="ssdlite320_mobilenet_v3_large")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--processor", action="append",
                   help="post-processor 'module:factory' or a registered onvif_m.processors "
                        "name (repeatable; ONVIF_M_PROCESSORS)")

    p.add_argument("--format", action="append", help="json and/or xml (repeatable; ONVIF_M_FORMAT)")
    p.add_argument("--sink", action="append",
                   help="file, stdout, mqtt, or a registered onvif_m.publishers plugin "
                        "(repeatable; ONVIF_M_SINK)")
    p.add_argument("--output-root", default=_env("ONVIF_M_OUTPUT_ROOT", "."),
                   help="file sink output dir")
    p.add_argument("--mqtt-host", default=_env("ONVIF_M_MQTT_HOST", "localhost"))
    p.add_argument("--mqtt-port", type=int, default=int(_env("ONVIF_M_MQTT_PORT", "1883") or 1883))
    p.add_argument("--mqtt-topic-prefix", default=_env("ONVIF_M_MQTT_TOPIC_PREFIX", "onvif-m"))
    p.add_argument("--health-port", type=int, default=0, help="serve /healthz (0=off)")
    p.add_argument("--log-level", default="INFO")
    return p


def resolve_multi(
    cli_values: list[str] | None, env_key: str, default: list[str], choices: set[str]
) -> list[str]:
    """Resolve a repeatable multi-value option. CLI wins; else env (comma/space
    separated); else default. Values are flattened, validated, and de-duplicated
    in first-seen order."""
    raw = cli_values if cli_values else (_split(os.environ.get(env_key)) or default)
    out: list[str] = []
    for value in raw:
        for part in value.replace(",", " ").split():
            if part not in choices:
                raise SystemExit(f"invalid value {part!r}; choose from {sorted(choices)}")
            if part not in out:
                out.append(part)
    return out


def _split(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [p for p in raw.replace(",", " ").split() if p]


def resolve_url(cli_url: str | None) -> str:
    url = cli_url or _env("ONVIF_M_URL")
    if not url:
        raise SystemExit("an RTSP URL is required (positional arg or ONVIF_M_URL)")
    return url


def load_processors(specs: list[str] | None) -> list[PostProcessor]:
    """Import ``module:factory`` specs into PostProcessor instances. The factory
    is called with no arguments and must return an object with a ``process``
    method (a class works). None / empty → no processors."""
    processors: list[PostProcessor] = []
    for spec in specs or []:
        if ":" in spec:
            module_name, _, attr = spec.partition(":")
            if not module_name or not attr:
                raise SystemExit(f"bad --processor spec {spec!r}; expected 'module:factory'")
            factory = getattr(importlib.import_module(module_name), attr)
        else:
            factory = plugins.load_plugin(plugins.PROCESSORS, spec)
            if factory is None:
                raise SystemExit(
                    f"unknown processor {spec!r}; use 'module:factory' or a registered "
                    "onvif_m.processors plugin name"
                )
        processors.append(factory())
    return processors


def make_publisher(sinks: list[str], formats: list[str], args: argparse.Namespace) -> Publisher:
    """Build one publisher per sink (each emitting every selected format), wrapped
    in a MultiPublisher when more than one sink is selected."""
    built: list[Publisher] = []
    for sink in sinks:
        if sink == "stdout":
            built.append(StdoutPublisher(formats=formats))
        elif sink == "mqtt":
            built.append(MqttPublisher(
                args.mqtt_host, args.mqtt_port,
                topic_prefix=args.mqtt_topic_prefix, formats=formats,
            ))
        elif sink == "file":
            built.append(FilePublisher(args.output_root, formats=formats))
        else:
            factory = plugins.load_plugin(plugins.PUBLISHERS, sink)
            if factory is None:
                raise SystemExit(
                    f"unknown sink {sink!r}; use file/stdout/mqtt or a registered "
                    "onvif_m.publishers plugin name"
                )
            built.append(factory())
    return built[0] if len(built) == 1 else MultiPublisher(built)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    url = resolve_url(args.url)
    formats = resolve_multi(args.format, "ONVIF_M_FORMAT", ["json"], FORMATS)
    sinks = resolve_multi(args.sink, "ONVIF_M_SINK", ["file"], sink_choices())
    processors = load_processors(args.processor or _split(os.environ.get("ONVIF_M_PROCESSORS")))

    detector = create_detector(backend=args.detector, model=args.model, conf=args.conf,
                               device=args.device)
    publisher = make_publisher(sinks, formats, args)
    camera = Camera(args.name, profile_token=args.profile_token)
    logger.info("producer: stream=%s detector=%s format=%s sink=%s processors=%d fps=%s",
                args.name, args.detector, formats, sinks, len(processors), args.fps)

    health = None
    if args.health_port > 0:
        from .health import HealthRegistry, serve_health
        health = HealthRegistry(stale_after=max(30.0, 3.0 / max(args.fps, 0.1)))
        serve_health(health, args.health_port)
        logger.info("health endpoint on :%d/healthz", args.health_port)

    source = RtspCaptureSource(url, fps=args.fps, transport=args.transport)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: source.close())

    try:
        run_camera(camera, source, detector, publisher, module=args.detector,
                   max_frames=args.max_frames, health=health, processors=processors)
    finally:
        source.close()
        publisher.close()
    logger.info("clean shutdown")


if __name__ == "__main__":
    main()
