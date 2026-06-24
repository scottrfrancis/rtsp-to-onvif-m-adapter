"""Third-party plugin discovery via ``importlib.metadata`` entry points.

A package extends onvif_m by declaring entry points in its ``pyproject.toml``;
the CLI then accepts the registered name in ``--sink`` / ``--detector`` /
``--processor``:

    [project.entry-points."onvif_m.publishers"]
    s3 = "my_pkg:S3Publisher"

    [project.entry-points."onvif_m.detectors"]
    yolo11 = "my_pkg:Yolo11Detector"

    [project.entry-points."onvif_m.processors"]
    reid = "my_pkg:ReID"

Each entry point resolves to a zero-argument factory (a class works) returning,
respectively, a ``Publisher``, ``Detector``, or ``PostProcessor``. Plugins
configure themselves (e.g. from environment variables).
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

PUBLISHERS = "onvif_m.publishers"
DETECTORS = "onvif_m.detectors"
PROCESSORS = "onvif_m.processors"


def discover(group: str) -> dict[str, Any]:
    """Map registered plugin name → entry point for an ``onvif_m.*`` group."""
    return {ep.name: ep for ep in entry_points(group=group)}


def load_plugin(group: str, name: str) -> Any | None:
    """Load a plugin factory by registered name, or ``None`` if not registered."""
    ep = discover(group).get(name)
    return ep.load() if ep is not None else None
