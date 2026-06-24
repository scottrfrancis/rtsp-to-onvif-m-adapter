"""Example post-processors for the extension hook.

A post-processor runs after detection and before the metadata is built. It
implements ``onvif_m.pipeline.PostProcessor``:

    def process(self, objects: list[DetectedObject],
                frame: CapturedFrame) -> list[DetectedObject]: ...

Wire one in from the CLI by dotted ``module:factory`` path (the factory is called
with no arguments and returns the processor). This file lives in the source tree
(it is not in the installed wheel), so from a checkout you can reference it
directly; from a PyPI install, copy it to your own module and reference that:

    # from a source checkout
    python -m onvif_m rtsp://host/stream --processor examples.processors:HumansOnly

    # from a PyPI install, with your own my_hooks.py on the path
    python -m onvif_m rtsp://host/stream --processor my_hooks:HumansOnly

A packaged plugin can instead register an ``onvif_m.processors`` entry point and
be used by name (``--processor <name>``). Or wire from the library:

    run_camera(camera, source, detector, publisher,
               processors=[HumansOnly(), MyReID()])

These are illustrative only — real ReID, tracking, histogram tagging, or face
blurring are the user's to implement.
"""

from __future__ import annotations

from onvif_m.model import DetectedObject


class Passthrough:
    """Returns the objects unchanged. The minimal valid processor."""

    def process(self, objects: list[DetectedObject], frame: object) -> list[DetectedObject]:
        return objects


def passthrough() -> Passthrough:
    """Factory referenced as ``examples.processors:passthrough``."""
    return Passthrough()


class HumansOnly:
    """Drop every object whose top class is not ``Human`` — a trivial filter and
    a template for richer hooks (ReID, attribute tagging, …)."""

    def process(self, objects: list[DetectedObject], frame: object) -> list[DetectedObject]:
        return [o for o in objects if o.classes and o.classes[0].type == "Human"]
