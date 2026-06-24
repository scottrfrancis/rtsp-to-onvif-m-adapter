"""Detector latency benchmark.

    python -m onvif_m.bench --device auto --runs 50 --size 640x480
    python -m onvif_m.bench --device cuda --model retinanet_resnet50_fpn

Reports per-frame detection latency (mean / p50 / p95) and throughput for the
configured detector + device — useful for capacity planning and comparing
accelerators (CPU vs MPS vs CUDA).
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any


def benchmark(detector: Any, image: Any, runs: int = 50, warmup: int = 5) -> dict[str, float]:
    """Time ``detector.detect(image)`` over ``runs`` (after ``warmup``)."""
    for _ in range(warmup):
        detector.detect(image)
    latencies: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        detector.detect(image)
        latencies.append((time.perf_counter() - start) * 1000.0)
    latencies.sort()
    mean = statistics.mean(latencies)
    return {
        "runs": float(runs),
        "mean_ms": mean,
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "fps": 1000.0 / mean if mean > 0 else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="onvif_m.bench", description=__doc__)
    p.add_argument("--detector", default="torchvision")
    p.add_argument("--model", default="ssdlite320_mobilenet_v3_large")
    p.add_argument("--device", default="auto")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--size", default="640x480", help="WxH of the synthetic frame")
    args = p.parse_args(argv)

    import numpy as np

    from .detect import create_detector

    w, h = (int(x) for x in args.size.lower().split("x"))
    image = np.zeros((h, w, 3), dtype=np.uint8)
    detector = create_detector(backend=args.detector, model=args.model, device=args.device)
    device = getattr(detector, "device", "n/a")

    s = benchmark(detector, image, runs=args.runs)
    print(f"detector={args.detector} model={args.model} device={device} size={args.size}")
    print(f"  mean={s['mean_ms']:.1f}ms  p50={s['p50_ms']:.1f}ms  "
          f"p95={s['p95_ms']:.1f}ms  ~{s['fps']:.1f} fps  (runs={int(s['runs'])})")


if __name__ == "__main__":
    main()
