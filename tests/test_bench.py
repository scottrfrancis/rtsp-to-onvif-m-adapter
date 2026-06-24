"""Benchmark utility unit test (mock detector — no model)."""

from onvif_m.bench import benchmark
from onvif_m.detect import MockDetector


def test_benchmark_returns_latency_stats():
    stats = benchmark(MockDetector(objects=[]), object(), runs=5, warmup=1)
    assert set(stats) >= {"runs", "mean_ms", "p50_ms", "p95_ms", "fps"}
    assert stats["runs"] == 5
    assert stats["mean_ms"] >= 0.0
    assert stats["fps"] >= 0.0
