"""OpenVINO / min_size integration — real conversion + inference.

Self-skips unless torch + torchvision + openvino + numpy are installed (like the
other integration suites). Small min/max keep it fast.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("openvino")

from onvif_m.detect import OpenVINODetector, create_detector  # noqa: E402


def _frame(h=240, w=320):
    return (np.random.rand(h, w, 3) * 255).astype("uint8")


def test_torchvision_min_max_reaches_the_model():
    det = create_detector(backend="torchvision", model="fasterrcnn_mobilenet_v3_large_fpn",
                          min_size=320, max_size=640, device="cpu")
    # the passthrough must reach the model's internal transform
    assert det._model.transform.min_size == (320,)
    assert det._model.transform.max_size == 640


def test_openvino_converts_and_detects():
    det = create_detector(backend="openvino", model="fasterrcnn_mobilenet_v3_large_fpn",
                          min_size=320, max_size=640, conf=0.0)
    assert isinstance(det, OpenVINODetector)
    assert det.suppress_biometrics is True
    objs = det.detect(_frame())
    assert isinstance(objs, list)  # ran end-to-end through OV IR


def test_openvino_num_threads_reaches_compiled_model():
    """num_threads must cap the compiled model's INFERENCE_NUM_THREADS — the
    multi-camera pack fix. 0 (default) leaves the LATENCY hint alone."""
    det = create_detector(backend="openvino", model="fasterrcnn_mobilenet_v3_large_fpn",
                          min_size=320, max_size=640, conf=0.0, num_threads=1)
    assert det.num_threads == 1
    det.detect(_frame())  # forces first-frame compile with the capped config
    assert det._compiled.get_property("INFERENCE_NUM_THREADS") == 1


def test_concurrent_compile_is_serialized():
    """Two detectors compiling their IR at the same instant must both succeed —
    torch.jit.trace/ov.convert_model aren't thread-safe, so _COMPILE_LOCK serializes
    them. Without the lock this raises 'dtype float64 != float32'. Regression for the
    multi-camera startup crash."""
    import threading
    frame = _frame(1080, 1920)
    results: dict[int, str] = {}

    def work(i: int) -> None:
        try:
            det = create_detector(backend="openvino",
                                  model="fasterrcnn_mobilenet_v3_large_fpn",
                                  min_size=640, max_size=1024, conf=0.25, device="cpu")
            det.detect(frame)          # concurrent first-frame compile
            results[i] = "ok"
        except Exception as e:         # noqa: BLE001 - record any failure
            results[i] = f"fail: {e}"

    threads = [threading.Thread(target=work, args=(i,)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert all(v == "ok" for v in results.values()), results


def test_openvino_and_torch_agree_on_object_count():
    """FP32 OV should be numerically ~equal to eager torch (same detections)."""
    frame = _frame()
    torch_det = create_detector(backend="torchvision",
                                model="fasterrcnn_mobilenet_v3_large_fpn",
                                min_size=320, max_size=640, conf=0.5, device="cpu")
    ov_det = create_detector(backend="openvino",
                             model="fasterrcnn_mobilenet_v3_large_fpn",
                             min_size=320, max_size=640, conf=0.5)
    assert len(ov_det.detect(frame)) == len(torch_det.detect(frame))
