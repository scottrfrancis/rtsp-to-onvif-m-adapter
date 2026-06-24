"""Capture unit tests — arg building, mock source, JPEG decode (no RTSP)."""

import io

import numpy as np
from PIL import Image

from onvif_m.capture import (
    CapturedFrame,
    MockCaptureSource,
    RtspCaptureSource,
    build_grab_args,
)


class TestBuildGrabArgs:
    def test_tcp_single_frame_to_stdout(self):
        args = build_grab_args("rtsp://cam/live", transport="tcp", timeout_s=10)
        assert args[0] == "ffmpeg"
        assert args[args.index("-rtsp_transport") + 1] == "tcp"
        assert "rtsp://cam/live" in args
        assert args[args.index("-frames:v") + 1] == "1"
        # ffmpeg timeout is microseconds
        assert args[args.index("-timeout") + 1] == str(10 * 1_000_000)
        assert args[-1] == "-"

    def test_timeout_omitted_when_zero(self):
        assert "-timeout" not in build_grab_args("rtsp://x", timeout_s=0)


class TestMockCaptureSource:
    def test_yields_count_frames(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        frames = list(MockCaptureSource(img, count=3).frames())
        assert len(frames) == 3
        assert all(isinstance(f, CapturedFrame) for f in frames)
        assert frames[0].image is img


class TestRtspDecode:
    def test_decode_jpeg_bytes_to_rgb_array(self):
        # encode a known image, decode it back through the capture path
        arr = np.dstack([
            np.full((8, 12), 200, np.uint8),
            np.zeros((8, 12), np.uint8),
            np.zeros((8, 12), np.uint8),
        ])
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        out = RtspCaptureSource._decode(buf.getvalue())
        assert out.shape == (8, 12, 3)
        assert out.dtype == np.uint8
        assert out[0, 0, 0] > 150  # red channel survived
