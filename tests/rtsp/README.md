# RTSP capture integration tests

`RtspCaptureSource` is unit-tested for arg-building and JPEG decode (no stream)
in `tests/test_capture.py`. The end-to-end test in `tests/test_rtsp_integration.py`
captures from a **real RTSP server** and runs the full pipeline.

The server is **not committed** — spin one up on demand:

```bash
# 1. Serve a synthetic looping stream (Docker mediamtx + ffmpeg publisher)
bash tests/rtsp/run-rtsp.sh                 # rtsp://localhost:8554/test

# 2. Run the integration test
pip install -e ".[detect,compliance]"
pytest tests/test_rtsp_integration.py -v

# 3. Stop it
bash tests/rtsp/run-rtsp.sh stop
```

The test **self-skips** if nothing is listening on `localhost:8554`, so the
default `pytest` run stays green without a server. Set `RTSP_PORT` to override.

Server: [mediamtx](https://github.com/bluenviron/mediamtx) (Docker
`bluenviron/mediamtx`); the synthetic stream is `ffmpeg` lavfi `testsrc`.
