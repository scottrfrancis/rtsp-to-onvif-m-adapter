# TODO / Backlog

The initial build order was delivered in 0.1.0 — capture, detector plugins,
publishers, health, and Docker packaging all shipped. Git history is the record
of that work. What remains is below.

**Requests welcome:** please [file a GitHub issue](https://github.com/scottrfrancis/rtsp-to-onvif-m-adapter/issues)
for bugs, feature requests, or new detector/publisher/post-processor ideas.

## Open

- **Latency profiler (`python -m onvif_m.profile`)** — read published sidecars and
  emit a per-stage latency histogram (p50 / p95 / p99), ASCII for the terminal
  plus optional JSON. Distinct from `onvif_m.bench`, which measures *live*
  detection latency rather than replaying sidecars.
  *Definition of done:* a report over 10K sidecars in <2 s.

## Future (don't block on these)

- HTTP webhook publisher
- SOAP / WS-Notification publisher (only if a real user asks)
- gRPC publisher
- Pluggable detector ecosystem (drop-in detectors via entrypoints)
- Hot config reload
- Prometheus metrics endpoint alongside `/healthz`
- **Profile on edge SBCs — Orange Pi 5 and Radxa Rock 5A** (Rockchip
  RK3588/RK3588S). Measure `onvif_m.bench` on the A76/A55 CPU cores as a baseline,
  then evaluate the on-board 6-TOPS NPU via the RKNN toolkit (needs an RKNN
  detector backend). Establishes edge-deployment feasibility and whether a stream
  can run on a fanless SBC. *Pending: boards to be set up first.*
