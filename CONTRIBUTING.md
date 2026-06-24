# Contributing

This project is a vendor-neutral **ONVIF Profile-M metadata producer**. Setup,
tests, and benchmarking live in [docs/development.md](docs/development.md).

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). To report
a security issue, see [SECURITY.md](SECURITY.md) — please don't open a public issue.
Optionally run the same checks as CI locally with `pre-commit install`.

## Ground rules

- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `chore:`,
  `refactor:`).
- `ruff check .` and `mypy --strict` must pass; `pytest -q` green.
- **ONVIF conformance is non-negotiable.** Changes to the metadata shape must
  keep `tests/test_compliance.py` green — it validates against the official
  ONVIF `metadatastream.xsd`, not our own example.
- **Plugins, not forks.** Add detector/publisher backends behind the existing
  `Detector` / `Publisher` protocols; import heavy deps (torch, paho) lazily so
  the core stays light.
- **Licensing.** The core is Apache-2.0 and stays AGPL-free: Ultralytics YOLOv8
  is opt-in only, never a default or a hard dependency.

## Pull requests

- One logical change per PR, with tests.
- CI runs lint + types + the suite on Python 3.11 and 3.12.
- Integration tests self-skip without their server/fixture; if you touch them,
  verify locally with the scripted servers (see the dev docs).
