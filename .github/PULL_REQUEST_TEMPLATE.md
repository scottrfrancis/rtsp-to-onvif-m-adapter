<!-- One logical change per PR. See CONTRIBUTING.md. -->

## What & why

<!-- What does this change and why? Link any related issue (Fixes #N). -->

## Checklist

- [ ] Conventional Commit title (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`)
- [ ] `ruff check .` clean
- [ ] `mypy --strict` clean
- [ ] `pytest -q` green (incl. ONVIF XSD compliance suite)
- [ ] Tests added/updated for the change
- [ ] If the metadata shape changed, `tests/test_compliance.py` still validates against the ONVIF XSD
- [ ] New detector/publisher backends go behind the existing protocols with lazy heavy-dependency imports
- [ ] No new AGPL (or otherwise copyleft) hard dependency; YOLOv8/Ultralytics stays opt-in
