# Releasing to PyPI

This project publishes to PyPI automatically when you push a version tag (`v*`),
using **Trusted Publishing** — PyPI's OIDC integration with GitHub Actions. No
API tokens, no secrets: PyPI verifies the release came from this exact repo +
workflow + environment. Workflow:
[`.github/workflows/release.yml`](../.github/workflows/release.yml).

> Publishing is already configured for this repo (trusted publisher registered on
> PyPI; a `pypi` environment restricted to `v*` tags). Day to day you only need
> **Cutting a release** below. The configuration reference at the end is for a new
> maintainer, a fork publishing under its own name, or re-creating the setup.

---

## Cutting a release

1. **Bump the version** in [`pyproject.toml`](../pyproject.toml) (`[project].version`).
2. **Update [`CHANGELOG.md`](../CHANGELOG.md)** — set the date on the new version section.
3. Commit those changes.
4. **Tag and push** (the tag must be `v` + the version):

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

5. The `release` workflow builds an sdist + wheel, runs `twine check`, and
   publishes to PyPI. Watch it under the repo's **Actions** tab. If a required
   reviewer is configured, approve the `pypi` environment when prompted.
6. Verify: `pip install onvif-m-producer` from a clean environment.

The version in `pyproject.toml` and the git tag must agree (`0.1.0` ↔ `v0.1.0`);
PyPI rejects re-uploading an existing version, so bump for every release.

### Rehearse on TestPyPI (optional)

Register a second pending publisher on [TestPyPI](https://test.pypi.org) (same
fields as below) and temporarily add `repository-url:
https://test.pypi.org/legacy/` to the `pypa/gh-action-pypi-publish` step. Revert
before the real release.

---

## Configuration reference

Already done for this repo. Recreate this only as a new maintainer, or in a fork
publishing under its own name (substitute your own PyPI project name and
`owner/repo`).

**PyPI trusted publisher** — Account settings → Publishing → add a GitHub
publisher:

| Field | Value |
|---|---|
| PyPI Project Name | `onvif-m-producer` |
| Owner | `scottrfrancis` |
| Repository name | `rtsp-to-onvif-m-adapter` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

(For a brand-new project name, this registers a **pending publisher** that
creates the project on first successful upload.)

**GitHub environment** — Settings → Environments → `pypi`:

- **Deployment branches and tags** → *Selected branches and tags* → add `v*`, so
  only version tags can publish. (Works on any plan.)
- **Deployment protection rules** → *Required reviewers* (optional manual
  approval). This section appears only on **public** repos, or private repos on
  **GitHub Pro/Team/Enterprise**.

Trusted Publishing's security comes from the OIDC identity match (repo +
`release.yml` + `pypi` environment); the required reviewer is an extra gate, not
a requirement.
