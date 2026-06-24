# Security Policy

## Supported versions

This project is pre-1.0. Security fixes are applied to the latest released
version on the `main` branch only.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's [Security Advisories](https://github.com/scottrfrancis/rtsp-to-onvif-m-adapter/security/advisories/new),
or by email to **scott.russell.francis@gmail.com**.

Please include: a description of the issue, steps to reproduce, affected
version/commit, and any suggested remediation. You can expect an initial
response within a few days. Once a fix is available, we will coordinate
disclosure and credit the reporter unless anonymity is requested.

## Scope notes

This component is a metadata **producer** intended for trusted-network
deployments. It does not implement WS-Security or transport authentication —
those are the responsibility of the network boundary and of transport-layer
publishers (MQTT TLS, HTTP auth). Reports about missing auth on the producer
itself are by-design (see ARCHITECTURE.md), but reports about the producer
mishandling untrusted input (malformed RTSP streams, crafted detector output,
path traversal in the file publisher, resource exhaustion) are in scope.
