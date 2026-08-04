# Security Policy

## Supported Versions

Security fixes are applied to the current default branch. Historical source
snapshots and locally generated AscendC output are not maintained as separate
supported releases unless a release page explicitly says otherwise.

## Reporting a Vulnerability

Use the repository's **Security → Report a vulnerability** form to submit a
private report. Do not disclose suspected vulnerabilities in a public issue,
discussion, pull request, or generated log.

Include the affected revision, reproduction steps, impact, and any proposed
mitigation. Remove credentials, proprietary source code, device identifiers,
and other secrets from attachments. If private vulnerability reporting is not
enabled on the hosting service, contact a verified repository maintainer
through a private channel shown on that service before sharing details.

Maintainers should acknowledge a complete report within seven days and will
coordinate validation, remediation, and disclosure with the reporter. This is
a response target, not a guaranteed service-level agreement.

## Generated Code

Treat generated Host, Kernel, Tiling, runtime, and FlowPlan files as build
artifacts. Review them before compiling or deploying, and use only a trusted
CANN toolchain and SDK. Translator debug output may contain user source
structure; enable `--debug` only when the resulting logs can be protected.

