# Release process

This document describes how `vllm-gaudi` is versioned, branched, tagged, and
how its release artifacts are signed and verified.

## Versioning and branching

The project's versioning and branching model follows these rules:

- `vllm-gaudi` release versions track upstream `vllm`. A `vllm-gaudi` release
  `vX.Y.Z` is built on and compatible with the corresponding upstream `vllm`
  `vX.Y.Z` release.
- Each release line lives on a `releases/vX.Y.Z` branch cut from `main`.
  Fixes for a release are backported to that branch via PRs targeting it,
  as described in [`AGENTS.md`](AGENTS.md).
- Releases are tagged `vX.Y.Z`. Pre-releases use an `rcN` suffix,
  such as `v0.24.0rc0`, and post-releases use a `.postN` suffix, such as
  `v0.19.1.post1`.
- `main` is the active development branch. All changes are submitted through PRs; direct
  pushes to `main` and release branches are not permitted.

## Signed release artifacts

Every published release is accompanied by a signed source tarball, so
consumers can verify the origin and integrity of what they download.

Signing is performed automatically by the
[`Sign release artifacts`](.github/workflows/release-sign.yml) workflow when a
release is published. The workflow uses Sigstore/Cosign keyless signing.
The signature is tied to the release workflow's GitHub OIDC identity via a
short-lived Sigstore (Fulcio) certificate and recorded in the public Rekor
transparency log. There is no long-lived private signing key.

Each release includes these assets:

| Asset | Description |
| :--- | :--- |
| `vllm-gaudi-<tag>.tar.gz` | Source tarball for the tag |
| `vllm-gaudi-<tag>.tar.gz.sha256` | SHA-256 checksum of the tarball |
| `vllm-gaudi-<tag>.tar.gz.sig` | Cosign signature |
| `vllm-gaudi-<tag>.tar.gz.pem` | Fulcio signing certificate |

> Note: GitHub also auto-generates its own `Source code (tar.gz/zip)` links on
> every release. Prefer the signed `vllm-gaudi-<tag>.tar.gz` asset above when
> integrity matters, as only that artifact is covered by the signature.

## Verifying a signed release

Install [Cosign](https://docs.sigstore.dev/system_config/installation/), then
run the following commands for a tag such as `v0.24.0`:

```bash
TAG=v0.24.0
BASE="https://github.com/vllm-project/vllm-gaudi/releases/download/${TAG}"
ART="vllm-gaudi-${TAG}.tar.gz"

# Download the tarball, signature, and certificate.
curl -sSLO "${BASE}/${ART}"
curl -sSLO "${BASE}/${ART}.sig"
curl -sSLO "${BASE}/${ART}.pem"
```

Verify the downloaded tarball against its signature and certificate using
Cosign, checking that the signature was produced by this repository's release
workflow and issued through GitHub's OIDC provider. You can additionally check
the checksum with `sha256sum -c "${ART}.sha256"`.
