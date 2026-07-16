---
title: Security
---
[](){ #security }

# Security

This document provides security guidance for deploying and operating the vLLM Hardware Plugin for Intel® Gaudi®.

## Follow the upstream vLLM security guidance

The vLLM Hardware Plugin for Intel® Gaudi® is a hardware plugin that runs on top of the [vLLM serving engine](https://docs.vllm.ai/). It shares the same architecture, deployment model, and threat surface as upstream vLLM.

**Whatever applies to vLLM applies to the vLLM Hardware Plugin for Intel® Gaudi® as well.** Before deploying this project, read and follow the upstream vLLM security documentation:

- [vLLM Security Guide](https://docs.vllm.ai/en/latest/usage/security.html) — security assumptions, deployment recommendations, and hardening advice.
- [vLLM Security Policy (`SECURITY.md`)](https://github.com/vllm-project/vllm/blob/main/SECURITY.md) — vulnerability severity model and coordinated disclosure process.

The recommendations on those pages — network isolation between nodes, running behind a reverse proxy, restricting endpoint exposure, avoiding development mode in production, and loading models and adapters only from trusted sources — apply directly to Intel® Gaudi® deployments.

## General deployment recommendations

The following advice complements the upstream guidance and is worth reviewing before you expose a deployment:

- **Run behind a trusted network boundary.** vLLM does not authenticate most of its endpoints by default, and inter-node communication is insecure by default. Place your deployment on an isolated network and put it behind a reverse proxy or API gateway that enforces authentication and TLS. Do not expose the server directly to the public internet.

- **Restrict file and directory permissions.** Limit access to model weights, cache directories (for example the Hugging Face cache and any recipe/graph cache), calibration measurement files, and configuration files to the user account that runs the vLLM process. A user who can write to a cache directory that vLLM loads from may be able to achieve arbitrary code execution. Do not share these directories with untrusted users or mount them from untrusted storage.

- **Load only trusted models and adapters.** Model files, quantization/calibration artifacts, and LoRA adapters can execute code when loaded. Only use artifacts from sources you trust, and avoid enabling runtime/dynamic adapter loading on untrusted input.

- **Never enable development mode in production.** Do not set `VLLM_SERVER_DEV_MODE=1` on a production deployment, as it exposes additional debug endpoints.

- **Protect secrets.** Keep tokens (for example `HF_TOKEN`) and other credentials out of source control and out of container images. Provide them through environment variables or a secrets manager, and scope them to the minimum required access.

- **Keep dependencies up to date.** Track upstream vLLM and Intel® Gaudi® software releases and apply security fixes promptly.

## Reporting a vulnerability

To report a security vulnerability in the vLLM Hardware Plugin for Intel® Gaudi®, follow the process described in the project's [Security Policy](https://github.com/vllm-project/vllm-gaudi/blob/main/SECURITY.md).

For vulnerabilities in the upstream vLLM engine, use the [vLLM vulnerability reporting process](https://github.com/vllm-project/vllm/security/advisories/new).
