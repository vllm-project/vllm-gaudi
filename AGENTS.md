# Project Guidelines for vllm-gaudi

## Project Overview

This is **vllm-gaudi** — the vLLM Hardware Plugin for Intel® Gaudi® AI accelerators. It is a plugin package (not a fork) that integrates Intel Gaudi HPUs with the upstream [vLLM](https://github.com/vllm-project/vllm) inference engine via the vLLM plugin architecture.

- **Package**: `vllm_gaudi` (installed as `vllm-gaudi`)
- **License**: Apache-2.0
- **Python**: Supported versions are defined in `pyproject.toml` (`requires-python`)
- **Entry point**: `vllm_gaudi/__init__.py` registers the HPU platform, ops, models, and utilities into vLLM's plugin system.

## Getting Started

See the [README — Getting Started](README.md#getting-started) for installation instructions (vLLM checkout, plugin install, Docker).

## Architecture

The plugin follows vLLM's pluggable architecture. For Intel® Gaudi® hardware details (MME, TPC, profiling, etc.), see the [Intel® Gaudi® Documentation](https://docs.habana.ai/en/latest/index.html).

Key subsystems:

| Directory | Purpose |
|-----------|---------|
| `vllm_gaudi/ops/` | HPU-specific operator implementations (attention, FP8, MoE, LoRA, etc.) |
| `vllm_gaudi/models/` | Model-specific overrides and registrations for HPU |
| `vllm_gaudi/attention/` | HPU attention backends and paged attention ops |
| `vllm_gaudi/extension/` | Bucketing, profiling, quantization, runtime config, unified batching |
| `vllm_gaudi/v1/` | V1 engine worker and model runner for HPU |
| `vllm_gaudi/distributed/` | HPU communicator and KV transfer connectors (NIXL) |
| `vllm_gaudi/lora/` | LoRA layer support for HPU-specific layers |
| `docs/` | MkDocs-based documentation |
| `tests/` | Unit tests (`unit_tests/`), full model tests (`full_tests/`), upstream compat tests (`upstream_tests/`) |
| `calibration/` | FP8 calibration pipeline scripts |
| `.cd/` | Docker build files and CI/CD configs |

## Code Style

- **Formatter**: `yapf` (column limit 120) + `ruff` (line length 120)
- **Linter**: `ruff` with pycodestyle (E), pyflakes (F), pyupgrade (UP), flake8-bugbear (B), flake8-simplify (SIM). Note: `vllm_gaudi/extension/` is currently excluded from ruff
- **Pre-commit hooks**: Install via `pip install -r requirements-lint.txt && pre-commit install`
- **SPDX headers**: New Python files should include `# SPDX-License-Identifier: Apache-2.0` as the first comment line
- **Imports**: Use absolute imports. Avoid star imports where possible (F405/F403 are currently suppressed)
- **Line length**: 120 characters max

## Commit and PR Conventions

- **Never commit directly to `main`**. Always work on a feature branch.
  - If you have write access: create a branch in the repo (e.g., `git checkout -b my-feature`)
  - If you don't have write access: fork the repo and create a branch in your fork
- Create a Pull Request from your branch targeting the appropriate base branch. Default target is `main` unless a specific release branch is specified (e.g., `releases/v0.16.0`)
- All PRs are squash-merged with PR number appended: `Title (#NNN)`
- Commit messages must include `Signed-off-by:` (enforced by pre-commit hook via DCO)
- Common title patterns:
  - Features: `Feature name (#NNN)` — e.g., `RowParallel NIC chunking (#896)`
  - Fixes: `Fix description (#NNN)` — e.g., `Fix mamba cumsum padded calculations (#1009)`
  - Upstream compatibility: `[FIX_FOR_VLLM_CUSTOM=<hash>] Description (#NNN)`
  - CI changes: `[CI] Description (#NNN)`
  - Reverts: `Revert "Original title" (#NNN)`
  - Topic tags: `[warmup][multimodal] Description (#NNN)`

### Confidentiality and content rules

- **No internal references**: Never include internal ticket numbers (e.g., Jira IDs), links to internal sites (Jira, Confluence, etc.), or internal file paths (e.g., `/mnt/weka/...`) in PR titles, descriptions, commit messages, or code comments
- **No raw performance numbers**: Do not include absolute performance metrics or profiling data/screenshots. Relative improvements are acceptable (e.g., "~20% throughput gain for long contexts") but raw numbers, benchmarks, or profiler outputs must not appear in PRs
- **PR titles must be descriptive**: Use a clear summary of the change, not a ticket reference

## Testing

- **Unit tests**: `tests/unit_tests/` — test individual ops, workers, samplers
- **Full tests**: `tests/full_tests/` — end-to-end model generation tests using model cards
- **Run tests**: Tests require Intel Gaudi hardware (HPU). Use `pytest` to run
- When adding a new operator in `vllm_gaudi/ops/`, add corresponding unit tests in `tests/unit_tests/ops/`
- When adding new model support, add model cards in `tests/models/`

## Environment Variables

- HPU-specific env vars are defined in `vllm_gaudi/envs.py`
- Document new env vars in `docs/configuration/env_variables.md`
- Feature flags belong in `vllm_gaudi/extension/features.py`

## Adding New Ops

1. Create the op file in `vllm_gaudi/ops/hpu_<name>.py`
2. Register it in `vllm_gaudi/__init__.py` under `register_ops()`
3. Add unit tests in `tests/unit_tests/ops/test_hpu_<name>.py`
4. Add documentation if the feature is user-facing

## Documentation

- Uses MkDocs with navigation defined in `docs/.nav.yml`
- Feature docs go in `docs/features/`
- Configuration docs go in `docs/configuration/`
- Developer guides go in `docs/dev_guide/`
