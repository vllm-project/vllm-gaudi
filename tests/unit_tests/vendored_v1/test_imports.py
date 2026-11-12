# SPDX-License-Identifier: Apache-2.0
"""Checks for the Model Runner V1 code vendored into vllm_gaudi/v1/_vendored/ (see tools/vendor_v1/)."""

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
VENDORED_PKG = "vllm_gaudi.v1._vendored"

# Modules deleted by the upstream Model Runner V1 removal (vllm aac1b2a4f3, vllm-project/vllm#63373).
REMOVED_MODULES = {
    "vllm.model_executor.warmup.mamba_triton_warmup",
    "vllm.utils.cpu_triton_utils",
    "vllm.v1.cudagraph_dispatcher",
    "vllm.v1.sample.logits_processor",
    "vllm.v1.sample.metadata",
    "vllm.v1.sample.ops.bad_words",
    "vllm.v1.sample.ops.logprobs",
    "vllm.v1.sample.ops.penalties",
    "vllm.v1.sample.rejection_sampler",
    "vllm.v1.sample.sampler",
    "vllm.v1.sample.thinking_budget_state",
    "vllm.v1.spec_decode",
    "vllm.v1.worker.block_table",
    "vllm.v1.worker.cpu_model_runner",
    "vllm.v1.worker.ec_connector_model_runner_mixin",
    "vllm.v1.worker.gpu.spec_decode.v1_speculator_configs",
    "vllm.v1.worker.gpu_input_batch",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu_ubatch_wrapper",
    "vllm.v1.worker.kv_connector_model_runner_mixin",
}
# Names the same removal deleted from modules that still exist.
REMOVED_NAMES = {
    "vllm.compilation.monitor":
    {"cudagraph_capturing_enabled", "set_cudagraph_capturing_enabled", "validate_cudagraph_capturing_enabled"},
    "vllm.config.ec_manager_config": {"EncoderCacheManagerMetadata"},
    "vllm.model_executor.models.deepseek_ocr": {"NGramPerReqLogitsProcessor", "NoRepeatNGramLogitsProcessor"},
    "vllm.utils.collection_utils": {"swap_dict_values"},
    "vllm.utils.import_utils": {"has_arctic_inference"},
    "vllm.utils.nvtx_pytorch_hooks": {"PytHooks"},
    "vllm.utils.torch_utils": {"make_tensor_with_pad"},
    "vllm.v1.attention.backends.utils": {"reorder_batch_to_split_decodes_and_prefills"},
    "vllm.v1.sample.ops.topk_topp_sampler":
    {"TopKTopPSampler", "empty_exponential_noise_like", "random_sample", "sample_with_exponential_noise"},
    "vllm.v1.structured_output.utils": {"apply_grammar_bitmask"},
    "vllm.v1.utils": {"copy_slice"},
    "vllm.v1.worker.cp_utils": {"get_dcp_dummy_context_len", "prepare_dcp_dummy_context_metadata"},
    "vllm.v1.worker.mamba_utils": {
        "MambaBuffers", "MambaCopyBuffers", "batch_memcpy", "batch_memcpy_kernel", "cleanup_mamba_state_idx",
        "collect_mamba_copy_meta", "do_mamba_copy_block", "postprocess_mamba_gpu", "preprocess_mamba",
        "stage_postprocess_inputs_to_gpu"
    },
    "vllm.v1.worker.ubatch_utils": {"slice_query_start_locs", "split_attn_metadata"},
    "vllm.v1.worker.xpu_model_runner": {"XPUModelRunnerV2"},
}
# Known sites still being migrated off removed modules: (file, module). Remove an entry with its last import.
PENDING = {
    ("vllm_gaudi/v1/worker/hpu_model_runner.py", "vllm.v1.worker.kv_connector_model_runner_mixin"),
}
GUARD_EXCEPTIONS = {"ImportError", "ModuleNotFoundError"}


def _vendored_modules() -> list[str]:
    pkg = importlib.import_module(VENDORED_PKG)
    return [VENDORED_PKG] + [m.name for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{VENDORED_PKG}.")]


def test_vendored_tree_matches_manifest():
    manifest = [
        line.strip() for line in (REPO / "tools/vendor_v1/manifest.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    vendored = REPO / "vllm_gaudi/v1/_vendored"
    expected = {p[len("vllm/v1/"):] for p in manifest} | {"_compat.py"}
    files = {str(p.relative_to(vendored)) for p in vendored.rglob("*.py")}
    # Generated package __init__ files are not in the manifest.
    assert {f for f in files if f in expected or not f.endswith("__init__.py")} == expected


@pytest.mark.parametrize("patch", sorted((REPO / "tools/vendor_v1/patches").glob("*.patch")), ids=lambda p: p.name)
def test_vendored_patch_is_applied(patch):
    """Every line a local patch adds is in the committed vendored file (tools/vendor_v1/vendor_v1.py --check
    proves the whole tree, but needs a vLLM checkout)."""
    target = None
    for line in patch.read_text().splitlines():
        if line.startswith("+++ b/"):
            target = (REPO / "vllm_gaudi/v1/_vendored" / line[len("+++ b/"):]).read_text().splitlines()
        elif line.startswith("+") and target is not None:
            assert line[1:] in target, f"{patch.name}: missing line {line[1:]!r}"


@pytest.mark.parametrize("module", _vendored_modules())
def test_vendored_module_imports(module):
    importlib.import_module(module)


def _load_hpu_plugins():
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    assert current_platform.device_type == "hpu"
    load_general_plugins()


def test_sampler_patches_hit_the_vendored_copy():
    from vllm_gaudi import patches

    _load_hpu_plugins()
    sampler = importlib.import_module(f"{VENDORED_PKG}.sample.sampler")
    logprobs = importlib.import_module(f"{VENDORED_PKG}.sample.ops.logprobs")
    assert sampler.Sampler.gather_logprobs is patches._hpu_gather_logprobs
    assert sampler.batched_count_greater_than is patches._hpu_batched_count_greater_than
    assert logprobs.batched_count_greater_than is patches._hpu_batched_count_greater_than


def test_installed_logits_processor_plugins_are_not_loaded(monkeypatch):
    import importlib.metadata

    from vllm_gaudi.v1._vendored.sample import logits_processor

    class _EntryPoint:
        name = value = "fake:Plugin"

        def load(self):
            raise AssertionError("an installed logits processor plugin was loaded")

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_: [_EntryPoint()])
    assert logits_processor._load_logitsprocs_plugins() == []


def test_custom_logits_processors_are_rejected():
    from vllm_gaudi.v1._vendored.sample.logits_processor import build_logitsprocs

    with pytest.raises(ValueError, match="not supported on HPU"):
        build_logitsprocs(None, None, False, False, custom_logitsprocs=["some.module:LogitsProcessor"])


def _absolute(module: str | None, level: int, path: Path) -> str:
    if not level:
        return module or ""
    # For a package __init__ the last part is "__init__", so the same slice yields the package.
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    return ".".join(parts[:len(parts) - level] + ([module] if module else []))


def _guarded(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.Try) and any(
                h.type is not None and {n.id
                                        for n in ast.walk(h.type) if isinstance(n, ast.Name)} & GUARD_EXCEPTIONS
                for h in node.handlers):
            return True
    return False


def _removed(module: str) -> bool:
    return any(module == m or module.startswith(f"{m}.") for m in REMOVED_MODULES)


def test_no_imports_of_code_removed_with_model_runner_v1():
    """Every import (or alias attribute read) of removed V1 code must use the vendored copy, or be guarded."""
    offenders = []
    for path in sorted(p for d in ("vllm_gaudi", "tests") for p in (REPO / d).rglob("*.py")):
        rel = str(path.relative_to(REPO))
        tree = ast.parse(path.read_text(), filename=rel)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        # Local names bound to a module in REMOVED_NAMES, so `alias.removed_name` reads are caught too.
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases.update({a.asname: a.name for a in node.names if a.asname and a.name in REMOVED_NAMES})
            elif isinstance(node, ast.ImportFrom):
                module = _absolute(node.module, node.level, path)
                aliases.update({
                    a.asname or a.name: f"{module}.{a.name}"
                    for a in node.names if f"{module}.{a.name}" in REMOVED_NAMES
                })
        for node in ast.walk(tree):
            if not isinstance(node, (ast.stmt, ast.expr)):  # only these carry a line number
                continue
            lineno = node.lineno
            hits: list[tuple[str, str | None]]
            if isinstance(node, ast.Import):
                hits = [(a.name, None) for a in node.names if _removed(a.name)]
            elif isinstance(node, ast.ImportFrom):
                module = _absolute(node.module, node.level, path)
                hits = [
                    (module, f"import {a.name}") for a in node.names
                    if _removed(module) or _removed(f"{module}.{a.name}") or a.name in REMOVED_NAMES.get(module, ())
                ]
            elif (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                  and isinstance(node.value, ast.Name)
                  and node.attr in REMOVED_NAMES.get(aliases.get(node.value.id, ""), ())):
                hits = [(aliases[node.value.id], f"attribute {node.attr}")]
            else:
                continue
            for module, what in hits:
                if (rel, module) not in PENDING and not _guarded(node, parents):
                    offenders.append(f"{rel}:{lineno}: {module}" + (f" {what}" if what else ""))
    assert not offenders, ("imports of vLLM code removed in aac1b2a4f3; use vllm_gaudi.v1._vendored instead:\n" +
                           "\n".join(offenders))
