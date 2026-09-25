"""
Minimal example of FP8 quantization for a single Linear layer on HPU.

Compares four schemes against a bf16 baseline:
  1) bf16 baseline (no quantization)
  2) FP8 dynamic per-tensor   — one scalar scale per activation tensor
  3) FP8 dynamic per-channel  — per-token activation scale + per-out-channel
                                 weight scale (the scheme used by Llama-3.x FP8
                                 in compressed-tensors)
  4) FP8 static  per-tensor   — pre-calibrated activation scale

Uses the same HPU ops as `vllm_gaudi.extension.ops.apply_fp8_linear_hpu`:
  torch.ops.hpu.cast_to_fp8_v2   — bf16 -> float8_e4m3fn
  torch.ops.hpu.fp8_gemm_v2      — FP8 matmul with on-the-fly dequant

Run (eager only, traces + FX dumps):
  python bench_fp8_dynamic_quant.py

Run with torch.compile(hpu_backend), adds compiled traces:
  COMPILE=1 python bench_fp8_dynamic_quant.py

Dump dynamo/inductor graph code to stderr:
  TORCH_LOGS="graph_code,output_code" COMPILE=1 python bench_fp8_dynamic_quant.py

Artifacts in ./traces_fp8_dynamic_quant/:
  *.json   — open in chrome://tracing or https://ui.perfetto.dev
  *.fx.txt — torch.fx GraphModule dump (via make_fx)
"""
import os

# Disable torch's autoload of habana_frameworks (we import it ourselves below).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
# Required for HPU profiler activity to be recorded into the chrome trace.
os.environ.setdefault("HABANA_PROFILE", "1")

import sys
import importlib.abc
import importlib.machinery
import torch


# ----------------------------------------------------------------------------
# Workaround: installed habana_frameworks expects symbols in torch internals
# (torch.utils._config_module, torch._dynamo.config) that the torch in this
# venv no longer exposes. Its `_patch_config_module_getattr()` then injects a
# `_fast_getattr` that breaks every torch config access. We keep the rest of
# habana_frameworks intact but turn that one patcher into a no-op via an
# import hook before the module is exec'd.
# ----------------------------------------------------------------------------
class _DisableHabanaConfigPatch(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    TARGET = "habana_frameworks.torch.core.torch_overwrites"

    def find_spec(self, name, path, target=None):
        if name != self.TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        self._real_loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        if hasattr(self._real_loader, "create_module"):
            return self._real_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self._real_loader.exec_module(module)
        module._patch_config_module_getattr = lambda: None


sys.meta_path.insert(0, _DisableHabanaConfigPatch())

import habana_frameworks.torch  # noqa: E402, F401  registers torch.hpu / hpu_backend
from torch.fx.experimental.proxy_tensor import make_fx  # noqa: E402

# ---------------------------------------------------------------------------
# Bench configuration — tweak here.
# ---------------------------------------------------------------------------

# Device / dtypes
DEVICE = torch.device("hpu")
DTYPE = torch.bfloat16
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max

# Linear shape: x[M, K] @ W[N, K]^T -> y[M, N].
# Defaults match Llama-3.3-70B q_proj (out=8192, in=8192) with M = 4096 tokens.
M, N, K = 4096, 8192, 8192

# Profiler loop: warmup runs first (compilation, allocator), then ITERS
# iterations are recorded into the chrome trace.
WARMUP = 5
ITERS = 20

# Output directory for *.json (chrome traces) and *.fx.txt (FX graph dumps).
TRACE_DIR_NAME = "traces_fp8_dynamic_quant"


# ---------------------------------------------------------------------------
# 1. bf16 baseline
# ---------------------------------------------------------------------------
def linear_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, w)


# ---------------------------------------------------------------------------
# 2. Dynamic per-tensor:
#    one scalar scale per activation, one scalar scale per weight
# ---------------------------------------------------------------------------
def quant_per_tensor(t: torch.Tensor):
    scale = (t.abs().max() + 1e-8) / FP8_MAX  # scalar
    t_fp8 = torch.ops.hpu.cast_to_fp8_v2(t, 1.0 / scale, False, False, FP8_DTYPE)[0]
    return t_fp8, scale.float().view(1)  # stored as shape [1]


def linear_fp8_per_tensor(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    x_fp8, x_scale = quant_per_tensor(x)
    return torch.ops.hpu.fp8_gemm_v2(
        A=x_fp8,
        trans_A=False,
        B=w_fp8,
        trans_B=True,
        D=None,
        out_dtype=DTYPE,
        A_scale_inv=x_scale,  # shape [1]   per-tensor
        B_scale_inv=w_scale,  # shape [1]   per-tensor
        bias=None,
        accumulate=False,
    )


# ---------------------------------------------------------------------------
# 3. Dynamic per-token activations + per-channel weights:
#    x scale shape [M, 1] (one per token, computed on the fly)
#    w scale shape [N, 1] (one per output channel, computed once at load)
#
#    NOTE: amax(dim=-1, keepdim=True) is preferred over max(dim=-1).values +
#    unsqueeze(-1). aten.max.dim returns (values, indices); the indices tensor
#    is computed and immediately discarded, costing an extra reduce pass and
#    an i32->i64 cast on HPU. amax does not produce indices, and keepdim=True
#    folds the unsqueeze into the reduce kernel — under torch.compile the
#    hpu_backend then fuses abs+amax into a single TPC kernel (~20% device
#    time win on the quant step).
# ---------------------------------------------------------------------------
def quant_per_token(t: torch.Tensor):
    scale = (t.abs().amax(dim=-1, keepdim=True) + 1e-8) / FP8_MAX  # [M, 1]
    t_fp8 = torch.ops.hpu.cast_to_fp8_v2(t, 1.0 / scale, False, False, FP8_DTYPE)[0]
    return t_fp8, scale.float()


def quant_per_channel(w: torch.Tensor):
    # w: [N, K], reduce along input axis K -> one scale per output channel
    scale = (w.abs().amax(dim=-1, keepdim=True) + 1e-8) / FP8_MAX  # [N, 1]
    w_fp8 = torch.ops.hpu.cast_to_fp8_v2(w, 1.0 / scale, False, False, FP8_DTYPE)[0]
    return w_fp8, scale.float()


def linear_fp8_per_channel(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    x_fp8, x_scale = quant_per_token(x)  # [M, 1]
    # HPU fp8_gemm_v2 expects B_scale_inv along the output dim of the result,
    # i.e. transposed wrt the [N, 1] storage shape — same as apply_fp8_linear_hpu.
    w_scale_t = w_scale.transpose(0, 1) if w_scale.dim() > 1 else w_scale
    return torch.ops.hpu.fp8_gemm_v2(
        A=x_fp8,
        trans_A=False,
        B=w_fp8,
        trans_B=True,
        D=None,
        out_dtype=DTYPE,
        A_scale_inv=x_scale,  # shape [M, 1]    per-token
        B_scale_inv=w_scale_t,  # shape [1, N]    per-channel (transposed)
        bias=None,
        accumulate=False,
    )


# ---------------------------------------------------------------------------
# 4. Static per-tensor:
#    x_scale is pre-calibrated (shape [1], fixed), w_scale is per-tensor [1].
#    The forward has NO abs/max/div/unsqueeze — only one cast + one gemm.
#    In compressed-tensors this corresponds to "activation_scheme: static".
# ---------------------------------------------------------------------------
def calibrate_static_scale(samples: torch.Tensor) -> torch.Tensor:
    """One-shot calibration: take max over a few activation samples — that's
    the static input_scale. In production this is done over a calibration
    dataset (see calibration/step-4-quantize-scales.py)."""
    scale = (samples.abs().max() + 1e-8) / FP8_MAX
    return scale.float().view(1)


def linear_fp8_static(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor,
                      x_scale: torch.Tensor) -> torch.Tensor:
    # The scale is used as 1/scale for the cast (same as apply_fp8_linear_hpu)
    # but as scale_inv for the gemm — matches the production code path.
    x_fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / x_scale, False, False, FP8_DTYPE)[0]
    return torch.ops.hpu.fp8_gemm_v2(
        A=x_fp8,
        trans_A=False,
        B=w_fp8,
        trans_B=True,
        D=None,
        out_dtype=DTYPE,
        A_scale_inv=x_scale,  # shape [1]   fixed, from calibration
        B_scale_inv=w_scale,  # shape [1]   per-tensor
        bias=None,
        accumulate=False,
    )


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------
def profile_variant(name: str, fn, args, trace_dir: str):
    for _ in range(WARMUP):
        fn(*args)
    torch.hpu.synchronize()

    trace_path = os.path.join(trace_dir, f"{name}.json")
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            record_shapes=True,
            with_stack=False,
    ) as prof:
        for _ in range(ITERS):
            out = fn(*args)
        torch.hpu.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"[{name:32s}] trace -> {trace_path}")
    # Sort by HPU time if the profiler captured device activity, otherwise CPU.
    averages = prof.key_averages()
    sort_key = "self_hpu_time_total"
    if not averages or not hasattr(averages[0], sort_key):
        sort_key = "self_cpu_time_total"
    try:
        print(averages.table(sort_by=sort_key, row_limit=8))
    except Exception as e:
        print(f"  (table render failed: {e})")
    return out


def dump_fx_graph(name: str, fn, args, trace_dir: str):
    """Capture the torch.fx GraphModule via make_fx and write it to .fx.txt.

    Shows the real low-level graph with all aten/hpu nodes:
    cast_to_fp8_v2, fp8_gemm_v2, abs, amax, etc.
    """
    try:
        gm = make_fx(fn, tracing_mode="fake")(*args)
    except Exception as e:
        print(f"[{name:32s}] make_fx failed: {e}")
        return
    fx_path = os.path.join(trace_dir, f"{name}.fx.txt")
    with open(fx_path, "w") as f:
        f.write(f"# Graph for: {name}\n\n")
        f.write("## graph.print_tabular()\n")
        # print_tabular writes to stdout — capture via io.
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gm.graph.print_tabular()
        f.write(buf.getvalue())
        f.write("\n\n## gm.code\n")
        f.write(gm.code)
    print(f"[{name:32s}] fx graph -> {fx_path}")


def main():
    trace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRACE_DIR_NAME)
    os.makedirs(trace_dir, exist_ok=True)

    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    w = torch.randn(N, K, dtype=DTYPE, device=DEVICE) * 0.02

    # Weights are quantized once up-front, like at checkpoint load time.
    w_fp8_pt, w_scale_pt = quant_per_tensor(w)  # scale: [1]
    w_fp8_pc, w_scale_pc = quant_per_channel(w)  # scale: [N, 1]

    # Static input_scale calibration — usually done over multiple batches.
    # Here we use one sample x; in production this is the role of
    # calibration/step-4-quantize-scales.py.
    x_scale_static = calibrate_static_scale(x)  # scale: [1], fixed

    print(f"x              : {tuple(x.shape)} {x.dtype}")
    print(f"w              : {tuple(w.shape)} {w.dtype}")
    print(f"w_fp8_pt       : {tuple(w_fp8_pt.shape)}   scale {tuple(w_scale_pt.shape)}  (per-tensor)")
    print(f"w_fp8_pc       : {tuple(w_fp8_pc.shape)}   scale {tuple(w_scale_pc.shape)}  (per-channel)")
    print(f"x_scale_static : {tuple(x_scale_static.shape)}              (calibrated, fixed)")
    print()

    # ---- FX graphs (via make_fx) ----
    dump_fx_graph("01_bf16_baseline", linear_bf16, (x, w), trace_dir)
    dump_fx_graph("02_fp8_dyn_per_tensor", linear_fp8_per_tensor, (x, w_fp8_pt, w_scale_pt), trace_dir)
    dump_fx_graph("03_fp8_dyn_per_channel", linear_fp8_per_channel, (x, w_fp8_pc, w_scale_pc), trace_dir)
    dump_fx_graph("04_fp8_static_per_tensor", linear_fp8_static, (x, w_fp8_pt, w_scale_pt, x_scale_static), trace_dir)
    print()

    # ---- eager traces ----
    y_bf16 = profile_variant("01_bf16_baseline", linear_bf16, (x, w), trace_dir)
    y_dpt = profile_variant("02_fp8_dyn_per_tensor", linear_fp8_per_tensor, (x, w_fp8_pt, w_scale_pt), trace_dir)
    y_dpc = profile_variant("03_fp8_dyn_per_channel", linear_fp8_per_channel, (x, w_fp8_pc, w_scale_pc), trace_dir)
    y_spt = profile_variant("04_fp8_static_per_tensor", linear_fp8_static, (x, w_fp8_pt, w_scale_pt, x_scale_static),
                            trace_dir)

    # ---- (optional) torch.compile with hpu_backend ----
    if os.environ.get("COMPILE", "0") == "1":
        print("\n=== torch.compile(backend='hpu_backend') ===")
        compile_kwargs = dict(backend="hpu_backend", fullgraph=False, dynamic=False)
        bf16_c = torch.compile(linear_bf16, **compile_kwargs)
        dpt_c = torch.compile(linear_fp8_per_tensor, **compile_kwargs)
        dpc_c = torch.compile(linear_fp8_per_channel, **compile_kwargs)
        spt_c = torch.compile(linear_fp8_static, **compile_kwargs)
        # Compilation happens on the first call — warm up out of the profiler.
        for _ in range(3):
            bf16_c(x, w)
            dpt_c(x, w_fp8_pt, w_scale_pt)
            dpc_c(x, w_fp8_pc, w_scale_pc)
            spt_c(x, w_fp8_pt, w_scale_pt, x_scale_static)
        torch.hpu.synchronize()
        profile_variant("05_bf16_baseline_compiled", bf16_c, (x, w), trace_dir)
        profile_variant("06_fp8_dyn_per_tensor_compiled", dpt_c, (x, w_fp8_pt, w_scale_pt), trace_dir)
        profile_variant("07_fp8_dyn_per_channel_compiled", dpc_c, (x, w_fp8_pc, w_scale_pc), trace_dir)
        profile_variant("08_fp8_static_per_tensor_compiled", spt_c, (x, w_fp8_pt, w_scale_pt, x_scale_static),
                        trace_dir)

    # ---- numeric check ----
    err_dpt = (y_dpt.float() - y_bf16.float()).abs().mean().item()
    err_dpc = (y_dpc.float() - y_bf16.float()).abs().mean().item()
    err_spt = (y_spt.float() - y_bf16.float()).abs().mean().item()
    ref = y_bf16.float().abs().mean().item()
    print("\nMAE vs bf16:")
    print(f"  dynamic per-tensor  : {err_dpt:.4e}  (rel {err_dpt/ref:.2%})")
    print(f"  dynamic per-channel : {err_dpc:.4e}  (rel {err_dpc/ref:.2%})")
    print(f"  static  per-tensor  : {err_spt:.4e}  (rel {err_spt/ref:.2%})")


if __name__ == "__main__":
    main()
