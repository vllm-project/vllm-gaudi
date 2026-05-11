# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Online Model Swapping Test for Gaudi (API-based)
==================================================
Runs N phases (default 5) alternating between Model A and Model B.
Uses the OpenAI-compatible /v1/models/switch endpoint to switch models.
Parses server logs to extract actual warmup timings.

Collects per-phase metrics (switch time, gen time, warmup time, memory) and
prints a summary table.

Requires:
  VLLM_SERVER_DEV_MODE=1
  VLLM_ALLOW_INSECURE_SERIALIZATION=1
  Multi-model YAML config

Usage:
  VLLM_SERVER_DEV_MODE=1 \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  python tests/full_tests/online_model_swap_test.py \
    --config tests/full_tests/multi_models.yaml \
    --phases 5 \
    --api-host localhost \
    --api-port 8080
"""

import argparse
import asyncio
import contextlib
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import requests
import yaml

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.trust_env = False

SEED_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Explain quantum computing in simple terms:",
    "The tallest mountain in the world is",
    "Write a short poem about the ocean:",
    "The speed of light is approximately",
    "In the year 2050, technology will",
    "The most important invention in history is",
    "Describe the process of photosynthesis:",
    "The largest ocean on Earth is",
    "Artificial intelligence can help with",
    "The first person to walk on the moon was",
    "Climate change affects our planet by",
    "The meaning of life according to philosophy is",
    "Python programming is useful because",
    "The human brain contains approximately",
    "Renewable energy sources include",
    "The history of the internet began with",
]


def generate_prompts(n=100):
    """Generate n prompts by cycling through seed prompts with variations."""
    prompts = []
    for i in range(n):
        base = SEED_PROMPTS[i % len(SEED_PROMPTS)]
        if i < len(SEED_PROMPTS):
            prompts.append(base)
        else:
            prompts.append(f"{base} (variation {i // len(SEED_PROMPTS)})")
    return prompts


PROMPTS = generate_prompts(20)

# Fixed token budget for accuracy checks.
ACCURACY_MAX_TOKENS = 128


def _server_api_host(api_host: str) -> str:
    """Normalize host used by the local test server."""
    return '127.0.0.1' if api_host == 'localhost' else api_host


def _client_api_host(api_host: str) -> str:
    """Normalize host used by the local HTTP client."""
    return '127.0.0.1' if api_host in {'localhost', '0.0.0.0'} else api_host


def _api_url(api_host: str, api_port: int, path: str) -> str:
    return f"http://{_client_api_host(api_host)}:{api_port}{path}"


def _display_model_name(model_name: str, width: int = 38) -> str:
    if len(model_name) <= width:
        return model_name
    return model_name[:width - 3] + "..."


def _mb_to_gb(memory_mb: float | None) -> float | None:
    if memory_mb is None:
        return None
    return memory_mb / 1024.0


def _filter_present_floats(values: list[float | None]) -> list[float]:
    present_values: list[float] = []
    for value in values:
        if value is not None:
            present_values.append(value)
    return present_values


def find_free_port():
    """Find an available port to bind to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def create_multi_model_config(model_a, model_b, max_model_len=4096, max_num_batched_tokens=8192):
    """Create a temporary multi-model YAML config."""
    config = {
        'default_model': 'model_a',
        'models': {
            'model_a': {
                'model': model_a,
                'tensor_parallel_size': 1,
                'max_model_len': max_model_len,
                'max_num_batched_tokens': max_num_batched_tokens,
            },
            'model_b': {
                'model': model_b,
                'tensor_parallel_size': 1,
                'max_model_len': max_model_len,
                'max_num_batched_tokens': max_num_batched_tokens,
            },
        },
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmpfile:
        yaml.dump(config, tmpfile, default_flow_style=False)
        return tmpfile.name


def read_default_model_id(config_path: str) -> str | None:
    """Read default_model from a multi-model YAML config."""
    try:
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"  [warning] Failed to read config {config_path}: {exc}")
        return None

    if not isinstance(config, dict):
        print(f"  [warning] Config {config_path} is not a YAML mapping")
        return None

    default_model = config.get('default_model')
    if not isinstance(default_model, str) or not default_model.strip():
        print(f"  [warning] Config {config_path} has no valid default_model")
        return None

    return default_model.strip()


def select_models_for_test(available_models: list[dict[str, str]],
                           default_model_id: str | None,
                           count: int = 2) -> list[dict[str, str]]:
    """Pick models for test and ensure the configured default model is first."""
    if not available_models:
        return []

    if not default_model_id:
        return available_models[:count]

    default_model = next((m for m in available_models if m.get('id') == default_model_id), None)
    if default_model is None:
        print(f"  [warning] default_model '{default_model_id}' not found in /v1/models response")
        return available_models[:count]

    selected_models = [default_model]
    for model in available_models:
        if model.get('id') == default_model_id:
            continue
        selected_models.append(model)
        if len(selected_models) >= count:
            break
    return selected_models


class ServerLogCapture:
    """Capture and parse server logs to extract warmup timings."""

    def __init__(self):
        self.logs = []
        self.lock = threading.Lock()
        self.warmup_start_time = None
        self.warmup_events = []  # List of (timestamp, warmup_secs) tuples

    def add_line(self, line: str):
        """Add a log line and extract warmup markers."""
        with self.lock:
            self.logs.append(line)
            ts = time.time()

            # Pattern: "Warmup finished in <N> secs"
            warmup_match = re.search(r'Warmup finished in (\d+) secs', line)
            if warmup_match:
                elapsed = int(warmup_match.group(1))
                self.warmup_events.append((ts, elapsed))

    def get_warmup_times(self):
        """Return list of captured warmup times in seconds."""
        with self.lock:
            return [warmup_s for _, warmup_s in self.warmup_events]

    def clear_warmup_events(self):
        """Clear captured warmup events (call before each switch to isolate measurements)."""
        with self.lock:
            self.warmup_events = []


def run_server(config_path: str,
               api_host: str,
               api_port: int,
               log_capture: ServerLogCapture,
               max_num_batched_tokens: int | None = None):
    """Start the multi-model API server as a subprocess."""
    server_host = _server_api_host(api_host)
    env = os.environ.copy()
    env['VLLM_SERVER_DEV_MODE'] = '1'
    env['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    env['VLLM_HPU_MULTI_MODEL_CONFIG'] = config_path
    env['NO_PROXY'] = ','.join(filter(None, [env.get('NO_PROXY'), '127.0.0.1,localhost,0.0.0.0']))
    env['no_proxy'] = ','.join(filter(None, [env.get('no_proxy'), '127.0.0.1,localhost,0.0.0.0']))

    cmd = [
        sys.executable,
        '-m',
        'vllm_gaudi.entrypoints.openai.multi_model_api_server',
        '--host',
        server_host,
        '--port',
        str(api_port),
    ]
    if max_num_batched_tokens is not None:
        cmd.extend([
            '--max-num-batched-tokens',
            str(max_num_batched_tokens),
        ])

    print(f"\n>>> Starting server: {' '.join(cmd)}")
    print(f"    Config: {config_path}")
    print(f"    Listening on {server_host}:{api_port}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    def capture_logs():
        """Read stdout in background and capture log lines."""
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip('\n')
                if line:
                    print(f"[SERVER] {line}")
                    log_capture.add_line(line)
        except Exception as e:
            print(f"[LOG_CAPTURE_ERROR] {e}")

    log_thread = threading.Thread(target=capture_logs, daemon=True)
    log_thread.start()

    return proc, log_thread


async def wait_for_server(api_host: str,
                          api_port: int,
                          timeout: int = 600,
                          proc: subprocess.Popen | None = None) -> list[dict[str, str]]:
    """Wait for server to be ready and return list of available models."""
    url = _api_url(api_host, api_port, '/v1/models')
    start = time.time()
    last_error = None

    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Server exited before readiness check succeeded (exit code {proc.returncode})")
        try:
            resp = await asyncio.to_thread(
                _HTTP_SESSION.get,
                url,
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = []
                for model_card in data.get('data', []):
                    model_id = model_card.get('id')
                    if not model_id:
                        continue
                    model_root = model_card.get('root') or model_card.get('model_path') or model_id
                    models.append({
                        'id': model_id,
                        'display_name': model_root,
                    })
                print("  ✓ Server is ready")
                print("  Available models:")
                for model in models:
                    print(f"    - {model['id']} -> {model['display_name']}")
                return models
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            last_error = str(e)
        await asyncio.sleep(2)

    suffix = f" Last error: {last_error}" if last_error else ""
    raise RuntimeError(f"Server did not start after {timeout}s.{suffix}")


async def generate_text(
    api_host: str,
    api_port: int,
    model_name: str,
    prompt: str,
    max_tokens: int = ACCURACY_MAX_TOKENS,
    seed: int = 42,
) -> str | None:
    """Generate text and return the raw string output, or None on failure.

    Uses temperature=0 and a fixed seed so results are deterministic:
    a correctly loaded model must produce the exact same output on every call.
    """
    url = _api_url(api_host, api_port, '/v1/chat/completions')
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
    }
    try:
        resp = await asyncio.to_thread(
            _HTTP_SESSION.post,
            url,
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get('choices', [{}])[0].get('message', {}).get('content', '')
    except Exception as exc:  # noqa: BLE001
        print(f"  [accuracy] generate_text failed: {exc}")
    return None


async def generate_texts_for_prompts(
    api_host: str,
    api_port: int,
    model_name: str,
    prompts: list[str],
    seed: int = 42,
) -> list[str | None]:
    """Generate outputs for all prompts; returns one output per prompt."""
    outputs: list[str | None] = []
    for prompt in prompts:
        output = await generate_text(
            api_host,
            api_port,
            model_name,
            prompt,
            max_tokens=ACCURACY_MAX_TOKENS,
            seed=seed,
        )
        outputs.append(output)
    return outputs


def _wrap_text(text: str, width: int = 45) -> list[str]:
    """Word-wrap text to at most `width` characters per line."""
    words = text.split()
    lines: list[str] = []
    line: list[str] = []
    current = 0
    for word in words:
        if line and current + 1 + len(word) > width:
            lines.append(' '.join(line))
            line = [word]
            current = len(word)
        else:
            current += (1 if line else 0) + len(word)
            line.append(word)
    if line:
        lines.append(' '.join(line))
    return lines or [""]


def _short(text: str | None, width: int) -> str:
    """Shorten text for compact table display."""
    if text is None:
        return "<no output>"
    stripped = " ".join(text.split())
    if len(stripped) <= width:
        return stripped
    return stripped[:width - 3] + "..."


def print_accuracy_comparison(
    model_display_name: str,
    prompts: list[str],
    baseline_outputs: list[str | None],
    current_outputs: list[str | None],
) -> tuple[int, int]:
    """Print compact visual comparison for all prompts and return (matched, total)."""
    total = min(len(prompts), len(baseline_outputs), len(current_outputs))
    matched = 0

    print("\n  " + "-" * 118)
    print(f"  ACCURACY CHECK · {_display_model_name(model_display_name, 55)}")
    print(f"  {'#':>2}  {'Match':<5}  {'Prompt':<34}  {'Baseline':<36}  {'After switch':<36}")
    print("  " + "-" * 118)

    for idx in range(total):
        prompt = prompts[idx]
        baseline = baseline_outputs[idx]
        current = current_outputs[idx]
        same = (baseline or "").strip() == (current or "").strip()
        if same:
            matched += 1
        marker = "✓" if same else "✗"
        print(f"  {idx + 1:>2}  {marker:<5}  "
              f"{_short(prompt, 34):<34}  "
              f"{_short(baseline, 36):<36}  "
              f"{_short(current, 36):<36}")

    print("  " + "-" * 118)
    print(f"  Result: {matched}/{total} prompts match exactly")
    return matched, total


async def switch_model(api_host: str, api_port: int, model_name: str, drain_timeout: int = 60) -> dict[str, Any]:
    """Call /v1/models/switch endpoint and return metrics."""
    url = _api_url(api_host, api_port, '/v1/models/switch')
    payload = {
        "model": model_name,
        "drain_timeout": drain_timeout,
    }

    start = time.perf_counter()
    try:
        resp = await asyncio.to_thread(
            _HTTP_SESSION.post,
            url,
            json=payload,
            timeout=600,
        )
        elapsed_s = time.perf_counter() - start

        if resp.status_code == 200:
            data = resp.json()
            return {
                'status': 'ok',
                'duration_s': elapsed_s,
                'api_duration_ms': data.get('duration_ms', 0),
                'reconfigure_ms': data.get('reconfigure_ms'),
                'switched': data.get('switched', False),
                'model': data.get('current_model'),
                'memory_before_mb': data.get('memory_before_mb'),
                'memory_after_unload_mb': data.get('memory_after_unload_mb'),
                'freed_memory_mb': data.get('freed_memory_mb'),
                'stash_memory_after_mb': data.get('stash_memory_after_mb'),
            }
        else:
            return {
                'status': 'error',
                'duration_s': elapsed_s,
                'error': resp.text,
            }
    except Exception as e:
        elapsed_s = time.perf_counter() - start
        return {
            'status': 'error',
            'duration_s': elapsed_s,
            'error': str(e),
        }


async def generate(api_host: str,
                   api_port: int,
                   model_name: str,
                   prompt: str,
                   seed: int = 42,
                   max_tokens: int = 1600,
                   strict_tokens: bool = True) -> dict:
    """Call /v1/chat/completions and return metrics."""
    url = _api_url(api_host, api_port, '/v1/chat/completions')
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,  # Fixed seed for reproducibility
    }
    if strict_tokens:
        payload["min_tokens"] = max_tokens
        payload["ignore_eos"] = True

    start = time.perf_counter()
    try:
        resp = await asyncio.to_thread(
            _HTTP_SESSION.post,
            url,
            json=payload,
            timeout=120,
        )
        elapsed_s = time.perf_counter() - start

        if resp.status_code == 200:
            data = resp.json()
            tokens = len(data.get('choices', [{}])[0].get('message', {}).get('content', '').split())
            usage = data.get('usage', {})
            return {
                'status': 'ok',
                'duration_s': elapsed_s,
                'output_tokens': usage.get('completion_tokens', tokens),
                'total_tokens': usage.get('total_tokens', 0),
            }
        else:
            if strict_tokens and resp.status_code == 400:
                payload.pop("min_tokens", None)
                payload.pop("ignore_eos", None)
                retry_resp = await asyncio.to_thread(
                    _HTTP_SESSION.post,
                    url,
                    json=payload,
                    timeout=120,
                )
                retry_elapsed_s = time.perf_counter() - start
                if retry_resp.status_code == 200:
                    data = retry_resp.json()
                    tokens = len(data.get('choices', [{}])[0].get('message', {}).get('content', '').split())
                    usage = data.get('usage', {})
                    return {
                        'status': 'ok',
                        'duration_s': retry_elapsed_s,
                        'output_tokens': usage.get('completion_tokens', tokens),
                        'total_tokens': usage.get('total_tokens', 0),
                        'strict_tokens_fallback': True,
                    }
            return {
                'status': 'error',
                'duration_s': elapsed_s,
                'error': resp.text,
            }
    except Exception as e:
        elapsed_s = time.perf_counter() - start
        return {
            'status': 'error',
            'duration_s': elapsed_s,
            'error': str(e),
        }


def print_metrics_table(all_metrics):
    """Print a summary table of per-phase metrics."""
    hdr = (f"{'Phase':>5}  {'Model':<38}  "
           f"{'Init/Switch(s)':>14}  {'Warmup(s)':>9}  "
           f"{'Gen(s)':>7}  "
           f"{'Tokens':>7}  "
           f"{'Freed(GB)':>9}  "
           f"{'StashUsed(GB)':>13}")
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for m in all_metrics:
        warmup_s = m.get('warmup_s', 'N/A')
        warmup_str = f"{warmup_s:>9.1f}" if isinstance(warmup_s, (int, float)) else f"{str(warmup_s):>9}"

        freed_gb = _mb_to_gb(m.get('freed_memory_mb'))
        freed_str = f"{freed_gb:>9.2f}" if isinstance(freed_gb, (int, float)) else f"{'N/A':>9}"

        stash_used_gb = _mb_to_gb(m.get('stash_memory_after_mb'))
        stash_used_str = f"{stash_used_gb:>13.2f}" if isinstance(stash_used_gb, (int, float)) else f"{'N/A':>13}"

        print(f"{m['phase']:>5}  "
              f"{_display_model_name(m['model'], 38):<38}  "
              f"{m['reconfigure_s']:>13.1f}  "
              f"{warmup_str}  "
              f"{m['gen_s']:>7.2f}  "
              f"{m['tokens']:>7}  "
              f"{freed_str}  "
              f"{stash_used_str}")
    print(sep)

    n = len(all_metrics)
    if n > 0:
        avg = {
            'reconfigure_s': sum(m['reconfigure_s'] for m in all_metrics) / n,
            'gen_s': sum(m['gen_s'] for m in all_metrics) / n,
            'tokens': sum(m['tokens'] for m in all_metrics) / n,
        }
        warmup_times = [m.get('warmup_s') for m in all_metrics if isinstance(m.get('warmup_s'), (int, float))]
        if warmup_times:
            avg_warmup = sum(warmup_times) / len(warmup_times)
            warmup_str = f"{avg_warmup:>9.1f}"
        else:
            warmup_str = f"{'N/A':>9}"

        freed_gbs = _filter_present_floats([_mb_to_gb(m.get('freed_memory_mb')) for m in all_metrics])
        if freed_gbs:
            avg_freed = sum(freed_gbs) / len(freed_gbs)
            freed_str = f"{avg_freed:>9.2f}"
        else:
            freed_str = f"{'N/A':>9}"

        stash_used_gbs = _filter_present_floats([_mb_to_gb(m.get('stash_memory_after_mb')) for m in all_metrics])
        if stash_used_gbs:
            avg_stash_used = sum(stash_used_gbs) / len(stash_used_gbs)
            stash_used_str = f"{avg_stash_used:>13.2f}"
        else:
            stash_used_str = f"{'N/A':>13}"

        print(f"{'AVG':>5}  {'':<38}  "
              f"{avg['reconfigure_s']:>13.1f}  "
              f"{warmup_str}  "
              f"{avg['gen_s']:>7.2f}  "
              f"{avg['tokens']:>7.0f}  "
              f"{freed_str}  "
              f"{stash_used_str}")
        print(sep)


async def main():
    parser = argparse.ArgumentParser(description="Online Model Swapping Test")
    parser.add_argument("--model-a", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model-b", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--config",
                        type=str,
                        default=None,
                        help="Multi-model YAML config (auto-generated if not provided)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens",
                        type=int,
                        default=8192,
                        help="Align with offline baseline (default 8192)")
    parser.add_argument("--phases", type=int, default=5)
    parser.add_argument("--fixed-output-tokens", type=int, default=1600, help="Target completion tokens per request")
    parser.add_argument("--api-host", type=str, default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=None)
    parser.add_argument("--no-accuracy-check",
                        action="store_true",
                        help="Skip accuracy comparison between baseline and post-switch outputs")
    args = parser.parse_args()
    config_provided = args.config is not None

    if args.api_port is None:
        args.api_port = find_free_port()

    # Create config if not provided
    config_path = args.config
    if config_path is None:
        config_path = create_multi_model_config(
            args.model_a,
            args.model_b,
            args.max_model_len,
            args.max_num_batched_tokens,
        )
        print(f"Generated config: {config_path}")

    print("=" * 60)
    print("  ONLINE MODEL SWAPPING TEST")
    print("=" * 60)
    if config_provided:
        print("  Models: see config (--config)")
    else:
        print(f"  Model A: {args.model_a}")
        print(f"  Model B: {args.model_b}")
    print(f"  Phases: {args.phases}")
    if config_provided:
        print("  Max model len: see config")
        print("  Max num batched tokens: see config")
    else:
        print(f"  Max model len: {args.max_model_len}")
        print(f"  Max num batched tokens: {args.max_num_batched_tokens}")
    print(f"  Fixed output tokens: {args.fixed_output_tokens}")
    print(f"  API: {args.api_host}:{args.api_port}")
    print(f"  Config: {config_path}")
    print("=" * 60)

    log_capture = ServerLogCapture()
    proc = None

    try:
        default_model_id = read_default_model_id(config_path)

        # Start server
        startup_begin = time.perf_counter()
        proc, log_thread = run_server(
            config_path,
            args.api_host,
            args.api_port,
            log_capture,
            None if config_provided else args.max_num_batched_tokens,
        )
        available_models = await wait_for_server(args.api_host, args.api_port, proc=proc)
        initial_load_s = time.perf_counter() - startup_begin
        startup_warmup_times = log_capture.get_warmup_times()
        startup_warmup_s = startup_warmup_times[-1] if startup_warmup_times else None

        if len(available_models) < 2:
            raise RuntimeError(f"Expected at least 2 models, got {len(available_models)}: {available_models}")

        models = select_models_for_test(available_models, default_model_id, count=len(available_models))
        print("  Using models for test:")
        for model in models:
            print(f"    - {model['id']} -> {model['display_name']}")

        all_metrics: list[dict[str, Any]] = []
        phase_failures: list[str] = []
        # baseline_texts[model_id] = outputs for all prompts on first load.
        baseline_texts: dict[str, list[str | None]] = {}
        accuracy_stats: dict[str, tuple[int, int]] = {}
        test_start = time.time()

        for phase in range(1, args.phases + 1):
            model_idx = (phase - 1) % len(models)
            model_info = models[model_idx]
            model_name = model_info['id']
            model_display_name = model_info['display_name']

            print("\n" + "=" * 60)
            print(f"  PHASE {phase}/{args.phases}: {model_display_name}")
            print("=" * 60)

            # Clear previous warmup events before this phase
            log_capture.clear_warmup_events()

            # Switch model (only if not on first phase with default model)
            print(f"\n>>> Switching to model: {model_display_name} ({model_name})")

            if phase == 1:
                print("  (Skipping switch on first phase - model already loaded)")
                switch_result: dict[str, Any] = {
                    'status': 'ok',
                    'duration_s': 0,
                    'reconfigure_ms': 0,
                    'switched': False,
                    'model': model_name,
                }
                warmup_s = None  # Phase 1 warmup not measured separately
                # Capture baseline output on first load (no switch yet).
                if not args.no_accuracy_check:
                    print(f"  [accuracy] Capturing baseline for {model_display_name} on {len(PROMPTS)} prompts...")
                    baseline_texts[model_name] = await generate_texts_for_prompts(
                        args.api_host,
                        args.api_port,
                        model_name,
                        PROMPTS,
                    )
            else:
                switch_result = await switch_model(args.api_host, args.api_port, model_name)

                if switch_result['status'] != 'ok':
                    error_msg = switch_result.get('error')
                    print(f"  ✗ Switch failed: {error_msg}")
                    phase_failures.append(f"phase{phase} switch failed: {error_msg}")
                    continue

                reconfigure_ms = switch_result.get('reconfigure_ms')
                if isinstance(reconfigure_ms, (int, float)):
                    print(f"  ✓ Sleep+load(reconfigure) duration: {reconfigure_ms / 1000.0:.1f}s")
                else:
                    print(f"  ✓ Switch API duration: {switch_result['duration_s']:.1f}s")

                # Extract warmup time from logs captured during switch
                # Wait a bit for final logs to be captured
                await asyncio.sleep(0.5)
                warmup_times = log_capture.get_warmup_times()
                warmup_s = warmup_times[-1] if warmup_times else None
                if warmup_s is not None:
                    print(f"  ✓ Warmup time from logs: {warmup_s}s")

                freed_gb = _mb_to_gb(switch_result.get('freed_memory_mb'))
                if freed_gb is not None:
                    print(f"  ✓ Freed HPU memory: {freed_gb:.2f} GB")

                stash_used_gb = _mb_to_gb(switch_result.get('stash_memory_after_mb'))
                if stash_used_gb is not None:
                    print(f"  ✓ HPU memory still used after stashing: {stash_used_gb:.2f} GB")

                # Accuracy check: compare post-switch output to baseline.
                if not args.no_accuracy_check:
                    if model_name not in baseline_texts:
                        # First time seeing this model — establish its baseline now.
                        print(f"  [accuracy] Capturing baseline for {model_display_name} "
                              f"on {len(PROMPTS)} prompts...")
                        baseline_texts[model_name] = await generate_texts_for_prompts(
                            args.api_host,
                            args.api_port,
                            model_name,
                            PROMPTS,
                        )
                    else:
                        current_outputs = await generate_texts_for_prompts(
                            args.api_host,
                            args.api_port,
                            model_name,
                            PROMPTS,
                        )
                        matched, total = print_accuracy_comparison(
                            model_display_name,
                            PROMPTS,
                            baseline_texts[model_name],
                            current_outputs,
                        )
                        accuracy_stats[f"phase{phase}"] = (matched, total)

            # Generate
            print(f">>> Generating with model: {model_display_name}")
            prompt = PROMPTS[(phase - 1) % len(PROMPTS)]
            gen_result = await generate(
                args.api_host,
                args.api_port,
                model_name,
                prompt,
                seed=42,  # Fixed seed for reproducibility
                max_tokens=args.fixed_output_tokens,
                strict_tokens=True,
            )

            if gen_result['status'] != 'ok':
                error_msg = gen_result.get('error')
                print(f"  ✗ Generation failed: {error_msg}")
                phase_failures.append(f"phase{phase} generation failed: {error_msg}")
                continue

            print(f"  ✓ Generated {gen_result['output_tokens']} tokens in {gen_result['duration_s']:.2f}s")

            phase_metrics = {
                'phase': len(all_metrics) + 1,
                'model': model_display_name,
                'reconfigure_s': (switch_result.get('reconfigure_ms') or 0) / 1000.0,
                'warmup_s': warmup_s,
                'gen_s': gen_result['duration_s'],
                'tokens': gen_result['output_tokens'],
            }
            if phase == 1:
                phase_metrics['reconfigure_s'] = initial_load_s
                phase_metrics['warmup_s'] = startup_warmup_s
            else:
                phase_metrics['freed_memory_mb'] = switch_result.get('freed_memory_mb')
                phase_metrics['stash_memory_after_mb'] = switch_result.get('stash_memory_after_mb')
            all_metrics.append(phase_metrics)

        total_time = time.time() - test_start

        # Print summary
        print("\n" + "=" * 60)
        print("  METRICS SUMMARY")

        # Print accuracy summary.
        if not args.no_accuracy_check and accuracy_stats:
            total_phases = len(accuracy_stats)
            passed_phases = sum(1 for matched, total in accuracy_stats.values() if matched == total)
            icon = "✓" if passed_phases == total_phases else "✗"
            print(f"  {icon} Accuracy checks: {passed_phases}/{total_phases} phases fully matched")
            for phase_key, (matched, total) in accuracy_stats.items():
                mark = "✓" if matched == total else "✗"
                print(f"      {mark} {phase_key}: {matched}/{total} prompts")
        print("=" * 60)
        print_metrics_table(all_metrics)
        print(f"\n  Total time: {total_time:.2f}s")

        if phase_failures:
            print("\n" + "=" * 60)
            print("  TEST FAILED ✗")
            print("=" * 60)
            print(f"  ✗ {len(phase_failures)} phase operation(s) failed")
            for failure in phase_failures:
                print(f"    - {failure}")
            return 1

        # Success
        print("\n" + "=" * 60)
        print("  TEST COMPLETED ✓")
        print("=" * 60)
        print(f"  ✓ {args.phases} phases")
        print("  ✓ Model switching via API")
        print("  ✓ Warmup times captured from logs")

    except Exception:
        print("\n" + "=" * 60)
        print("  TEST FAILED ✗")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if proc:
            print("\n>>> Shutting down server...")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print("  ✓ Server stopped")

        # Clean up temp config if we created it
        if args.config is None and config_path:
            with contextlib.suppress(Exception):
                os.unlink(config_path)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
