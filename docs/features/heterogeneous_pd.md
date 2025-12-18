---
title: Heterogeneous PD Disaggregation on CUDA+Gaudi Multi‑Node System
---

# Heterogeneous PD Disaggregation on CUDA+Gaudi Multi‑Node System

## Overview

PD Disaggregation splits model execution into prefill and decode stages. The validated topology uses CUDA GPUs for prefill nodes and Gaudi HPUs for decode nodes connected through UCX over InfiniBand. Reverse roles are not yet supported.

## Prerequisites

- Docker containers must run with `--privileged --net=host --ipc=host` to expose InfiniBand devices and UCX transports.
- Tested images:
  - CUDA node: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
  - Gaudi node: `vault.habana.ai/gaudi-docker/1.22.2/ubuntu24.04/habanalabs/pytorch-installer-2.7.1`
- Each node requires vLLM; install vLLM Gaudi only on the Gaudi node.

## Installation

The following installation is supported with vLLM and vLLM Gaudi versions:

| vLLM commit hash / branch                  | vLLM-Gaudi commit hash / branch            |
| ------------------------------------------ | ------------------------------------------ |
| `f21f5ea38c6fa0e824bc00d5762d17e049199cd3` | `e26948506df2ec861fd5280f0c1eefdb97a5d956` |

### CUDA Node Installation

```sh
# Install default NIXL (comes with UCX)
pip install nixl[cu12]

# Install vLLM from source
git clone https://github.com/vllm-project/vllm
cd vllm
git checkout $vllm_commit
pip install -e .
```

### Gaudi Node Installation

```sh
# Install NIXL with UCX support
curl -O https://raw.githubusercontent.com/intel-staging/ucx/refs/heads/intel_gaudi_gdr_enabling_0/setup_nixl_ucx.sh
chmod +x setup_nixl_ucx.sh
./setup_nixl_ucx.sh

# Necessary environment variables for NIXL
export LD_LIBRARY_PATH=/tmp/ucx_install/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export UCX_MEMTYPE_CACHE=0

# Install vLLM from source
git clone https://github.com/vllm-project/vllm
cd vllm && git checkout $vllm_commit
pip install -r <(sed '/^torch/d' requirements/build.txt)
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .

# Install vLLM Gaudi from source
git clone https://github.com/vllm-project/vllm-gaudi
cd vllm-gaudi && git checkout $vllm_gaudi_commit
pip install -e .
```

## Examples

For each of the example cases below, the following parameters are generalized

| Parameter            | Purpose                                                      | Scope                |
| -------------------- | ------------------------------------------------------------ | -------------------- |
| `prefill_kv_layout`  | KV cache layout for prefill (e.g., NHD or HND)               | prefill              |
| `prefill_block_size` | Block size for prefill                                       | prefill              |
| `prefill_port`       | Prefill <-> proxy communication port                         | prefill/proxy        |
| `decode_kv_layout`   | KV cache layout for decode (e.g., NHD or HND)                | decode               |
| `decode_block_size`  | Block size for decode                                        | decode               |
| `decode_port`        | Decode <-> proxy communication port                          | decode/proxy         |
| `port`               | External port exposed by the proxy                           | proxy                |
| `model`              | Define model served, must be synced (i.e. `Qwen/Qwen3-0.6B`) | prefill/decode/proxy |

The following toy proxy script is also valid for all examples and should be run on the server sending requests:

```sh
PREFILL_HOST="$cuda_ip"
PREFILL_PORT="$prefill_port"
DECODE_HOST="$gaudi_ip"
DECODE_PORT="$decode_port"
PROXY_PORT="$port"

python examples/nixl/toy_proxy_server.py \
  --port "$PROXY_PORT" \
  --prefiller-hosts "$PREFILL_HOST" \
  --prefiller-ports "$PREFILL_PORT" \
  --decoder-hosts "$DECODE_HOST" \
  --decoder-ports "$DECODE_PORT"
```

### Case 1: CUDA Prefill + Gaudi Decode

CUDA launch script:

```sh
VLLM_KV_CACHE_LAYOUT="$prefill_kv_layout"
HOST="$cuda_ip"
PORT="$prefill_port"
BLOCK="$prefill_block_size"
MODEL="$model"

export VLLM_KV_CACHE_LAYOUT
export UCX_TLS=ib,rc,cuda_copy
export UCX_MEMTYPE_CACHE=0
export LD_LIBRARY_PATH="/tmp/ucx_install/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export VLLM_NIXL_SIDE_CHANNEL_HOST=$HOST
export VLLM_NIXL_DEVICE_TO_DEVICE=1
export VLLM_USE_V1=1

vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --block-size "$BLOCK" \
  --kv-transfer-config '{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_buffer_device": "cuda",
  "kv_connector_extra_config": {"enforce_handshake_compat": false}
}'
```

Gaudi launch script:

```sh
VLLM_KV_CACHE_LAYOUT="$decode_kv_layout"
HOST="$gaudi_ip"
PORT="$decode_port"
BLOCK="$decode_block_size"
MODEL="$model"

export VLLM_KV_CACHE_LAYOUT
export UCX_TLS=ib,rc,gaudi_gdr
export UCX_MEMTYPE_CACHE=0
export LD_LIBRARY_PATH="/tmp/ucx_install/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export VLLM_NIXL_SIDE_CHANNEL_HOST=$HOST
export VLLM_NIXL_DEVICE_TO_DEVICE=1
export VLLM_USE_V1=1

vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --block-size "$BLOCK" \
  --kv-transfer-config '{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_buffer_device": "hpu",
  "enable_permute_local_kv": "True",
  "kv_connector_extra_config": {"enforce_handshake_compat": false}
}'
```

> [!NOTE]
> The installation process detailed in the [installation section](#installation) only supports either heterogeneous block size OR heterogeneous kv layout.
> [#30275](https://github.com/vllm-project/vllm/pull/30275) implements support for heterogeneous block size AND kv layout.
> To enable this, set the vLLM branch to [dev/decode_KV_post_process](https://github.com/xuechendi/vllm-fork/tree/dev/decode_KV_post_process)
> or cherry-pick the commits on both nodes before installing via `pip install`.

### Case 2: Gaudi Decode + CUDA Prefill

> [!IMPORTANT]
> This case is enabled with [#30448](https://github.com/vllm-project/vllm/pull/30448). To enable this example, set the vLLM branch to
> [dev/prefill_KV_process](https://github.com/xuechendi/vllm-fork/tree/dev/prefill_KV_process) or cherry-pick the commits on both
> nodes before installing via `pip install`.

CUDA launch script:

```sh
VLLM_KV_CACHE_LAYOUT="$decode_kv_layout"
HOST="$cuda_ip"
PORT="$decode_port"
BLOCK="$decode_block_size"
MODEL="$model"

export VLLM_KV_CACHE_LAYOUT
export UCX_TLS=ib,rc,cuda_copy
export UCX_MEMTYPE_CACHE=0
export LD_LIBRARY_PATH="/tmp/ucx_install/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export VLLM_NIXL_SIDE_CHANNEL_HOST=$HOST
export VLLM_NIXL_DEVICE_TO_DEVICE=1
export VLLM_USE_V1=1

vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --block-size "$BLOCK" \
  --kv-transfer-config '{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_buffer_device": "cuda",
  "kv_connector_extra_config": {"enforce_handshake_compat": false, "agreed_block_size": 16}
}'
```

Gaudi launch script:

```sh
VLLM_KV_CACHE_LAYOUT="$prefill_kv_layout"
HOST="$gaudi_ip"
PORT="$prefill_port"
BLOCK="$prefill_block_size"
MODEL="$model"

export VLLM_KV_CACHE_LAYOUT
export UCX_TLS=ib,rc,gaudi_gdr
export UCX_MEMTYPE_CACHE=0
export LD_LIBRARY_PATH="/tmp/ucx_install/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export VLLM_NIXL_SIDE_CHANNEL_HOST=$HOST
export VLLM_NIXL_DEVICE_TO_DEVICE=1
export VLLM_USE_V1=1

vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --block-size "$BLOCK" \
  --no-enable-prefix-caching \
  --kv-transfer-config '{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_buffer_device": "hpu",
  "enable_permute_local_kv": "True",
  "kv_connector_extra_config": {"enforce_handshake_compat": false, "agreed_block_size": 16}
}'
```

## Validation

Run the request below from the proxy host to confirm end-to-end connectivity and kv cache integrity for any example:

```bash
MODEL="$model"
PROXY_PORT="$port"

curl http://localhost:$PROXY_PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": '"$MODEL"',
    "prompt": "Mark Elliot Zuckerberg is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. Born in White Plains, New York, Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the worlds wealthiest individuals. According to Forbes, Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025, making him the second-richest individual in the world.",
    "max_tokens": 100,
    "temperature": 0
  }'
```

Example output:

```json
{
  "id": "cmpl-7039b3dc-49b9-4943-84a1-6889d2a5c2cf",
  "object": "text_completion",
  "created": 1766014426,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "index": 0,
      "text": " He has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. He has also been the subject of criticism for his actions in the past, including the use of his company's resources to support the education of his children, and for his actions in the past, including the use of his company's resources to support the education of his children, and for his actions in the past, including the use of his company's resources to support the education",
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "token_ids": null,
      "prompt_logprobs": null,
      "prompt_token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 199,
    "total_tokens": 299,
    "completion_tokens": 100,
    "prompt_tokens_details": null
  },
  "kv_transfer_params": null
}
```
