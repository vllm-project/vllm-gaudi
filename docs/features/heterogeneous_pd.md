# PD Disaggregation on CUDA+Gaudi Multi‑Node System

## Overview

PD Disaggregation enables splitting model execution into prefill and decode stages,
allowing heterogeneous compute utilization. Currently, we only support CUDA for prefill
nodes and Gaudi for decode nodes, with the reverse configuration currently still
in progress.

## Requirements

All experiments were tested in docker environments with `--privileged` to allow `UCX_TLS=rc,ib`,
and `--net=host --ipc=host` to allow for network connections. The following docker images
were used for testing:

- CUDA: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
- Gaudi: `vault.habana.ai/gaudi-docker/1.22.2/ubuntu24.04/habanalabs/pytorch-installer-2.7.1`

## Installation

The installation script for building NIXL with proper UCX support can be obtained [here](https://raw.githubusercontent.com/intel-staging/ucx/refs/heads/intel_gaudi_gdr_enabling_0/setup_nixl_ucx.sh).

```sh
curl https://raw.githubusercontent.com/intel-staging/ucx/refs/heads/intel_gaudi_gdr_enabling_0/setup_nixl_ucx.sh -O
chmod +x ./setup_nixl_ucx.sh
./setup_nixl_ucx.sh
```

Post install, these environment variables are required for NIXL to register UCX properly and must be set for both nodes:

```sh
export LD_LIBRARY_PATH=/tmp/ucx_install/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export UCX_MEMTYPE_CACHE=0
```

Lastly, follow standard vLLM installation of vLLM on the CUDA node and vLLM + vLLM Gaudi on the Gaudi node.

## Launching Services

Launching requires three independent services:

1. CUDA Prefill Service (launched in CUDA node with ip `<cuda_ip>`)
2. Gaudi Decode Service (launched in Gaudi node with ip `<gaudi_ip>`)
3. Proxy Service linking the two (launched from anywhere with network access to the two nodes with those IP addresses)

| Name           | Description                                                                  | Node(s)        |
| -------------- | ---------------------------------------------------------------------------- | -------------- |
| `kv_layout`    | KV cache layout for each node, can be different between CUDA/Gaudi (NHD/HND) | prefill/decode |
| `block_size`   | Block size for each node, can be different between CUDA/Gaudi                | prefill/decode |
| `decode_port`  | Port for communications between decode and proxy services                     | decode/proxy   |
| `prefill_port` | Port for communications between prefill and proxy services                   | prefill/proxy  |
| `port`         | Port exposed for external requests by proxy                                  | proxy          |

For the prefill (CUDA) service:

```sh
# Configuration
VLLM_KV_CACHE_LAYOUT="<prefill_kv_layout>"
HOST="<cuda_ip>"
PORT="<prefill_port>"
BLOCK="<prefill_block_size>"
MODEL=Qwen/Qwen3-0.6B

# Exports
export VLLM_KV_CACHE_LAYOUT
export UCX_TLS=ib,rc,cuda_copy # Proper ucx config
export UCX_MEMTYPE_CACHE=0
export LD_LIBRARY_PATH="/tmp/ucx_install/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export VLLM_NIXL_SIDE_CHANNEL_HOST=$HOST

vllm serve $MODEL \
  --port $PORT \
  --gpu-memory-utilization 0.8 \
  --block-size $BLOCK \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"enforce_handshake_compat": false}}'
```

For the decode (Gaudi) service:

```sh
# Configuration
VLLM_KV_CACHE_LAYOUT="<decode_kv_layout>"
HOST="<gaudi_ip>"
PORT="<decode_port>"
BLOCK="<decode_block_size>"
MODEL=Qwen/Qwen3-0.6B

# Exports
export VLLM_KV_CACHE_LAYOUT
export UCX_TLS=ib,rc,gaudi_gdr # Proper ucx config
export UCX_MEMTYPE_CACHE=0
export LD_LIBRARY_PATH="/tmp/ucx_install/lib:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export VLLM_NIXL_SIDE_CHANNEL_HOST=$HOST

vllm serve $MODEL \
  --port $PORT \
  --gpu-memory-utilization 0.8 \
  --block-size $BLOCK \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_buffer_device": "hpu", "enable_permute_local_kv": "True", "kv_connector_extra_config": {"enforce_handshake_compat": false}}'
```

For the proxy service:

```sh
# Configuration
PREFILL_HOST="<cuda_ip>"
PREFILL_PORT="<prefill_port>"
DECODE_HOST="<gaudi_ip>"
DECODE_PORT="<decode_port>"
PROXY_PORT="<port>"

python toy_proxy_server.py \
  --port $PROXY_PORT \
  --prefiller-hosts $PREFILL_HOST \
  --prefiller-ports $PREFILL_PORT \
  --decoder-hosts $DECODE_HOST \
  --decoder-ports $DECODE_PORT
```

## Configuration

Two validated configurations currently exist:

1. Homogeneous layout: `kv_layout=NHD` and `block_size=64`
2. Heterogeneous layout (heterogeneous KV layout _AND_ block size): `prefill_kv_layout=HND`, `decode_kv_layout=NHD`, `prefill_block_size=16`, `decode_block_size=128`

## Verification

After launching all three services, the following curl command can be run to validate the setup:

```bash
# On proxy service node
MODEL=Qwen/Qwen3-0.6B
PROXY_PORT="<port>"

curl http://localhost:$PROXY_PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '
{
  "model": '"$MODEL"',
  "prompt": "Mark Elliot Zuckerberg is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. Born in White Plains, New York, Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the worlds wealthiest individuals. According to Forbes, Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025, making him the second-richest individual in the world.",
  "max_tokens": 100,
  "temperature": 0
}'
```
