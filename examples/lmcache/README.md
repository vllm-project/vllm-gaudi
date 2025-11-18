# LMCache Examples
Please Note: HPU integration for LMCache will be upstreamed. After that, the following test cases can be used.

This folder demonstrates how to use LMCache for disaggregated prefilling and  KV cache sharing.

The test scripts are dependent on [vllm/benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks) scripts.
Please download them and set their path in disagg_example.sh.

## 1. Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill using lm or redis on a single node.

### Prerequisites
- At least 2 HPU cards
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct
- https://github.com/LMCache/LMCache/pull/1066 needed for lmcache

### Usage

Run
`cd disagg_prefill_lmcache_v1`
to get into `disagg_prefill_lmcache_v1` folder, and then run

```bash
bash disagg_example_gaudi_lm.sh
```

to run disaggregated prefill and benchmark the performance.

lmserver is default and it's configurable as well as tensor_parallel_size and model name.

For tp>1

```bash
bash disagg_example_gaudi_lm_tp2.sh
```

### Components

#### Server Scripts
- `disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `../disagg_prefill_lmcache_v1/disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_prefill_lmcache_v1/disagg_example.sh` - Main script to run the example through lm/redis remote server

#### Configuration
- `disagg_prefill_lmcache_v1/configs/lmcache-config-lm.yaml` - Configuration for prefiller/decoder server through lm server

#### Log Files
The main script generates several log files:
- `prefiller.log` - Logs from the prefill server
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server

## 2. KV Cache Sharing

The `kv_cache_sharing_lmcache_v1.py` example demonstrates how to share KV caches between vLLM v1 instances.

### Usage

```bash
python kv_cache_sharing_lmcache_v1.py
```

lmserver is default and it's configurable as well as tensor_parallel_size.

For tp > 1

```bash
python kv_cache_sharing_lmcache_v1_tp2.py
```
