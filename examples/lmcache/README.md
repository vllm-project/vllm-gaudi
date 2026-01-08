# LMCache Examples

LMCache is a system for optimizing large language model inference through efficient key-value cache management. The `lmcache` folder demonstrates an example usage of LMCache for disaggregated prefilling and KV cache sharing.

## Disaggregated Prefilling

This example demonstrates how to run LMCache with disaggregated prefill on a single node in vLLM v1. The example supports two cache backend options: LMCache built-in server and the external Redis server.

### Prerequisites

Before you start, make sure you have:

- At least 2 HPU cards
- Valid Hugging Face token (`HF_TOKEN`) for Llama 3.1 8B Instruct
- LMCache version supporting Intel® Gaudi® HPU.

### Usage

1. Navigate to the `disagg_prefill_lmcache_v1` directory.

    ```bash
    cd disagg_prefill_lmcache_v1
    ```

2. Run the disaggregated prefill script based on your tensor parallelism configuration:

   - Tensor parallelism equal to 1:
  
       ```bash
       bash disagg_example_gaudi_lm.sh
       ```
  
   - Tensor parallelism higher than 1:
  
       ```bash
       bash disagg_example_gaudi_lm_tp2.sh
       ```

The scripts use LMCache server as the default cache backend. You can configure the cache backend type, `tensor_parallel_size`, and the model name by editing the scripts.

### Components

The disaggregated prefill example consists of the following server scripts and configuration files.

Server scripts:

- `disagg_prefill_lmcache_v1/disagg_vllm_launcher_gaudi_lm.sh`: Launches individual prefill and decode vLLM servers and the proxy server for tensor parallelism equal to 1.
- `disagg_prefill_lmcache_v1/disagg_vllm_launcher_gaudi_lm_tp2.sh`: Launches individual prefill and decode vLLM servers and the proxy server for tensor parallelism higher than 1.
- `disagg_prefill_lmcache_v1/disagg_proxy_server.py`: Coordinates requests between the prefiller and decoder servers using FastAPI.
- `disagg_prefill_lmcache_v1/disagg_example_gaudi_lm.sh`: Runs the example using LMCache or Redis server for tensor parallelism equal to 1.
- `disagg_prefill_lmcache_v1/disagg_example_gaudi_lm_tp2.sh`: Runs the example using LMCache or Redis server for tensor parallelism higher than 1.

Configuration files:

- `disagg_prefill_lmcache_v1/configs/lmcache-config-lm.yaml`: Contains configuration for prefiller and decoder servers through the LMCache server.
- `disagg_prefill_lmcache_v1/configs/lmcache-decoder-config`: Contains LMCache configuration specific to the decoder server.
- `disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config`: Contains LMCache configuration specific to the prefiller server.

The main script generates the following log files:

- `prefiller.log`: Logs from the prefill server.
- `decoder.log`: Logs from the decode server.
- `proxy.log`: Logs from the proxy server.

## KV Cache Sharing

The `kv_cache_sharing_lmcache_v1.py` example demonstrates how to share KV caches between vLLM v1 instances.

### Usage

Run the KV cache sharing script based on your tensor parallelism configuration:

- Tensor parallelism equal to 1:

    ```bash
    python kv_cache_sharing_lmcache_v1.py
    ```

- Tensor parallelism higher than 1:  

    ```bash
    python kv_cache_sharing_lmcache_v1_tp2.py
    ```
