# Disaggregated Prefill Server Launcher

A flexible bash script for launching disaggregated vLLM servers for both prefill and decode operations with NIXL (Network Inference eXtended Library) support.

## Overview

This tool enables deployment of disaggregated inference systems where prefill and decode operations can be separated across different nodes/instances, improving resource utilization and scalability for large language model serving.

## Usage

### Basic Usage

```bash
# Launch prefill server with default settings
./disaggregated_prefill_server_launcher

# Launch decode server
./disaggregated_prefill_server_launcher --role decode

# Show all options
./disaggregated_prefill_server_launcher --help
```

### Common Scenarios

#### Single Node Setup

```bash
# Prefill server with 4 instances and data parallelism
./disaggregated_prefill_server_launcher --role prefill --num-instances 4 --dp-size 4

# Decode server with 4 instances and data parallelism
./disaggregated_prefill_server_launcher --role decode --num-instances 4 --dp-size 4

# Start proxy server to coordinate prefill and decode servers (with proxy bypass and validation)
python toy_proxy_server.py \
  --port 9192 \
  --prefill-instances localhost:8300,8301,8302,8303 \
  --decode-instances localhost:9300,9301,9302,9303
```

#### Multi-Node Setup

```bash
# 4-node setup: 2 prefill nodes + 2 decode nodes
# Each prefill node: 8 instances with DP=16 total
# Each decode node: 8 instances with DP=16 total

# Prefill Node 0 (DP ranks 0-7)
./disaggregated_prefill_server_launcher \
  --role prefill \
  --num-instances 8 \
  --dp-size 16 \
  --node-size 2 \
  --node-rank 0 \
  --node-ip 192.168.1.100 \
  --dp-master-ip 192.168.1.100

# Prefill Node 1 (DP ranks 8-15)
./disaggregated_prefill_server_launcher \
  --role prefill \
  --num-instances 8 \
  --dp-size 16 \
  --node-size 2 \
  --node-rank 1 \
  --node-ip 192.168.1.101 \
  --dp-master-ip 192.168.1.100

# Decode Node 0 (DP ranks 0-7)
./disaggregated_prefill_server_launcher \
  --role decode \
  --num-instances 8 \
  --dp-size 16 \
  --node-size 2 \
  --node-rank 0 \
  --node-ip 192.168.1.102 \
  --dp-master-ip 192.168.1.102

# Decode Node 1 (DP ranks 8-15)
./disaggregated_prefill_server_launcher \
  --role decode \
  --num-instances 8 \
  --dp-size 16 \
  --node-size 2 \
  --node-rank 1 \
  --node-ip 192.168.1.103 \
  --dp-master-ip 192.168.1.102
```

## Command Line Options

### Basic Options
- `-h, --help`: Show help message
- `-m, --model MODEL`: Model to serve (default: ibm-research/PowerMoE-3b)
- `-r, --role ROLE`: Server role - prefill or decode (default: prefill)
- `-n, --num-instances NUM`: Number of local instances (default: 1)

### Parallelism Options
- `-t, --tp-size SIZE`: Tensor parallel size (default: 1)
- `-d, --dp-size SIZE`: Data parallel size (default: 1)

### Network Configuration
- `--base-port PORT`: Base port for servers (default: 8300)
- `--base-channel-port PORT`: Base channel port (default: 4300)
- `--node-ip IP`: IP address of this node (default: localhost)
- `--dp-master-ip IP`: Data parallel master IP (default: localhost)
- `--dp-master-port PORT`: Data parallel master port (default: 6300)

### Multi-Node Options
- `--node-size SIZE`: Total number of nodes for this role's data parallel group (default: 1)
- `--node-rank RANK`: Data parallel node rank within this role's group (default: 0)

### NIXL Configuration
- `--nixl-buffer-device DEVICE`: Buffer device - cpu or hpu (default: cpu)
- `--nixl-backend BACKEND`: NIXL backend (default: UCX)
- `--ucx-tls TLS`: UCX transport layer (default: rc,ud,ib)

### Performance Tuning
- `--max-model-len LENGTH`: Maximum model length (default: 8192)
- `--max-num-batched-tokens TOKENS`: Maximum batched tokens (default: 8192)
- `--max-num-seqs SEQS`: Maximum number of sequences (default: 256)

### Execution Options
- `--enforce-eager`: Enable eager execution mode

## Port Allocation

The script automatically calculates ports to avoid conflicts:

- **Prefill servers**: Base port + (8 × node_rank) + instance_id
- **Decode servers**: (Base port + 1000) + (8 × node_rank) + instance_id
- **Side channels**: Base channel port + (8 × node_rank) + instance_id

### Example Port Layout

```
Node 0, Prefill instances: 8300, 8301
Node 0, Decode instances:  9300, 9301
Node 1, Prefill instances: 8308, 8309
Node 1, Decode instances:  9308, 9309
```

## Configuration Details

### Data Parallel Configuration
- `DP_SIZE=1`: No data parallelism
- `DP_SIZE=NUM_INSTANCES`: Each instance is a separate DP rank
- `DP_SIZE=NUM_INSTANCES×NODE_SIZE`: Distributed DP across nodes

### NIXL Buffer Devices
- `cpu`: Uses CPU memory for KV cache transfers
- `hpu`: Uses HPU memory with direct device-to-device transfers

## Logging

Each server instance creates a timestamped log file:

```
vllm_server_YYYYMMDD_HHMMSS_<role>_<instance>.log
```

## Router

Validated with [vllm-router](https://github.com/vllm-project/router). E.g.,

```bash
vllm-router --vllm-pd-disaggregation \
  --port 9192 \
  --prefill-policy cache_aware \
  --decode-policy cache_aware \
  --prefill http://192.168.1.100:8300 \
  --prefill http://192.168.1.100:8301 \
  --prefill http://192.168.1.100:8302 \
  --prefill http://192.168.1.100:8303 \
  --decode http://192.168.1.101:9300 \
  --decode http://192.168.1.101:9301 \
  --decode http://192.168.1.101:9302 \
  --decode http://192.168.1.101:9303 \
  --model-path ...

```
