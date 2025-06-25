
# Multi-node Configuration

vLLM works with a multi-node environment setup via Ray. To run models on multiple nodes, follow the procedure below.

## Prerequisites
Perform the following on all nodes:

- Install the latest [vllm-fork](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#build-and-install-vllm).

- Check if all Gaudi NIC ports are up by running:

!!! note
    Following commands should be run on the host and NOT inside the container.

```bash
cd /opt/habanalabs/qual/gaudi2/bin 
./manage_network_ifs.sh --status 
# All the ports should be in 'up' state. Try flipping the state
./manage_network_ifs.sh --down 
./manage_network_ifs.sh --up
# Give it a minute for the NIC's to flip and check the status again
```

- Set the following flags:

```bash
# Check the network interface for outbound/inbound comms. Command 'ip a' or 'ifconfig' should list all the interfaces
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
```

## 1. Start Ray on the head node:

```bash
ray start --head --port=6379
```

## 2. Add workers to the Ray cluster:

```bash
ray start --address='<ip-of-ray-head-node>:6379'
```

## 3. Start the vLLM server:

```bash
vllm serve meta-llama/Llama-3.1-405B-Instruct --dtype bfloat16 --max-model-len  2048 --block-size 128 --max-num-seqs 32 --tensor-parallel-size 16 --distributed-executor-backend ray
```

!!! note
    Running FP8 models with a multi-node setup is described in the documentation of FP8 calibration procedure: [README](https://github.com/HabanaAI/vllm-hpu-extension/blob/main/calibration/README.md).

# Other Online Serving Examples

Please refer to this [collection](https://github.com/HabanaAI/Gaudi-tutorials/tree/main/PyTorch/vLLM_Tutorials/Benchmarking_on_vLLM/Online_Static#quick-start) of static-batched online serving example scripts designed to help the user reproduce performance numbers with vLLM on Gaudi for various types of models and varying context lengths. Below is a list of the models and example scripts provided for 2K and 4K context length scenarios:
- deepseek-r1-distill-llama-70b_gaudi3_1.20_contextlen-2k
- deepseek-r1-distill-llama-70b_gaudi3_1.20_contextlen-4k
- llama-3.1-70b-instruct_gaudi3_1.20_contextlen-2k
- llama-3.1-70b-instruct_gaudi3_1.20_contextlen-4k
- llama-3.1-8b-instruct_gaudi3_1.20_contextlen-2k
- llama-3.1-8b-instruct_gaudi3_1.20_contextlen-4k
- llama-3.3-70b-instruct_gaudi3_1.20_contextlen-2k
- llama-3.3-70b-instruct_gaudi3_1.20_contextlen-4k