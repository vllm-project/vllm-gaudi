# Calibrating Multiple Nodes

This procedure explains how to perform calibration for multiple Intel® Gaudi® nodes with more than 8 cards. It requires execution within a [Gaudi Pytorch container](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#use-intel-gaudi-containers).

As an example, we use the Llama 3.1 405B model running in tensor parallelism 16 mode spanning two Intel® Gaudi® 2 nodes.

## Prerequisites

Before you start:

- Familiarize with [notes and recommendations](calibration.md#notes-and-recommendations).
- Ensure that all nodes in your multi-node setup are connected to an Network File System (NFS) mount.
- Ensure you have a multi-node configuration with more than 8 cards.

## Calibration procedure

To perform calibration, follow these steps in a [Gaudi Pytorch container](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#use-intel-gaudi-containers).

1. Build and install the latest version of vLLM Hardware Plugin for Intel® Gaudi® by following the [Installation](../../getting_started/installation.md) procedure.

2. Create workspace directory on NFS, clone the calibration scripts repository, and create an empty `quant_config_buffer.json` file in the calibration directory.

    ```bash
    mkdir <nfs-mount-path>/my_workspace && cd <nfs-mount-path>/my_workspace
    git clone https://github.com/vllm-project/vllm-gaudi
    cd vllm-gaudi/calibration
    pip install -r requirements.txt
    touch quant_config_buffer.json
    ```

3. Check if all Intel® Gaudi® NIC ports are up and running by using the following commands on the host, not inside the container.

    ```bash
    cd /opt/habanalabs/qual/gaudi2/bin 
    ./manage_network_ifs.sh --status 
    # All the ports should be in the 'up' state, you may try flipping the state
    ./manage_network_ifs.sh --down 
    ./manage_network_ifs.sh --up
    # Give it a minute for the NIC to flip and check the status again
    ```

4. Set the following environment variables for all nodes to verify the network interface for inbound and outbound communication.

    ```bash
    # Use the 'ip a' or 'ifconfig' command to list all available network interfaces.
    export GLOO_SOCKET_IFNAME=eth0
    export HCCL_SOCKET_IFNAME=eth0
    export QUANT_CONFIG="<nfs-path-to-config>/quant_config_buffer.json"
    ```

5. Start a Ray cluster with enough nodes to accommodate the required tensor parallelism size.

    ```bash
    # Start Ray on the head node
    ray start --head --port=6379

    # Add worker nodes to the Ray cluster
    ray start --address='<ip-of-ray-head-node>:6379'

    # Check if the cluster has the required number of HPU's
    ray status
    ```

6. Run the model calibration script. It will create calibration measurement files in the specified output directory, organized into into subdirectories for each model.

    ```bash
    ./calibrate_model.sh -m meta-llama/Llama-3.1-405B-Instruct -d <path-to-dataset>/open_orca_gpt4_tokenized_llama.calibration_1000.pkl -o <nfs-path-to-calibration-output>/fp8_output -l 4096 -t 16 -b 128
    ```

7. Optionally, you can reduce the target tensor parallelism level by unifying the measurement scales. For example, you can perform FP8 calibration on the Llama 3.1 405B model using two Intel® Gaudi® 2 nodes with tensor parallelism set to 16, and then use the unification script to reduce the tensor parallelism to 8. To achieve this, you can add the optional `-r` parameter, to the `calibration_model.sh` script. This parameter specifies the rank number of the unified measurements. For example, to convert scales from tensor parallelism 16 to 8, set `-r 8`.

    ```bash
    ./calibrate_model.sh -m meta-llama/Llama-3.1-405B-Instruct -d <path-to-dataset>/open_orca_gpt4_tokenized_llama.calibration_1000.pkl -o <nfs-path-to-calibration-output>/fp8_output -l 4096 -t 16 -b 128 -r 8
    ```

    If you have already performed calibration, you can use the `step-5-unify_measurements` script to convert existing scales, as in the following example. In this case, the `-m <path/ID>` parameter has to be set to the calibration output directory containing the measurement files.

    ```bash
    python3 step-5-unify_measurements.py -r 8 -m <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/ -o <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/
    ```

    If the model contains Mixture of Experts (MoE) layers and is calibrated with expert parallelism, use the `-u` parameter to unify the original measurement results according to expert parallelism rules, as in the following example:

    ```bash
    python3 step-5-unify_measurements.py -r 4 -m <nfs-path-to-calibration-output>/fp8_output/model_name/g2 -o <nfs-path-to-calibration-output>/fp8_output/model_name/g2 -u
    ```

8. Serve the FP8 quantized model.

    ```bash
    export QUANT_CONFIG='<nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/maxabs_quant_g2.json'
    vllm serve meta-llama/Llama-3.1-405B-Instruct --quantization inc --kv-cache-dtype fp8_inc --tensor-parallel-size 8 --max-model-len 2048
    ```

## Recommendations for Advanced Usage for MoE Models

For models with Mixture of Experts (MoE), such as [DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d), you can run the calibration once and reuse the results across different expert parallelism and data parallelism configurations, for example, 8, 16, or 32 cards. This process requires:

1. Unifying all measurement files onto a single card (TP1).
2. Optionally, postprocessing the unified measurements to improve performance.
3. Expanding the unified results to the desired number of expert-parallel cards. The `step-6-expand-measurements` script distributes the expert measurements across the target number of cards, while other values are reused.

The following diagram presents an example in which calibration is performed on 2 cards and deployment occurs on 4 cards.

![](../../assets/calibration/unify-and-expand.png)

The following example demonstrates calibration with DeepSeek-R1 on 8 cards, followed by deployment on 16 and 32 cards.

```bash
# Unify measurements: TP8 -> TP1
python step-5-unify_measurements.py -m /path/to/measurements/deepseek-r1/g3/ -r 1 -o /path/to/measurements/deepseek-r1/g3-unified-tp1/ -u -s

# (Optional) Postprocess unified TP1
python step-3-postprocess-measure.py -m /path/to/measurements/deepseek-r1/g3-unified-tp1/ -o /path/to/measurements/deepseek-r1/g3-unified-tp1-post/ -d

# Expand to EP16TP1
python step-6-expand-measurements.py -m /path/to/measurements/deepseek-r1/g3-unified-tp1-post/ -o /path/to/measurements/deepseek-r1/g3-unified-tp1-post-expand-ep16 -w 16

# Expand to EP32TP1
python step-6-expand-measurements.py -m /path/to/measurements/deepseek-r1/g3-unified-tp1-post/ -o /path/to/measurements/deepseek-r1/g3-unified-tp1-post-expand-ep32 -w 32
```
