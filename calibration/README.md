# FP8 Calibration Procedure

Running inference via [vLLM](https://github.com/vllm-project/vllm) on HPU with FP8 precision is achieved using [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8) package. This approach requires a model calibration procedure to generate measurements, quantization files, and configurations first. To simplify this process, we've provided the `calibrate_model.sh` script. It requires the following arguments:

- `-m`, i.e., **model stub or path:** Path to your model (if stored locally) or the model ID from the Hugging Face Hub.
- `-d`, i.e., **path to the source dataset:** Path to your dataset in pickle format (".pkl").
- `-o`, i.e., **output path:** Path to the directory where the generated measurements, etc., will be stored.

There are also optional arguments, and you can read about them by executing the script with the `-h` option.

The calibration procedure works with any dataset that contains following fields: `system_prompt` and `question`. These fields are used to prepare a calibration dataset with prompts formatted specifically for your model. We recommend to use a public dataset used by MLCommons in Llama2-70b inference submission: https://github.com/mlcommons/inference/tree/master/language/llama2-70b#preprocessed.

> [!TIP]
> For the [DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) series models, which contains 256 experts, it’s important to provide a diverse and
> sufficiently large sample set to ensure that all experts are properly activated during calibration.
> Through our experiments, we found that using [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) and selecting 512 samples with at least 1024 tokens each yields good calibration coverage.

## Options and Usage

To run the ```calibrate_model.sh``` script, follow the steps below:

1. Build and install latest [vllm-plugin](https://vllm-gaudi.readthedocs.io/en/latest/getting_started/installation.html).
2. Go ```calibration``` subdirectory:

```bash
cd calibration
pip install -r requirements.txt
```

1. Download the dataset.
> [!NOTE]
> For [DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) series models, it is recommended to use `NeelNanda/pile-10k` as the dataset.

1. Run the ```calibrate_model.sh``` script. Refer to the script options and run examples below. The script generates the ```maxabs_quant_g3.json``` file, which is used for FP8 inference.

### Here are some examples of how to use the script:

```bash
./calibrate_model.sh -m /path/to/local/llama3.1/Meta-Llama-3.1-405B-Instruct/ -d dataset-processed.pkl -o /path/to/measurements/vllm-benchmarks/inc -b 128 -t 8 -l 4096
# OR
./calibrate_model.sh -m facebook/opt-125m -d dataset-processed.pkl -o inc/
# OR Calibrate DeepSeek models with dataset NeelNanda/pile-10k
PT_HPU_LAZY_MODE=1 ./calibrate_model.sh -m deepseek-ai/DeepSeek-R1  -d NeelNanda/pile-10k -o inc/ -t 8
```

> [!WARNING]
> Measurements are device-dependent, so you can't use scales collected on Gaudi3 on Gaudi2 accelerators. This behavior can cause accuracy issues.

> [!TIP]
> If you get following error, ensure you set a valid tensor parallelism value, e.g. `-t 8`:
>
> ```
> RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed for size::939524096 (896)MB
> ```

# Run inference with FP8 models

An inference with FP8 precision models using vLLM has been described in [Documentation](https://vllm-gaudi.readthedocs.io/en/latest/configuration/model_calibration.html).

# Multi-node FP8 Calibration

Following section details the procedure for calibrating models that do not fit into a single Gaudi node. For illustration we have used the Llama 3.1 405B model running in Tensor Parallelism(TP)-16 mode spanning two Gaudi2 nodes.<br>

> [!NOTE]
> Following steps are to be executed within a [Gaudi Pytorch container](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#use-intel-gaudi-containers)

## Step 1: Pre-requisites

- Install latest [vllm-plugin](https://vllm-gaudi.readthedocs.io/en/latest/getting_started/installation.html)
- Ensure that all nodes in the multi-node setup are connected to an NFS mount (Network File System).
- Create workspace directory on NFS, clone the calibration scripts repo and create an empty file `quant_config_buffer.json`.

```bash
mkdir <nfs-mount-path>/my_workspace && cd <nfs-mount-path>/my_workspace
cd <path-to-vllm-gaudi>/calibration
touch quant_config_buffer.json
```

- Check if all Gaudi NIC ports are up <br>
Note : Following commands should be run on the host and NOT inside the container. <br>

```bash
cd /opt/habanalabs/qual/gaudi2/bin 
./manage_network_ifs.sh --status 
# All the ports should be in 'up' state. Try flipping the state
./manage_network_ifs.sh --down 
./manage_network_ifs.sh --up
# Give it a minute for the NIC's to flip and check the status again
```

- Set following envs at all nodes:

```bash
# Check the network interface for outbound/inbound comms. Command 'ip a' or 'ifconfig' should list all the interfaces
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export QUANT_CONFIG="<nfs-path-to-config>/quant_config_buffer.json"
```

### Step 2: Start a Ray cluster to accommodate the required TP size.

```bash
# Start Ray on head node
ray start --head --port=6379

# Add worker nodes to the Ray cluster
ray start --address='<ip-of-ray-head-node>:6379'

# Check if the cluster has required number of HPU's
ray status
```

#### Step 3: Run model calibration script

```bash
./calibrate_model.sh -m meta-llama/Llama-3.1-405B-Instruct -d <path-to-dataset>/open_orca_gpt4_tokenized_llama.calibration_1000.pkl -o <nfs-path-to-calibration-output>/fp8_output -l 4096 -t 16 -b 128
```

Running the above command will create calibration measurement files in the specified output directory, organized into model-specific subdirectories.

> [!NOTE]
> The current calibration procedure works correctly only when the multi-node configuration has more than 8 cards.

#### Step 4: (Optional) Measurement unification

This is an optional step and is used to reduce the target tensor parallelism level by unifying the measurement scales. For example, you can perform FP8 calibration on the Llama 3.1 405B model using 2x Gaudi2 nodes with Tensor Parallelism (TP) set to 16, and then use the unification script to reduce the TP to 8. This can be achieved in two ways:
1. Add `-r` optional parameter to `calibration_model.sh` script, e.g.

```bash
./calibrate_model.sh -m meta-llama/Llama-3.1-405B-Instruct -d <path-to-dataset>/open_orca_gpt4_tokenized_llama.calibration_1000.pkl -o <nfs-path-to-calibration-output>/fp8_output -l 4096 -t 16 -b 128 -r 8
```

1. If calibration has already been performed, use the following command to convert existing scales:

```bash
python3 step-5-unify_measurements.py -r 8  -m <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/ -o <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/
```

- `-r`, i.e. **rank number** of unified measurements.
- `-m`, i.e. **calibration output path** containing the measurement files.
- `-o`, i.e. **unification output directory** where unification output will be written.
- `-u`, i.e. unify original measurement results based on **expert parallelism** rules.

> [!TIP]
> It is a good practice to store unification results in the source directory. This allows you to run the vLLM server with FP8 precision and different TP values without modifying the directory specified in the `QUANT_CONFIG` environment variable.

Below examples in case you want to convert scales from TP=16 to TP=4 and 2:
- conversion of scales TP=16 -> TP=4:

```bash
python3 step-5-unify_measurements.py -r 4  -m <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/ -o <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/
```

- conversion of scales TP=16 -> TP=2:

```bash
python3 step-5-unify_measurements.py -r 2  -m <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/ -o <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/
```

In case the model contains MoE layers and is calibrated with expert parallelism, `-u` is required for unification:

```bash
python3 step-5-unify_measurements.py -r 4 -m <nfs-path-to-calibration-output>/fp8_output/model_name/g2 -o <nfs-path-to-calibration-output>/fp8_output/model_name/g2 -u
```

#### Step 5: Serving the FP8 quantized model

```bash
export QUANT_CONFIG='<nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/maxabs_quant_g2.json'
vllm serve meta-llama/Llama-3.1-405B-Instruct --quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu --tensor-parallel-size 8 --max-model-len 2048
```

> [!NOTE]
> Detailed information about serving with vLLM (including multi-node serving) you can find in [Documentation](https://vllm-gaudi.readthedocs.io/en/latest/configuration/model_calibration.html).

#### Advanced Usage for MoE Models

For models with Mixture of Experts (MoE), like Deepseek-R1, you may want to run calibration once and use the results for different expert parallelism and data parallelism scenarios (e.g., 8, 16, or 32 cards). To do this:

1. Unify all measurement files onto a single card (TP1).
2. (Optional) Postprocess the unified measurement for better performance.
3. Expand the unified results to the number of expert-parallel cards you need. The `step-6-expand-measurements.py` splits expert measurements across the target number of cards, while other values are reused.

The diagram below shows an example where calibration is done on 2 cards and deployment is on 4 cards.

![unify-and-expand](./unify-and-expand.png)

Here is a real example that calibrates Deepseek-R1 on 8 cards and deploys on 16 or 32 cards:

```bash
# Unify measurements: TP8 -> TP1
python step-5-unify_measurements.py -m  /path/to/measurements/deepseek-r1/g3/ -r 1 -o /path/to/measurements/deepseek-r1/g3-unified-tp1/  -u -s

# (Optional) Postprocess unified TP1
python step-3-postprocess-measure.py -m /path/to/measurements/deepseek-r1/g3-unified-tp1/ -o /path/to/measurements/deepseek-r1/g3-unified-tp1-post/ -d

# Expand to EP16TP1
python step-6-expand-measurements.py -m /path/to/measurements/deepseek-r1/g3-unified-tp1-post/ -o /path/to/measurements/deepseek-r1/g3-unified-tp1-post-expand-ep16 -w 16

# Expand to EP32TP1
python step-6-expand-measurements.py -m /path/to/measurements/deepseek-r1/g3-unified-tp1-post/ -o /path/to/measurements/deepseek-r1/g3-unified-tp1-post-expand-ep32 -w 32
```
