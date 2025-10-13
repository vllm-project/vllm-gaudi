---
title: Installation
---

This guide provides instructions on running vLLM with Intel Gaudi devices.

## Requirements

- Python 3.10
- Intel Gaudi 2 or 3 AI accelerators
- Intel Gaudi software version 1.22.0 or above

!!! note
    To set up the execution environment, please follow the instructions in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).
    To achieve the best performance on HPU, please follow the methods outlined in the
    [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).

## Running vLLM on Gaudi with Docker Compose

Starting with the 1.22 release, we are introducing ready-to-run container images that bundle vLLM and Gaudi software. Please follow the [instruction](https://github.com/vllm-project/vllm-gaudi/tree/main/.cd) to quickly launch vLLM on Gaudi using a prebuilt Docker image and Docker Compose, with options for custom parameters and benchmarking.

## Quick Start Using Dockerfile

## --8<-- [start:docker_quickstart]

Set up the container with the latest Intel Gaudi Software Suite release using the Dockerfile.

=== "Ubuntu"

    ```
    $ docker build -f .cd/Dockerfile.ubuntu.pytorch.vllm -t vllm-hpu-env  .
    $ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --entrypoint='' --rm vllm-hpu-env
    ```

!!! tip
    If you are facing the following error: `docker: Error response from daemon: Unknown runtime specified habana.`, please refer to the "Install Optional Packages" section
    of [Install Driver and Software](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html#install-driver-and-software) and "Configure Container
    Runtime" section of [Docker Installation](https://docs.habana.ai/en/latest/Installation_Guide/Installation_Methods/Docker_Installation.html#configure-container-runtime).
    Make sure you have ``habanalabs-container-runtime`` package installed and that ``habana`` container runtime is registered.

## --8<-- [end:docker_quickstart]

## Build from Source

### Environment Verification

To verify that the Intel Gaudi software was correctly installed, run the following:

    $ hl-smi # verify that hl-smi is in your PATH and each Gaudi accelerator is visible
    $ apt list --installed | grep habana # verify that habanalabs-firmware-tools, habanalabs-graph, habanalabs-rdma-core, habanalabs-thunk and habanalabs-container-runtime are installed
    $ pip list | grep habana # verify that habana-torch-plugin, habana-torch-dataloader, habana-pyhlml and habana-media-loader are installed
    $ pip list | grep neural # verify that neural-compressor is installed

Refer to [System Verification and Final Tests](https://docs.habana.ai/en/latest/Installation_Guide/System_Verification_and_Final_Tests.html) for more details.

### Run Docker Image

It is highly recommended to use the latest Docker image from the Intel Gaudi vault.
Refer to the [Intel Gaudi documentation](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pull-prebuilt-containers) for more details.

Use the following commands to run a Docker image. Make sure to update the versions below as listed in the [Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html):

    docker pull vault.habana.ai/gaudi-docker/1.22.0/ubuntu22.04/habanalabs/pytorch-installer-2.7.1:latest
    docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.22.0/ubuntu22.04/habanalabs/pytorch-installer-2.7.1:latest

### Build and Install vLLM

=== "Step 1: Get Last good commit on vllm"

    !!! note
        Vllm-gaudi always follows the latest vllm commit. However, updates to the upstream vLLM
        API may cause vLLM-Gaudi to crash. This saved commit has been verified with vLLM-Gaudi
        on an hourly basis.

    ```bash
    git clone https://github.com/vllm-project/vllm-gaudi
    cd vllm-gaudi
    export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)
    ```

=== "Step 2: Install vLLM"

    Install vLLM with `pip` or  [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source)
    
    ```bash
    # Build vLLM from source for empty platform, reusing existing torch installation
    git clone https://github.com/vllm-project/vllm
    cd vllm
    git checkout $VLLM_COMMIT_HASH
    pip install -r <(sed '/^[torch]/d' requirements/build.txt)
    VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
    cd ..
    ```

=== "Step 3: Install vLLM Plugin"

    Install  vLLM-Gaudi from source:
    ```bash
    cd vllm-gaudi
    pip install -e .
    cd ..
    ```

### Build and Install vLLM with nixl

=== "Install vLLM Plugin with nixl"

    ```bash
    cd vllm-gaudi
    python install_nixl.sh
    cd ..
    ```

=== "Install vLLM Gaudi and nixl with Docker file"

    ```bash
    docker build -t ubuntu.pytorch.vllm.nixl.latest \
      -f .cd/Dockerfile.ubuntu.pytorch.vllm.nixl.latest github.com/vllm-project/vllm-gaudi
    docker run -it --rm --runtime=habana \
      --name=ubuntu.pytorch.vllm.nixl.latest \
      --network=host \
      -e HABANA_VISIBLE_DEVICES=all \
      vllm-gaudi-for-llmd /bin/bash
    ```

=== "Full installation from source vLLM Gaudi with nixl"

    ```bash
    # Fetch last good commit on vllm
    git clone https://github.com/vllm-project/vllm-gaudi
    cd vllm-gaudi
    export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)
    
    # Build vLLM from source for empty platform, reusing existing torch installation
    git clone https://github.com/vllm-project/vllm
    cd vllm
    git checkout $VLLM_COMMIT_HASH
    pip install -r <(sed '/^[torch]/d' requirements/build.txt)
    VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
    cd ..
    
    # Build vLLM-Gaudi from source
    cd vllm-gaudi
    pip install -e .
    
    # Build nixl
    python install_nixl.sh
    ```
