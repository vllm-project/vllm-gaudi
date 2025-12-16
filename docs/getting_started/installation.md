---
title: Installation
---

# Installation

There are three ways to run vLLM Hardware Plugin for Intel® Gaudi®:

- **Using Docker Compose**: The easiest method that requires no image building and is supported only in 1.22 and later releases on Ubuntu. For more information and detailed instructions, see the [Quick Start](quickstart/quickstart.md) guide.
- **Using a Dockerfile**: Allows building a container with the Intel Gaudi software suite using the provided Dockerfiles, either Ubuntu-based or UBI-based.
- **Building from source**: Allows installing and running vLLM directly on your Intel Gaudi machine by building from source. It's supported as a standard installation and an enhanced setup with NIXL.

This guide explains how to run vLLM Hardware Plugin for Intel Gaudi from source and using a Dockerfile.

## Requirements

Before you start, ensure that your environment meets the following requirements:

- Python 3.10
- Intel Gaudi 2 or 3 AI accelerator
- Intel Gaudi software version 1.21.0 or later

Additionally, ensure that the Gaudi execution environment is properly set up. If
it is not, complete the setup by using the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) instructions.

## Running vLLM Hardware Plugin for Intel Gaudi Using Dockerfile

## --8<-- [start:docker_quickstart]

vLLM Hardware Plugin for Intel Gaudi provides two Dockerfile options:

- **Ubuntu-based Dockerfile**: A setup for Ubuntu systems.
- **UBI-based Dockerfile**: A setup for Red Hat Enterprise Linux (RHEL) Universal Base Image (UBI) 9 environments.

Choose the option that matches your deployment environment and requirements.

### Ubuntu-based Dockerfile

Use the following commands to set up the container with the latest Intel Gaudi software suite release using the Ubuntu-based Dockerfile.

    $ docker build -f .cd/Dockerfile.ubuntu.pytorch.vllm -t vllm-hpu-env  .
    $ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --entrypoint='' --rm vllm-hpu-env

!!! tip
    If you are facing the following error: `docker: Error response from daemon: Unknown runtime specified habana.`, refer to the "Install Optional Packages" section
    of [Install Driver and Software](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html#install-driver-and-software) and "Configure Container
    Runtime" section of [Docker Installation](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#configure-container-runtime).
    Make sure you have ``habanalabs-container-runtime`` package installed and that ``habana`` container runtime is registered.

To achieve the best performance on HPU, please follow the methods outlined in the
[Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).

### UBI-based Dockerfile

The UBI-based Dockerfile is designed for RHEL UBI 9 environments and provides extensive customization through build arguments.

To build with default settings, run the following command from the repository root:

    $ docker build -f .cd/Dockerfile.rhel.ubi.vllm -t vllm-gaudi:ubi .

The Dockerfile supports the following build arguments:

| Build argument | Default value | Description |
|----------------|---------------|-------------|
| `ARTIFACTORY_URL` | `vault.habana.ai` | Intel Gaudi software repository URL. |
| `SYNAPSE_VERSION` | `1.22.2` | Intel Gaudi software suite version. |
| `SYNAPSE_REVISION` | `32` | Specific revision of the software suite. |
| `BASE_NAME` | `rhel9.6` | Base RHEL UBI image version. |
| `PT_VERSION` | `2.7.1` | PyTorch version. |
| `TORCH_TYPE` | `upstream` | PyTorch distribution type. |
| `VLLM_GAUDI_COMMIT` | `main` | vLLM Hardware Plugin for Intel Gaudi commit or branch. |
| `VLLM_PROJECT_COMMIT` | empty | Specific vLLM project commit. |

To override build arguments, use the `--build-arg` flag, as in this example:

        $ docker build -f .cd/Dockerfile.rhel.ubi.vllm -t vllm-gaudi:ubi \
                --build-arg SYNAPSE_VERSION=1.22.2 \
                --build-arg SYNAPSE_REVISION=32 \
                --build-arg PT_VERSION=2.7.1 \
                --build-arg TORCH_TYPE=upstream \
                --build-arg VLLM_GAUDI_COMMIT=main \
                --build-arg VLLM_PROJECT_COMMIT= \
                .

The UBI Dockerfile includes commented lines to install additional Python packages and copy scripts from `.cd/templates`, `.cd/entrypoints`, `.cd/server`, and `.cd/benchmark` into `/root/scripts/`. If you need these packages in your workflow, uncomment the relevant lines in `.cd/Dockerfile.rhel.ubi.vllm` and adjust the container `ENTRYPOINT`.

## --8<-- [end:docker_quickstart]

## Building vLLM Hardware Plugin for Intel Gaudi from Source

There are two ways to install vLLM Hardware Plugin for Intel Gaudi from source: a standard installation for typical usage, and an enhanced setup with NIXL for optimized performance with large-scale or distributed inference.

### Standard Plugin Deployment

1. Verify that the Intel Gaudi software was correctly installed.

        $ hl-smi # verify that hl-smi is in your PATH and each Gaudi accelerator is visible
        $ apt list --installed | grep habana # verify that habanalabs-firmware-tools, habanalabs-graph, habanalabs-rdma-core, habanalabs-thunk and habanalabs-container-runtime are installed
        $ pip list | grep habana # verify that habana-torch-plugin, habana-torch-dataloader, habana-pyhlml and habana-media-loader are installed
        $ pip list | grep neural # verify that neural-compressor is installed
  
    For more information about verification, see [System Verification and Final Tests](https://docs.habana.ai/en/latest/Installation_Guide/System_Verification_and_Final_Tests.html).

2. Run the latest Docker image from the Intel Gaudi vault as in the following code sample. Make sure to provide your versions of vLLM Hardware Plugin for Intel Gaudi, operating system, and PyTorch. Ensure that these versions are supported, according to the [Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html).
  
        docker pull vault.habana.ai/gaudi-docker/1.22.2/ubuntu22.04/habanalabs/pytorch-installer-2.7.1:latest
        docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.22.2/ubuntu22.04/habanalabs/pytorch-installer-2.7.1:latest
  
    For more information, see the [Intel Gaudi documentation](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pull-prebuilt-containers).
  
3. Get the last verified vLLM commit. While vLLM Hardware Plugin for Intel Gaudi follows the latest vLLM commits, upstream API updates may introduce compatibility issues. The saved commit has been thoroughly validated.
  
        git clone https://github.com/vllm-project/vllm-gaudi
        cd vllm-gaudi
        export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)
        cd ..
  
4. Install vLLM using `pip` or build it [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).
  
        # Build vLLM from source for empty platform, reusing existing torch installation
        git clone https://github.com/vllm-project/vllm
        cd vllm
        git checkout $VLLM_COMMIT_HASH
        pip install -r <(sed '/^torch/d' requirements/build.txt)
        VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
        cd ..
  
5. Install vLLM Hardware Plugin for Intel Gaudi from source.
  
        cd vllm-gaudi
        pip install -e .
        cd ..
  
To achieve the best performance on HPU, please follow the methods outlined in the
[Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).

### Plugin Deployment with NIXL

Verify that the Intel Gaudi software was correctly installed.

        $ hl-smi # verify that hl-smi is in your PATH and each Gaudi accelerator is visible
        $ apt list --installed | grep habana # verify that habanalabs-firmware-tools, habanalabs-graph, habanalabs-rdma-core, habanalabs-thunk and habanalabs-container-runtime are installed
        $ pip list | grep habana # verify that habana-torch-plugin, habana-torch-dataloader, habana-pyhlml and habana-media-loader are installed
        $ pip list | grep neural # verify that neural-compressor is installed
  
    For more information about verification, see [System Verification and Final Tests](https://docs.habana.ai/en/latest/Installation_Guide/System_Verification_and_Final_Tests.html).

#### Dockerfile deployment

To install vLLM Hardware Plugin for Intel Gaudi and NIXL using a Dockerfile:
  
        git clone https://github.com/vllm-project/vllm-gaudi
        docker build -t ubuntu.pytorch.vllm.nixl.latest \
          -f vllm-gaudi/.cd/Dockerfile.ubuntu.pytorch.vllm.nixl.latest vllm-gaudi
        docker run -it --rm --runtime=habana \
          --name=ubuntu.pytorch.vllm.nixl.latest \
          --network=host \
          -e HABANA_VISIBLE_DEVICES=all \
          ubuntu.pytorch.vllm.nixl.latest /bin/bash
  
#### Building Plugin with NIXL using sources

1. Get the last verified vLLM commit. While vLLM Hardware Plugin for Intel Gaudi follows the latest vLLM commits, upstream API updates may introduce compatibility issues. The saved commit has been thoroughly validated
  
        git clone https://github.com/vllm-project/vllm-gaudi
        cd vllm-gaudi
        export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)
  
2. Build vLLM from source for empty platform, reusing existing torch installation.

        cd ..
        git clone https://github.com/vllm-project/vllm
        cd vllm
        git checkout $VLLM_COMMIT_HASH
        pip install -r <(sed '/^torch/d' requirements/build.txt)
        VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
        cd ..

3. Install vLLM Hardware Plugin for Intel Gaudi from source.

        cd vllm-gaudi
        pip install -e .
  
4. Build NIXL.
  
        python install_nixl.py
  
To achieve the best performance on HPU, please follow the methods outlined in the
[Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
