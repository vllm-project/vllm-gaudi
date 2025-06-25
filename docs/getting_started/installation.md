---
title: Installation
---
[](){ #installation }
This guide provides instructions on running vLLM with Intel Gaudi devices.

## Requirements

- Python 3.10
- Intel Gaudi 2 or 3 AI accelerators
- Intel Gaudi software version 1.21.0 or above

!!! note
    To set up the execution environment, please follow the instructions in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).
    To achieve the best performance on HPU, please follow the methods outlined in the
    [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).


## Quick Start Using Dockerfile
# --8<-- [start:docker_quickstart]
Set up the container with the latest Intel Gaudi Software Suite release using the Dockerfile.

=== "Ubuntu"

    ```
    $ docker build -f Dockerfile.hpu -t vllm-hpu-env  .
    $ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --rm vllm-hpu-env
    ```

=== "Red Hat Enterprise Linux for Use with Red Hat OpenShift AI"

    ```
    $ docker build -f Dockerfile.hpu.ubi -t vllm-hpu-env  .
    $ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --rm vllm-hpu-env
    ```

!!! tip
    If you are facing the following error: `docker: Error response from daemon: Unknown runtime specified habana.`, please refer to the "Install Optional Packages" section
    of [Install Driver and Software](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html#install-driver-and-software) and "Configure Container
    Runtime" section of [Docker Installation](https://docs.habana.ai/en/latest/Installation_Guide/Installation_Methods/Docker_Installation.html#configure-container-runtime).
    Make sure you have ``habanalabs-container-runtime`` package installed and that ``habana`` container runtime is registered.
# --8<-- [end:docker_quickstart]

## Build from Source

### Environment Verification
To verify that the Intel Gaudi software was correctly installed, run the following:

```{.console}
$ hl-smi # verify that hl-smi is in your PATH and each Gaudi accelerator is visible
$ apt list --installed | grep habana # verify that habanalabs-firmware-tools, habanalabs-graph, habanalabs-rdma-core, habanalabs-thunk and habanalabs-container-runtime are installed
$ pip list | grep habana # verify that habana-torch-plugin, habana-torch-dataloader, habana-pyhlml and habana-media-loader are installed
$ pip list | grep neural # verify that neural-compressor is installed
```

Refer to [System Verification and Final Tests](https://docs.habana.ai/en/latest/Installation_Guide/System_Verification_and_Final_Tests.html) for more details.

### Run Docker Image

It is highly recommended to use the latest Docker image from the Intel Gaudi vault.
Refer to the [Intel Gaudi documentation](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pull-prebuilt-containers) for more details.

Use the following commands to run a Docker image. Make sure to update the versions below as listed in the [Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html):

```{.console}
docker pull vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

### Build and Install vLLM

Currently, multiple ways are provided which can be used to install vLLM with Intel® Gaudi®:

=== "Stable vLLM-fork version"

    vLLM releases are being performed periodically to align with Intel® Gaudi® software releases. The stable version is released with a tag, and supports fully validated features and performance optimizations in Gaudi's [vLLM-fork](https://github.com/HabanaAI/vllm-fork). To install the stable release from [HabanaAI/vLLM-fork](https://github.com/HabanaAI/vllm-fork), run the following:

    ```{.console}
    git clone https://github.com/HabanaAI/vllm-fork.git
    cd vllm-fork
    git checkout v0.7.2+Gaudi-1.21.0
    pip install -r requirements-hpu.txt
    python setup.py develop
    ```

=== "Latest vLLM-fork"

    Currently, the latest features and performance optimizations are being developed in Gaudi's [vLLM-fork](https://github.com/HabanaAI/vllm-fork) and periodically upstreamed to the vLLM main repository.
    To install latest [HabanaAI/vLLM-fork](https://github.com/HabanaAI/vllm-fork), run the following:

    ```{.console}
    git clone https://github.com/HabanaAI/vllm-fork.git
    cd vllm-fork
    git checkout habana_main
    pip install --upgrade pip
    pip install -r requirements-hpu.txt
    python setup.py develop
    ```

=== "vLLM Upstream"

    If you prefer to build and install directly from the main vLLM source, where periodically we are upstreaming new features, run the following:

    ```{.console}
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    pip install -r requirements-hpu.txt
    python setup.py develop
    ```

=== "[EXPERIMENTAL] vLLM Upstream + Plugin"

    You're on the bleeding edge, good luck to you:

    ```{.console}
    VLLM_TARGET_DEVICE=hpu pip install git+https://github.com/HabanaAI/vllm-fork.git@dev/upstream_vllm_for_plugin
    pip uninstall -y triton
    git clone -b plugin_poc https://github.com/HabanaAI/vllm-hpu-extension.git vllm-hpu
    cd vllm-hpu
    pip install -e .
    ```

