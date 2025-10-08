> [!IMPORTANT]  
> This is an early developer preview of the vLLM Gaudi Plugin and is not yet intended for general use. For a more stable experience, consider using the [HabanaAI/vllm-fork](https://github.com/HabanaAI/vllm-fork) or the in-tree Gaudi implementation available in [vllm-project/vllm](https://github.com/vllm-project/vllm).

# Welcome to vLLM x Intel Gaudi

<p align="center">
  <img src="./docs/assets/logos/vllm-logo-text-light.png" alt="vLLM" width="30%">
  <span style="font-size: 24px; font-weight: bold;">x</span>
  <img src="./docs/assets/logos/gaudi-logo.png" alt="Intel-Gaudi" width="30%">
</p>

vLLM Gaudi plugin (vllm-gaudi) integrates Intel Gaudi accelerators with vLLM to optimize large language model inference.

This plugin follows the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162) and [[RFC]: Enhancing vLLM Plugin Architecture](https://github.com/vllm-project/vllm/issues/19161) principles, providing a modular interface for Intel Gaudi hardware.

Learn more:

📚 [Intel Gaudi Documentation](https://docs.habana.ai/en/v1.21.1/index.html)  
🚀 [vLLM Plugin System Overview](https://docs.vllm.ai/en/latest/design/plugin_system.html)

## Getting Started
1. Get Last good commit on vllm
   NOTE: vllm-gaudi is always follow latest vllm commit, however, vllm upstream
   API update may crash vllm-gaudi, this commit saved is verified with vllm-gaudi
   in a hourly basis

    ```bash
    git clone https://github.com/vllm-project/vllm-gaudi
    cd vllm-gaudi
    export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)
    ```

2. Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

    ```bash
    # Build vLLM from source for empty platform, reusing existing torch installation
    git clone https://github.com/vllm-project/vllm
    cd vllm
    git checkout $VLLM_COMMIT_HASH
    pip install -r <(sed '/^[torch]/d' requirements/build.txt)
    wget -nv https://vault.habana.ai/artifactory/gaudi-installer/[latest-version]/habanalabs-installer.sh
    chmod +x habanalabs-installer-[latest-version].sh
    export HABANALABS_VIRTUAL_DIR=[YOUR_PYTHON_VENv]
    ./habanalabs-installer-[latest-version].sh install --type pytorch --venv
    VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
    cd ..
    ```

3. Install vLLM-Gaudi from source:

    ```bash
    cd vllm-gaudi
    pip install -e .
    cd ..
    ```

4. (Optional) Install nixl:

    ```bash
    cd vllm-gaudi
    python install_nixl.sh
    cd ..
    ```

## Install with Docker file

```bash
docker build -t ubuntu.pytorch.vllm.nixl.latest \
  -f .cd/Dockerfile.ubuntu.pytorch.vllm.nixl.latest github.com/vllm-project/vllm-gaudi
docker run -it --rm --runtime=habana \
  --name=ubuntu.pytorch.vllm.nixl.latest \
  --network=host \
  -e HABANA_VISIBLE_DEVICES=all \
  vllm-gaudi-for-llmd /bin/bash
```

### Full installation from source (vLLM and vLLM-Gaudi):

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
