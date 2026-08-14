#!/bin/bash
CUDA_VER=130
TORCH_VER=213

current_path=$(pwd)
base_dir="${current_path%%/BALI*}/BALI"

if [[ "$current_path" != *"/BALI"* ]]; then
    echo "Error: not inside a BALI directory" >&2
    exit 1
fi

source $base_dir/setup/activate_pyenv.sh --quiet "$CUDA_VER" "$TORCH_VER"

if [ "$_PYENV_ACTIVATE_OK" = "1" ]; then
    echo "Env already exists and is activated: $VIRTUAL_ENV"
    echo "Skipping setup."
    return 0 2>/dev/null || exit 0
fi

echo "No existing env found for cuda${CUDA_VER}_torch${TORCH_VER}, proceeding with setup..."

python -m venv $BALI_REPO/pyenv_inferbench_cuda130_torch213
source $BALI_REPO/pyenv_inferbench_cuda130_torch213/bin/activate

pip install --upgrade pip setuptools

# Core PyTorch 
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130

# Basic ML Dependencies with precise versions
pip install transformers  sentencepiece  tqdm --no-build-isolation

# LLM Lingua
pip install --no-build-isolation llmlingua accelerate

# Flash Attention dependencies
pip install ninja packaging pandas tabulate --no-build-isolation

# Flash Attention
pip install --no-build-isolation flash-attn

# FlashInfer
pip install flashinfer-python --no-build-isolation

# DeepSpeed
pip install --no-build-isolation deepspeed  deepspeed-mii

pip install -e . --no-build-isolation

# VLLM with cuda main version 13.0
pip install --no-build-isolation vllm

echo "Setup complete!"
deactivate
