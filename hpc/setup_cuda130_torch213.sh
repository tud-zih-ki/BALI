#!/bin/bash

source hpc/env_cuda130_torch213.sh
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
