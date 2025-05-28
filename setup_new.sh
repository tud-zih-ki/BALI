#!/bin/bash

source env_new.sh
python -m venv $BALI_REPO/pyenv_inferbench_new
source $BALI_REPO/pyenv_inferbench_new/bin/activate

pip install --upgrade pip setuptools
pip install --upgrade pip

# Core PyTorch 2.6.0 on CUDA 12.6
pip install torch==2.6.0

# Basic ML Dependencies with precise versions
pip install numpy==1.26.4 transformers==4.50.3 sentencepiece==0.2.0 tqdm==4.67.1

# VLLM
pip install --no-build-isolation vllm==0.8.2

# LLM Lingua
pip install --no-build-isolation llmlingua==0.2.2 accelerate==1.6.0

# Flash Attention dependencies
pip install ninja==1.11.1.4 packaging==24.2

# Flash Attention
pip install --no-build-isolation flash-attn==2.7.4.post1

# Utilities
pip install pandas tabulate

# FlashInfer
pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

# DeepSpeed
pip install --no-build-isolation deepspeed==0.16.5 deepspeed-mii==0.3.3

echo "Setup complete!"
deactivate