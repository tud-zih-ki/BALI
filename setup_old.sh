#!/bin/bash

source env_old.sh
python -m venv $BALI_REPO/pyenv_inferbench_old
source $BALI_REPO/pyenv_inferbench_old/bin/activate

pip install --upgrade pip setuptools
pip install --upgrade pip

# Core PyTorch 2.1.2 on CUDA 12.1
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Basic ML Dependencies with precise versions
pip install numpy==1.26.4 sentencepiece==0.2.0 tqdm==4.67.1 

# VLLM
pip install vllm==0.3.3

# LLM Lingua
pip install --no-build-isolation llmlingua==0.2.2 accelerate==1.7.0

# Flash Attention dependencies
pip install ninja==1.11.1.4 packaging==25.0

# Flash Attention
pip install flash-attn==2.3.1.post1 --no-build-isolation

# Utilities
pip install pandas tabulate

# DeepSpeed
pip install --no-build-isolation transformers==4.41.2 setuptools==69.5.1
pip install deepspeed==0.14.0
pip install deepspeed-mii==0.2.3

# OpenLLM - to be fixed
# pip install --no-build-isolation openllm==0.4.44
# pip install --upgrade pydantic

echo "Setup complete!" 
deactivate