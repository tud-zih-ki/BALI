#!/bin/bash

source env_cuda121_torch212.sh
python -m venv $BALI_REPO/pyenv_inferbench_cuda121_torch212
source $BALI_REPO/pyenv_inferbench_cuda121_torch212/bin/activate

# pip install rich
pip install --upgrade pip setuptools

# Baseline torch 2.1.2 on CUDA 12.1
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy transformers sentencepiece tqdm

# VLLM
pip install vllm==0.3.3
# LLM Lingua
pip install --no-build-isolation llmlingua accelerate


# FlashDecoding from flashAttention
pip install ninja packaging

pip install flash-attn==2.3.1.post1 --no-build-isolation

# FlashDecoding from Xformers on existing torch version
# vllm comes with xformers 0.0.23post1 --> used here to avoid dependency issues
pip install fire

# openLLM
pip install attrs==23.2.0
pip install --no-build-isolation openllm==0.4.44
pip install --upgrade pydantic

# deepspeed
pip install --no-build-isolation transformers==4.41.2
pip install deepspeed==0.14.0
pip install --no-build-isolation deepspeed-mii==0.2.3 --no-build-isolation

# apply the needed patches
cd patches
bash apply_patches.sh

cd ${PWD_PREV}
