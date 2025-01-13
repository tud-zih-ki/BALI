#!/bin/bash

python -m virtualenv --system-site-packages "../bali_final"
source ../bali_final/bin/activate
pip install --upgrade pip setuptools
# # pip install rich
pip install --upgrade pip


# # # # # #Baseline torch 2.1.2 on CUDA 12.1
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy transformers sentencepiece tqdm

# # # # # #VLLM
pip install vllm==0.3.3
#LLM Lingua
pip install --no-build-isolation llmlingua accelerate


# # # # # #FlashDecoding from flashAttention
pip install ninja packaging

pip install flash-attn==2.3.1.post1 --no-build-isolation

# # #FlashDecoding from Xformers on existing torch version
# # # #vllm comes with xformers 0.0.23post1 --> used here to avoid dependency issues
pip install fire
cd acceleration_frameworks/external_scripts
git clone git@github.com:facebookresearch/xformers.git


#openLLM
pip install --no-build-isolation openllm==0.4.44
pip install --upgrade pydantic

# deepspeed
pip install --no-build-isolation transformers==4.41.2 setuptools==69.5.1
pip install deepspeed==0.14.0
pip install deepspeed-mii

#apply the needed patches
cd patches
bash apply_patches.sh
