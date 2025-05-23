#!/bin/bash
set -e

export BALI_REPO=`pwd`
export XDG_CACHE_HOME="${BALI_REPO}/.cache"
export TRITON_CACHE_DIR="${BALI_REPO}/.triton"
export BENTOML_HOME="${BALI_REPO}/.bentoml"

python -m venv $BALI_REPO/pyenv_inferbench
source $BALI_REPO/pyenv_inferbench/bin/activate

pip install --upgrade pip setuptools
pip install --upgrade pip

pip install ninja==1.11.1.4 packaging==24.2 tqdm==4.67.1 tabulate==0.9.0 pandas==2.2.3 numpy==1.26.4
pip install torch==2.6.0 transformers==4.50.3 sentencepiece==0.2.0

pip install --no-build-isolation vllm==0.8.2

pip install --no-build-isolation accelerate==1.6.0 llmlingua==0.2.2

pip install --no-build-isolation flash-attn==2.7.4.post1

pip install --no-build-isolation deepspeed==0.16.5 deepspeed-mii==0.3.3

pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

deactivate