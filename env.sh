#!/bin/bash
# TUD ZIH Capella setup
#module load release/24.04 GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0
#export HF_TOKEN=

export BALI_REPO=`pwd`
export XDG_CACHE_HOME="${BALI_REPO}/cuda126_torch260/.cache"
export TRITON_CACHE_DIR="${BALI_REPO}/cuda126_torch260/.triton"
export BENTOML_HOME="${BALI_REPO}/cuda126_torch260/.bentoml"
export HF_HOME="${BALI_REPO}/cuda126_torch260/.huggingface"
# VLLM_FLASH_ATTN_VERSION=2 

python -m venv $BALI_REPO/pyenv_inferbench_cuda126_torch260
source $BALI_REPO/pyenv_inferbench_cuda126_torch260/bin/activate
