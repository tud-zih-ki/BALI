#!/bin/bash
# TUD ZIH Alpha setup
module load release/24.04 GCCcore/11.3.0 Python/3.10.4 CUDA/12.4.0
#export HF_TOKEN=

export BALI_REPO=`pwd`
export XDG_CACHE_HOME="${BALI_REPO}/old/.cache"
export TRITON_CACHE_DIR="${BALI_REPO}/old/.triton"
export BENTOML_HOME="${BALI_REPO}/old/.bentoml" 