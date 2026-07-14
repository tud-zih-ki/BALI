#!/bin/bash
# TUD ZIH Alpha setup
module load release/24.04 GCCcore/11.3.0 Python/3.10.4 CUDA/12.4.0
#export HF_TOKEN=

export BALI_REPO=`pwd`
export XDG_CACHE_HOME="${BALI_REPO}/cuda121_torch212/.cache"
export TRITON_CACHE_DIR="${BALI_REPO}/cuda121_torch212/.triton"
export BENTOML_HOME="${BALI_REPO}/cuda121_torch212/.bentoml"
export HF_HOME="${BALI_REPO}/cuda121_torch212/.huggingface"