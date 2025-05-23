#!/bin/bash

MODEL_NAME=$1
INPUT_VAL=$2
OUTPUT_VAL=$3
CONFIG_FILE=$4
RESULTS_DIR=$5

export BALI_REPO=`pwd`
export XDG_CACHE_HOME=$BALI_REPO/.cache
export TRITON_CACHE_DIR=$BALI_REPO.triton
export BENTOML_HOME=$BALI_REPO/.bentoml

module load release/24.04  GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0 NCCL/2.22.3-CUDA-12.6.0

echo "Starting benchmark for $MODEL_NAME with input=$INPUT_VAL, output=$OUTPUT_VAL"

cd $BALI_REPO
source ./pyenv_inferbench/bin/activate
huggingface-cli login --token $HF_TOKEN
python inferbench.py --config-file "$CONFIG_FILE" > "${RESULTS_DIR}/bench.log" 2>&1

echo "Completed benchmark for $MODEL_NAME with input=$INPUT_VAL, output=$OUTPUT_VAL"