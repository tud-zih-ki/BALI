#!/bin/bash
cd ..
source setup/env_cuda124_torch260.sh

export HF_HOME=`pwd`/../huggingface
export BENTOML_HOME=`pwd`/../bentoML
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BAR=1

python $BALI_REPO/bali/inferbench.py --save-slurm-config --config-file "${BALI_REPO}/examples/configs/example-opt-125m.json" --data "${BALI_REPO}/examples/data/prompts.txt" --input-len 128 --output-len 128 --batch-size 4 --generate-from-token --output-dir "./test_bali_res" 
