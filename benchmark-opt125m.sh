#!/bin/bash
source modules.sh
source pyenv_inferbench/bin/activate

export HF_HOME=`pwd`/../huggingface
export BENTOML_HOME=`pwd`/../bentoML
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BAR=1

python inferbench.py --save-slurm-config --config-file 'configs/default_config_opt125m.json' --input-len 128 --output-len 128 --batch-size 4 --generate-from-token --output-dir "./test_bali_res" 
