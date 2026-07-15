#!/bin/bash
# TUD ZIH Alpha setup
module load release/24.04 GCCcore/11.3.0 Python/3.10.4 CUDA/12.4.0
#export HF_TOKEN=


current_path=$(pwd)
base_dir="${current_path%%/BALI*}/BALI"

if [[ "$current_path" != *"/BALI"* ]]; then
    echo "Error: not inside a BALI directory" >&2
    exit 1
fi

export BALI_REPO="${current_path%%/BALI*}/BALI"
export XDG_CACHE_HOME="${BALI_REPO}/cuda121_torch212/.cache"
export TRITON_CACHE_DIR="${BALI_REPO}/cuda121_torch212/.triton"
export BENTOML_HOME="${BALI_REPO}/cuda121_torch212/.bentoml"
export HF_HOME="${BALI_REPO}/cuda121_torch212/.huggingface"

if [ -d "${BALI_REPO}/pyenv_inferbench_cuda121_torch212" ]; then
	echo "Activating BALI environment ${BALI_REPO}/pyenv_inferbench_cuda121_torch212"
	source $BALI_REPO/pyenv_inferbench_cuda121_torch212/bin/activate
else
	echo "BALI env does not exist."

fi

