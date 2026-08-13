#!/bin/bash
# TUD ZIH Capella setup
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0

#export HF_TOKEN=

current_path=$(pwd)
base_dir="${current_path%%/BALI*}/BALI"

if [[ "$current_path" != *"/BALI"* ]]; then
    echo "Error: not inside a BALI directory" >&2
    exit 1
fi

export BALI_REPO="${current_path%%/BALI*}/BALI"
export XDG_CACHE_HOME="${BALI_REPO}/cuda130_torch213/.cache"
export TRITON_CACHE_DIR="${BALI_REPO}/cuda130_torch213/.triton"
export BENTOML_HOME="${BALI_REPO}/cuda130_torch213/.bentoml"
export HF_HOME="${BALI_REPO}/cuda130_torch213/.huggingface"
# VLLM_FLASH_ATTN_VERSION=2

if [ -d "${BALI_REPO}/pyenv_inferbench_cuda130_torch213" ]; then
        echo "Activating BALI environment ${BALI_REPO}/pyenv_inferbench_cuda130_torch213"
	source $BALI_REPO/pyenv_inferbench_cuda130_torch213/bin/activate

else
        echo "BALI environment does not exist."

fi
