#!/bin/bash
set -e

BALI_REPO=`pwd`
if [ -n "$1" ]; then
    export BALI_ENV="$1"
else
    export BALI_ENV="${BALI_REPO}/pyenv_inferbench"
fi

export XDG_CACHE_HOME="${BALI_REPO}/.cache"
export TRITON_CACHE_DIR="${BALI_REPO}/.triton"
export BENTOML_HOME="${BALI_REPO}/.bentoml"

python -m venv ${BALI_ENV}
source ${BALI_ENV}/bin/activate

pip install --upgrade pip setuptools
pip install --upgrade pip

pip install ninja==1.11.1.4 packaging==24.2 tqdm==4.67.1 tabulate==0.9.0 pandas==2.2.3 numpy==1.26.4
pip install torch==2.6.0 transformers==4.50.3 sentencepiece==0.2.0

pip install --no-build-isolation vllm==0.8.2

pip install --no-build-isolation accelerate==1.6.0 llmlingua==0.2.2

pip install --no-build-isolation openllm==0.6.23
pip install --upgrade pydantic

pip install --no-build-isolation flash-attn==2.7.4.post1

pip install --no-build-isolation deepspeed==0.16.5 deepspeed-mii==0.3.3

deactivate

#apply the needed patches
cd ${BALI_REPO}/patches
bash apply_patches.sh

cd ${BALI_REPO}
