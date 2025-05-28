#!/bin/bash

EXP_DIR=$1
SOFTWARE_SETUP=$2
export BALI_REPO=$(pwd)

# Use old or new software setup
source env_${SOFTWARE_SETUP}.sh
PYENV_DIR="./pyenv_inferbench_${SOFTWARE_SETUP}"

# Extract experiment info from directory structure
# experiments_{old|new}/model_name/framework/input_size/output_size/
exp_path_parts=($(echo "$EXP_DIR" | tr '/' ' '))
model_name="${exp_path_parts[-4]}"
framework="${exp_path_parts[-3]}"
input_size="${exp_path_parts[-2]}"
output_size="${exp_path_parts[-1]}"

echo "Starting benchmark: $model_name/$framework/${input_size}/${output_size} with $SOFTWARE_SETUP software setup"

# Collect metadata before running benchmark
printenv > "$EXP_DIR/metadata/environment.txt"
module list 2> "$EXP_DIR/metadata/loaded_modules.txt"

{
    echo "=== SLURM Job Information ==="
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
    echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
    echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
    echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
    echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
    echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
    echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
    echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
    echo -e "\n=== GPU Information ==="
    nvidia-smi
    echo -e "\n=== CPU Information ==="
    lscpu
    echo -e "\n=== Memory Information ==="
    free -h
    echo -e "\n=== System Information ==="
    uname -a
    cat /etc/os-release
} > "$EXP_DIR/metadata/hardware_info.txt"

# Python environment and package versions
cd "$BALI_REPO"
source "$PYENV_DIR/bin/activate"
{
    echo "=== Python Version ==="
    python --version
    echo -e "\n=== Pip List ==="
    pip list
    echo -e "\n=== Virtual Environment Path ==="
    which python
    echo "$VIRTUAL_ENV"
} > "$EXP_DIR/metadata/python_environment.txt"

# Copy SLURM script itself for reproducibility
cp "$BALI_REPO/benchmark_job.slurm" "$EXP_DIR/metadata/slurm_script.sh"

# Save the exact config used
cp "$EXP_DIR/config.json" "$EXP_DIR/metadata/config.json"

echo "Metadata collection completed."

echo "Starting benchmark execution..."

python inferbench.py --config-file "$EXP_DIR/config.json" > "$EXP_DIR/logs/bench.log" 2>&1

echo "Completed benchmark: $model_name/$framework/${input_size}/${output_size} with $SOFTWARE_SETUP software setup"