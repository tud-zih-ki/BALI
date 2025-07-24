#!/bin/bash

# Run with:
# sbatch --mem=16GB --time=02:00:00 start_comp_job.sh <flags (see below)>

# Note:
# For llmlingua, device="balanced" speeds up compression by a lot, for llmlingua2 it doesn't make a difference

#SBATCH --job-name=comp_benchmark_job
#SBATCH --output=logs/comp_benchmark_job_%j.out
#SBATCH --error=logs/comp_benchmark_job_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --account=p_gptx

NAME="alpha"
MODEL="llmlingua2"
DEVICE="balanced"
CUDA_VISIBLE="0"
EXTRA_ARGS=""
TMUX=false
TMUX_SESSION="slurm_job_$SLURM_JOB_ID"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) NAME="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --cuda_visible) CUDA_VISIBLE="$2"; shift ;;
        --tmux) TMUX=true ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1" ;;
    esac
    shift
done

CMD="python eval_latency.py --name \"$NAME\" --model \"$MODEL\" --device \"$DEVICE\" --cuda_visible \"$CUDA_VISIBLE\" $EXTRA_ARGS"

if [ "$TMUX" = true ]; then
    echo "Running command in tmux session $TMUX_SESSION: $CMD"

    tmux new-session -d -s $TMUX_SESSION "$CMD"

    while tmux has-session -t $TMUX_SESSION 2>/dev/null; do
        sleep 10
    done

    echo "Tmux session has ended, job complete."
else
    echo "Running command: $CMD"
    eval $CMD
fi