#!/bin/bash

# Usage:
# sbatch --mem=50GB --gres=gpu:2 --gpus-per-task=2 --cpus-per-task=12 --time=12:00:00 start_e2e_job.sh <flags (see below)>
# Example (Llama 3.1 70b, exclusive):
# sbatch --mem=512GB --gres=gpu:8 --gpus-per-task=8 --cpus-per-task=48 --time=02:30:00 --exclusive start_e2e_job.sh --vllm --name alpha_llama_vllm --target_model meta-llama/Meta-Llama-3.1-70B-Instruct --cuda_visible 7 --cuda_visible_target "0,1,2,3,4,5,6,7"

# Note:
# For llmlingua, device="balanced" speeds up compression by a lot, for llmlingua2 it doesn't make a difference
# For llmlingua2 set device="cuda" if running with HF target model

#SBATCH --job-name=e2e_benchmark_job
#SBATCH --output=logs/e2e_benchmark_job_%j.out
#SBATCH --error=logs/e2e_benchmark_job_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

VLLM=false
NAME="alpha_mistral_vllm_compute"
COMP_MODEL="llmlingua2"
TARGET_MODEL="mistralai/Mistral-7B-v0.1"
HF=false
MAX_STD=0.1
DEVICE="balanced"
CUDA_VISIBLE="0"
DEVICE_TARGET="balanced"
CUDA_VISIBLE_TARGET="1"
API_KEY="OPENAI_API_KEY"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --vllm) VLLM=true ;;
        --name) NAME="$2"; shift ;;
        --comp_model) COMP_MODEL="$2"; shift ;;
        --target_model) TARGET_MODEL="$2"; shift ;;
        --hf) HF=true ;;
        --max_std) MAX_STD="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --cuda_visible) CUDA_VISIBLE="$2"; shift ;;
        --device_target) DEVICE_TARGET="$2"; shift ;;
        --cuda_visible_target) CUDA_VISIBLE_TARGET="$2"; shift ;;
        --api_key) API_KEY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Make sure conda env llmlingua is activated
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmlingua

echo "Params of job: mem=$SBATCH_MEM_PER_NODE, gres=$SBATCH_GRES, gpus=$SBATCH_GPUS_PER_TASK, cpus=$SLURM_CPUS_PER_TASK, time=$SBATCH_TIMELIMIT, exclusive=$SBATCH_EXCLUSIVE"
if [ "$VLLM" = true ]; then
  echo "Starting vLLM server - cuda_visible_target: $CUDA_VISIBLE_TARGET, model: $TARGET_MODEL"
  VLLM_PID=$(./launch_vllm.sh --device $CUDA_VISIBLE_TARGET --model $TARGET_MODEL --background)
  echo "vLLM server started in background with PID: $VLLM_PID"

  cleanup() {
      echo "Interrupt detected. Cleaning up..."
      kill $VLLM_PID
      echo "vLLM server stopped."
      exit 0
  }

  trap cleanup SIGINT SIGTERM EXIT

  while ! curl -s http://localhost:8000/health > /dev/null; do
    echo "vLLM server not ready, waiting..."
    sleep 5
  done

  echo "vLLM server is ready. Starting benchmark..."
fi

CMD="python eval_latency_e2e.py --name \"$NAME\" --comp_model \"$COMP_MODEL\" --target_model \"$TARGET_MODEL\" --max_std \"$MAX_STD\" --device \"$DEVICE\" --cuda_visible \"$CUDA_VISIBLE\" --device_target \"$DEVICE_TARGET\" --api_key \"$API_KEY\""

if [ "$HF" = true ]; then
  CMD="$CMD --hf"
fi

if [ "$VLLM" = true ]; then
  CMD="$CMD --no_api"
fi

echo "Running command: $CMD"
eval $CMD

if [ "$VLLM" = true ]; then
  echo "Benchmark completed. Shutting down vLLM server..."
  kill $VLLM_PID

  echo "vLLM server stopped."
fi
