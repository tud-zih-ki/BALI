#!/bin/bash

# Usage:
# sbatch --time=12:00:00 start_repro_job.sh --scenario origin <other flags (see below)>

#SBATCH --job-name=repro_benchmark_job
#SBATCH --output=logs/repro_benchmark_job_%j.out
#SBATCH --error=logs/repro_benchmark_job_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=75GB
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6

BENCHMARK="longbench"
MODEL_NAME="mistralai/Mistral-7B-v0.1"
SCENARIO="origin"
VLLM=false
DEVICE="0"
PORT=8000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --benchmark) BENCHMARK="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        --scenario) SCENARIO="$2"; shift ;;
        --vllm) VLLM=true ;;
        --device) DEVICE="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Make sure conda env llmlingua is activated
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmlingua

if [ "$VLLM" = true ]; then
  echo "Starting vLLM server, model: $MODEL_NAME"
  VLLM_PID=$(./launch_vllm.sh --device $DEVICE --port $PORT --model $MODEL_NAME --background)
  echo "vLLM server started in background with PID: $VLLM_PID"

  cleanup() {
      echo "Interrupt detected. Cleaning up..."
      kill $VLLM_PID
      echo "vLLM server stopped."
      exit 0
  }

  trap cleanup SIGINT SIGTERM EXIT

  while ! curl -s http://localhost:$PORT/health > /dev/null; do
    echo "vLLM server not ready, waiting..."
    sleep 5
  done

  echo "vLLM server is ready. Starting benchmark..."
fi

cd reproduction

BENCH_SCRIPT="eval_$BENCHMARK.py"
if [[ "$BENCHMARK" == *"_"* ]]; then
  BENCHMARK=${BENCHMARK%%_*}
fi
BENCH_CONFIG="scripts/evaluate_$BENCHMARK.yaml"
CMD="python $BENCH_SCRIPT --config \"$BENCH_CONFIG\" --model_name \"$MODEL_NAME\" --scenarios \"$SCENARIO\""

export VLLM_BASE_URL="http://localhost:$PORT/v1"
echo "Running command: $CMD"
eval $CMD
