#!/bin/bash

# Before running evaluation: export VLLM_BASE_URL=http://localhost:8000/v1

PORT=8000
DEVICE=7
MODEL="mistralai/Mistral-7B-v0.1"
RUN_IN_BACKGROUND=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --background) RUN_IN_BACKGROUND=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES="$DEVICE"
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)

CMD="vllm serve \"$MODEL\" --port \"$PORT\" --tensor-parallel-size $NUM_DEVICES"

if [[ "$MODEL" == "mistralai/Mistral-7B-v0.1" ]]; then
    CMD="$CMD --tokenizer-mode mistral"
elif [[ "$MODEL" == "meta-llama/Meta-Llama-3.1-70B-Instruct" ]]; then
    CMD="$CMD --enable-chunked-prefill=False --max-seq-len-to-capture 16384 --num-scheduler-steps 15 --max_model_len 16384"
fi

if [ "$RUN_IN_BACKGROUND" = true ]; then
    LOG_DIR="logs/vllm"
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
    fi
    bash -c "$CMD --disable-log-requests" &> "$LOG_DIR/vllm_server_$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID.log" &
    echo $!
else
    echo "Running vLLM command: $CMD"
    eval $CMD
fi
