#!/bin/bash
[[ $# -lt 2 ]] && { echo "Usage: $0 <software_setup> <run_mode>
  software_setup: old|new
  run_mode: incomplete|failed

Available frameworks: deepspeed, hf_accelerate, llmlingua, vllm, vllm_async

Run modes:
  incomplete - Run jobs for combinations that don't have a summary file
  failed     - Rerun jobs for combinations where bench.log contains error

Examples:
  $0 old incomplete
  $0 new failed"; exit 1; }

SOFTWARE_SETUP=$1 RUN_MODE=$2
[[ "$SOFTWARE_SETUP" != "old" && "$SOFTWARE_SETUP" != "new" ]] && { echo "Error: Invalid software setup parameter. Use 'old' or 'new'"; exit 1; }
[[ "$RUN_MODE" != "incomplete" && "$RUN_MODE" != "failed" ]] && { echo "Error: Invalid run mode parameter. Use 'incomplete' or 'failed'"; exit 1; }

export BALI_REPO=$(pwd)

# Allocated time (hours) and memory (GB) for each model
declare -A models=(["facebook/opt-1.3b"]="3:16" 
                    ["microsoft/phi-1_5"]="3:16" 
                    ["facebook/opt-2.7b"]="4:24" 
                    ["google/gemma-2b"]="4:24" 
                    ["facebook/opt-6.7b"]="6:32" 
                    ["mistralai/Mistral-7B-v0.1"]="6:32" 
                    ["mosaicml/mpt-7b"]="6:32" 
                    ["google/gemma-7b"]="6:48" 
                    ["TheBloke/Llama-2-7B-fp16"]="6:32")
all_frameworks=("deepspeed" "hf_accelerate" "llmlingua" "vllm" "vllm_async")
unsupported_deepspeed=("google/gemma-2b" "google/gemma-7b" "mosaicml/mpt-7b")
# Models that require VLLM_USE_V1=0
vllm_v0_models=("facebook/opt-2.7b" "mosaicml/mpt-7b")

is_framework_compatible() { [[ "$2" == "deepspeed" && " ${unsupported_deepspeed[*]} " =~ " $1 " ]] && return 1 || return 0; }
is_job_running() { squeue --name="$1" --noheader --format="%j" 2>/dev/null | grep -q "^$1$"; }

EXPERIMENTS_DIR="$BALI_REPO/experiments_$SOFTWARE_SETUP"
mkdir -p "$EXPERIMENTS_DIR"
job_count=0 skipped_count=0

submit_job() {
    local model_name="$1" framework="$2" input_len="$3" output_len="$4" time_hours="$5" memory_gb="$6"
    is_framework_compatible "$model_name" "$framework" || { echo "Warning: Framework $framework not compatible with $model_name, skipping"; ((skipped_count++)); return; }
    
    local model_dir_name=$(echo "$model_name" | sed 's/\//_/g')
    local exp_dir="$EXPERIMENTS_DIR/$model_dir_name/$framework/$input_len/$output_len"
    local job_name="bench_${SOFTWARE_SETUP}_${model_dir_name}_${framework}_${input_len}_${output_len}"
        
    if [[ "$RUN_MODE" == "incomplete" ]]; then
        [[ -f "$exp_dir/logs/bench.log" ]] && grep -q "Saved Benchmark summary to" "$exp_dir/logs/bench.log" && { ((skipped_count++)); return; }
    elif [[ "$RUN_MODE" == "failed" ]]; then
        [[ ! -f "$exp_dir/logs/bench.log" ]] || ! grep -qi "Error for Framework" "$exp_dir/logs/bench.log" && { ((skipped_count++)); return; }
    fi

    is_job_running "$job_name" && { echo "Skipping $model_name/$framework/${input_len}/${output_len}: job already running ($job_name)"; ((skipped_count++)); return; }

    mkdir -p "$exp_dir"/{metadata,logs,results}
    # Fill template config with model name, framework,input/output lengths, and output directory
    jq --arg model_name "$model_name" \
            --argjson input_len "$input_len" \
            --argjson output_len "$output_len" \
            --arg output_dir "$exp_dir/results" \
            --argjson frameworks "[\"$framework\"]" \
            '
            .model_name = $model_name | 
            .input_len = $input_len | 
            .output_len = $output_len | 
            .output_dir = $output_dir | 
            .frameworks = $frameworks' "$BALI_REPO/configs/template.json" > "$exp_dir/config.json"
    
    local export_opts="--export=ALL"; [[ " ${vllm_v0_models[*]} " =~ " ${model_name} " ]] && export_opts="--export=ALL,VLLM_USE_V1=0"
    
    local job_id=$(sbatch --parsable \
                    --account=p_scads \
                    --time=${time_hours}:00:00 \
                    --nodes=1 --ntasks=1 \
                    --cpus-per-task=1 \
                    --gres=gpu:1 \
                    --mem=${memory_gb}G \
                    --job-name="$job_name" \
                    --output="$exp_dir/logs/slurm_%j.out" \
                    --partition="capella" \
                    $export_opts "$BALI_REPO/benchmark_job.slurm" "$exp_dir" "$SOFTWARE_SETUP")
    
    echo "Submitted job $job_id: $model_name/$framework/${input_len}/${output_len} on $SOFTWARE_SETUP"
    ((job_count++))
}

echo "Starting benchmark jobs with $SOFTWARE_SETUP software setup in '$RUN_MODE' mode"
echo "Running for all models: ${!models[*]}"
echo "Running for all frameworks: ${all_frameworks[*]}"

for model_name in "${!models[@]}"; do
    IFS=':' read -r time_hours memory_gb <<< "${models[$model_name]}"
    echo "Processing $model_name (${time_hours}h, ${memory_gb}GB)"
    for framework in "${all_frameworks[@]}"; do
        for input_pow in {0..9}; do
            for output_pow in {0..9}; do
                submit_job "$model_name" "$framework" $((2**input_pow)) $((2**output_pow)) "$time_hours" "$memory_gb"
            done
        done
    done
done

echo "Submitted $job_count new jobs, skipped $skipped_count experiments with $SOFTWARE_SETUP software setup in '$RUN_MODE' mode (reasons: already completed, already running, or incompatible frameworks)."