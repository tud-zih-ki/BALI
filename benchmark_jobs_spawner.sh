#!/bin/bash
export BALI_REPO=`pwd`

# hours allocations
declare -A model_times
model_times=(
    ["facebook/opt-1.3b"]=6
    ["microsoft/phi-1_5"]=6
    ["facebook/opt-2.7b"]=8
    ["google/gemma-2b"]=8
    ["facebook/opt-6.7b"]=12
    ["mistralai/Mistral-7B-v0.3"]=12
    ["mosaicml/mpt-7b"]=12
    ["google/gemma-7b"]=12
    ["TheBloke/Llama-2-7B-fp16"]=12
)

declare -A model_memory
model_memory=(
    ["facebook/opt-1.3b"]=16
    ["microsoft/phi-1_5"]=16
    ["facebook/opt-2.7b"]=24
    ["google/gemma-2b"]=24
    ["facebook/opt-6.7b"]=32
    ["mistralai/Mistral-7B-v0.3"]=32
    ["mosaicml/mpt-7b"]=32
    ["google/gemma-7b"]=48
    ["TheBloke/Llama-2-7B-fp16"]=32
)

models=(
    "facebook/opt-1.3b"
    "microsoft/phi-1_5"
    "facebook/opt-2.7b"
    "google/gemma-2b"
    "facebook/opt-6.7b"
    "mistralai/Mistral-7B-v0.3"
    "mosaicml/mpt-7b"
    "google/gemma-7b"
    "TheBloke/Llama-2-7B-fp16"
)

BENCH_CONFIGS=$BALI_REPO/benchmark/configs
BENCH_RESULTS=$BALI_REPO/benchmark/results

mkdir -p $BENCH_CONFIGS
mkdir -p $BENCH_RESULTS

job_count=0
skipped_count=0

for model in "${models[@]}"; do
    model_name=$(echo "$model" | sed 's/.*\///')
    time_hours=${model_times[$model]}
    memory_gb=${model_memory[$model]}

    echo "Processing model: $model_name (${time_hours}h per job)"
    
    for input_pow in {0..9}; do
        input_val=$((2**$input_pow))
        
        for output_pow in {0..9}; do
            output_val=$((2**$output_pow))

            base_results_dir="${BENCH_RESULTS}/${model_name}_${input_val}_${output_val}"
            
            # Check if this benchmark has been completed successfully
            completed=false
            bench_log="${base_results_dir}/bench.log"
            
            if [[ -f "$bench_log" ]] && grep -q "Saved Benchmark summary to" "$bench_log"; then
                ((skipped_count++))
                completed=true
            fi
            
            if [ "$completed" = true ]; then
                continue
            fi
            
            config_file="${BENCH_CONFIGS}/config_${model_name}_${input_val}_${output_val}.json"
            
            # Create config file from template
            jq --arg model_name "$model_name" \
               --argjson input_len "$input_val" \
               --argjson output_len "$output_val" \
               --arg output_dir "$base_results_dir" \
               '
               .model_name = $model_name |
               .input_len = $input_len | 
               .output_len = $output_len |
               .output_dir = $output_dir |
               ' $BALI_REPO/configs/template.json > "$config_file"
            
            # Models not supported by DeepSpeed-MII
            if [[ "$model_name" == "gemma-2b" || "$model_name" == "gemma-7b" || "$model_name" == "mpt-7b" ]]; then
                jq '.frameworks = (.frameworks | map(select(. != "deepspeed")))' "$config_file" > "$config_file.tmp" && mv "$config_file.tmp" "$config_file"
            fi
            
            # Selective VLLM "downgrade"
            export_options="--export=ALL"
            if [[ "$model_name" == "opt-2.7b" || "$model_name" == "mpt-7b" ]]; then
                export_options="--export=ALL,VLLM_USE_V1=0"
            fi

            mkdir -p "$base_results_dir"
            
            job_id=$(sbatch --parsable \
                   --account=p_scads \
                   --time=${time_hours}:00:00 \
                   --nodes=1 \
                   --ntasks=1 \
                   --cpus-per-task=1 \
                   --gres=gpu:1 \
                   --mem=${memory_gb}G \
                   --job-name="bench_${model_name}_${input_val}_${output_val}" \
                   --output="${base_results_dir}/slurm_%j.out" \
                   --partition=capella \
                   $export_options \
                   $BALI_REPO/benchmark_job.slurm \
                   "$model_name" "$input_val" "$output_val" "$config_file" "$base_results_dir")
            
            echo "Scheduled job ID: $job_id for $model_name [in=$input_val, out=$output_val]"
            ((job_count++))
        done
    done
    
    echo "----------------------------------------"
done

echo "Submitted $job_count new benchmark jobs."
echo "Skipped $skipped_count benchmarks that were already completed successfully."