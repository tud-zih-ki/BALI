import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tiktoken
import torch
from llmlingua.prompt_compressor import PromptCompressor

# from memory_profiler import profile
from tqdm.contrib import itertools

tokenizer = tiktoken.encoding_for_model("gpt-3.5")


def compress(input_text: str, rate: float):
    if args.model_name.startswith("long"):
        compressor.compress_prompt(
            [input_text],
            question=question,
            rate=rate,
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
            condition_compare=True,
            context_budget="+100",
            rank_method="longllmlingua",
        )
    else:
        compressor.compress_prompt(input_text, rate=rate)


# @profile
def benchmark(original_context: str, size: int, rate: float, device: str, repetitions: int):
    context_encoded = tokenizer.encode(original_context)
    context_length = len(context_encoded)
    if args.random_sampling:
        start_idx = np.random.randint(0, context_length - size + 1)
        input_text = tokenizer.decode(tokenizer.encode(original_context)[start_idx : start_idx + size])
    else:
        input_text = tokenizer.decode(tokenizer.encode(original_context)[:size])
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)  # Reset peak memory stats
        start_mem = torch.cuda.memory_allocated(device=device)  # Memory used before
        compress(input_text, rate)
        end_mem = torch.cuda.memory_allocated(device=device)  # Memory used after
        peak_mem = torch.cuda.max_memory_allocated(device=device)  # Peak memory during the call
    elif device == "mps":
        with ThreadPoolExecutor() as executor:

            def measure_peak_memory(peak_mem):
                while not future.done():
                    current = torch.mps.current_allocated_memory()
                    peak_mem = max(peak_mem, current)
                    time.sleep(0.01)
                return peak_mem

            # Warmup
            compress(input_text, rate)
            start, end, peak = [], [], []
            for _ in range(repetitions):
                torch.mps.empty_cache()
                start_mem = torch.mps.current_allocated_memory()
                future = executor.submit(compress, input_text, rate)
                poller = executor.submit(measure_peak_memory, peak_mem=start_mem)
                peak_mem = poller.result()
                end_mem = torch.mps.current_allocated_memory()
                start, end, peak = [*start, start_mem], [*end, end_mem], [*peak, peak_mem]
            start_mem, end_mem, peak_mem = sum(start) / repetitions, sum(end) / repetitions, sum(peak) / repetitions

    res = {"start_mem": start_mem, "end_mem": end_mem, "peak_mem": peak_mem}
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the benchmark, e.g. a100, m1pro...", required=True)
    parser.add_argument("--model_name", help="LLM used to compress", default="llmlingua2")
    parser.add_argument(
        "--device",
        help='Set this to "balanced" on CUDA machines, except for LLMLingua-2-small where it has to be "cuda". On MPS, set to "mps".',
        required=True,
    )
    parser.add_argument("--cuda_visible", help="Visible cuda devices", default="7")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--random_sampling", action="store_true")
    args = parser.parse_args()
    is_cuda = args.device in ["balanced", "cuda"]
    device = "cuda" if is_cuda else "mps"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible
    models = {
        "llmlingua": "NousResearch/Llama-2-7b-hf",
        "llmlingua_small": "openai-community/gpt2",
        "longllmlingua": "NousResearch/Llama-2-7b-hf",
        "longllmlingua_small": "openai-community/gpt2",
        "llmlingua2": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "llmlingua2_small": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    }
    print(f"Benchmark: {args.model_name}_{args.name} - Loading model...")

    start_mem = torch.cuda.memory_allocated(device) if is_cuda else torch.mps.current_allocated_memory()
    compressor = PromptCompressor(
        model_name=models[args.model_name],
        device_map=args.device,
        use_llmlingua2=args.model_name.startswith("llmlingua2"),
        llmlingua2_config={"max_batch_size": args.batch_size},
    )
    end_mem = torch.cuda.memory_allocated(device) if is_cuda else torch.mps.current_allocated_memory()
    peak_mem = torch.cuda.max_memory_allocated(device) if is_cuda else torch.mps.current_allocated_memory()
    model_usage = {"start_mem": start_mem, "end_mem": end_mem, "peak_mem": peak_mem}
    print(f"Loading model - mem_usage: {model_usage}")

    input_sizes = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 40000, 48000]  # Input token sizes
    rates = [2 / 3, 1 / 2, 1 / 3, 1 / 5]  # Compression rates
    repetitions = 4
    output_file = f"./results/memory/{args.model_name}/{args.name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    longbench_data = json.load(open("results/longbench/origin/longbench_test_formatted.json", "r"))
    original_context = longbench_data[1665]["context"]  # Prompt from longbench gov_report task (length: 51394 tokens)
    question = "Now, write a one-page summary of the report.\n\nSummary:"
    benchmark_results = {
        "batch_size": args.batch_size,
        "model": args.model_name,
        "device": args.device,
        "cuda_visible": args.cuda_visible,
        "repetitions": repetitions,
        "random_sampling": args.random_sampling,
        "origina_context": original_context[:100],
        "model_usage": model_usage,
        "results": {},
    }

    start_time = time.perf_counter()
    for rate, size in (prog := itertools.product(rates, input_sizes)):
        print(f"## Benchmarking: {size} tokens - rate: {rate}")
        if rate not in benchmark_results["results"]:
            benchmark_results["results"][rate] = {}

        mem_usage = benchmark(original_context, size, rate, device, repetitions)
        benchmark_results["results"][rate][size] = mem_usage
        print(f"Input (tokens): {size} - mem_usage: {mem_usage}")
        with open(output_file, "w") as f:
            json.dump(benchmark_results, f, indent=4)

    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"Benchmarking results saved to {output_file}. Total runtime: {time.perf_counter() - start_time:.2f} seconds")
