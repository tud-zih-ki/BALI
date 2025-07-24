# Requires the following branch of LLMLingua: https://github.com/cornzz/LLMLingua/tree/timings

import argparse
import json
import os
import time

import numpy as np
import tiktoken
import torch
from llmlingua.prompt_compressor import PromptCompressor
from tqdm.contrib import itertools

tokenizer = tiktoken.encoding_for_model("gpt-3.5")


def benchmark(original_context: str, size: int, rate: float, warmup_repetitions: int, repetitions: int):
    context_length = len(tokenizer.encode(original_context))
    total, model = [], []
    for _ in range(warmup_repetitions + repetitions):
        if args.device == "mps":
            torch.mps.empty_cache()
        start_idx = np.random.randint(0, context_length - size + 1)
        input_text = tokenizer.decode(tokenizer.encode(original_context)[start_idx : start_idx + size])
        if args.model.startswith("long"):
            res = compressor.compress_prompt(
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
            res = compressor.compress_prompt(input_text, rate=rate)
        total.append(res["timings"]["total"])
        model.append(res["timings"]["model"])
    res = [
        [[np.mean(t), np.std(t), t], [np.mean(m), np.std(m), m]]
        for t, m in [
            (total[:warmup_repetitions], model[:warmup_repetitions]),
            (total[warmup_repetitions:], model[warmup_repetitions:]),
        ]
    ]

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the benchmark, e.g. a100, m1pro...", required=True)
    parser.add_argument("--model", help="LLM used to compress", default="llmlingua2")
    parser.add_argument("--device", default="balanced")
    parser.add_argument("--cuda_visible", help="Visible cuda devices", default="6")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_input_size", type=int, default=48000)
    args = parser.parse_args()

    models = {
        "llmlingua": "NousResearch/Llama-2-7b-hf",
        "llmlingua_small": "openai-community/gpt2",
        "longllmlingua": "NousResearch/Llama-2-7b-hf",
        "longllmlingua_small": "openai-community/gpt2",
        "llmlingua2": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "llmlingua2_small": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    }

    input_sizes = [
        s
        for s in [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 40000, 48000]
        if s <= args.max_input_size
    ]  # Input token sizes
    rates = [2 / 3, 1 / 2, 1 / 3, 1 / 5]  # Compression rates
    repetitions = 8
    warmup_repetitions = 4
    output_file = f"./results/latency/{args.model}/{args.name}/{args.name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    longbench_data = json.load(open("results/longbench/origin/longbench_test_formatted.json", "r"))
    original_context = longbench_data[1665]["context"]  # Prompt from longbench gov_report task (length: 51394 tokens)
    question = "Now, write a one-page summary of the report.\n\nSummary:"
    benchmark_results = {
        "repetitions": repetitions,
        "warmup_repetitions": warmup_repetitions,
        "batch_size": args.batch_size,
        "model": args.model,
        "device": args.device,
        "cuda_visible": args.cuda_visible,
        "origina_context": original_context[:100],
        "results": {},
        "total_runtime": 0,
    }

    if os.path.exists(output_file):
        print(f"Results already exist at {output_file}. Loading...")
        benchmark_results = json.load(open(output_file, "r"))

        def convert_keys_to_int(d, depth=0):
            if isinstance(d, dict) and depth < 2:
                return {float(k) if "." in k else int(k): convert_keys_to_int(v, depth + 1) for k, v in d.items()}
            else:
                return d

        benchmark_results["results"] = convert_keys_to_int(benchmark_results["results"])

    print(f"Benchmark: {args.model}_{args.name} - Loading model...")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible
    compressor = PromptCompressor(
        model_name=models[args.model],
        device_map="cuda" if args.model == "llmlingua2_small" and args.device == "balanced" else args.device,
        use_llmlingua2=args.model.startswith("llmlingua2"),
        llmlingua2_config={"max_batch_size": args.batch_size},
    )
    start_time = time.perf_counter()
    results = benchmark_results["results"]
    for rate, size in (prog := itertools.product(rates, input_sizes)):
        print(f"## Benchmarking: {size} tokens - rate: {rate}")
        if rate in results:
            if size in results[rate]:
                print(f"Already benchmarked, skipping...")
                continue
        else:
            results[rate] = {}

        [wu_total, wu_model], [total, model] = benchmark(original_context, size, rate, warmup_repetitions, repetitions)
        results[rate][size] = {
            "avg_total": total[0],
            "std_total": total[1],
            "avg_model": model[0],
            "std_model": model[1],
            "times_total": total[2],
            "times_model": model[2],
            "warmup": {
                "wu_avg_total": wu_total[0],
                "wu_std_total": wu_total[1],
                "wu_avg_model": wu_model[0],
                "wu_std_model": wu_model[1],
                "times_total": wu_total[2],
                "times_model": wu_model[2],
            },
        }

        print(
            f"Input (tokens): {size} - total: {total[0]:.4f}s (±{total[1]:.4f}), "
            f"model: {model[0]:.4f}s (±{model[1]:.4f})\n"
        )
        with open(output_file, "w") as f:
            json.dump(benchmark_results, f, indent=4)

    total_runtime = time.perf_counter() - start_time
    benchmark_results["results"] = results
    benchmark_results["total_runtime"] += total_runtime
    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"Benchmarking results saved to {output_file}. Total runtime: {total_runtime:.2f} seconds")
