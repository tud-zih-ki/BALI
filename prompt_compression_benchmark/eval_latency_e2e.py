import argparse
import json
import os
import random
import time

import numpy as np
import requests
import tiktoken
from dotenv import load_dotenv
from llmlingua.prompt_compressor import PromptCompressor
from tqdm.contrib import itertools

load_dotenv("reproduction/.env")
tokenizer = tiktoken.encoding_for_model("gpt-3.5")

SCADS_MODELS = ["meta-llama/Meta-Llama-3.1-70B-Instruct", "CohereForAI/c4ai-command-r-plus"]
VLLM_MODELS = ["mistralai/Mistral-7B-v0.1", "meta-llama/Meta-Llama-3.1-70B-Instruct"]
OPENAI_MODELS = ["gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18"]
SLEEP_TIME_FAILED = 30
MAX_BACKOFF = 600


def query_llm(prompt: str, model_name: str, n_gen_tokens: int, **kwargs) -> str:
    if args.hf and model_name == "mistralai/Mistral-7B-v0.1":
        return model(
            prompt,
            max_new_tokens=n_gen_tokens,
            min_new_tokens=n_gen_tokens,
            return_full_text=False,
            pad_token_id=target_tok.eos_token_id,
        )[0]["generated_text"]
    elif model_name in [*SCADS_MODELS, *VLLM_MODELS, *OPENAI_MODELS]:
        # Use requests library since openai library does not support min_tokens
        api_url = (
            os.getenv("SCADS_BASE_URL")
            if not args.no_api and model_name in SCADS_MODELS
            else (
                os.getenv("OPENAI_BASE_URL")
                if not args.no_api and model_name in OPENAI_MODELS
                else "http://localhost:8000/v1"
            )
        )
        api_key = os.getenv("SCADS_API_KEY") if model_name in SCADS_MODELS else os.getenv(args.api_key)
        request = {
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 1.0),
            "max_tokens": n_gen_tokens,
            "n": 1,
            "model": model_name,
        }
        if model_name in SCADS_MODELS and not args.no_api:
            del request["top_p"]
        if model_name in OPENAI_MODELS:
            request["messages"] = [{"role": "user", "content": prompt}]
            endpoint_url = f"{api_url}/chat/completions"
        else:
            request["min_tokens"] = n_gen_tokens
            request["prompt"] = prompt
            endpoint_url = f"{api_url}/completions"

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        response = requests.post(endpoint_url, headers=headers, data=json.dumps(request))
        if response.status_code != 200:
            raise ValueError(f"Error querying model {model_name}: {response.text}")
        response = response.json()
        assert response["usage"]["completion_tokens"] == n_gen_tokens
        return (
            response["choices"][0]["message"]["content"]
            if model_name in OPENAI_MODELS
            else response["choices"][0]["text"]
        )
    raise NotImplementedError("Model not supported.")


def benchmark(
    prompt: str,
    original_context: str,
    input_size: int,
    rate: int,
    n_gen_tokens: int,
    warmup_repetitions: int,
    repetitions: int,
):
    context_length = len(tokenizer.encode(original_context))
    std_total, avg_total, retries = 999, 0, 0
    while std_total > max(args.max_std, avg_total * 0.15):
        try:
            total, compression = [], []
            for _ in range(warmup_repetitions + repetitions):
                start_idx = np.random.randint(0, context_length - input_size + 1)
                context = tokenizer.decode(tokenizer.encode(original_context)[start_idx : start_idx + input_size])
                start = time.perf_counter()
                if rate < 1:
                    res_comp = compressor.compress_prompt(context, rate=rate)
                    context = res_comp["compressed_prompt"]
                prompt = prompt.format(context=context)
                res_llm = query_llm(prompt, args.target_model, n_gen_tokens)
                end = time.perf_counter()
                total.append(end - start)
                compression.append(res_comp["timings"]["total"] if rate < 1 else 0)

            res = [
                [[np.mean(t), np.std(t), t], [np.mean(c), np.std(c), c]]
                for t, c in [
                    (total[:warmup_repetitions], compression[:warmup_repetitions]),
                    (total[warmup_repetitions:], compression[warmup_repetitions:]),
                ]
            ]
            avg_total = res[1][0][0]
            std_total = res[1][0][1]
            if std_total > max(args.max_std, avg_total * 0.15):
                print(f"Retrying, std_total: {std_total}, avg_total: {avg_total}")
        except Exception as e:
            backoff_time = min(SLEEP_TIME_FAILED * (2**retries), MAX_BACKOFF) + random.uniform(0, 4)
            print(f"Error during benchmark run - {type(e).__name__}: {e}.\nRetrying in {backoff_time} seconds")
            time.sleep(backoff_time)
            retries += 1
            continue
    return res


parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Name of the benchmark, e.g. a100, m1pro...", required=True)
parser.add_argument("--comp_model", help="LLM used to compress", default="llmlingua2")
parser.add_argument("--target_model", default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--hf", action="store_true")
parser.add_argument("--no_api", action="store_true")  # Use vLLM instance instead of API
parser.add_argument("--device", help="Device for compression model", default="balanced")
parser.add_argument("--cuda_visible", help="Visible cuda devices", default="6")
parser.add_argument("--device_target", help="Device for target model")
parser.add_argument("--batch_size", help="Batch size for LLMLingua-2", type=int, default=50)
parser.add_argument(
    "--max_std",
    help="Initial max std tolerance for a benchmark run. Either this value or avg_total * 0.15 is the maximum, whichever is bigger",
    type=float,
    default=0.1,
)
parser.add_argument("--api_key", default="OPENAI_API_KEY")
args = parser.parse_args()
args.device_target = args.device_target or args.device
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible
output_file = f"results/latency_e2e/{args.comp_model}/{args.name}/results.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Import only after setting CUDA_VISIBLE_DEVICES
from reproduction.utils import get_n_max_token, load_model_and_tokenizer

longbench_data = json.load(open("results/longbench/origin/longbench_test_formatted.json", "r"))
original_context = longbench_data[1665]["context"]  # Prompt from longbench gov_report task (length: 51394 tokens)
prompt = "Write a 200 word summary of this report:\n\n{context}\n\nNow, write the summary of the report.\n\nSummary:"  # 29 tokens
n_max_token = get_n_max_token(args.target_model)
models = {
    "llmlingua_small": "openai-community/gpt2",
    "longllmlingua_small": "openai-community/gpt2",
    "llmlingua2": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    "llmlingua2_small": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
}
input_sizes = [
    i for i in [100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 12000, 16000] if i <= n_max_token
]  # Input token sizes
rates = [1, 2 / 3, 1 / 2, 1 / 3, 1 / 5]  # 1x, 1.5x, 2x, 3x, 5x compression rates
output_sizes = [1, 10, 25, 50, 100]  # Response token sizes
warmup_repetitions = 4
repetitions = 8
print(f"Benchmark: {args.name} comp_model: {args.comp_model} target: {args.target_model} - Loading models...")

use_llmlingua2 = args.comp_model.startswith("llmlingua2")
compressor = PromptCompressor(
    model_name=models[args.comp_model],
    device_map="cuda" if args.comp_model == "llmlingua2_small" and args.device == "balanced" else args.device,
    use_llmlingua2=use_llmlingua2,
    llmlingua2_config={"max_batch_size": args.batch_size},
)
print("Compressor loaded, loading target model...")
if args.hf and args.target_model == "mistralai/Mistral-7B-v0.1":
    model, target_tok = load_model_and_tokenizer(args.target_model, device=args.device_target, hf=args.hf)

benchmark_results = {
    "prompt": prompt,
    "n_max_token": n_max_token,
    "repetitions": repetitions,
    "warmup_repetitions": warmup_repetitions,
    "target_model": args.target_model,
    "device": args.device,
    "cuda_visible": args.cuda_visible,
    "device_target": args.device_target,
    "batch_size": args.batch_size,
    "max_std": args.max_std,
    "original_context": original_context[:100],
    "results": {},
    "total_runtime": 0,
}
if os.path.exists(output_file):
    existing_results = json.load(open(output_file, "r"))

    def convert_keys_to_int(d, depth=0):
        if isinstance(d, dict) and depth < 3:
            return {float(k) if "." in k else int(k): convert_keys_to_int(v, depth + 1) for k, v in d.items()}
        else:
            return d

    benchmark_results["results"] = convert_keys_to_int(existing_results["results"])


start_time = time.perf_counter()
results = benchmark_results["results"]
for output_size, input_size, rate in (prog := itertools.product(output_sizes, input_sizes, rates)):
    print(f"## Output size: {output_size} - Input size: {input_size} - Rate: {rate}")
    if output_size in results:
        if input_size in results[output_size]:
            if rate in results[output_size][input_size]:
                std_total = results[output_size][input_size][rate]["std_total"]
                if std_total < args.max_std:
                    print("Already benchmarked, skipping...")
                    continue
                else:
                    print(f"std_total {std_total} greater than {args.max_std}, retrying benchmark...")
        else:
            results[output_size][input_size] = {}
    else:
        results[output_size] = {input_size: {}}
    [wu_total, wu_comp], [total, comp] = benchmark(
        prompt, original_context, input_size, rate, output_size, warmup_repetitions, repetitions
    )
    results[output_size][input_size][rate] = {
        "avg_total": total[0],
        "std_total": total[1],
        "avg_comp": comp[0],
        "std_comp": comp[1],
        "times_total": total[2],
        "times_comp": comp[2],
        "warmup": {
            "wu_avg_total": wu_total[0],
            "wu_std_total": wu_total[1],
            "wu_avg_comp": wu_comp[0],
            "wu_std_comp": wu_comp[1],
            "times_total": wu_total[2],
            "times_comp": wu_comp[2],
        },
    }

    print(f"Total: {total[0]:.4f}s (±{total[1]:.4f}), Compression: {comp[0]:.4f}s (±{comp[1]:.4f})\n")
    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=4)

total_runtime = time.perf_counter() - start_time
benchmark_results["results"] = results
benchmark_results["total_runtime"] += total_runtime
with open(output_file, "w") as f:
    json.dump(benchmark_results, f, indent=4)
print(f"Benchmarking results saved to {output_file}. Total runtime: {total_runtime:.2f} seconds")
