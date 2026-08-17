import gc
import json
import os
from collections import defaultdict

import tiktoken
import torch
from tqdm import tqdm
from llmlingua import PromptCompressor


def compress(prompt, rate, question=""):
    if model.startswith("long"):
        res = compressor.compress_prompt(
            [prompt],
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
        res = compressor.compress_prompt(prompt, rate=rate)
    return res


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model = "llmlingua2"
models = {
    "llmlingua": "NousResearch/Llama-2-7b-hf",
    "llmlingua_small": "openai-community/gpt2",
    "longllmlingua": "NousResearch/Llama-2-7b-hf",
    "longllmlingua_small": "openai-community/gpt2",
    "llmlingua2": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    "llmlingua2_small": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
}
tokenizer = tiktoken.encoding_for_model("gpt-3.5")
longbench_data = json.load(open("results/longbench/origin/longbench_test_formatted.json", "r"))
input_sizes = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 40000, 48000]
rates = [2 / 3, 1 / 2, 1 / 3, 1 / 5]
original_context = longbench_data[1665]["context"]  # Length: 51394 tokens
question = "Now, write a one-page summary of the report.\n\nSummary:"

for model in (p := tqdm(models)):
    p.set_description(f"Model: {model}")
    print(f"Loading model...\n")
    compressor = PromptCompressor(
        model_name=models[model],
        device_map="cuda" if model == "llmlingua2_small" else "balanced",
        use_llmlingua2=model.startswith("llmlingua2"),
        llmlingua2_config={"max_batch_size": 50},
    )
    print("Model loaded.")

    if model.endswith("_small"):
        out_file = f"results/rates/{model.removesuffix("_small")}/actual_rates_small.json" 
    else:
        out_file = f"results/rates/{model}/actual_rates.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if os.path.exists(out_file):
        actual_rates = json.load(open(out_file, "r"))

        def convert_keys_to_int(d, depth=0):
            if isinstance(d, dict) and depth < 2:
                return {float(k) if "." in k else int(k): convert_keys_to_int(v, depth + 1) for k, v in d.items()}
            else:
                return d

        actual_rates = convert_keys_to_int(actual_rates)
    else:
        actual_rates = defaultdict(dict)
    for size in (p2 := tqdm(input_sizes, leave=False)):
        p2.set_description(f"Size: {size}")
        context = tokenizer.decode(tokenizer.encode(original_context)[:size])
        for rate in rates:
            if rate not in actual_rates[size]:
                res = compress(context, rate, question)
                actual_rate = 1 / (
                    1 if res["compressed_tokens"] == 0 else res["origin_tokens"] / res["compressed_tokens"]
                )
                actual_rates[size][rate] = actual_rate
            else:
                print(f"Skipping {size} tokens - rate: {rate}")
    json.dump(actual_rates, open(out_file, "w"), indent=4)

    print("Cleaning up model...\n")
    del compressor
    gc.collect()
    torch.cuda.empty_cache()
