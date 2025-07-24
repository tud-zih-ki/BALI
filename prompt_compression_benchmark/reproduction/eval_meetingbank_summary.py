# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
from collections import defaultdict

import yaml
from metrics import evaluate_sim
from tqdm import tqdm
from utils import get_n_max_token, load_model_and_tokenizer, query_llm


def predict(load_prompt_from: str, save_path: str, load_key: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = json.load(open(load_prompt_from))
    if isinstance(data, dict):
        data = data.values()
    print(f"Num data: {len(data)}")

    results = defaultdict(dict)
    if os.path.exists(save_path):
        results = json.load(open(save_path))

    prompt = "Summarize the provided meeting transcript (which may be compressed).\n{transcript}\nSummary:"
    n_max_token = get_n_max_token(args.model_name)
    for sample in tqdm(data):
        if isinstance(sample, float):
            continue
        idx = int(sample["idx"])
        if idx in results or str(idx) in results:
            print(f"Sample {idx} is already processed")
            continue
        if args.num_sample > 0 and int(idx) > args.num_sample:
            break

        transcript = sample[load_key]
        token_ids = tokenizer.encode(transcript, bos=False)
        if len(token_ids) > n_max_token - args.n_max_token_ans:
            transcript = tokenizer.decode(token_ids[: n_max_token - args.n_max_token_ans], skip_special_tokens=True)

        query = prompt.format(transcript=transcript)
        model_summary = query_llm(
            query, model, args.model_name, args.n_max_token_ans, tokenizer=tokenizer, save_path=save_path, custom_id=idx
        )
        summary = sample["gpt4_summary"]

        results[str(idx)]["transcript"] = transcript
        results[str(idx)]["model_summary"] = model_summary
        results[str(idx)]["gpt4_summary"] = summary

    json.dump(results, open(save_path, "w"), indent=4)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--n_max_token_ans", type=int, default=400)
    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--scenarios", nargs="+", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    model_config = config["models"][args.model_name]
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, model_config["device"] if "device" in model_config else "cuda:0"
    )

    if args.scenarios is None:
        scenarios = config["scenarios"]
    else:
        scenarios = [sc for sc in config["scenarios"] if sc["name"] in args.scenarios]
    for scenario in scenarios:
        for load_from in scenario["load_from"]:
            print(f"Scenario: {scenario['name']}, Load from: {load_from}")
            load_prompt_from = f'{config["results_dir"]}/{scenario["name"]}/{load_from}'
            out_file = f"answer_{'ratio33_' if 'llmlingua' in scenario['name'] else ''}meetingbank_summary.json"
            save_path = f'{config["results_dir"]}/{scenario["name"]}/{model_config["out_dir"]}/{out_file}'

            results = predict(load_prompt_from, save_path, scenario["load_key"] if "load_key" in scenario else None)

            model_summaries = [r["model_summary"] for r in results.values()]
            gpt4_summaries = [r["gpt4_summary"] for r in results.values()]
            score_dict = evaluate_sim(model_summaries, gpt4_summaries)
            json.dump(
                score_dict,
                open(
                    os.path.join(
                        os.path.dirname(save_path),
                        os.path.basename(save_path).replace("answer", "metrics"),
                    ),
                    "w",
                ),
                indent=4,
            )
