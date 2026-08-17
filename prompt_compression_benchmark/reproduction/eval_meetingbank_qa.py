# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import yaml
from metrics import evaluate_with_gt
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

    prompt = "Write a high-quality answer for the given question using the provided meeting transcript (which may be compressed).\n{transcript}\nQuestion:{question}\nAnswer:"
    n_max_token = get_n_max_token(args.model_name)
    with ThreadPoolExecutor(max_workers=3) as executor:
        for sample in tqdm(data):
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

            qa_list = sample["QA_pairs"]
            q_list, a_list, a_list_model = [], [], []

            def process_qa(qa, q_i: int):
                q, a = qa["question"], qa["answer"]
                query = prompt.format(transcript=transcript, question=q)
                model_a = query_llm(
                    query,
                    model,
                    args.model_name,
                    args.n_max_token_ans,
                    tokenizer=tokenizer,
                    save_path=save_path,
                    custom_id=f"{idx}_{q_i}",
                )
                return q, a, model_a

            responses = list(executor.map(process_qa, enumerate(qa_list))) if args.parallel else [process_qa(qa, i) for i, qa in enumerate(qa_list)]
            for q, a, model_a in responses:
                q_list, a_list, a_list_model = [*q_list, q], [*a_list, a], [*a_list_model, model_a]

            results[str(idx)]["transcript"] = transcript
            results[str(idx)]["questions"] = q_list[:]
            results[str(idx)]["answers"] = a_list[:]
            results[str(idx)]["model_answers"] = a_list_model[:]

    json.dump(results, open(save_path, "w"), indent=4)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--n_max_token_ans", type=int, default=100)
    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--parallel", action="store_true")
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
            out_file = f"answer_{'ratio33_' if 'llmlingua' in scenario['name'] else ''}meetingbank_QA.json"
            save_path = f'{config["results_dir"]}/{scenario["name"]}/{model_config["out_dir"]}/{out_file}'

            results = predict(load_prompt_from, save_path, scenario["load_key"] if "load_key" in scenario else None)

            model_answers = [a for r in results.values() for a in r["model_answers"]]
            answers = [[a] for r in results.values() for a in r["answers"]]
            score_dict = evaluate_with_gt(model_answers, answers)
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
