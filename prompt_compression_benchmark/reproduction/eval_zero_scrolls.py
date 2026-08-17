# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
import shutil
from collections import defaultdict

import datasets
import yaml
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from utils import get_n_max_token, load_model_and_tokenizer, query_llm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--num_sample", type=int, default=-1)
parser.add_argument("--scenarios", nargs="+", default=None)
args = parser.parse_args()
config = yaml.safe_load(open(args.config))

summarization_tasks = ["gov_report", "summ_screen_fd", "qmsum", "squality"]
n_max_tokens_ans = {
    "gov_report": 1024,
    "summ_screen_fd": 512,
    "qmsum": 512,
    "qasper": 128,
    "narrative_qa": 64,
    "quality": 10,
    "musique": 32,
    "squality": 512,
    "space_digest": 36,
    "book_sum_sort": 256,
}


def eval(predict_path: str):
    zero_scrolls_metric_path = hf_hub_download(
        repo_id="tau/zero_scrolls",
        repo_type="dataset",
        filename="metrics/zero_scrolls.py",
    )
    preds = json.load(open(predict_path))
    preds_g, refers_g = defaultdict(list), defaultdict(list)
    for v in preds.values():
        task, refer, pred = [v[k] for k in ["task", "reference", "pred"]]
        # if task == "narrative_qa":
        pred = (
            pred.split("\n\nQuestion:", 1)[0]
            .split("\n\nExplanation:", 1)[0]
            .replace("<|im_end|>", "")
            .replace("\end{document}", "")
            .strip()
        )
        # .split("\n\nExplanation:", 1)[0]
        if task == "space_digest":
            if pred.startswith("0.") and "%" not in pred[:4]:
                pred = "{:.2f}%".format(float(pred[:4]) * 100)
            else:
                pred = pred[:5].strip().replace("%", "") + "%"
        preds_g[task].append(pred)
        refers_g[task].append([refer])

    zero_scrolls = []
    score_dict = {}
    for task in n_max_tokens_ans.keys():
        if task not in preds_g:
            zero_scrolls.append(0)
            continue
        p, r = preds_g[task], refers_g[task]
        zero_scrolls_metric = datasets.load_metric(zero_scrolls_metric_path, task, trust_remote_code=True)
        results = zero_scrolls_metric.compute(predictions=p, references=r)
        print(task, len(p), results)
        zero_scrolls.append(results["zero_scrolls_score"])
        score_dict[task] = {
            "zero_scrolls_score": results["zero_scrolls_score"],
            "length": len(p),
        }
    print(",".join([f"{ii:.2f}" for ii in zero_scrolls]))
    score_avg = sum(zero_scrolls) / len(zero_scrolls)
    score_dict["avg"] = score_avg
    return score_dict


def insert_format_request(example, load_key, format_request):
    instruction_end_index = example[load_key].find("\n\n")
    instruction = example[load_key][:instruction_end_index]
    example[load_key] = f"{instruction.strip()} {format_request}{example[load_key][instruction_end_index:]}"
    for key in ["doc_start", "doc_end", "query_start", "query_end"]:
        example[key] += len(format_request) + 1
    return example


def build_prompt(sample, load_key, n_max_token, n_max_gen, is_chat_api):
    prompt = sample[load_key]
    if is_chat_api:
        # See ZeroScrolls paper appendix A
        if sample["task"] not in summarization_tasks:
            format_request = "Do not provide any explanation."
            sample = insert_format_request(sample, load_key, format_request)
        prompt = sample[load_key][: sample["query_end"]]

    tokenized = tokenizer.encode(prompt, bos=False)
    if len(tokenized) <= n_max_token - n_max_gen:
        return prompt

    query_and_answer_prompt = prompt[sample["query_start"] :]
    truncation_separator = sample["trunc_sep"]
    suffix_tokenized = tokenizer.encode(truncation_separator + query_and_answer_prompt, bos=False)
    max_tokens_for_input = n_max_token - len(suffix_tokenized) - n_max_gen - 1
    prompt = (
        tokenizer.decode(tokenized[:max_tokens_for_input], skip_special_tokens=True)
        + truncation_separator
        + query_and_answer_prompt
    )
    return prompt


def predict(load_prompt_from: str, save_path: str, load_key: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset = json.load(open(load_prompt_from))
    if isinstance(dataset, dict):
        dataset = dataset.values()
    print(f"Num data: {len(dataset)}")

    results = {}
    if os.path.exists(save_path):
        results = json.load(open(save_path))
    n_max_token = get_n_max_token(args.model_name)
    for sample in tqdm(dataset):
        idx = int(sample["idx"])
        if idx in results or str(idx) in results:
            print(f"Sample {idx} is already processed")
            continue
        if args.num_sample > 0 and int(idx) > args.num_sample:
            break

        max_gen = n_max_tokens_ans[sample["task"]]
        prompt = build_prompt(sample, load_key, n_max_token, max_gen, is_chat_api)
        pred = query_llm(prompt, model, args.model_name, max_gen, tokenizer, save_path=save_path, custom_id=idx)
        results[idx] = {
            "idx": idx,
            "task": sample["task"],
            "pred": pred,
            "reference": sample["answer"],
        }
    json.dump(results, open(save_path, "w"), indent=4)


model_config = config["models"][args.model_name]
is_chat_api = "chat_model" in model_config and model_config["chat_model"]
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
        out_file = f'answer_{load_from.replace("compression_", "")}'
        save_path = f'{config["results_dir"]}/{scenario["name"]}/{model_config["out_dir"]}/{out_file}'

        predict(load_prompt_from, save_path, scenario["load_key"] if "load_key" in scenario else None)
        score_dict = eval(save_path)
        json.dump(
            score_dict,
            open(
                os.path.join(
                    os.path.dirname(save_path),
                    os.path.basename(save_path).replace("answer", "metrics"),
                ),
                "w",
            ),
        )
