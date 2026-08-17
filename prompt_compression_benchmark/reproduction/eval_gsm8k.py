# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
import re

import yaml
from tqdm import tqdm
from utils import get_n_max_token, load_model_and_tokenizer, query_llm


def extract_ans(ans_model):
    ans_model = ans_model.split("\n")
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if "answer is" in al:
            break
    residual = list(ans_model[li + 1 :])
    ans = "\n".join(ans)
    residual = "\n".join(residual)
    return ans, residual


def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = "none"
    questions = []
    ans_pred = []
    ans_gold = []
    for l in lines:
        l = l.replace(",", "")
        if l.startswith("Q: "):
            if am is not None and a is not None:
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if test_answer(am, a):
                    acc += 1
            current_mode = "q"
            q = l
            num_q += 1
        elif l.startswith("A_model:"):
            current_mode = "am"
            am = l
        elif l.startswith("A:"):
            current_mode = "a"
            a = l
        else:
            if current_mode == "q":
                q += l
            elif current_mode == "am":
                am += l
            elif current_mode == "a":
                a += l
            else:
                raise ValueError(current_mode)

    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    if test_answer(am, a):
        acc += 1
    score = float(acc / num_q)
    print(f"num_q {num_q} correct {acc} ratio {score:.4f}")
    return {"score": score}


def get_result(text: str):
    pattern = "\d*\.?\d+"
    res = re.findall(pattern, text)
    return res[-1] if res else ""


def test_answer(pred_str, ans_str):
    pred, gold = get_result(pred_str), get_result(ans_str)
    return pred == gold


def predict(load_prompt_from: str, save_path: str, load_key: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Load question dataset
    dataset = json.load(open("../results/gsm8k/origin/gsm8k_test.json"))
    print(f"Num data: {len(dataset)}")

    # Load previous results
    results = {}
    if os.path.exists(save_path):
        results = json.load(open(save_path))

    # Load CoT demonstrations (if not zero-shot)
    if load_prompt_from is not None:
        demon_dict = json.load(open(load_prompt_from))
        if isinstance(demon_dict, dict):
            demon_dict = list(demon_dict.values())
        demonstrations = []
        for demon in demon_dict[0][load_key]:
            demonstrations.append("\n\nQuestion: " + demon)
        demonstrations = "".join(demonstrations)

    n_max_token = get_n_max_token(args.model_name)
    res_txt = ""
    for sample in tqdm(dataset):
        idx = sample["idx"]
        if idx in results or str(idx) in results:
            print(f"Sample {idx} is already processed")
            continue
        q, a = sample["question"], sample["answer"]

        # Build prompt
        query = f"Question: {q}" + "\nLet's think step by step."
        prompt = None
        if load_prompt_from is not None:
            prompt = f"Please reference the following examples to answer the math question. \n {demonstrations}"
            token_ids = tokenizer.encode(prompt)
            len2 = len(tokenizer.encode("\n\n" + query))
            if len(token_ids) > (n_max_token - args.n_max_token_ans - len2):
                half = int((n_max_token - args.n_max_token_ans - len2) / 2) - 1
                prompt = (
                    tokenizer.decode(token_ids[:half], skip_special_tokens=True)
                    + tokenizer.decode(token_ids[-half:], skip_special_tokens=True)
                )
            prompt += "\n\n"
            
        prompt = (prompt or "") + query

        answer = query_llm(
            prompt, model, args.model_name, args.n_max_token_ans, tokenizer, save_path=save_path, custom_id=idx
        )

        results[idx] = {"question": q, "model_answer": answer, "truth_answer": a}
        ans_, _ = extract_ans(answer)
        res_txt += f'Q: {q}\nA_model:\n{ans_.replace("Q:", "").replace("A:", "")}\nA:\n{a}\n\n'
    with open(save_path.replace(".json", ".txt"), "a") as fd:
        fd.write(res_txt)
    json.dump(results, open(save_path, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--n_max_token_ans", type=int, default=400)
    parser.add_argument("--num_sample", default=-1, type=int)
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
        load_from_list = scenario["load_from"] if scenario["name"] != "zero_shot" else [None]
        for load_from in load_from_list:
            print(f"Scenario: {scenario['name']}, Load from: {load_from}")
            load_prompt_from = f'{config["results_dir"]}/{scenario["name"]}/{load_from}' if load_from else None
            out_file = f'answer_{load_from.replace("compression_", "")}' if load_from else "answer_zero_shot.json"
            save_path = f'{config["results_dir"]}/{scenario["name"]}/{model_config["out_dir"]}/{out_file}'

            predict(load_prompt_from, save_path, scenario["load_key"] if "load_key" in scenario else None)
            scores = parse_pred_ans(save_path.replace(".json", ".txt"))
            json.dump(
                scores,
                open(
                    os.path.join(
                        os.path.dirname(save_path),
                        os.path.basename(save_path).replace("answer", "metrics"),
                    ),
                    "w",
                ),
            )
