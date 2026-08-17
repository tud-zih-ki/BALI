# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
import re
from collections import defaultdict

import tiktoken
from tqdm import tqdm
import yaml
from utils import get_n_max_token, load_model_and_tokenizer, query_llm


MULTIPLE_CHOICE_TASKS = [
    "temporal_sequences",
    "disambiguation_qa",
    "date_understanding",
    "tracking_shuffled_objects_three_objects",
    "penguins_in_a_table",
    "geometric_shapes",
    "snarks",
    "ruin_names",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_five_objects",
    "logical_deduction_three_objects",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "salient_translation_error_detection",
    "reasoning_about_colored_objects",
]
FREE_FORM_TASKS = [
    "multistep_arithmetic_two",
    "navigate",
    "dyck_languages",
    "word_sorting",
    "sports_understanding",
    "boolean_expressions",
    "object_counting",
    "formal_fallacies",
    "causal_judgement",
    "web_of_lies",
]


def extract_ans(ans, mode):
    ans_line = ans.split("answer is", 1)
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()

    if mode == "multiple_choice":
        # fmt: off
        options = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)", "(O)", "(P)", "(Q)", "(R)", "(S)", "(T)", "(U)", "(V)", "(W)", "(X)", "(Y)", "(Z)"]
        match_g = []
        for option in options:
            if option in ans:
                # ans = option[1]
                match_g.append((ans.index(option), option[1]))
        if match_g:
            match_g.sort(key=lambda x: x[0])
            return match_g[0][1]
    elif mode == "free_form":
        ans = ans.split(".", 1)[0]
        if ans and ans[-1] == ".":
            ans = ans[:-1]
        return ans


def analyze_cases(good, bad, task):
    _, good_questions, good_ans_pred, good_ans_gold = good
    _, bad_questions, bad_ans_pred, bad_ans_gold = bad
    mode = "multiple_choice" if task in MULTIPLE_CHOICE_TASKS else "free_form"
    true_map, x_map = {}, {}
    for q, p, g in zip(good_questions[task], good_ans_pred[task], good_ans_gold[task]):
        p_ans, g_ans = extract_ans(p, mode), g
        if p_ans == g_ans:
            true_map[q] = (p, g, p_ans, g_ans)
        x_map[q] = (p, g, p_ans, g_ans)
    false_map = {}
    for q, p, g in zip(bad_questions[task], bad_ans_pred[task], bad_ans_gold[task]):
        p_ans, g_ans = extract_ans(p, mode), g
        if p_ans != g_ans and q in true_map:
            false_map[q] = (p, g, p_ans, g_ans)


def parse_pred_ans(path: str):
    res = open(path).read()
    pattern = "Task:(.*?)\n(.*?)\nA_model:(.*?)\nA_target:(.*?)\n\n"
    g, ans = defaultdict(int), defaultdict(list)
    questions, ans_models, ans_targets = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for m in re.findall(pattern, res, re.S):
        task, question, ans_model, ans_target = m
        task = task.strip()
        mode = "multiple_choice" if task in MULTIPLE_CHOICE_TASKS else "free_form"
        question = question.strip()
        ans_model = ans_model.strip()
        ans_target = ans_target.strip()
        p, gg = extract_ans(ans_model, mode), ans_target
        g[task] += int(p == gg)
        ans[task].append((ans_model, gg))
        questions[task].append(question)
        ans_models[task].append(ans_model)
        ans_targets[task].append(ans_target)
    scores = defaultdict(dict)
    total_num = 0
    for task, correct in g.items():
        scores[task]["acc"] = correct / len(ans[task])
        scores[task]["num"] = len(ans[task])
        print(task, correct, len(ans[task]), correct / len(ans[task]))
        total_num += len(ans[task])
    print(total_num)
    score_list = [v["acc"] for v in scores.values()]
    scores["avg"] = sum(score_list) / len(score_list)
    # return ans, questions, ans_models, ans_targets
    return scores


def get_generation_token_length(path):
    res = open(path, "r").read()
    pattern = "Task:(.*?)\n(.*?)\nA_model:(.*?)\nA_target:(.*?)\n\n"
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokens = []
    for m in re.findall(pattern, res, re.S):
        task, question, ans_model, ans_target = m
        tokens.append(len(tokenizer.encode(ans_model)))
    return sum(tokens) / len(tokens)


def save_res(results, res_txt, save_path):
    with open(save_path.replace(".json", ".txt"), "a") as fd:
        fd.write(res_txt)
    json.dump(results, open(save_path, "w"), indent=4)


def predict(load_prompt_from: str, save_path: str, load_key: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results = {}
    if os.path.exists(save_path):
        results = json.load(open(save_path))

    demonstrations = json.load(open(load_prompt_from))
    if isinstance(demonstrations, dict):
        demonstrations = list(demonstrations.values())
    prompts, instructions = {}, {}
    for demon in demonstrations:
        task = demon["task"]
        prompt = demon[load_key] if load_key is not None else ""
        instructions[task] = demon["instruction"]
        prompts[task] = prompt
    # print(prompts, "\n", instructions)

    dataset = json.load(open("../results/bbh/origin/bbh.json"))
    n_max_token = get_n_max_token(args.model_name)
    res_txt = ""
    for sample in tqdm(dataset):
        idx = sample["idx"]
        task = sample["task"]
        task_type = "multiple_choice" if task in MULTIPLE_CHOICE_TASKS else "free_form"
        cot_prompt = prompts[task]
        instruction = instructions[task]
        if args.num_sample > 0 and int(idx) > args.num_sample:
            break
        if idx in results or str(idx) in results:
            print(f"Sample {idx} is already processed")
            continue
        q, a = sample["question"], sample["answer"]

        if cot_prompt and cot_prompt[0] != "\n":
            cot_prompt = "\n\n" + cot_prompt
        prompt = f"{instruction}{cot_prompt}\n\nQ: {q}" + "\nA: Let's think step by step.\n"
        token_ids = tokenizer.encode(prompt)
        n_max_token_ans = 400 if task != "geometric_shapes" else 800
        if len(token_ids) > (n_max_token - n_max_token_ans):
            half = int((n_max_token - n_max_token_ans) / 2) - 1
            prompt = tokenizer.decode(token_ids[:half], skip_special_tokens=True) + tokenizer.decode(
                token_ids[-half:], skip_special_tokens=True
            )
        answer = query_llm(
            prompt,
            model,
            args.model_name,
            max_tokens=n_max_token_ans,
            tokenizer=tokenizer,
            save_path=save_path,
            custom_id=idx,
        )

        results[idx] = {"question": q, "model_answer": answer, "truth_answer": a}

        if task_type == "multiple_choice":
            a = a[1]
        res_txt += "%dTask:%s\n%s\nA_model:%s\nA_target:%s\n\n" % (
            idx,
            task,
            q.replace("\n", ""),
            answer.replace("\n", "").replace("Q:", "").replace("A:", ""),
            a.replace("\n", ""),
        )
        if idx % 15 == 0:
            save_res(results, res_txt, save_path)
            res_txt = ""
    save_res(results, res_txt, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", required=True)
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
    print(scenarios)
    for scenario in scenarios:
        for load_from in scenario["load_from"]:
            print(f"Scenario: {scenario['name']}, Load from: {load_from}")
            load_prompt_from = f'{config["results_dir"]}/{scenario["name"]}/{load_from}'
            out_file = f'answer_{load_from.replace("compression_", "")}'
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
