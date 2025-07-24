# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import atexit
import json
import os
import random
from collections import defaultdict
from time import sleep

import openai
import tiktoken
import torch
from dotenv import load_dotenv
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

load_dotenv()
SLEEP_TIME_FAILED = 5
MAX_BACKOFF = 320
SCADS_MODELS = ["meta-llama/Meta-Llama-3.1-70B-Instruct", "CohereForAI/c4ai-command-r-plus"]
VLLM_MODELS = ["mistralai/Mistral-7B-v0.1"]
BATCH_MODELS = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]


# Wrapper for the tokenizer to handle different tokenizers
class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, prompt, **kwargs):
        if isinstance(self.tokenizer, SentencePieceTokenizer):
            bos, eos = kwargs.get("bos", True), kwargs.get("eos", False)
            return self.tokenizer.encode(prompt, bos=bos, eos=eos)
        elif isinstance(self.tokenizer, tiktoken.Encoding):
            return self.tokenizer.encode(prompt)
        add_special_tokens = kwargs.get("add_special_tokens", True)
        return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    def decode(self, tokens, **kwargs):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast) or isinstance(self.tokenizer, CohereTokenizerFast):
            skip_special_tokens = kwargs.get("skip_special_tokens", False)
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        return self.tokenizer.decode(tokens)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


class BatchModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.batches_by_path = defaultdict(list)
        self.responses_by_path = {}
        atexit.register(self.save_batches)

    def save_batches(self):
        for path, batch in self.batches_by_path.items():
            with open(path, "w") as f:
                for item in batch:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def get_batch_item(self, prompt: str, custom_id: int, request: object):
        return {
            "custom_id": f"request-{custom_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], **request},
        }

    def call(self, prompt: str, custom_id: int, request: object, save_path: str):
        batch_save_path = save_path.replace(".json", "_batch.jsonl")
        response_path = save_path.replace(".json", "_output.jsonl")

        if response_path not in self.responses_by_path:
            # Unseen response_path, check if exists and load, else set to False
            if os.path.exists(response_path):
                with open(response_path) as batch_output:
                    batch_output = [json.loads(line) for line in batch_output]
                    self.responses_by_path[response_path] = {
                        item["custom_id"]: item["response"]["body"] for item in batch_output
                    }
            else:
                self.responses_by_path[response_path] = False

        if self.responses_by_path[response_path]:
            res_object = self.responses_by_path[response_path][f"request-{custom_id}"]
            return res_object["choices"][0]["message"]["content"]

        item = self.get_batch_item(prompt, custom_id, request)
        self.batches_by_path[batch_save_path].append(item)
        return "placeholder"


def query_llm(
    prompt: str,
    model: object,
    model_name: str,
    max_tokens: int,
    tokenizer: Tokenizer,
    max_retries: int = -1,
    **kwargs,
) -> str:
    request = {
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
        "seed": kwargs.get("seed", 42),
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    if isinstance(model, BatchModel):
        save_path = kwargs.get("save_path", None)
        custom_id = kwargs.get("custom_id", None)
        return model.call(prompt, custom_id, request, save_path)
    elif isinstance(model, openai.OpenAI):
        # OpenAI API compatible models
        chat_completion = kwargs.get("chat_completion", False)
        if chat_completion:
            request["messages"] = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            model = model.chat.completions
        else:
            request["prompt"] = prompt
            model = model.completions
        if model_name in SCADS_MODELS:
            del request["top_p"]  # HF TGI doesn't support top_p=1.0 but top_p=None is equivalent

        answer, response = None, None
        retries = 0
        while answer is None and (max_retries < 0 or retries < max_retries):
            try:
                response = model.create(model=model_name, **request)
                answer = response.choices[0].message.content if chat_completion else response.choices[0].text
                break
            except Exception as e:
                answer = None
                backoff_time = min(SLEEP_TIME_FAILED * (2**retries), MAX_BACKOFF) + random.uniform(0, 4)
                print(f"Error calling API: {e}, response: {response}\nRetrying in {backoff_time} seconds")
                sleep(backoff_time)
                retries += 1

        return answer
    elif isinstance(model, TextGenerationPipeline):
        # Hugging Face pipeline (greedy decoding by default)
        return model(
            prompt,
            max_new_tokens=max_tokens,
            min_new_tokens=kwargs.get("min_new_tokens", None),
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"]
    elif isinstance(model, Transformer):
        # Mistral 7b v0.1 using mistral-inference
        tokens = tokenizer.encode(prompt)
        newline_id = 13
        out_tokens, logprobs = generate(
            [tokens],
            model,
            max_tokens=max_tokens,
            temperature=request["temperature"],
            # eos_id=newline_id
        )
        result = tokenizer.decode(out_tokens[0])
        return result
    else:
        raise ValueError(f"Model {model_name} not supported.")


def load_model_and_tokenizer(
    model_name: str, device="cuda:0", hf: bool = False, no_flash_attn: bool = False
) -> tuple[object, Tokenizer]:
    print(f"Loading model {model_name}")
    if model_name in BATCH_MODELS:
        model, tokenizer = BatchModel(model_name), tiktoken.encoding_for_model(model_name)
    elif model_name.startswith("gpt-"):
        model, tokenizer = openai.OpenAI(), tiktoken.encoding_for_model(model_name)
    elif hf and model_name == "mistralai/Mistral-7B-v0.1":
        if not no_flash_attn:
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map=device,
            )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        model = pipeline("text-generation", model=model if not no_flash_attn else model_name, tokenizer=tokenizer, device_map=device)
        tokenizer = model.tokenizer
    elif model_name in [*SCADS_MODELS, *VLLM_MODELS]:
        os.environ["OPENAI_API_KEY"] = os.getenv("SCADS_API_KEY")
        os.environ["OPENAI_BASE_URL"] = (
            os.getenv("SCADS_BASE_URL")
            if model_name in SCADS_MODELS
            else os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        )
        model, tokenizer = openai.OpenAI(), AutoTokenizer.from_pretrained(model_name)
    elif model_name == "../models/mistral-7B-v0.1":
        model = Transformer.from_folder(model_name, device=device)
        tokenizer = MistralTokenizer.from_file(f"{model_name}/tokenizer.model").instruct_tokenizer.tokenizer
    else:
        raise ValueError(f"Model {model_name} not supported.")

    print(f"Model {model_name} loaded")
    return model, Tokenizer(tokenizer)


def get_n_max_token(model_name: str) -> str:
    return {
        "../models/mistral-7B-v0.1": 8100,
        "mistralai/Mistral-7B-v0.1": 8100,
        "meta-llama/Meta-Llama-3.1-70B-Instruct": 16300,
        "CohereForAI/c4ai-command-r-plus": 7150,
        "gpt-3.5-turbo": 16384,
        "gpt-3.5-turbo-0125": 16384,
        "gpt-4o-2024-08-06": 32000,
        "gpt-4o-mini-2024-07-18": 32000,
    }[model_name]
