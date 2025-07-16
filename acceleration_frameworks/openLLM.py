import asyncio
import logging
from itertools import chain

import acceleration_frameworks.openLLM as openLLM
from transformers import AutoTokenizer

from acceleration_frameworks.acceleration_framework import AccelerationFramework

batch_size = 1


class OpenLLM(AccelerationFramework):
    def __init__(self, config, data, flops, generate_from_token: bool = True):
        super(OpenLLM, self).__init__(config, data, flops, generate_from_token)
        self.data = list(chain.from_iterable(self.data))

    def tokenize_data(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'], model_max_length=self.config['input_len'],
                                                  **self.config['tokenizer_init_config'])
        tokenizer.pad_token = tokenizer.eos_token
        # chain batched data again, since async engine cant use batched data
        self.tokenized_data = tokenizer(self.data, **self.config['tokenize_config'])
        self.timer.stop_pure_tokenization_timer()
        logging.info(f"Tokenized data shape:{self.tokenized_data['input_ids'].shape}")

    def setup(self):
        llm = openLLM.LLM(self.config['model_name'],
                          backend=self.config["open_llm_backend"],
                          max_new_tokens=self.config['output_len'],
                          min_length=self.config['output_len'],
                          ignore_eos=True,
                          disable_log_requests=True,
                          disable_log_stats=True,
                          batch_size=self.config['batch_size'])
        self.model = llm

    async def generate(self):
        if self.generate_from_token:
            tokens = [self.model.generate(prompt=p,
                                          prompt_token_ids=t.tolist(),
                                          model_max_length=self.config['input_len'] + self.config['output_len'],
                                          max_new_tokens=self.config['output_len'],
                                          min_length=self.config['output_len'],
                                          ignore_eos=True) for p, t in zip(self.data, self.tokenized_data['input_ids'])]
        else:
            tokens = [self.model.generate(d,
                                          max_new_tokens=self.config['output_len'],
                                          min_length=self.config['output_len'],
                                          ignore_eos=True) for d in self.data]

        tokens, _ = await asyncio.wait(tokens)
        tokens = [t.result() for t in tokens]
        print(self.model.__dict__)
        return [t.outputs[0].token_ids for t in tokens]
