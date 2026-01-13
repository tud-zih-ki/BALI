import logging

import torch
import tqdm
from llmlingua import PromptCompressor
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from acceleration_frameworks.acceleration_framework import AccelerationFramework


class LLMLingua(AccelerationFramework):
    def __init__(self, config, data, generate_from_token: bool = True, random_tokens = True):
        super(LLMLingua, self).__init__(config, data, generate_from_token, random_tokens)

    def tokenize_data(self):
        logging.info('Note that for acceleration via prompt compression the given input len is always ignored.')
        tokenized_batch = []
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        tokenizer.pad_token = tokenizer.eos_token
        for b in self.data:
            inputs = tokenizer(b, padding='longest', return_tensors='pt')
            tokenized_batch.append(inputs)
        self.timer.stop_pure_tokenization_timer()
        logging.info(f"Input IDs shape per batch:{[t['input_ids'].shape[1] for t in tokenized_batch]}")
        self.tokenized_data = [inputs.to(self.device) for inputs in tokenized_batch]

    def setup(self):
        model = AutoModelForCausalLM.from_pretrained(self.config['model_name'])
        gen_config = GenerationConfig(max_new_tokens=self.config['output_len'],
                                      min_new_token=self.config['input_len'],
                                      do_sample=True)
        model.generation_config = gen_config

        self._compress_prompts()

        self.model = model.to(self.device)

    def generate(self):
        batch_results = torch.Tensor().to(self.device)
        if self.generate_from_token:
            logging.info('Using LLM lingua with generate from token Flag. Note that input length will not match the fixed len here -'
                'Will default to generation from prompt but from pre-tokenized prompts.')
            assert self.tokenized_data is not None
        else:
            assert self.tokenized_data is None, "Use tokenized data is false but data was still tokenized!"
            self.tokenize_data()

        for batch in tqdm.tqdm(self.tokenized_data, desc='batch', colour='CYAN'):
            result = self.model.generate(**batch)
            result = torch.split(result, [len(batch['input_ids'][0]), self.config['output_len']], dim=1)[1]
            batch_results = torch.cat((batch_results, result))

        return batch_results

    def _compress_prompts(self):
        llm_lingua = PromptCompressor()
        data = []
        for batch in self.data:
            compressed_batch = []
            for prompt in batch:
                compressed_prompt = llm_lingua.compress_prompt(prompt, rate=self.config['compression_config']['rate'])[
                    'compressed_prompt']
                compressed_batch.append(compressed_prompt)
            data.append(compressed_batch)

        self.data = data
