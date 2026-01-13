import logging
import sys
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import BaseStreamer

from acceleration_frameworks.acceleration_framework import AccelerationFramework

class TimerStreamer(BaseStreamer):
    def __init__(self, timer):
        self.timer = timer

    def start(self):
        self.timer.time_token_start(start_timing=True)

    def put(self, value):
        self.timer.time_token()

    def end(self):
        self.timer.time_token_final(end_timing=False)

class HFAccelerate(AccelerationFramework):
    def __init__(self, config, data, generate_from_token: bool = True, random_tokens = True):
        super(HFAccelerate, self).__init__(config, data, generate_from_token, random_tokens)

    def tokenize_data(self):
        tokenized_batch = []
        if self.generate_from_token:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'],
                                                           model_max_length=self.config['input_len'],
                                                           trust_remote_code=self.config['trust_remote_code'],
                                                           **self.config['tokenizer_init_config'])

            self.tokenizer.pad_token = self.tokenizer.eos_token
            for b in self.data:
                inputs = self.tokenizer(b, **self.config['tokenize_config'])
                tokenized_batch.append(inputs)

        else:
            # take max seq len as input len per batch
            tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'], trust_remote_code=self.config['trust_remote_code'])
            tokenizer.pad_token = tokenizer.eos_token
            for b in self.data:
                inputs = tokenizer(b, padding='longest', return_tensors='pt')
                tokenized_batch.append(inputs)
        self.timer.stop_pure_tokenization_timer()
        logging.info(f"Input IDs shape per batch:{inputs['input_ids'].shape}")
        # tokenized_batch is a list of dicts {'input_ids': [list], 'attention_mask':[list]}, [list] is cut to input length
        self.tokenized_data = [inputs.to(self.device) for inputs in tokenized_batch]

    def setup(self):
        model = AutoModelForCausalLM.from_pretrained(self.config['model_name'],
                                                     trust_remote_code=self.config['trust_remote_code'],
                                                     device_map="cuda",
                                                     max_length=self.config['output_len'] + self.config[
                                                         'input_len'] if self.generate_from_token else None)

        gen_config = GenerationConfig(max_new_tokens=self.config['output_len'], eos_token_id=50256, pad_token_id=50256)
        self.model = model.to(self.device), gen_config

    def generate(self):
        batch_results = torch.Tensor().to(self.device)
        # optionally, enable the streamer to capture token latencies
        streamer = TimerStreamer(self.timer) if self.config["token_latencies"] else None
        if self.generate_from_token:
            assert self.tokenized_data is not None
            for batch in tqdm.tqdm(self.tokenized_data, desc='batch', colour='CYAN'):
                if streamer:
                    streamer.start()

                try:
                    result = self.model[0].generate(**batch, generation_config=self.model[1], streamer=streamer)
                except ValueError as e:
                    if "The following `model_kwargs` are not used by the model: ['token_type_ids'] "\
                        "(note: typos in the generate arguments will also show up in this list)" in repr(e):
                        logging.error("generate() failed due to wrong tokenizer configuration! Try adding 'return_token_type_ids': "
                            "false to tokenize_config")
                        sys.exit(1)
                batch_results = torch.cat((batch_results, result))

            return torch.split(batch_results, [self.config['input_len'], self.config['output_len']], dim=1)[1]

        assert self.tokenized_data is None, "Use tokenized data is false but data was still tokenized!"
        self.tokenize_data()
        # no ways of feeding prompts and using on the fly tokenization
        for batch in tqdm.tqdm(self.tokenized_data, desc='batch', colour='CYAN'):
            if streamer:
                streamer.start()

            result = self.model[0].generate(**batch, generation_config=self.model[1], streamer=streamer)
            result = torch.split(result, [len(batch['input_ids'][0]), self.config['output_len']], dim=1)[1]
            batch_results = torch.cat((batch_results, result))

        return batch_results
