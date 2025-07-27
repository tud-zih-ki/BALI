import logging
from itertools import chain
import os

import mii
import tqdm
from deepspeed.accelerator import get_accelerator
from transformers import AutoTokenizer

from acceleration_frameworks.acceleration_framework import AccelerationFramework


class Deepspeed(AccelerationFramework):
    def __init__(self, config, data, generate_from_token: bool = True, random_tokens=True):
        super(Deepspeed, self).__init__(config, data, generate_from_token, random_tokens)
        self.model_name_or_path = config['model_name']

    def tokenize_data(self):
        all_data = []
        if self.generate_from_token:
            tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'],
                                                      model_max_length=self.config['input_len'],
                                                      **self.config['tokenizer_init_config'])

            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Pad token set to EOS token.")

            for batches in self.data:
                batch_data = []
                if isinstance(batches, str):
                    inputs = tokenizer([batches], **self.config['tokenize_config'])
                    decoded_texts = [tokenizer.decode(input_id, skip_special_tokens=False) for input_id in
                                     inputs['input_ids']]
                    batch_data.append(decoded_texts)
                elif isinstance(batches, list) and all(isinstance(item, str) for item in batches):
                    for batch in batches:
                        inputs = tokenizer(batch, **self.config['tokenize_config'])
                        decoded_texts = [tokenizer.decode(input_id, skip_special_tokens=False) for input_id in
                                         inputs['input_ids']]
                        batch_data.extend(decoded_texts)
                else:
                    raise ValueError("Each batch must be a string or a list of strings.")

                all_data.append(batch_data)
        else:

            tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])

            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Pad token set to EOS token.")

            for batches in self.data:
                batch_data = []
                if isinstance(batches, str):
                    inputs = tokenizer([batches], padding='longest', truncation=False, return_tensors='pt')
                    decoded_texts = [tokenizer.decode(input_id, skip_special_tokens=False) for input_id in
                                     inputs['input_ids']]
                    batch_data.append(decoded_texts)
                elif isinstance(batches, list) and all(isinstance(item, str) for item in batches):
                    for batch in batches:
                        inputs = tokenizer(batch, padding='longest', truncation=False, return_tensors='pt')
                        decoded_texts = [tokenizer.decode(input_id, skip_special_tokens=False) for input_id in
                                         inputs['input_ids']]
                        batch_data.extend(decoded_texts)
                else:
                    raise ValueError("Each batch must be a string or a list of strings.")

                all_data.append(batch_data)

        self.timer.stop_tokenize_timer()
        self.timer.stop_pure_tokenization_timer()
        self.mii_data = all_data

    def setup(self):
        # Initialize the MII pipeline
        slurm_job_id = int(os.environ.get('SLURM_JOB_ID', 0))
        job_id = slurm_job_id if slurm_job_id > 0 else os.getpid()
        mii_configs = {"tensor_parallel": self.config['num_gpus'], "tokenizer": self.model_name_or_path,
                       "profile_model_time": True, "torch_dist_port": 29500 + (job_id % 10000),
                       "zmq_port_number": 25555 + (job_id % 10000)}

        self.pipe = mii.pipeline(
            self.model_name_or_path,
            model_config=mii_configs)

        get_accelerator().empty_cache()
        logging.info("MII pipeline setup complete")

    def generate(self):
        """
        Generator function using DeepSpeed MII.
        """
        batch_results = []
        for batch in tqdm.tqdm(self.mii_data, desc='batch', colour='CYAN'):
            outputs = self.pipe(prompts=batch, return_full_text=False, temperature=0.1, top_p=1.0, ignore_eos=True,
                                min_new_tokens=self.config['output_len'], max_new_tokens=self.config['output_len'])
            batch_results.append(outputs)

        # get_accelerator().empty_cache()

        self.pipe.destroy()
        return batch_results

    def postprocess_outputs(self, outputs):
        batch_results = []
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        tokenizer.pad_token = tokenizer.eos_token
        if len(outputs[0]) > 1:
            for i in outputs:
                encoded_outputs = [
                    tokenizer.encode(str(output), truncation=True, max_length=self.config['output_len'],
                                     padding='max_length')
                    for output in i]
                batch_results.append(encoded_outputs)
            batch_results = list(chain.from_iterable(batch_results))
        else:
            for i in outputs:
                encoded_output = tokenizer.encode(str(i), truncation=True, max_length=self.config['output_len'],
                                                  padding='max_length')
                batch_results.append(encoded_output)
        return batch_results
