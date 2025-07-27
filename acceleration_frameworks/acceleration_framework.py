import asyncio
import logging

import torch
from transformers import AutoTokenizer

from timer import InferenceTimer


class AccelerationFramework():
    def __init__(self, config, data, generate_from_token: bool = True, random_tokens=True):
        self.timer = InferenceTimer()
        self.data = data
        self.config = config
        self.generate_from_token = generate_from_token
        self.random_tokens = random_tokens

        if not self.generate_from_token:
            assert self.random_tokens == False, "Random tokens can only be used if generate_from_token is True."

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            raise ValueError("No GPU was found. Exiting...")
        self.model = None
        self.tokenized_data = None

    def forward(self):
        self.timer.start_timer()
        # create self.model
        self.setup()
        self.timer.stop_setup_time()
        if self.generate_from_token:
            # load tokenizer and save self.tokenized_data
            self.tokenize_data()
            self.timer.stop_tokenize_timer()

        if type(self).__name__ in ["VLLM_Async"]:
            outputs = asyncio.run(self.generate())
        else:
            outputs = self.generate()

        self.timer.end_timer()
        outputs = self.postprocess_outputs(outputs)

        if not torch.is_tensor(outputs):
            print(len(outputs), [len(outputs[i]) for i in range(len(outputs))])
            outputs = torch.FloatTensor(outputs)

        if self.generate_from_token:
            assert outputs.shape[1] == self.config[
                'output_len'], f"Output length {outputs.shape[1]} of framework {self.__class__.__name__} does not match the configs output length!"

        return {'total_time': self.timer.total_prediction_time(),
                'output_shape': outputs.shape,
                'batch_size': self.config['batch_size'],
                'generate_from_token': self.generate_from_token,
                'random_tokens':self.random_tokens,
                'setup_time': self.timer.time_for_setup(),
                'tokenize_time': self.timer.time_for_pre_tokenization() if self.timer.tokenize_time else 'No pretokenization',
                'token_transfer_time': self.timer.token_transfer_time() if self.timer.tokenize_time else 'No pretokenization',
                'generation_time': self.timer.generation_time(),
                'time_per_token': self.timer.time_per_token(outputs),
                'token_timestamps': self.timer.token_timings,
                'token_per_sec': self.timer.token_per_sec(outputs),
                'num_output_token': self.timer.num_output_token,
                'sequences/s': self.timer.seq_per_sec(outputs),
                'start_timestamp': self.timer.start_time,
                'end_timestamp': self.timer.end_time,
                'setup_timestamp': self.timer.setup_time,
                'tokenize_timestamp': self.timer.tokenize_time}

    def tokenize_data(self):
        """
        Default Tokenization batched using Tranforners AutoTokenizer
        """
        tokenized_batch = []
        if self.generate_from_token:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'],
                                                           model_max_length=self.config['input_len'],
                                                           trust_remote_code=self.config['trust_remote_code'],
                                                           **self.config['tokenizer_init_config'])

            if self.random_tokens:
                logging.info("Using Random tokens as input data")
                vocab_size = self.tokenizer.vocab_size

                assert self.config['num_samples']%self.config['batch_size']==0, "The number of samples is not divisible by the batch size!"
                batches = int(self.config['num_samples']/self.config['batch_size'])
                for b in range(batches):
                    tokenized_batch.append(torch.randint(0,vocab_size,(self.config['batch_size'],self.config['input_len'])))
            else:
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

        self.tokenized_data = [inputs.to(self.device) for inputs in tokenized_batch]

    def setup(self):
        """
        Loading the model or model pipeline respectively,defined in subclass and adapt prompts if nessecary.
        Returns a model pipeline with a kind of generate() function for inference
        """
        raise NotImplementedError

    def generate(self):
        """
        Generator Function of model or model pipeline, defined in subclasses
        """
        NotImplementedError

    def postprocess_outputs(self, outputs):
        return outputs

