import logging
from itertools import chain

import tqdm

from acceleration_frameworks.acceleration_framework import AccelerationFramework
from vllm import LLM, SamplingParams


class VLLM(AccelerationFramework):
    """
    Created from vllm's own benchmark
    https://github.com/vllm-project/vllm/blob/ab406446691f289ef51d1abd8d1ff66760eda36f/benchmarks/benchmark_throughput.py#L61
    """

    def __init__(self, config, data, generate_from_token: bool = True, random_tokens = True):
        super(VLLM, self).__init__(config, data, generate_from_token, random_tokens)

    def setup(self):
        """
        Loading the model or model pipeline respectively,defined in subclass
        """
        # maybe -- depending on version -- add disable_log_stats=False for token logging
        llm = LLM(
            model=self.config['model_name'],
            revision="main",
            tensor_parallel_size=self.config['num_gpus'],
            enforce_eager=False,
            trust_remote_code=self.config['trust_remote_code'],
            max_model_len=self.config['input_len'] + self.config['output_len'] if self.generate_from_token else None)
        self.model = llm

    def generate(self):
        """
        Generator Function of model or model pipeline, defined in subclasses
        """
        sampling_params = SamplingParams(n=1, temperature=0.0, top_p=1.0, ignore_eos=True,
                                         max_tokens=self.config['output_len'])
        batch_results = []
        if self.generate_from_token:
            assert self.tokenized_data is not None
            for token_batch in tqdm.tqdm(self.tokenized_data, desc='batch', colour='CYAN'):
                outputs = self.model.generate(prompt_token_ids=[t.tolist() for t in token_batch['input_ids']],
                                              sampling_params=sampling_params, use_tqdm=False)
                batch_results.append([o.outputs[0].token_ids for o in outputs])
                # access per token stats with model.llm_engine._get_stats(None) and update in timings.py
        else:
            assert self.tokenized_data is None
            for batch in tqdm.tqdm(self.data, desc='batch', colour='CYAN'):
                outputs = self.model.generate(prompts=batch, sampling_params=sampling_params, use_tqdm=False)
                batch_results.append([o.outputs[0].token_ids for o in outputs])
                logging.info(f'Prompt batch had {[len(o.prompt_token_ids) for o in outputs]} input tokens')

        return list(chain.from_iterable(batch_results))
