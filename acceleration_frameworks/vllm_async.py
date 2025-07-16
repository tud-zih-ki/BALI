import asyncio
import logging
from itertools import chain

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import Counter

# Try to import new vLLM inputs, fall back to None if not available
try:
    from vllm.inputs import TextPrompt, TokensPrompt
    HAS_VLLM_INPUTS = True
except ImportError:
    HAS_VLLM_INPUTS = False

from acceleration_frameworks.acceleration_framework import AccelerationFramework
from vllm import SamplingParams


class VLLM_Async(AccelerationFramework):
    """
    Using VLLMs asynchronous engine
    """

    def __init__(self, config, data, flops, generate_from_token: bool = True):
        super(VLLM_Async, self).__init__(config, data, flops, generate_from_token)
        self.data = list(chain.from_iterable(self.data))

    def tokenize_data(self):
        """
        Init tokenizer and return tokenized data. Tokenization od the data done per sample withing the LLM engine and not as a pre-step.
        No possibility to tokenize the data first.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'], model_max_length=self.config['input_len'],
                                                  **self.config['tokenizer_init_config'])
        tokenizer.pad_token = tokenizer.eos_token
        # chain batched data again, since async engine cant use batched data
        self.tokenized_data = tokenizer(self.data, **self.config['tokenize_config'])
        self.timer.stop_pure_tokenization_timer()
        logging.info(f"Tokenized data shape:{len(self.tokenized_data['input_ids'])}")

    def setup(self):
        """
        Loading the model or model pipeline respectively,defined in subclass
        """
        engine_args = AsyncEngineArgs(model=self.config['model_name'],
                                      revision="main",
                                      tensor_parallel_size=self.config['num_gpus'],
                                      max_model_len=self.config['input_len'] + self.config[
                                          'output_len'] if self.generate_from_token else None,
                                      enforce_eager=True,
                                      max_num_seqs=self.config['batch_size'],
                                      gpu_memory_utilization=0.95,
                                      disable_log_requests=True,
                                      disable_log_stats=True)

        llm = AsyncLLMEngine.from_engine_args(engine_args)
        self.model = llm

    async def generate(self):
        "The async engine has no manual batched execution. it batches multiple request when they come at once. "
        request_tracker = Counter()
        sampling_params = SamplingParams(n=1, ignore_eos=True, max_tokens=self.config['output_len'])
        final_output = []
        if self.generate_from_token:
            for token_ids in tqdm(self.tokenized_data['input_ids'], desc='Samples', colour='CYAN'):
                final_output.append(asyncio.create_task(self.run_single_prompt(sampling_params, request_tracker, token_ids=token_ids)))
        else:
            for prompt in tqdm(self.data, desc='Samples', colour='CYAN'):
                final_output.append(asyncio.create_task(self.run_single_prompt(sampling_params, request_tracker, prompt=prompt)))

        tokens = await asyncio.wait(final_output)
        tokens = [t.result() for t in tokens[0]]
        return [t.outputs[0].token_ids for t in tokens]

    async def run_single_prompt(self, sampling_params, request_tracker, prompt=None, token_ids=None):
        if HAS_VLLM_INPUTS:
            # New vLLM version with inputs module
            if self.generate_from_token:
                inputs = TokensPrompt(prompt_token_ids=token_ids.tolist())
            else:
                inputs = TextPrompt(prompt=prompt)
            results_generator = self.model.generate(request_id=str(next(request_tracker)),
                                                    prompt=inputs,
                                                    sampling_params=sampling_params)
        else:
            # Old vLLM version without inputs module
            results_generator = self.model.generate(request_id=str(next(request_tracker)),
                                                    prompt=prompt if not self.generate_from_token else None,
                                                    prompt_token_ids=token_ids.tolist() if self.generate_from_token else None,
                                                    sampling_params=sampling_params)

        async for request_output in results_generator:
            final = request_output
        return final
