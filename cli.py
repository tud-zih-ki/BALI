import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser = arguments(parser)
    return parser


def arguments(parser):
    # Model loading parameters
    parser.add_argument("--model-name", type=str,
                        help="Path or huggingface-directory of the model to benchmark")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer to use, default is same as model")
    parser.add_argument("--tokenizer-init-config", type=dict,
                        default={"padding": "max_length", "padding_side": "left", "truncation": "only_first"},
                        help="Configuration dict holding tokenizer initialization parameters")
    parser.add_argument("--trust-remote-code", type=bool, default=False,
                        help="Whether to trust remote code for models/tokenizers that would require it")

    # Benchmark configuration
    parser.add_argument("--frameworks", type=str, nargs='+',
                        default=['hf_accelerate', 'vllm', 'vllm_async', 'llmlingua', 'openllm', 'deepspeed'],
                        help='Inference frameworks to benchmark. Select form hf_accelerate, vllm, vllm_async, llmlingua, openllm, deepspeed')
    parser.add_argument("--tokenize-config", type=dict,
                        default={"return_tensors": "pt", "padding": "max_length", "truncation": True, "return_token_type_ids": None},
                        help="Config Dictionary for tokenize function parameters")
    parser.add_argument("--data", type=str, default='data/prompts.txt',
                        help="Path to the prompts text file")
    parser.add_argument("--num-samples", type=int, default=128,
                    help="Amount of Prompts to sample from data")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for prompts")
    parser.add_argument("--input-len", type=int, default=128,
                        help="Number of input tokens per sample")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Number of tokens to generate per sample")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Model data type. Select from available torch datatypes like float32, bfloat16")
    parser.add_argument("--warm-up-reps", type=int, default=1,
                        help="Warm up repetitions per framework")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Repetitions of inference benchmark per framework")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use for benchmark")
    parser.add_argument("--generate-from-token", action="store_true",
                        help="BALI setting, measures inference speed from token ids with fixed input length")

    # File I/O: config, results, loglevel
    parser.add_argument("--output-dir", type=str, default="../results/example",
                        help="Results directory")
    parser.add_argument("--config-file", type=str, default="configs/example-gpt2.json",
                        help="Config file for running the benchmark.")
    parser.add_argument("--save-slurm-config", action="store_true",
                        help="Save SLURM environment variables")
    parser.add_argument("--loglevel", default='info',
                        help="Provide logging level, default is info. Use debug for detailed log")

    # Inference framework specific parameters
    parser.add_argument("--compression-config", type=dict, default={"model": "", "rate": 0.5},
                        help="Prompt Compression Configuration for LLMLingua")
    parser.add_argument("--open-llm-backend", type=str, default="vllm",
                        help="Backend used for OpenLLM Framework")

    return parser
