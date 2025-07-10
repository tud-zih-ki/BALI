import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser = arguments(parser)
    return parser


def arguments(parser):
    parser.add_argument("--model-name", type=str,
                        help="LLM to use for Benchmark run")
    parser.add_argument("--trust-remote-code", type=bool, default=False,
                        help="")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer to use, default is same as model")
    parser.add_argument("--frameworks", type=str, nargs='+', default=['hf_accelerate'],
                        help='Inference accelerations frameworks to measure performance')
    parser.add_argument("--data", type=str, default='data/prompts_mini.txt',
                        help="Path to the prompts text file")
    parser.add_argument("--output-dir", type=str, default="../inferbench_results",
                        help="Results directory")
    parser.add_argument("--config-file", type=str, default="default_config.json",
                        help="Config file for running the benchmark.")
    parser.add_argument("--save-slurm-config", action="store_true",
                        help="Save SLURM environment variables")
    parser.add_argument("--loglevel", default='info',
                        help="Provide logging level, default is info. Use debug for detailed log")
    parser.add_argument("--input-len", type=int, default=128,
                        help="Sequence len per sample")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Sequence length to generate per prompt")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Inference data type")
    parser.add_argument("--warm-up-reps", type=int, default=1,
                        help="Warm up repetitions of benchmark per framework")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Repetitions of inference benchmark per framework")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use for benchmark")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch Size for prompts")
    parser.add_argument("--generate-from-token", action="store_true",
                        help="BALI setting, measures inference speed from token ids with fixed input length")
    parser.add_argument("--num-samples", type=int, default=128,
                    help="Amount of Prompts to sample from data")
    parser.add_argument("--tokenizer-init-config", type=dict,
                        default={"padding": "max_length", "padding_side": "left", "truncation": "only_first"},
                        help="Config Dictionary to initialize the tokenizer")
    parser.add_argument("--tokenize-config", type=dict,
                        default={"return_tensors": "pt", "padding": "max_length", "truncation": True, "return_token_type_ids": False})
    parser.add_argument("--compression-config", type=dict, default={"model": "", "rate": 0.5},
                        help="Prompt Compression Configuration for LLMLingua")
    parser.add_argument("--open-llm-backend", type=str, default="vllm",
                        help="Backend used for OpenLLM Framework")

    return parser
