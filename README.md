<img src="grafics/BALI%20transparent.png" align="left" width="115"/> 

# BALI - Benchmark for Accelerated <br> Language Model Inference

BALI is an Open-source Benchmark to compare LLM Inference Frameworks.
It allows a gine grained configuration of the inference, tailored to application needs.

## BALI Pipeline
![Overview of BALI Pipeline](grafics/BALI%20pipeline_morefancy.png)

## List of included Acceleration Frameworks
|Framework|Link|
|----|----|
|VLLM|https://docs.vllm.ai/en/latest/|
|Huggingface Transformers (baseline)|https://huggingface.co/docs/transformers/index|
|LLMLingua|https://github.com/microsoft/LLMLingua/tree/main|
|OpenLLM|https://github.com/bentoml/OpenLLM|
|DeepSpeed|https://github.com/microsoft/DeepSpeed-MII|



## Installation
```bash
source setup_cuda121_torch212.sh
source env_cuda121_torch212.sh
```


## Usage
BALI can be used via a JSON config file, defining the intented parameters:
```bash
source pyenv_inferbench/bin/activate
python inferbench.py --config-file 'configs/template.json'
```
Additionally, all parameters are available via the command line interface:
```bash
python inferbench.py --model-name 'gpt2' --data  'data/prompts.txt' --batch-size 1 --input_len 100 --output-len 100 
```
Note, that the config is read and overwritten by the command line arguments.

For Convenience, you might use `benchmark_jobs_spawner.sh` that will launch a wave of Slurm jobs, one for each model x input_size x output_size configuration, using `template.json` as base for configuration.

### Parameters
```
  --model-name                  LLM to use for Benchmark run
  --tokenizer                   Tokenizer to use, default is same as model
  --frameworks                  List of Inference accelerations frameworks to measure performance
  --data                        Path to the prompts text file
  --output-dir                  Results directory
  --config_file                 Config file for running the benchmark.
  --save-slurm-config           Save SLURM environment variables
  --loglevel                    Provide logging level, default is info. Use debug for detailed log
  --input-len                   Sequence len per sample
  --output-len                  Sequence length to generate per prompt
  --dtype                       Inference data type
  --warm-up-reps                Warm up repetitions of benchmark per framework
  --repeats                     Repetitions of inference benchmark per framework
  --num-gpus                    Number of GPUs to use for benchmark
  --batch-size                  Batch Size for prompts
  --generate-from-token         BALI setting, measures inference speed from token ids with fixed input length
  --num-samples                 Amount of Prompts to sample from data
  --tokenizer-init-config       Config Dictionary to initialize the tokenizer
  --tokenize-config             Config Dictionary for tokenize function parameters
  --compression-config          Prompt Compression Configuration for LLMLingua
```

## Citation
```
@ARTICLE{jurkschat2025bali,
  author={Jurkschat, Lena and Gattogi, Preetam and Vahdati, Sahar and Lehmann, Jens},
  journal={IEEE Access}, 
  title={BALI—A Benchmark for Accelerated Language Model Inference}, 
  year={2025},
  volume={13},
  pages={98976-98989},
  doi={10.1109/ACCESS.2025.3576898}}
```
