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
|DeepSpeed|https://github.com/microsoft/DeepSpeed-MII|



## Installation
```bash
source setup.sh
source ./pyenv_inferbench/bin/activate
```


## Usage
BALI can be used via a JSON config file, defining the intented parameters:
```bash
python inferbench.py --config-file 'configs/default_config_gpt2.json'
```
Additionally, all parameters are available via the command line interface:
```bash
python inferbench.py --model-name 'gpt2' --data  'data/prompts.txt' --batch-size 1 --input_len 100 --output-len 100 
```
Note, that the config is read and overwritten by the command line arguments.
For Convenience, you might want to set the following environment Variables:
```bash
cd InferBench
source ../pyenv_inferbench/bin/activate

export HF_HOME=`pwd`/../huggingface
export BENTOML_HOME=`pwd`/../bentoML
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BAR=1
srun python inferbench.py --config-file 'configs/default_config_gpt2.json' --save-slurm-config
```

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
TBA
