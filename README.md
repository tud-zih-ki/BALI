<img src="grafics/BALI%20transparent.png" align="left" width="115"/> 

# BALI - Benchmark for Accelerated Language Model Inference
<p align="center">
  <strong>An open-source benchmarking framework for evaluating and comparing accelerated LLM inference Frameworks.</strong>
</p>
<p align="center">
  <a href="#supported-frameworks">Supported Frameworks</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#jumplm">JumpLM</a> •
  <a href="#citation">Citation</a>
</p>
<p align="center">
  <img src="grafics/BALI%20pipeline_morefancy.png" alt="BALI Pipeline" width="900"/>
</p>

---

## Overview

**BALI** is a flexible and extensible benchmarking suite for evaluating the performance of Large Language Model (LLM) inference frameworks.

It enables:

- Fine-grained LLM inference configuration
- Comparative evaluation across inference backends
- GPU-aware performance measurements
- Integration with interactive Jupyter workflows

BALI is designed for researchers and practitioners who want reproducible and configurable LLM inference benchmarking tailored to specific workloads and deployment scenarios.

---
## Supported Frameworks

| Framework | Description |
|---|---|
| [vLLM](https://docs.vllm.ai/en/latest/) | High-throughput and memory-efficient LLM serving |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) | Baseline inference implementation |
| [LLMLingua](https://github.com/microsoft/LLMLingua/tree/main) | Prompt compression for efficient inference |
| [OpenLLM](https://github.com/bentoml/OpenLLM) | Open-source LLM serving platform |
| [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) | Optimized inference for large-scale transformer models |

---

## Installation

### Environment Setup

```bash
source setup/setup_cuda124_torch260.sh
```
The following setups are available:
- CUDA 12.1 Torch 2.1.2 `setup_cuda121_torch212.sh`
- CUDA 12.4 Torch 2.6   `setup/setup_cuda124_torch260.sh`
- CUDA 13.0 Torch 2.13  `setup/setup_cuda130_torch213.sh`

### (Re-) activate environment
To activate the Python environment, there is a general activation script. In case multiple setup exists, the cuda and torch versions can be handed to the script (e.g. `124 260`). If not given, the newest setup will be chosen.

```bash
source setup/activate_pyenv.sh --cuda 124 --torch 260
```

---

## Usage

### Run Benchmark Using a JSON Configuration

```bash
source setup/activate_pyenv.sh
python inferbench.py \
  --config-file configs/template.json
```

### Run Benchmark from the Command Line

```bash
python inferbench.py \
  --model-name gpt2 \
  --data data/prompts.txt \
  --batch-size 1 \
  --input-len 100 \
  --output-len 100
```

> [!NOTE]
> Configuration values provided via the command line override values defined in the JSON configuration file.


### Automated SLURM Benchmark Execution

For large-scale experiments, BALI provides the helper script:

```bash
examples/hpc/benchmark_jobs_spawner.sh
```

The script automatically launches benchmark sweeps across:

- Multiple models
- Input sequence lengths
- Output sequence lengths
- Framework configurations

using `template.json` as the base configuration.

---

## Configuration

### Core Parameters

| Parameter | Description |
|---|---|
| `--model-name` | LLM used for benchmarking |
| `--tokenizer` | Tokenizer to use (defaults to model tokenizer) |
| `--frameworks` | List of inference frameworks to benchmark |
| `--data` | Path to prompt dataset |
| `--output-dir` | Directory for benchmark results |
| `--config-file` | JSON configuration file |
| `--save-slurm-config` | Save SLURM environment variables |
| `--gpu_sampling` | Record GPU metrics using ['nvml'] |
| `--loglevel` | Logging level (`info`, `debug`, etc.) |
| `--input-len` | Input sequence length |
| `--output-len` | Generated output sequence length |
| `--dtype` | Inference precision / datatype |
| `--warm-up-reps` | Number of warm-up iterations |
| `--repeats` | Number of benchmark repetitions |
| `--num-gpus` | Number of GPUs used |
| `--batch-size` | Prompt batch size |
| `--generate-from-token` | Benchmark from token IDs with fixed input length |
| `--num-samples` | Number of prompts sampled |
| `--tokenizer-init-config` | Tokenizer initialization configuration |
| `--tokenize-config` | Tokenization configuration |
| `--compression-config` | LLMLingua compression configuration |
|`--flopcount-config` | Enable FLOP count module and initialize model parameters with the provided config |


#### FLOP Count Config
To enable the FLOP counter, a JSON config file including the following information must be given to `--flopcount-config`:
```python
#gpt2
{
    "d_embed": 768,
    "d_ffn": 3072,
    "ln_bias": true,
    "attn_bias": true,
    "mlp_bias": true,
    "activation": "gelu",
    "n_head": 12,
    "n_kvhead": 12,
    "n_vocab": 50257,
    "n_layer": 12,
    "kvcache": true,
    "type": "decoder"
}
```
---
# JumpLM

JumpLM combines:

- [JUmPER](https://github.com/ScaDS/jumper_ipython_extension/tree/bali-hook#)
- BALI

for integrated LLM benchmarking and hardware performance monitoring directly inside Jupyter notebooks.

<p align="center">
  <img src="grafics/visualization_interface_jumpLM(1).png" alt="JumpLM Interface" width="900"/>
</p>

---

## JumpLM Setup

### 1. Install BALI

Follow the installation instructions above.

### 2. Install JUmPER

```bash
git clone git@github.com:ScaDS/jumper_ipython_extension.git
cd jumper_ipython_extension

git switch bali-hook
pip install .
```

### 3. Install Notebook Dependencies

```bash
pip install ipywidgets ipympl ipykernel pynvml
```

### 4. Register Jupyter Kernel (Optional)

```bash
python -m ipykernel install \
  --prefix $BALI_REPO/BALI/pyenv_inferbench_cuda126_torch260 \
  --display-name jumpLM-kernel
```

---
## JumpLM Usage in Jupyter
JumpLM is available through a variety of cell magics.

### Load Extensions and Start Performance Monitoring

```python
%reload_ext jumper_extension
%reload_ext bali
%perfmonitor_start --monitor thread_bali
```

### Configure Benchmark

```yaml
%%bali_config
model_name: facebook/opt-1.3b
frameworks: hf_accelerate
batch_size: 128
input_len: 32
output_len: 32
warm_up_reps: 0
repeats: 1
```

### Available Commands

| Command | Description |
|---|---|
| `%bali_config_show` | Display current BALI configuration |
| `%bali_run` | Execute benchmark |
| `%bali_plot` | Generate inference speed heatmaps |
| `%perfmonitor_plot` | Show hardware performance visualization |

---

## Citation

If you use BALI in your research, please cite:

```bibtex
@ARTICLE{jurkschat2025bali,
  author={Jurkschat, Lena and Gattogi, Preetam and Vahdati, Sahar and Lehmann, Jens},
  journal={IEEE Access},
  title={BALI—A Benchmark for Accelerated Language Model Inference},
  year={2025},
  volume={13},
  pages={98976-98989},
  doi={10.1109/ACCESS.2025.3576898}
}
```

