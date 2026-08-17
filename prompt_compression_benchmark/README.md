# Installation of dependencies

1. Clone this repository including the llmlingua submodule:
```bash
git clone --recurse-submodules https://github.com/tud-zih-ki/BALI.git
```
2. Install Miniconda package manager (version 24.5.0) https://docs.anaconda.com/miniconda/install/
3. Install the required packages by running the following command in the terminal:
```bash
conda env create -f environment.yml
```
4. Activate the environment by running the following command in the terminal:
```bash
conda activate llmlingua
```
5. Uninstall LLMLingua and reinstall it in editable mode from the submodule:
```bash
pip uninstall llmlingua
pip install -e llmlingua/
```

# Repository Structure
- **`llmlingua/`**: The modified version of LLMLingua, used in the experiments. Modifications were done in `llmlingua/llmlingua/prompt_compressor.py`.
- **`reproduction/`**: The modified evaluation scripts from the original evaluation of LLMLingua, used for the response quality evaluation. See the `reproduction/README.md` for usage information.
- **`reproduction/calculations.ipynb`**: Jupyter notebook for calculation of results related to the response quality benchmarks, i.e. average scores, prompt lengths, reproduction costs, as well as rendering of LaTeX tables.
- **`results/`**: The results of all conducted experiments, including compression / end-to-end latency benchmarks, response quality benchmarks, target rates benchmarks and memory benchmarks. Plots of the experiment results are included in the respective subdirectories.
- **`compression.svg`**: Sketch of the LLMLingua-2 compression algorithm.
- **`download_datasets.ipynb`**: Jupyter notebook for downloading and formatting the datasets used in the experiments.
- **`environment.yml`**: Conda environment file for the installation of dependencies.
- **`environment_vllm_update.yml`**: Dependencies possibly required for vLLM after an update. Try to use the `environment.yml` file first.
- **`eval_latency_e2e.py`**: Script for the the end-to-end latency benchmark.
- **`eval_latency.py`**: Script for the the compression latency benchmark.
- **`eval_memory.py`**: Script for the evaluation of the memory benchmark.
- **`eval_rates.py`**: Script for the evaluation of the target rates benchmark.
- **`launch_vllm.sh`**: Helper script for serving of models with vLLM. Also used in slurm batch scripts.
- **`LICENSE`**: License file. Required by LLMLingua.
- **`optimization_test.py`**: (Legacy) Script for the profiling of LLMLingua-2.
- **`start_comp_job.sh`**: Slurm batch script for the submission of compression latency benchmarking jobs.
- **`start_e2e_job.sh`**: Slurm batch script for the submission of end-to-end latency benchmarking jobs.
- **`start_repro_job.sh`**: Slurm batch script for the submission of response quality benchmarking jobs.
- **`testing.ipynb`**: Jupyter notebook for various tests and experimentation, e.g. testing of the LLMLingua compressors, tokenizers, target LLMs and APIs, exploration of the datasets, different evaluation methods for the response quality benchmarks, etc.