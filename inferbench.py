import gc
import json
import logging
import os.path
import re
import traceback
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from deepspeed.accelerator import get_accelerator
from huggingface_hub import login
from tabulate import tabulate
from tqdm import tqdm

from acceleration_frameworks import *
from cli import get_parser

frameworks_available = {'hf_accelerate': HFAccelerate,
                        'vllm': VLLM,
                        'vllm_async': VLLM_Async,
                        'llmlingua': LLMLingua,
                        'openllm': OpenLLM,
                        'deepspeed': Deepspeed}


class InferBench:
    def __init__(self, parser: ArgumentParser):
        args = parser.parse_args()

        # overwrite cli defaults with config file
        if args.config_file is not None:
            with open(args.config_file, 'r') as file:
                parser.set_defaults(**json.load(file))

        # reload cli args
        args = parser.parse_args()
        self.config = vars(args)

        print(f'Benchmark config:\n {self.config}')

        # if out_dir exists_ add time_stamp to outdir
        if os.path.isdir(self.config['output_dir']):
            self.config['output_dir'] = '_'.join(
                (self.config['output_dir'], datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))

        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])

        # set logger
        logging.basicConfig(filename=os.path.join(self.config['output_dir'], 'logs.txt'), filemode='a',
                            encoding='utf-8', level=args.loglevel.upper(),
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Handler for stdout logging in addition
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)

        if 'hf_login_token' in self.config.keys():
            logging.info("Logging in to Huggingface Hub")
            login(token=self.config['hf_login_token'])

        logging.info('Starting Benchmark...')

        self.save_configs()

    def run_inference_benchmark(self) -> None:
        """
        Runs the standard inference benchmark based on an adapted or the default config
        :return: None
        """

        result_dict = {}

        for framework in tqdm(self.config['frameworks'], desc='Framework', colour='CYAN'):
            logging.info(f"Running accleration Framework {framework}...")
            result_dict[framework] = {}

            try:
                if self.config['warm_up_reps'] > 0:
                    logging.info(f"Starting Warm up for {framework}")
                for r in tqdm(range(self.config["warm_up_reps"]), desc='Warm Up', colour='GREEN'):
                    data = self.prepare_data()
                    result = self.single_framework_run(framework, data)
                    self.clean_gpu_memory()
                    logging.info(f'total time to run warm up repetition for {framework}: {result["total_time"]}s')

                logging.info("Starting actual Benchmark....")
                for r in tqdm(range(self.config["repeats"]), desc='Repeat', colour='CYAN'):
                    data = self.prepare_data()
                    result = self.single_framework_run(framework, data)
                    logging.info(f'total time to run Benchmark {framework}: {result["total_time"]}s')
                    result_dict[framework][r] = result
                    self.clean_gpu_memory()
            except Exception as e:
                logging.error(
                    f'Error for Framework {framework} or different error occured! Choose from the following frameworks: {frameworks_available.keys()}.\nError was: {e}')
                tb = traceback.format_exc()
                print(tb)

        self.evaluate_results(result_dict)
        self.save_results(result_dict)

    def prepare_data(self):
        if self.config['data'] is not None:
            with open(self.config['data'], 'r') as file:
                samples = file.readlines()
        else:
            ValueError("No data file provided!")

        def batch_data(samples):
            l = len(samples)
            for ndx in range(0, l, self.config["batch_size"]):
                yield samples[ndx:min(ndx + self.config["batch_size"], l)]

        def sample_docs(samples):
            np.random.seed(42)
            return np.random.choice(samples, size=self.config['num_samples'], replace=True).tolist()

        samples = batch_data(sample_docs(samples))
        return samples

    def single_framework_run(self, framework, data):
        framework_instance = frameworks_available[framework](self.config, data, self.config['generate_from_token'])
        return framework_instance.forward()

    def evaluate_results(self, result_dict):
        df = pd.DataFrame(result_dict)
        res = pd.DataFrame()
        for c in df.columns:
            avg = pd.DataFrame(pd.json_normalize(df[c]).mean(numeric_only=True).add_suffix('_avg')).T
            avg['framework'] = c
            avg = avg.set_index('framework')

            std = pd.DataFrame(pd.json_normalize(df[c]).std(numeric_only=True).add_suffix('_std')).T
            std['framework'] = c
            std = std.set_index('framework')

            res = pd.concat([res, avg.join(std, how='outer', on='framework', sort=True)])

        res = res.reindex(sorted(res.columns), axis=1).iloc[:, 4:]
        logging.info(
            f"RESULTS\n{tabulate(res[['total_time_avg', 'generation_time_avg', 'token_per_sec_avg', 'sequences/s_avg', 'setup_time_avg', 'tokenize_time_avg']], headers='keys', tablefmt='fancy_grid')}")

        res_path = os.path.join(self.config['output_dir'], 'benchmark_summary.csv')
        res.to_csv(res_path)
        logging.info(f"Saved Benchmark summary to {res_path}")

    def save_results(self, result_dict: dict) -> None:
        """:
        Saves the results of the inference framework benchmark as a json file and prints the table
        :param result_dict: dict from run_inference_benchmark()
        :return: None
        """
        assert os.path.exists(self.config['output_dir'])

        result_path = os.path.join(self.config['output_dir'], 'benchmark_results.json')

        with open(result_path, 'w') as out_file:
            json.dump(result_dict, out_file, indent=4)
        print(f"Saved results to {result_path}")

    def save_configs(self):
        result_path = os.path.join(self.config['output_dir'], 'config.json')

        with open(result_path, 'w') as out_file:
            json.dump(self.config, out_file, indent=4)

        logging.info(f"Saved configs to {result_path}")

        if self.config["save_slurm_config"]:
            slurm_conf_path = os.path.join(self.config['output_dir'], 'slurm_config.json')
            slurm_conf = {}
            pattern = re.compile(r'SLURM*')
            for key, value in os.environ.items():
                if pattern.match(key):
                    slurm_conf[key] = value

            with open(slurm_conf_path, 'w') as out_file:
                json.dump(slurm_conf, out_file, indent=4)

            logging.info(f"Saved slurm config of benchmark run to {slurm_conf_path}.")

    @staticmethod
    def clean_gpu_memory():
        logging.info(f'Memory allocated before clearing cache: {torch.cuda.memory_allocated()} bytes')
        torch.cuda.empty_cache()
        get_accelerator().empty_cache()
        gc.collect()
        logging.info(f'Memory allocated after clearing cache: {torch.cuda.memory_allocated()} bytes')


if __name__ == '__main__':
    parser = get_parser()
    benchmark = InferBench(parser=parser)
    benchmark.run_inference_benchmark()
