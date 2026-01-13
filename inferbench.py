#!/usr/bin/env python3

import gc
import json
import logging
import os.path
import re
import traceback
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from deepspeed.accelerator import get_accelerator
from huggingface_hub import login
from tabulate import tabulate
from tqdm import tqdm

from acceleration_frameworks import frameworks_available
from cli import get_parser


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
                (self.config['output_dir'], datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

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
            if framework not in frameworks_available:
                logging.warning(
                    f"Requested framework '{framework}' is not available in the current environment and will be skipped. "
                    f"Available frameworks: {list(frameworks_available.keys())}")
                continue

            logging.info(f"Running acceleration framework {framework}…")
            result_dict[framework] = {}

            try:
                if self.config['warm_up_reps'] > 0:
                    logging.info(f"Starting Warm up for {framework}")
                for r in tqdm(range(self.config["warm_up_reps"]), desc='Warm Up', colour='GREEN'):
                    data = self.prepare_data()
                    result = self.single_framework_run(framework, data)
                    self.clean_gpu_memory()
                    logging.info(
                        f'total time to run warm up repetition for {framework}: {result["total_time"]}s')

                logging.info("Starting actual benchmark...")
                for r in tqdm(range(self.config["repeats"]), desc='Repeat', colour='CYAN'):
                    data = self.prepare_data()
                    result = self.single_framework_run(framework, data)
                    logging.info(f'total time to run Benchmark {framework}: {result["total_time"]}s')
                    result_dict[framework][r] = result
                    self.clean_gpu_memory()
            except Exception as e:
                logging.error(
                    f'Error for Framework {framework} or different error occured! Choose from the following frameworks: '
                    f"{list(frameworks_available.keys())}.\nError was: {e}")
                tb = traceback.format_exc()
                print(tb)

        self.evaluate_results(result_dict)
        self.save_results(result_dict)

    def prepare_data(self):
        if self.config['data'] is not None:
            with open(self.config['data'], 'r') as file:
                samples = file.readlines()
        else:
            raise ValueError("No data file provided!")

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
        framework_instance = frameworks_available[framework](self.config, data, self.config['generate_from_token'], self.config['random_tokens'])
        return framework_instance.forward()

    def evaluate_results(self, result_dict):
        # don't litter the results file with individual timestamps but plot them instead
        token_timestamps = {}
        for framework in result_dict:
            token_timestamps[framework] = np.empty((0, len(result_dict[framework][0]["token_timestamps"][0])))
            for iteration in result_dict[framework]:
                # FIXME list can also be concatenated using []+[], maybe that's easier
                token_timestamps[framework] = np.concatenate((
                        token_timestamps[framework],
                        np.array(result_dict[framework][iteration].pop("token_timestamps"))
                ))
        prefill_times, decode_times = self.plot_token_times(token_timestamps)

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

        # update results_dict. no regard for individual runs as of now, so it's not in r[framework][0]["prefill_time"]
        for framework, median in prefill_times.items():
            result_dict[framework]["prefill_time_median"] = median
        # separate loops should maybe one framework only capture prefill, or the other way around, or whatever
        for framework, median in decode_times.items():
            result_dict[framework]["decode_times_median"] = median

        logging.info(
            f"RESULTS\n{tabulate(
                res[['total_time_avg', 'generation_time_avg', 'token_per_sec_avg', 'sequences/s_avg', 'setup_time_avg', 'tokenize_time_avg']],
                headers='keys',
                tablefmt='fancy_grid')}")

        res_path = os.path.join(self.config['output_dir'], 'benchmark_summary.csv')
        res.to_csv(res_path)
        logging.info(f"Saved Benchmark summary to {res_path}")

    def plot_token_times(self, token_timestamps):
        # FIXME these statistics should probably be calculated elsewhere. for now it's easiest to get them here
        prefill_times = {}
        decode_times = {}

        plt.rcParams.update({'font.size': 14})
        for framework in token_timestamps:
            # Only HFAccelerate measures this as of now
            if token_timestamps[framework].size == 0:
                continue

            # turn timestamps into latencies
            token_timestamps[framework] = list(token_timestamps[framework])
            for idx, t in enumerate(token_timestamps[framework]):
                # FIXME this is to accomodate for hf-accelerate emitting the prompt as the first token
                t = np.delete(t, 1)
                token_timestamps[framework][idx] = [t[i + 1] - t[i] for i in range(len(t) - 1)]

            # regroup from timings of a run to timings for each token
            token_timestamps[framework] = np.transpose(token_timestamps[framework])

            prefill_times[framework] = float(np.median(token_timestamps[framework][0]))
            if len(token_timestamps[framework]) > 1:
                decode_times[framework] = [np.median(t) for t in token_timestamps[framework][1:]]

            xs = list(range(len(token_timestamps[framework])))
            # avgs = [np.average(t) for t in token_timestamps[framework]]
            # flops = [f / a for f, a in zip(self.flops.get_flops(), avgs)]
            # stdevs = [np.std(t) for t in token_timestamps[framework]]
            # plt.bar(xs, avgs, yerr=stdevs)

            # error bars could also represent min and max (which i think is more informative
            #   but doesn't align with other benchmark errors)
            #   (better yet, 10th and 90th percentiles to account for outliers, but i could not be bothered)

            medians = [np.median(t) for t in token_timestamps[framework]]
            mins = [medians[i] - np.percentile(t, 5) for i, t in enumerate(token_timestamps[framework])]
            maxs = [np.percentile(t, 95) - medians[i] for i, t in enumerate(token_timestamps[framework])]
            colors = ["indigo", "orange"]
            patches = []
            fig, ax = plt.subplots(figsize=(10, 4))

            if len(xs) > 100:
                ax.bar(xs, medians, yerr=(mins, maxs), color=colors[0], width=1.001)
            else:
                ax.bar(xs, medians, yerr=(mins, maxs), color=colors[0])
            patches.append(Patch(color=colors[0], label="Batch Latencies"))
            fig.legend(ncols=1, loc="outside upper center", handles=patches, frameon=False)

            ax.set_xlim(-0.5, max(0.5, max(xs) - 0.5))
            ax.set_ylim(0, None)
            ax.set_ylabel("Batch latency [s]", fontsize=14)
            ax.set_xlabel("Output token ID", fontsize=14)
            # plt.title(f"Batch latencies for {framework}")
            plt.subplots_adjust(left=None, bottom=0.15, right=None, top=0.88)
            plt.savefig(os.path.join(self.config["output_dir"], f"token-timings-{framework}.png"), dpi=500)
            plt.close()
            print(f"Saved token latencies diagram to: {os.path.join(self.config['output_dir'], f'token-timings-{framework}.png')}")

        return prefill_times, decode_times

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
