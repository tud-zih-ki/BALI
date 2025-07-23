"""BALI IPython Extension - Magic commands for LLM inference benchmarking."""

from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
import json
import os
import itertools
import tempfile
import sys
import argparse
from datetime import datetime
from .inferbench import InferBench
from .cli import get_parser
from .heatmaps import create_heatmaps

@magics_class
class BALIMagics(Magics):
    """BALI magic commands for Jupyter notebooks."""
    
    def __init__(self, shell):
        super().__init__(shell)
        self.bali_config = self._load_bali_config()
    
    def _load_bali_config(self):
        """Load default configuration."""
        bali_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(bali_root, "configs", "bali_config.json")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'data' in config and not os.path.isabs(config['data']):
            config['data'] = os.path.join(bali_root, config['data'])
        
        return config
    
    @line_magic
    def bali_help(self, line):
        """Display help for BALI magic commands."""
        help_text = """
BALI Magic Commands:

%bali_help                   - Show this help
%bali_config_show           - Display current configuration

%%bali_config               - Configure benchmark parameters
    frameworks: vllm hf_accelerate
    input_len: 128 256 512
    output_len: 64 128
    model_name: facebook/opt-1.3b
    repeats: 3

%bali_run                   - Run benchmarks for all parameter combinations
                             Results saved to: bali_results_<timestamp>/

%bali_plot [options]        - Create performance heatmaps
    --output_dir DIR         - Specify results directory  
    --model_name M1 M2       - Filter specific models
    --frameworks F1 F2       - Filter specific frameworks

Examples:
    %bali_plot
    %bali_plot --model_name facebook_opt-1.3b --frameworks vllm hf_accelerate
        """
        print(help_text.strip())
    
    @line_magic
    def bali_config_show(self, line):
        """Print current BALI configuration."""
        print("BALI Configuration:")
        for key, value in self.bali_config.items():
            value_str = " ".join(map(str, value)) if isinstance(value, list) else str(value)
            print(f"{key}: {value_str}")

    @cell_magic
    def bali_config(self, line, cell):
        """Configure BALI benchmark parameters."""
        for line in cell.strip().split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            key, value_str = line.split(':', 1)
            key, value_str = key.strip(), value_str.strip()
            
            if key not in self.bali_config:
                raise ValueError(f"Unknown key: {key}")
            
            # Parse values
            raw_values = value_str.split()
            converted_values = []
            for val in raw_values:
                try:
                    converted_values.append(int(val))
                except ValueError:
                    converted_values.append(val)
            
            # Set value
            if isinstance(self.bali_config[key], list):
                self.bali_config[key] = converted_values
            else:
                self.bali_config[key] = converted_values[0] if converted_values else None
        
        print("Configuration updated!")

    @line_magic
    def bali_run(self, line):
        """Run BALI benchmark over all parameter combinations."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_output_dir = f"bali_results_{timestamp}"
        
        # Get parameters as lists
        def ensure_list(val):
            return val if isinstance(val, list) else [val]
        
        model_names = ensure_list(self.bali_config.get('model_name', ['facebook/opt-1.3b']))
        frameworks = ensure_list(self.bali_config.get('frameworks', ['hf_accelerate']))
        input_lens = ensure_list(self.bali_config.get('input_len', [128]))
        output_lens = ensure_list(self.bali_config.get('output_len', [128]))
        
        combinations = list(itertools.product(model_names, frameworks, input_lens, output_lens))
        
        print(f"Running {len(combinations)} combinations -> {base_output_dir}")
        
        for i, (model_name, framework, input_len, output_len) in enumerate(combinations, 1):
            model_dir_name = model_name.replace('/', '_')
            combo_output_dir = os.path.join(base_output_dir, model_dir_name, framework, str(input_len), str(output_len))
            
            print(f"[{i}/{len(combinations)}] {model_name} | {framework} | {input_len}→{output_len}")
            
            # Create config for this combination
            combo_config = self.bali_config.copy()
            combo_config.update({
                'model_name': model_name,
                'frameworks': [framework],
                'input_len': input_len,
                'output_len': output_len,
                'output_dir': combo_output_dir
            })
            
            # Run benchmark
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
                json.dump(combo_config, temp_config, indent=2)
                temp_config_path = temp_config.name
            
            try:
                original_argv = sys.argv.copy()
                sys.argv = ['bali_run', '--config-file', temp_config_path]
                
                parser = get_parser()
                benchmark = InferBench(parser)
                benchmark.run_inference_benchmark()
                
                print("✓")
                
            except Exception as e:
                print(f"✗ Error: {e}")
            
            finally:
                sys.argv = original_argv
                try:
                    os.unlink(temp_config_path)
                except:
                    pass
        
        print(f"Completed! Results: {base_output_dir}")

    @line_magic
    def bali_plot(self, line):
        """Create heatmaps from BALI results."""
        parser = argparse.ArgumentParser(prog='bali_plot', add_help=False)
        parser.add_argument('--output_dir', type=str)
        parser.add_argument('--model_name', nargs='+')
        parser.add_argument('--frameworks', nargs='+')
        
        try:
            args = parser.parse_args(line.split() if line.strip() else [])
            
            kwargs = {}
            if args.output_dir:
                kwargs['results_dir'] = args.output_dir
            if args.model_name:
                kwargs['model_names'] = args.model_name
            if args.frameworks:
                kwargs['frameworks'] = args.frameworks
            
            create_heatmaps(**kwargs)
            
        except SystemExit:
            print("Usage: %bali_plot [--output_dir DIR] [--model_name MODEL1 ...] [--frameworks FW1 ...]")
        except Exception as e:
            print(f"Error: {e}")


def load_ipython_extension(ipython):
    ipython.register_magics(BALIMagics)
    print("BALI extension loaded.")


def unload_ipython_extension(ipython):
    pass 