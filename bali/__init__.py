"""
BALI - Benchmark for Accelerated Language Model Inference

A comprehensive benchmarking framework for comparing LLM inference frameworks
including VLLM, Hugging Face Transformers, LLMLingua, DeepSpeed, and more.
"""

__version__ = "0.1.0"
__author__ = "BALI Development Team"
__license__ = "MIT"

# Import main classes for easy access
from .inferbench import InferBench
from .cli import get_parser, main as cli_main

# Import acceleration frameworks for programmatic access
try:
    from .acceleration_frameworks import frameworks_available
except ImportError:
    frameworks_available = {}

# Import utilities
from .timer import InferenceTimer

# Define what gets imported with "from bali import *"
__all__ = [
    "InferBench",
    "get_parser", 
    "cli_main",
    "frameworks_available",
    "InferenceTimer",
    "__version__",
]

# Package metadata
__title__ = "bali"
__description__ = "Benchmark for Accelerated Language Model Inference"
__url__ = "https://github.com/your-org/bali"
__email__ = ""

# IPython extension support
def load_ipython_extension(ipython):
    """Load the BALI IPython extension."""
    from .magics import load_ipython_extension as load_magics
    load_magics(ipython)

def unload_ipython_extension(ipython):
    """Unload the BALI IPython extension."""
    from .magics import unload_ipython_extension as unload_magics
    unload_magics(ipython)
