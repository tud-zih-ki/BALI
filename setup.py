import subprocess
import sys
import os
from setuptools import setup, find_packages
from setuptools.command.install import install

setup(
    name="bali",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "bali": ["configs/*.json", "data/*.txt", "data/*.csv"]
    },
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "pandas", 
        "tabulate",
        "tqdm",
        "matplotlib",
        "seaborn",
        "ipython",
        "ninja",
        "packaging",
    ],
    description="Benchmark for LLM inference",
    entry_points={
        'console_scripts': [
            'bali=bali.cli:main',
        ],
        'ipython.extensions': [
            'bali=bali.magics',
        ],
    },
)
