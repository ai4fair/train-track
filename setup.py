#!/usr/bin/env python
# coding: utf-8

import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="traintrack",
    version="0.1.4",
    packages=find_packages(),
    install_requires=[
        # Core ML dependencies
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        "lightning==2.1.3",
        # Scientific computing
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        # Graph Neural Networks
        "torch-scatter==2.1.2+pt21cu118",
        "torch-sparse==0.6.18+pt21cu118",
        "torch-geometric==2.4.0",
        # Logging and visualization
        "wandb",
        "tensorboard",
        "matplotlib",
        # Configuration and utilities
        "pyyaml>=5.1",
        "simple-slurm",
        "decorator",
        "more-itertools",
        "memory-profiler",
    ],
    extras_require={
        "dev": [
            # Development tools
            "ipython",
            "pexpect",
            "pygments",
            # Testing
            "pytest",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "traintrack=traintrack.command_line_pipe:main",
            "ttbatch=traintrack.run_pipeline:batch_stage",
        ]
    },
    python_requires=">=3.10",
    author="Daniel Murnane",
    author_email="your.email@example.com",
    description="A training pipeline for machine learning models",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    keywords=[
        "Machine Learning",
        "MLOps",
        "Pytorch",
        "PytorchLightning",
        "Lightning",
        "Pipeline",
    ],
    url="https://github.com/murnanedaniel/train-track",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
