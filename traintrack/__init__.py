#!/usr/bin/env python
# coding: utf-8

"""
TrainTrack: A PyTorch Lightning-based framework for machine learning pipelines.

TrainTrack organizes ML workflows into stages (data processing, edge construction, 
edge labeling, graph segmentation) that can be executed as a full pipeline or individually.
The framework provides a consistent interface to manage complex training workflows
while handling logging, checkpointing, and configuration management automatically.

The package supports both command-line usage and programmatic API integration.
"""

# Import logging configuration but don't configure yet
# (Configuration will be handled by the entry points based on command line args)
from .utils.logging_utils import configure_logging

# Version information
__version__ = "0.1.4"
