#!/usr/bin/env python
# coding: utf-8

"""
Utilities for model management in TrainTrack.

This module provides functions for finding, building, and configuring models,
trainers, and loggers. It handles the core model management logic that powers
the pipeline execution, with a focus on PyTorch Lightning integration.
"""

import importlib
import logging
import os

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from .config_utils import handle_config_cases


def find_model(model_set, model_name, model_library):
    """
    Find a model or callback class by name in the model library.

    Args:
        model_set (str): The module set to search in (e.g., 'edge_construction')
        model_name (str): The name of the model or callback class to find
        model_library (str): Path to the model library directory

    Returns:
        class: The found model or callback class, or None if not found
    """
    # List all modules in the set/ and set/Models directories
    module_list = [
        os.path.splitext(name)[0]
        for name in os.listdir(os.path.join(model_library, model_set, "Models"))
        if name.endswith(".py")
    ]

    # Import all modules in the set/Models directory and find model_name
    imported_module_list = [
        importlib.import_module(".".join([model_set, "Models", module]))
        for module in module_list
    ]
    names = [
        mod
        for mod in imported_module_list
        if model_name
        in getattr(mod, "__all__", [n for n in dir(mod) if not n.startswith("_")])
    ]
    if len(names) == 0:
        print("Couldn't find model or callback", model_name)
        return None

    # Return the class of model_name
    model_class = getattr(names[0], model_name)

    return model_class


def build_model(model_config):
    """
    Build a model class from configuration.

    This function finds and returns the appropriate model class
    based on the provided configuration.

    Args:
        model_config (dict): Configuration dictionary containing model specifications

    Returns:
        class: The model class to instantiate

    Raises:
        ValueError: If the model class cannot be found
    """
    model_set = model_config["set"]
    model_name = model_config["name"]
    model_library = model_config["model_library"]
    # config_file = model_config["config"]

    logging.info("Building model...")
    model_class = find_model(model_set, model_name, model_library)

    # Ensure model_class is a proper class that can be instantiated
    if model_class is None:
        raise ValueError(f"Could not find model {model_name} in {model_set}")

    logging.info(f"Model class found: {model_name}")
    return model_class


def get_training_logger(model_config):
    """
    Get the appropriate training logger (WandbLogger or TensorBoardLogger) based on configuration.

    Args:
        model_config (dict): The model configuration dictionary

    Returns:
        Union[WandbLogger, TensorBoardLogger, None]: The configured logger for model training
    """
    logger_choice = model_config["logger"]
    if "project" not in model_config.keys():
        model_config["project"] = "my_project"

    if logger_choice == "wandb":
        logger = WandbLogger(
            project=model_config["project"],
            save_dir=model_config["artifact_library"],
            id=model_config["resume_id"],
        )

    elif logger_choice == "tb":
        logger = TensorBoardLogger(
            name=model_config["project"],
            save_dir=model_config["artifact_library"],
            version=model_config["resume_id"],
        )

    elif logger_choice is None:
        logger = None

    logging.info("Logger retrieved")
    return logger


def callback_objects(model_config, lr_logger=False):
    """
    Build callback objects from configuration.

    This function creates and returns callback objects for PyTorch Lightning
    based on the provided configuration.

    Args:
        model_config (dict): Configuration dictionary
        lr_logger (bool, optional): Whether to include a learning rate monitor callback

    Returns:
        list: List of instantiated callback objects
    """
    callback_list = (
        model_config["callbacks"] if "callbacks" in model_config.keys() else None
    )
    callback_list = handle_config_cases(callback_list)

    model_set = model_config["set"]
    model_library = model_config["model_library"]
    callback_object_list = [
        find_model(model_set, callback, model_library)() for callback in callback_list
    ]

    if lr_logger:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callback_object_list = callback_object_list + [lr_monitor]

    logging.info(f"Callbacks found: {callback_list}")
    return callback_object_list


def build_trainer(model_config, logger):
    """
    Build a PyTorch Lightning Trainer from configuration.

    This function creates and configures a PyTorch Lightning Trainer
    with appropriate callbacks, devices, and other settings based on
    the provided configuration.

    Args:
        model_config (dict): Configuration dictionary
        logger: PyTorch Lightning logger instance

    Returns:
        Trainer: Configured PyTorch Lightning Trainer instance
    """

    fom = model_config["fom"] if "fom" in model_config.keys() else "val_loss"
    fom_mode = model_config["fom_mode"] if "fom_mode" in model_config.keys() else "min"

    checkpoint_callback = ModelCheckpoint(
        monitor=fom, save_top_k=2, save_last=True, mode=fom_mode
    )

    # Always use at least 1 device (1 for CPU or GPU, 'auto' for multiple GPUs)
    devices = "auto" if torch.cuda.is_available() else 1

    # Set default num_sanity_val_steps or get from config
    num_sanity_val_steps = model_config.get("sanity_steps", 2)

    # Always use 0 workers to prevent excessive process spawning
    # This can be overridden for production use
    num_workers = 0
    logging.info(f"Using {num_workers} worker processes for data loading")

    # Single trainer initialization for all cases
    # In Lightning 2.0+, resuming is handled in trainer.fit() with ckpt_path
    trainer = Trainer(
        max_epochs=model_config["max_epochs"],
        devices=devices,
        num_sanity_val_steps=num_sanity_val_steps,
        logger=logger,
        callbacks=callback_objects(model_config) + [checkpoint_callback],
        strategy="auto",  # Use "ddp" for distributed training, "auto" for single GPU
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,  # Disable default model summary
        log_every_n_steps=1,  # Reduce logging frequency
    )

    logging.info(f"Trainer built with devices={devices}")
    return trainer


def get_resume_id(stage):
    """
    Get the resume ID from a stage configuration if it exists.

    Args:
        stage (dict): Stage configuration dictionary

    Returns:
        str or None: The resume ID if specified, None otherwise
    """
    resume_id = stage["resume_id"] if "resume_id" in stage.keys() else None
    return resume_id
