#!/usr/bin/env python
# coding: utf-8

"""
Utilities for configuration management in TrainTrack.

This module provides functions for handling configuration files, loading and 
combining configurations, finding checkpoints, and submitting batch jobs.
It handles the core configuration logic that powers the pipeline execution.
"""

import logging
import os
from itertools import product

import torch
import yaml
from more_itertools import collapse
from simple_slurm import Slurm


def find_checkpoint(run_id, path):
    """
    Find a checkpoint file for a specific run ID.

    Args:
        run_id (str): The run ID to search for
        path (str): Base path to search in

    Returns:
        str: Path to the checkpoint file, or None if not found
    """
    for root_dir, dirs, files in os.walk(path):
        if run_id in dirs:
            latest_run_path = os.path.join(root_dir, run_id, "checkpoints/last.ckpt")
            return latest_run_path


def handle_config_cases(some_config):
    """
    Standardize configuration entries to always be a list.

    Args:
        some_config: Configuration item which may be a list, None, or a single value

    Returns:
        list: The input converted to a list
    """
    if type(some_config) is list:
        return some_config
    if some_config is None:
        return []
    else:
        return [some_config]


def submit_batch(config, project_config, running_id=None):
    """
    Submit a stage to a SLURM batch system.

    Args:
        config (dict): Configuration for the stage to run
        project_config (dict): Project-wide configuration
        running_id (str, optional): ID of a previous job to depend on

    Returns:
        str: The job ID of the submitted job
    """
    with open(config["batch_config"]) as f:
        batch_config = yaml.load(f, Loader=yaml.FullLoader)

    command_line_args = dict_to_args(config)
    slurm = Slurm(**batch_config)

    if running_id is not None and project_config["serial"]:
        logging.info(f"Dependency on ID: {running_id}")
        slurm.set_dependency(dict(afterok=running_id))

    custom_batch_setup = project_config["custom_batch_setup"]

    # Run the slurm submission command
    slurm_command = (
        "\n".join(custom_batch_setup) + """\n ttbatch """ + command_line_args
    )

    # If there are setup commands to run before the batch submission, prepend them here
    if (
        "batch_setup" in config
        and config["batch_setup"]
        and len(project_config["command_line_setup"]) > 0
    ):
        slurm_setup = ";".join(project_config["command_line_setup"] + ["sbatch"])
    else:
        slurm_setup = "sbatch"

    logging.info(slurm_command)
    job_id = slurm.sbatch(slurm_command, sbatch_cmd=slurm_setup, shell="/bin/bash")
    logging.info(f"Job ID: {job_id}")
    return job_id


def find_config(name, path):
    """
    Find a configuration file by name in a directory tree.

    Args:
        name (str): Filename to search for
        path (str): Base path to search in

    Returns:
        str: Full path to the found configuration file, or None if not found
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def load_config(stage, resume_id, project_config, run_args):
    """
    Load and prepare a configuration for a stage.

    This function handles loading stage-specific configuration, either from a file
    or from a checkpoint, and combines it with project-wide configuration and
    command-line arguments.

    Args:
        stage (dict): Stage configuration dictionary
        resume_id (str, optional): ID to resume from
        project_config (dict): Project-wide configuration
        run_args: Command-line arguments

    Returns:
        dict: Combined configuration for the stage
    """
    if resume_id is None:
        stage_config_file = find_config(
            stage["config"],
            os.path.join(project_config["libraries"]["model_library"], stage["set"]),
        )

        with open(stage_config_file) as f:
            config = yaml.load(os.path.expandvars(f.read()), Loader=yaml.FullLoader)
        config["logger"] = project_config["logger"]
        config["resume_id"] = resume_id

    else:
        ckpnt_path = find_checkpoint(
            resume_id, project_config["libraries"]["artifact_library"]
        )
        ckpnt = torch.load(ckpnt_path, map_location=torch.device("cpu"))
        config = ckpnt["hyper_parameters"]
        config["checkpoint_path"] = ckpnt_path

    if "override" in stage.keys():
        config.update(stage["override"])

    # Add pipeline configs to model_config
    config.update(project_config["libraries"])
    config.update(stage)
    if ("inference" in run_args) and run_args.inference:
        config["inference"] = True
    elif (
        ("inference" in run_args) and (not run_args.inference) and (not run_args.batch)
    ):
        config["inference"] = False

    logging.info("Config found and built")
    return config


def combo_config(config):
    """
    Generate combination configurations from a configuration with list values.

    This function creates a list of configuration dictionaries by taking the
    cartesian product of all list values in the input configuration.

    Args:
        config (dict): Configuration dictionary with possibly list values

    Returns:
        list: List of configuration dictionaries
    """
    total_list = {k: (v if type(v) == list else [v]) for (k, v) in config.items()}
    keys, values = zip(*total_list.items())

    # Build list of config dictionaries
    config_list = []
    [config_list.append(dict(zip(keys, bundle))) for bundle in product(*values)]

    return config_list


def dict_to_args(config):
    """
    Convert a configuration dictionary to command-line arguments string.

    Args:
        config (dict): Configuration dictionary

    Returns:
        str: Command-line arguments string
    """
    collapsed_list = list(collapse([["--" + k, v] for k, v in config.items()]))
    collapsed_list = [str(entry) for entry in collapsed_list]
    command_line_args = " ".join(collapsed_list)

    return command_line_args
