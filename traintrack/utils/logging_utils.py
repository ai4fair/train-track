#!/usr/bin/env python
# coding: utf-8

"""
Logging utilities for TrainTrack.

This module provides functions for configuring and accessing loggers
throughout the TrainTrack pipeline. It centralizes logging configuration
to ensure consistent log formatting and proper file/console output handling.
"""

import logging
import os
import sys


def configure_logging(log_dir=None, verbose=False):
    """
    Configure the logging system for TrainTrack.

    This function sets up logging with consistent formatting for both console
    and file outputs. It configures the root logger with appropriate handlers
    and formatting.

    Args:
        log_dir (str, optional): Path to log file. If None, only console logging is configured.
        verbose (bool, optional): Whether to set logging level to DEBUG (True)
                                or WARNING (False). Defaults to False.
                                When True: Shows DEBUG and above (most detailed)
                                When False: Shows WARNING and above (important warnings)

    Returns:
        logging.Logger: The configured root logger
    """
    # Determine log level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.WARNING

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with appropriate formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # Set up file logging if log_dir is provided
    if log_dir:
        try:
            # Create parent directory for log file if needed
            log_parent_dir = os.path.dirname(log_dir)
            if log_parent_dir and not os.path.exists(log_parent_dir):
                os.makedirs(log_parent_dir, exist_ok=True)

            # Set up file handler
            file_handler = logging.FileHandler(log_dir)
            file_handler.setLevel(logging.INFO)  # Always log INFO and above to file
            file_format = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)
            root_logger.addHandler(file_handler)
            logging.info(f"Log file created at {log_dir}")
        except Exception as e:
            print(f"Warning: Could not create log file {log_dir}: {e}")
            print("Continuing with console logging only")

    # Configure Lightning loggers
    for logger_name in ["lightning.pytorch", "pytorch_lightning"]:
        lightning_logger = logging.getLogger(logger_name)
        lightning_logger.setLevel(
            logging.WARNING
        )  # Always set Lightning loggers to WARNING

    # Set other verbose loggers to WARNING level
    for logger_name in ["matplotlib", "PIL"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.debug(f"Logging configured with level: {'DEBUG' if verbose else 'WARNING'}")

    return root_logger
