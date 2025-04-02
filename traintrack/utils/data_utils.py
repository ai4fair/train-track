#!/usr/bin/env python
# coding: utf-8

"""
Data utility functions for type conversion and automatic type casting.

This module provides utility functions for converting string representations
of values to their appropriate Python types, and includes functionality for
automatic type casting of function arguments.
"""

def boolify(s):
    """
    Convert a string representation of a boolean to a Python bool.

    Args:
        s (str): String to convert, either "True"/"true" or "False"/"false"

    Returns:
        bool: The boolean value represented by the string

    Raises:
        ValueError: If the string cannot be converted to a boolean
    """
    if s == "True" or s == "true":
        return True
    if s == "False" or s == "false":
        return False
    raise ValueError("Not Boolean Value!")


def nullify(s):
    """
    Convert a string representation of None to Python None.

    Args:
        s (str): String to convert, either "None" or "none"

    Returns:
        None: If the string represents None

    Raises:
        ValueError: If the string cannot be converted to None
    """
    if s == "None" or s == "none":
        return None
    raise ValueError("Not None type!")


def estimateType(var):
    """
    Guess and convert a variable to its most appropriate Python type.

    This function attempts to convert a variable (typically a string) into
    its most likely Python type by trying different type conversions in
    the following order: None, boolean, integer, float, and string.

    Args:
        var: The variable to convert, can be a single value or a list

    Returns:
        The variable converted to its estimated Python type. If the input
        is a list, returns a list with all elements converted.
    """
    if type(var) is list:
        if len(var) == 1:
            return estimateType(var[0])
        else:
            return [estimateType(varEntry) for varEntry in var]
    else:
        var = str(var)  # important if the parameters aren't strings...
        for caster in (nullify, boolify, int, float):
            try:
                return caster(var)
            except ValueError:
                pass
    return var


def autocast(dFxn):
    """
    Decorator that automatically casts function arguments to their appropriate types.

    This decorator processes both positional and keyword arguments of a function,
    attempting to convert each argument to its most appropriate Python type using
    the estimateType function.

    Args:
        dFxn: The function to be decorated

    Returns:
        function: A wrapped function that automatically casts its arguments
    """
    def wrapped(*c, **d):
        cp = [estimateType(x) for x in c]
        dp = dict((i, estimateType(j)) for (i, j) in d.items())
        return dFxn(*cp, **dp)

    return wrapped
