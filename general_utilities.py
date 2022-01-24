#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" General Utilities.

This file contains functions and classes to improve usability.

This file is part of the Truck Detection Algorithm.

EDC consortium / H. Fisser
"""
import os
import sys


class suppress_output:
    """Suppress output class.py

     Attributes:
        suppress_stdout (bool, optional): Suppress messages to stdout. Defaults to False.
        suppress_stderr (bool, optional): Suppress messages to stderr. Defaults to False.

    Methods:
        __enter__: start suppressing output.
        __exit__: stop suppressing output.
    """

    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        """Initialise attributes of supress_output class.

        Args:
            suppress_stdout (bool, optional): Suppress messages to stdout. Defaults to False.
            suppress_stderr (bool, optional): Suppress messages to stderr. Defaults to False.
        """
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        """Start suppressing output."""
        devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = devnull
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args):
        """Stop suppressing output."""
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr
