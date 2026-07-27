"""
Converters module for file format transformations.

This module provides utilities for converting Jupyter notebook files (.ipynb)
to various output formats, particularly HTML with customizable templates and
optional code execution.

Main Components
---------------
Converter : class
    Main conversion class for transforming notebook files.
convert : function
    Command-line interface function for conversion operations.

Examples
--------
Using the Converter class:

>>> from labanalysis.converters import Converter
>>> converter = Converter("notebook.ipynb")
>>> converter.to_html(execute=True)

Using the command-line interface:

>>> from labanalysis.converters import convert
>>> convert(["convert", "notebook.ipynb", "--execute"])
0
"""

from .converter import *
from .client import *
