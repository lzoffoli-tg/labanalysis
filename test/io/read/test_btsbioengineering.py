"""
Test suite for labanalysis.io.read.btsbioengineering module.

Tests verify functions for reading BTS Bioengineering file formats.
"""

import pytest
from labanalysis.io.read import btsbioengineering


def test_btsbioengineering_module_importable():
    """
    Test that btsbioengineering module imports without errors.

    Expected:
        Module should be importable
    """
    assert btsbioengineering is not None


def test_module_has_public_functions():
    """
    Test that module contains public functions or classes.

    Expected:
        Module should have at least one non-private attribute
    """
    public_attrs = [attr for attr in dir(btsbioengineering) if not attr.startswith('_')]
    assert len(public_attrs) > 0, "Module should have public functions/classes"
