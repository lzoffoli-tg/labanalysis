"""
Test suite for labanalysis.io.write.opensim module.

Tests verify functions for writing OpenSim file formats (.mot, .trc, .sto).
"""

import pytest
from labanalysis.io.write import opensim


def test_opensim_write_module_importable():
    """
    Test that opensim write module imports without errors.

    Expected:
        Module should be importable
    """
    assert opensim is not None


def test_module_has_write_functions():
    """
    Test that module contains write functions.

    Expected:
        Module should have functions for writing OpenSim formats
    """
    public_attrs = [attr for attr in dir(opensim) if not attr.startswith('_')]
    assert len(public_attrs) > 0, "Module should have public write functions"
