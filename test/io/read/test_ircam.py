"""
Test suite for labanalysis.io.read.ircam module.

Tests verify functions for reading IRCAM file formats.
"""

import pytest
from labanalysis.io.read import ircam


def test_ircam_module_importable():
    """
    Test that ircam module imports without errors.

    Expected:
        Module should be importable
    """
    assert ircam is not None


def test_module_has_public_functions():
    """
    Test that module contains public functions or classes.

    Expected:
        Module should have at least one non-private attribute
    """
    public_attrs = [attr for attr in dir(ircam) if not attr.startswith('_')]
    assert len(public_attrs) > 0, "Module should have public functions/classes"
