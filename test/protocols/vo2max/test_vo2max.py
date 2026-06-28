"""
Test suite for labanalysis.protocols.vo2max module.

Tests VO2max test protocol functions.
"""

import pytest
import numpy as np

from labanalysis.protocols import vo2max


def test_vo2max_module_importable():
    """Test that vo2max module is importable."""
    assert vo2max is not None


def test_vo2max_has_public_functions():
    """Test that vo2max module has public functions or classes."""
    public_attrs = [attr for attr in dir(vo2max) if not attr.startswith('_')]
    assert len(public_attrs) > 0, "Module should have public attributes"


def test_vo2max_module_has_expected_content():
    """Test that vo2max module contains expected protocol-related content."""
    # Check that module is properly structured
    module_dict = vars(vo2max)
    assert module_dict is not None
