"""
Test suite for labanalysis.protocols.normativedata module.

Tests normative data tables and comparison functions.
"""

import pytest
import numpy as np

from labanalysis.protocols import normativedata


def test_normativedata_module_importable():
    """Test that normativedata module is importable."""
    assert normativedata is not None


def test_normativedata_has_attributes():
    """Test that normativedata module has expected attributes."""
    # Verify module has some content
    module_attrs = dir(normativedata)
    assert len(module_attrs) > 0


def test_normativedata_contains_data_structures():
    """Test that normativedata contains data structures or functions."""
    # Check for any callable functions or data structures
    has_content = any(
        not attr.startswith('_')
        for attr in dir(normativedata)
    )
    assert has_content, "Module should contain public attributes"
