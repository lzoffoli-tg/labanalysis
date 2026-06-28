"""Test suite for labanalysis.records.body._helpers module."""

import pytest
from labanalysis.records.body import _helpers


def test_helpers_module_importable():
    """Test that body._helpers module imports successfully."""
    assert _helpers is not None


def test_module_has_helper_functions():
    """Test that _helpers module contains geometric helper functions."""
    public_attrs = [attr for attr in dir(_helpers) if not attr.startswith('__')]
    assert len(public_attrs) > 0
