"""Test suite for labanalysis.records.locomotion._base module."""

import pytest
from labanalysis.records.locomotion import _base


def test_base_module_importable():
    """Test that locomotion._base module imports successfully."""
    assert _base is not None


def test_module_has_base_classes():
    """Test that _base module contains base classes."""
    public_attrs = [attr for attr in dir(_base) if not attr.startswith('__')]
    assert len(public_attrs) > 0
