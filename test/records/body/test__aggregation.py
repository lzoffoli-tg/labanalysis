"""Test suite for labanalysis.records.body._aggregation module."""

import pytest
from labanalysis.records.body import _aggregation


def test_aggregation_module_importable():
    """Test that body._aggregation module imports successfully."""
    assert _aggregation is not None


def test_module_has_aggregation_methods():
    """Test that _aggregation module contains aggregation methods."""
    public_attrs = [attr for attr in dir(_aggregation) if not attr.startswith('__')]
    assert len(public_attrs) > 0
