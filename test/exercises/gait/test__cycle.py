"""Test suite for labanalysis.exercises.gait._cycle module."""

import pytest
from labanalysis.exercises.gait import _cycle


def test_cycle_module_importable():
    """Test that gait._cycle module imports successfully."""
    assert _cycle is not None


def test_module_has_cycle_classes():
    """Test that _cycle module contains cycle classes."""
    public_attrs = [attr for attr in dir(_cycle) if not attr.startswith('__')]
    assert len(public_attrs) > 0
