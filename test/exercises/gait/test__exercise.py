"""Test suite for labanalysis.records.locomotion._exercise module."""

import pytest
from labanalysis.records.locomotion import _exercise


def test_exercise_module_importable():
    """Test that locomotion._exercise module imports successfully."""
    assert _exercise is not None


def test_module_has_exercise_classes():
    """Test that _exercise module contains exercise classes."""
    public_attrs = [attr for attr in dir(_exercise) if not attr.startswith('__')]
    assert len(public_attrs) > 0
