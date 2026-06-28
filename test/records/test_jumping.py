"""Test suite for labanalysis.exercises jumping classes."""

import pytest
from labanalysis import exercises


def test_exercises_module_importable():
    """Test that exercises module imports successfully."""
    assert exercises is not None


def test_jump_classes_importable():
    """Test that jump classes import successfully."""
    from labanalysis.exercises import SingleJump, DropJump, RepeatedJumps
    assert SingleJump is not None
    assert DropJump is not None
    assert RepeatedJumps is not None


def test_module_has_public_classes():
    """Test that jumping module contains public classes."""
    public_attrs = [attr for attr in dir(exercises) if not attr.startswith('_')]
    assert len(public_attrs) > 0
