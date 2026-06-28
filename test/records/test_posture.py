"""Test suite for labanalysis.exercises posture classes."""

import pytest
from labanalysis import exercises


def test_exercises_module_importable():
    """Test that exercises module imports successfully."""
    assert exercises is not None


def test_posture_classes_importable():
    """Test that posture classes import successfully."""
    from labanalysis.exercises import UprightPosture, PronePosture
    assert UprightPosture is not None
    assert PronePosture is not None


def test_module_has_public_classes():
    """Test that posture module contains public classes."""
    public_attrs = [attr for attr in dir(exercises) if not attr.startswith('_')]
    assert len(public_attrs) > 0
