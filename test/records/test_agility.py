"""Test suite for labanalysis.exercises agility classes."""

import pytest
from labanalysis import exercises


def test_exercises_module_importable():
    """Test that exercises module imports successfully."""
    assert exercises is not None


def test_change_of_direction_exercise_importable():
    """Test that ChangeOfDirectionExercise imports successfully."""
    from labanalysis.exercises import ChangeOfDirectionExercise
    assert ChangeOfDirectionExercise is not None


def test_module_has_public_classes():
    """Test that agility module contains public classes."""
    public_attrs = [attr for attr in dir(exercises) if not attr.startswith('_')]
    assert len(public_attrs) > 0
