"""Test suite for labanalysis.records.strength.freeweight module."""

import pytest
from labanalysis.records.strength.freeweight import FreeWeightExercise


def test_freeweight_exercise_importable():
    """Test that FreeWeightExercise class is importable."""
    assert FreeWeightExercise is not None


def test_freeweight_exercise_is_class():
    """Test that FreeWeightExercise is a class."""
    assert isinstance(FreeWeightExercise, type)
