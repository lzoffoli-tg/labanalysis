"""Test suite for labanalysis.exercises.gait running classes."""

import pytest
from labanalysis.exercises.gait import RunningExercise, RunningStep


def test_running_exercise_importable():
    """Test that RunningExercise class is importable."""
    assert RunningExercise is not None


def test_running_step_importable():
    """Test that RunningStep class is importable."""
    assert RunningStep is not None


def test_running_exercise_is_class():
    """Test that RunningExercise is a class."""
    assert isinstance(RunningExercise, type)


def test_running_step_is_class():
    """Test that RunningStep is a class."""
    assert isinstance(RunningStep, type)
