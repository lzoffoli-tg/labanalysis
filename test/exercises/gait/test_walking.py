"""Test suite for labanalysis.exercises.gait walking classes."""

import pytest
from labanalysis.exercises.gait import WalkingExercise, WalkingStride


def test_walking_exercise_importable():
    """Test that WalkingExercise class is importable."""
    assert WalkingExercise is not None


def test_walking_stride_importable():
    """Test that WalkingStride class is importable."""
    assert WalkingStride is not None


def test_walking_exercise_is_class():
    """Test that WalkingExercise is a class."""
    assert isinstance(WalkingExercise, type)


def test_walking_stride_is_class():
    """Test that WalkingStride is a class."""
    assert isinstance(WalkingStride, type)
