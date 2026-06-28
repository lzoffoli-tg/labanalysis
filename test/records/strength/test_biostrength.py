"""Test suite for labanalysis.records.strength.biostrength module."""

import pytest
from labanalysis.records.strength.biostrength import BiostrengthExercise


def test_biostrength_exercise_importable():
    """Test that BiostrengthExercise class is importable."""
    assert BiostrengthExercise is not None


def test_biostrength_exercise_is_class():
    """Test that BiostrengthExercise is a class."""
    assert isinstance(BiostrengthExercise, type)
