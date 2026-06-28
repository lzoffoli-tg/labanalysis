"""Tests for change_of_direction module."""

import pytest


@pytest.mark.integration
class TestChangeOfDirectionExercise:
    """Test ChangeOfDirectionExercise class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.exercises.change_of_direction import ChangeOfDirectionExercise
        assert ChangeOfDirectionExercise is not None

    def test_change_of_direction_inheritance(self):
        """Test ChangeOfDirectionExercise inherits from WholeBody."""
        from labanalysis.exercises.change_of_direction import ChangeOfDirectionExercise
        from labanalysis.records.body import WholeBody

        assert issubclass(ChangeOfDirectionExercise, WholeBody)

    def test_change_of_direction_has_loading_phase_property(self):
        """Test ChangeOfDirectionExercise has loading_phase property."""
        from labanalysis.exercises.change_of_direction import ChangeOfDirectionExercise

        assert hasattr(ChangeOfDirectionExercise, 'loading_phase')

    def test_change_of_direction_docstring_exists(self):
        """Test ChangeOfDirectionExercise has comprehensive docstring."""
        from labanalysis.exercises.change_of_direction import ChangeOfDirectionExercise

        assert ChangeOfDirectionExercise.__doc__ is not None
        assert len(ChangeOfDirectionExercise.__doc__) > 100
        assert 'agility' in ChangeOfDirectionExercise.__doc__.lower() or 'direction' in ChangeOfDirectionExercise.__doc__.lower()
