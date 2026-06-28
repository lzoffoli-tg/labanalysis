"""Tests for single_jump module."""

import pytest


@pytest.mark.integration
class TestSingleJump:
    """Test SingleJump class - requires real data for proper testing."""

    def test_module_imports(self):
        """Test that SingleJump can be imported."""
        from labanalysis.exercises.single_jump import SingleJump
        assert SingleJump is not None

    def test_single_jump_inheritance(self):
        """Test SingleJump inherits from WholeBody."""
        from labanalysis.exercises.single_jump import SingleJump
        from labanalysis.records.body import WholeBody

        assert issubclass(SingleJump, WholeBody)

    def test_single_jump_has_flight_time_property(self):
        """Test SingleJump has flight_time property."""
        from labanalysis.exercises.single_jump import SingleJump

        assert hasattr(SingleJump, 'flight_time')
        assert isinstance(getattr(SingleJump, 'flight_time'), property)

    def test_single_jump_has_contact_time_property(self):
        """Test SingleJump has contact_time property."""
        from labanalysis.exercises.single_jump import SingleJump

        assert hasattr(SingleJump, 'contact_time')
        assert isinstance(getattr(SingleJump, 'contact_time'), property)

    def test_single_jump_docstring_exists(self):
        """Test SingleJump has comprehensive docstring."""
        from labanalysis.exercises.single_jump import SingleJump

        assert SingleJump.__doc__ is not None
        assert len(SingleJump.__doc__) > 100
        assert 'jump' in SingleJump.__doc__.lower()
