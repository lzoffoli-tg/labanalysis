"""Tests for repeated_jumps module."""

import pytest


@pytest.mark.integration
class TestRepeatedJumps:
    """Test RepeatedJumps class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps
        assert RepeatedJumps is not None

    def test_repeated_jumps_inheritance(self):
        """Test RepeatedJumps inherits from WholeBody."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps
        from labanalysis.records.body import WholeBody

        assert issubclass(RepeatedJumps, WholeBody)

    def test_repeated_jumps_has_bodymass_property(self):
        """Test RepeatedJumps has bodymass_kg property."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps

        assert hasattr(RepeatedJumps, 'bodymass_kg')
        assert isinstance(getattr(RepeatedJumps, 'bodymass_kg'), property)

    def test_repeated_jumps_has_jumps_property(self):
        """Test RepeatedJumps has jumps property."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps

        assert hasattr(RepeatedJumps, 'jumps')

    def test_repeated_jumps_docstring_exists(self):
        """Test RepeatedJumps has comprehensive docstring."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps

        assert RepeatedJumps.__doc__ is not None
        assert len(RepeatedJumps.__doc__) > 100
        assert 'fatigue' in RepeatedJumps.__doc__.lower() or 'endurance' in RepeatedJumps.__doc__.lower()
