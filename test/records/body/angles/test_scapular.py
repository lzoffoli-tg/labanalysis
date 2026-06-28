"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestShoulderAnglesMixin:
    """Test ShoulderAnglesMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles.shoulder import ShoulderAnglesMixin
        assert ShoulderAnglesMixin is not None

    def test_shoulder_angles_mixin_has_abduction_properties(self):
        """Test ShoulderAnglesMixin has abduction/adduction properties."""
        from labanalysis.records.body.angles.shoulder import ShoulderAnglesMixin

        assert hasattr(ShoulderAnglesMixin, 'left_shoulder_abductionadduction')
        assert isinstance(getattr(ShoulderAnglesMixin, 'left_shoulder_abductionadduction'), property)

    def test_shoulder_angles_mixin_has_flexion_properties(self):
        """Test ShoulderAnglesMixin has flexion/extension properties."""
        from labanalysis.records.body.angles.shoulder import ShoulderAnglesMixin

        assert hasattr(ShoulderAnglesMixin, 'left_shoulder_flexionextension')
        assert isinstance(getattr(ShoulderAnglesMixin, 'left_shoulder_flexionextension'), property)

    def test_shoulder_angles_mixin_docstring_exists(self):
        """Test ShoulderAnglesMixin has docstring."""
        from labanalysis.records.body.angles.shoulder import ShoulderAnglesMixin

        assert ShoulderAnglesMixin.__doc__ is not None
        assert 'shoulder' in ShoulderAnglesMixin.__doc__.lower()

