"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestHipAnglesMixin:
    """Test HipAnglesMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles.hip import HipAnglesMixin
        assert HipAnglesMixin is not None

    def test_hip_angles_mixin_has_flexionextension_properties(self):
        """Test HipAnglesMixin has flexion/extension properties."""
        from labanalysis.records.body.angles.hip import HipAnglesMixin

        assert hasattr(HipAnglesMixin, 'left_hip_flexionextension')
        assert isinstance(getattr(HipAnglesMixin, 'left_hip_flexionextension'), property)
        assert hasattr(HipAnglesMixin, 'right_hip_flexionextension')
        assert isinstance(getattr(HipAnglesMixin, 'right_hip_flexionextension'), property)

    def test_hip_angles_mixin_has_abduction_properties(self):
        """Test HipAnglesMixin has abduction/adduction properties."""
        from labanalysis.records.body.angles.hip import HipAnglesMixin

        assert hasattr(HipAnglesMixin, 'left_hip_abductionadduction')
        assert isinstance(getattr(HipAnglesMixin, 'left_hip_abductionadduction'), property)

    def test_hip_angles_mixin_docstring_exists(self):
        """Test HipAnglesMixin has docstring."""
        from labanalysis.records.body.angles.hip import HipAnglesMixin

        assert HipAnglesMixin.__doc__ is not None
        assert 'hip' in HipAnglesMixin.__doc__.lower()

