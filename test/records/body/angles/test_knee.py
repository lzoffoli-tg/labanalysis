"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestKneeAnglesMixin:
    """Test KneeAnglesMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles.knee import KneeAnglesMixin
        assert KneeAnglesMixin is not None

    def test_knee_angles_mixin_has_flexionextension_properties(self):
        """Test KneeAnglesMixin has flexion/extension properties."""
        from labanalysis.records.body.angles.knee import KneeAnglesMixin

        assert hasattr(KneeAnglesMixin, 'left_knee_flexionextension')
        assert isinstance(getattr(KneeAnglesMixin, 'left_knee_flexionextension'), property)
        assert hasattr(KneeAnglesMixin, 'right_knee_flexionextension')
        assert isinstance(getattr(KneeAnglesMixin, 'right_knee_flexionextension'), property)

    def test_knee_angles_mixin_docstring_exists(self):
        """Test KneeAnglesMixin has docstring."""
        from labanalysis.records.body.angles.knee import KneeAnglesMixin

        assert KneeAnglesMixin.__doc__ is not None
        assert 'knee' in KneeAnglesMixin.__doc__.lower()

