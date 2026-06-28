"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestAnkleAnglesMixin:
    """Test AnkleAnglesMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles.ankle import AnkleAnglesMixin
        assert AnkleAnglesMixin is not None

    def test_ankle_angles_mixin_has_flexionextension_properties(self):
        """Test AnkleAnglesMixin has flexion/extension properties."""
        from labanalysis.records.body.angles.ankle import AnkleAnglesMixin

        assert hasattr(AnkleAnglesMixin, 'left_ankle_flexionextension')
        assert isinstance(getattr(AnkleAnglesMixin, 'left_ankle_flexionextension'), property)
        assert hasattr(AnkleAnglesMixin, 'right_ankle_flexionextension')
        assert isinstance(getattr(AnkleAnglesMixin, 'right_ankle_flexionextension'), property)

    def test_ankle_angles_mixin_docstring_exists(self):
        """Test AnkleAnglesMixin has docstring."""
        from labanalysis.records.body.angles.ankle import AnkleAnglesMixin

        assert AnkleAnglesMixin.__doc__ is not None
        assert 'ankle' in AnkleAnglesMixin.__doc__.lower()

