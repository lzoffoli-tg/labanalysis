"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestNeckAnglesMixin:
    """Test NeckAnglesMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles.neck import NeckAnglesMixin
        assert NeckAnglesMixin is not None

    def test_neck_angles_mixin_has_flexionextension_property(self):
        """Test NeckAnglesMixin has flexion/extension property."""
        from labanalysis.records.body.angles.neck import NeckAnglesMixin

        assert hasattr(NeckAnglesMixin, 'neck_flexionextension')
        assert isinstance(getattr(NeckAnglesMixin, 'neck_flexionextension'), property)

    def test_neck_angles_mixin_docstring_exists(self):
        """Test NeckAnglesMixin has docstring."""
        from labanalysis.records.body.angles.neck import NeckAnglesMixin

        assert NeckAnglesMixin.__doc__ is not None
        assert 'neck' in NeckAnglesMixin.__doc__.lower()

