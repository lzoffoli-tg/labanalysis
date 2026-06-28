"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestElbowAnglesMixin:
    """Test ElbowAnglesMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles.elbow import ElbowAnglesMixin
        assert ElbowAnglesMixin is not None

    def test_elbow_angles_mixin_has_flexionextension_properties(self):
        """Test ElbowAnglesMixin has flexion/extension properties."""
        from labanalysis.records.body.angles.elbow import ElbowAnglesMixin

        assert hasattr(ElbowAnglesMixin, 'left_elbow_flexionextension')
        assert isinstance(getattr(ElbowAnglesMixin, 'left_elbow_flexionextension'), property)
        assert hasattr(ElbowAnglesMixin, 'right_elbow_flexionextension')
        assert isinstance(getattr(ElbowAnglesMixin, 'right_elbow_flexionextension'), property)

    def test_elbow_angles_mixin_docstring_exists(self):
        """Test ElbowAnglesMixin has docstring."""
        from labanalysis.records.body.angles.elbow import ElbowAnglesMixin

        assert ElbowAnglesMixin.__doc__ is not None
        assert 'elbow' in ElbowAnglesMixin.__doc__.lower()

