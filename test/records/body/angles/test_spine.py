"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestSpineAnglesMixin:
    """Test SpineAnglesMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles.spine import SpineAnglesMixin
        assert SpineAnglesMixin is not None

    def test_spine_angles_mixin_has_kyphosis_property(self):
        """Test SpineAnglesMixin has kyphosis property."""
        from labanalysis.records.body.angles.spine import SpineAnglesMixin

        assert hasattr(SpineAnglesMixin, 'dorsal_kyphosis')
        assert isinstance(getattr(SpineAnglesMixin, 'dorsal_kyphosis'), property)

    def test_spine_angles_mixin_has_lordosis_property(self):
        """Test SpineAnglesMixin has lordosis property."""
        from labanalysis.records.body.angles.spine import SpineAnglesMixin

        assert hasattr(SpineAnglesMixin, 'lumbar_lordosis')
        assert isinstance(getattr(SpineAnglesMixin, 'lumbar_lordosis'), property)

    def test_spine_angles_mixin_docstring_exists(self):
        """Test SpineAnglesMixin has docstring."""
        from labanalysis.records.body.angles.spine import SpineAnglesMixin

        assert SpineAnglesMixin.__doc__ is not None
        assert 'spine' in SpineAnglesMixin.__doc__.lower()

