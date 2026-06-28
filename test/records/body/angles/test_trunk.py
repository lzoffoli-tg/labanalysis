"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestTrunkAnglesMixin:
    """Test TrunkAnglesMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles.trunk import TrunkAnglesMixin
        assert TrunkAnglesMixin is not None

    def test_trunk_angles_mixin_has_rotation_property(self):
        """Test TrunkAnglesMixin has rotation property."""
        from labanalysis.records.body.angles.trunk import TrunkAnglesMixin

        assert hasattr(TrunkAnglesMixin, 'trunk_rotation')
        assert isinstance(getattr(TrunkAnglesMixin, 'trunk_rotation'), property)

    def test_trunk_angles_mixin_docstring_exists(self):
        """Test TrunkAnglesMixin has docstring."""
        from labanalysis.records.body.angles.trunk import TrunkAnglesMixin

        assert TrunkAnglesMixin.__doc__ is not None
        assert 'trunk' in TrunkAnglesMixin.__doc__.lower()

