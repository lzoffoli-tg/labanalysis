"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestAnglesHelpersMixin:
    """Test AnglesHelpersMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles._helpers import AnglesHelpersMixin
        assert AnglesHelpersMixin is not None

    def test_helpers_mixin_has_foot_plane_properties(self):
        """Test AnglesHelpersMixin has foot plane properties."""
        from labanalysis.records.body.angles._helpers import AnglesHelpersMixin

        assert hasattr(AnglesHelpersMixin, '_left_foot_plane')
        assert isinstance(getattr(AnglesHelpersMixin, '_left_foot_plane'), property)
        assert hasattr(AnglesHelpersMixin, '_right_foot_plane')
        assert isinstance(getattr(AnglesHelpersMixin, '_right_foot_plane'), property)

    def test_helpers_mixin_has_pelvis_plane_property(self):
        """Test AnglesHelpersMixin has pelvis plane property."""
        from labanalysis.records.body.angles._helpers import AnglesHelpersMixin

        assert hasattr(AnglesHelpersMixin, '_pelvis_plane')
        assert isinstance(getattr(AnglesHelpersMixin, '_pelvis_plane'), property)

    def test_helpers_mixin_docstring_exists(self):
        """Test AnglesHelpersMixin has docstring."""
        from labanalysis.records.body.angles._helpers import AnglesHelpersMixin

        assert AnglesHelpersMixin.__doc__ is not None

