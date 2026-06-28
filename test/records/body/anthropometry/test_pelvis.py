"""Tests for anthropometry module."""

import pytest


@pytest.mark.unit
class TestPelvisMeasuresMixin:
    """Test PelvisMeasuresMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.anthropometry.pelvis import PelvisMeasuresMixin
        assert PelvisMeasuresMixin is not None

    def test_pelvis_measures_has_width_property(self):
        """Test PelvisMeasuresMixin has width property."""
        from labanalysis.records.body.anthropometry.pelvis import PelvisMeasuresMixin

        assert hasattr(PelvisMeasuresMixin, 'pelvis_width')
        assert isinstance(getattr(PelvisMeasuresMixin, 'pelvis_width'), property)

    def test_pelvis_measures_has_height_property(self):
        """Test PelvisMeasuresMixin has height property."""
        from labanalysis.records.body.anthropometry.pelvis import PelvisMeasuresMixin

        assert hasattr(PelvisMeasuresMixin, 'pelvis_height')
        assert isinstance(getattr(PelvisMeasuresMixin, 'pelvis_height'), property)

    def test_pelvis_measures_docstring_exists(self):
        """Test PelvisMeasuresMixin has docstring."""
        from labanalysis.records.body.anthropometry.pelvis import PelvisMeasuresMixin

        assert PelvisMeasuresMixin.__doc__ is not None
        assert 'pelvis' in PelvisMeasuresMixin.__doc__.lower()

