"""Tests for anthropometry module."""

import pytest


@pytest.mark.unit
class TestTrunkMeasuresMixin:
    """Test TrunkMeasuresMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.anthropometry.trunk import TrunkMeasuresMixin
        assert TrunkMeasuresMixin is not None

    def test_trunk_measures_has_shoulder_width_property(self):
        """Test TrunkMeasuresMixin has shoulder width property."""
        from labanalysis.records.body.anthropometry.trunk import TrunkMeasuresMixin

        assert hasattr(TrunkMeasuresMixin, 'shoulder_width')
        assert isinstance(getattr(TrunkMeasuresMixin, 'shoulder_width'), property)

    def test_trunk_measures_has_trunk_length_property(self):
        """Test TrunkMeasuresMixin has trunk length property."""
        from labanalysis.records.body.anthropometry.trunk import TrunkMeasuresMixin

        assert hasattr(TrunkMeasuresMixin, 'trunk_length')
        assert isinstance(getattr(TrunkMeasuresMixin, 'trunk_length'), property)

    def test_trunk_measures_docstring_exists(self):
        """Test TrunkMeasuresMixin has docstring."""
        from labanalysis.records.body.anthropometry.trunk import TrunkMeasuresMixin

        assert TrunkMeasuresMixin.__doc__ is not None
        assert 'trunk' in TrunkMeasuresMixin.__doc__.lower()

