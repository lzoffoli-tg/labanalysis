"""Tests for anthropometry module."""

import pytest


@pytest.mark.unit
class TestLowerLimbMeasuresMixin:
    """Test LowerLimbMeasuresMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.anthropometry.lower_limb import LowerLimbMeasuresMixin
        assert LowerLimbMeasuresMixin is not None

    def test_lower_limb_measures_has_ankle_width_properties(self):
        """Test LowerLimbMeasuresMixin has ankle width properties."""
        from labanalysis.records.body.anthropometry.lower_limb import LowerLimbMeasuresMixin

        assert hasattr(LowerLimbMeasuresMixin, 'left_ankle_width')
        assert isinstance(getattr(LowerLimbMeasuresMixin, 'left_ankle_width'), property)
        assert hasattr(LowerLimbMeasuresMixin, 'right_ankle_width')
        assert isinstance(getattr(LowerLimbMeasuresMixin, 'right_ankle_width'), property)

    def test_lower_limb_measures_has_leg_length_properties(self):
        """Test LowerLimbMeasuresMixin has leg length properties."""
        from labanalysis.records.body.anthropometry.lower_limb import LowerLimbMeasuresMixin

        assert hasattr(LowerLimbMeasuresMixin, 'left_leg_length')
        assert isinstance(getattr(LowerLimbMeasuresMixin, 'left_leg_length'), property)

    def test_lower_limb_measures_docstring_exists(self):
        """Test LowerLimbMeasuresMixin has docstring."""
        from labanalysis.records.body.anthropometry.lower_limb import LowerLimbMeasuresMixin

        assert LowerLimbMeasuresMixin.__doc__ is not None
        assert 'lower' in LowerLimbMeasuresMixin.__doc__.lower() or 'limb' in LowerLimbMeasuresMixin.__doc__.lower()

