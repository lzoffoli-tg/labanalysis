"""Tests for anthropometry module."""

import pytest


@pytest.mark.unit
class TestUpperLimbMeasuresMixin:
    """Test UpperLimbMeasuresMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.anthropometry.upper_limb import UpperLimbMeasuresMixin
        assert UpperLimbMeasuresMixin is not None

    def test_upper_limb_measures_has_arm_length_properties(self):
        """Test UpperLimbMeasuresMixin has arm length properties."""
        from labanalysis.records.body.anthropometry.upper_limb import UpperLimbMeasuresMixin

        assert hasattr(UpperLimbMeasuresMixin, 'left_arm_length')
        assert isinstance(getattr(UpperLimbMeasuresMixin, 'left_arm_length'), property)
        assert hasattr(UpperLimbMeasuresMixin, 'right_arm_length')
        assert isinstance(getattr(UpperLimbMeasuresMixin, 'right_arm_length'), property)

    def test_upper_limb_measures_has_elbow_width_properties(self):
        """Test UpperLimbMeasuresMixin has elbow width properties."""
        from labanalysis.records.body.anthropometry.upper_limb import UpperLimbMeasuresMixin

        assert hasattr(UpperLimbMeasuresMixin, 'left_elbow_width')
        assert isinstance(getattr(UpperLimbMeasuresMixin, 'left_elbow_width'), property)

    def test_upper_limb_measures_docstring_exists(self):
        """Test UpperLimbMeasuresMixin has docstring."""
        from labanalysis.records.body.anthropometry.upper_limb import UpperLimbMeasuresMixin

        assert UpperLimbMeasuresMixin.__doc__ is not None
        assert 'upper' in UpperLimbMeasuresMixin.__doc__.lower() or 'limb' in UpperLimbMeasuresMixin.__doc__.lower()

