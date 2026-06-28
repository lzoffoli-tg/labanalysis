"""Tests for anthropometry module."""

import pytest


@pytest.mark.unit
class TestFootMeasuresMixin:
    """Test FootMeasuresMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.anthropometry.foot import FootMeasuresMixin
        assert FootMeasuresMixin is not None

    def test_foot_measures_mixin_has_height_properties(self):
        """Test FootMeasuresMixin has height properties."""
        from labanalysis.records.body.anthropometry.foot import FootMeasuresMixin

        assert hasattr(FootMeasuresMixin, 'left_foot_height')
        assert isinstance(getattr(FootMeasuresMixin, 'left_foot_height'), property)
        assert hasattr(FootMeasuresMixin, 'right_foot_height')
        assert isinstance(getattr(FootMeasuresMixin, 'right_foot_height'), property)

    def test_foot_measures_mixin_has_length_properties(self):
        """Test FootMeasuresMixin has length properties."""
        from labanalysis.records.body.anthropometry.foot import FootMeasuresMixin

        assert hasattr(FootMeasuresMixin, 'left_foot_length')
        assert isinstance(getattr(FootMeasuresMixin, 'left_foot_length'), property)

    def test_foot_measures_mixin_docstring_exists(self):
        """Test FootMeasuresMixin has docstring."""
        from labanalysis.records.body.anthropometry.foot import FootMeasuresMixin

        assert FootMeasuresMixin.__doc__ is not None
        assert 'foot' in FootMeasuresMixin.__doc__.lower()

