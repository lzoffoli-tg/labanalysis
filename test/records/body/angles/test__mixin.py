"""Tests for angles module."""

import pytest


@pytest.mark.unit
class TestAngularMeasuresMixin:
    """Test AngularMeasuresMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.angles._mixin import AngularMeasuresMixin
        assert AngularMeasuresMixin is not None

    def test_angular_measures_mixin_inheritance(self):
        """Test AngularMeasuresMixin inherits from all angle mixins."""
        from labanalysis.records.body.angles._mixin import AngularMeasuresMixin
        from labanalysis.records.body.angles.ankle import AnkleAnglesMixin
        from labanalysis.records.body.angles.knee import KneeAnglesMixin
        from labanalysis.records.body.angles.hip import HipAnglesMixin

        assert issubclass(AngularMeasuresMixin, AnkleAnglesMixin)
        assert issubclass(AngularMeasuresMixin, KneeAnglesMixin)
        assert issubclass(AngularMeasuresMixin, HipAnglesMixin)

    def test_angular_measures_mixin_docstring_exists(self):
        """Test AngularMeasuresMixin has comprehensive docstring."""
        from labanalysis.records.body.angles._mixin import AngularMeasuresMixin

        assert AngularMeasuresMixin.__doc__ is not None
        assert len(AngularMeasuresMixin.__doc__) > 100
        assert 'angular' in AngularMeasuresMixin.__doc__.lower() or 'angle' in AngularMeasuresMixin.__doc__.lower()

