"""Tests for anthropometry module."""

import pytest


@pytest.mark.unit
class TestAnthropometryMixin:
    """Test AnthropometryMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.anthropometry._mixin import AnthropometryMixin
        assert AnthropometryMixin is not None

    def test_anthropometry_mixin_inheritance(self):
        """Test AnthropometryMixin inherits from all measurement mixins."""
        from labanalysis.records.body.anthropometry._mixin import AnthropometryMixin
        from labanalysis.records.body.anthropometry.pelvis import PelvisMeasuresMixin
        from labanalysis.records.body.anthropometry.foot import FootMeasuresMixin
        from labanalysis.records.body.anthropometry.lower_limb import LowerLimbMeasuresMixin

        assert issubclass(AnthropometryMixin, PelvisMeasuresMixin)
        assert issubclass(AnthropometryMixin, FootMeasuresMixin)
        assert issubclass(AnthropometryMixin, LowerLimbMeasuresMixin)

    def test_anthropometry_mixin_docstring_exists(self):
        """Test AnthropometryMixin has comprehensive docstring."""
        from labanalysis.records.body.anthropometry._mixin import AnthropometryMixin

        assert AnthropometryMixin.__doc__ is not None
        assert len(AnthropometryMixin.__doc__) > 100
        assert 'anthropometric' in AnthropometryMixin.__doc__.lower()

