"""Tests for pytorch.utils.metrics module."""

import pytest


@pytest.mark.pytorch
class TestMAEMetric:
    """Test MAEMetric class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.modelling.pytorch.utils.metrics import MAEMetric
        assert MAEMetric is not None

    def test_mae_metric_inheritance(self):
        """Test MAEMetric inherits from torch.nn.Module."""
        from labanalysis.modelling.pytorch.utils.metrics import MAEMetric
        import torch

        assert issubclass(MAEMetric, torch.nn.Module)

    def test_mae_metric_has_forward_method(self):
        """Test MAEMetric has forward method."""
        from labanalysis.modelling.pytorch.utils.metrics import MAEMetric

        assert hasattr(MAEMetric, 'forward')
        assert callable(getattr(MAEMetric, 'forward'))
