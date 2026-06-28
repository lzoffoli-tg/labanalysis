"""Tests for pytorch.modules.features_generator module."""

import pytest


@pytest.mark.pytorch
class TestFeaturesGenerator:
    """Test FeaturesGenerator class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.modelling.pytorch.modules.features_generator import FeaturesGenerator
        assert FeaturesGenerator is not None

    def test_features_generator_inheritance(self):
        """Test FeaturesGenerator inherits from torch.nn.Module."""
        from labanalysis.modelling.pytorch.modules.features_generator import FeaturesGenerator
        import torch

        assert issubclass(FeaturesGenerator, torch.nn.Module)

    def test_features_generator_init_signature(self):
        """Test FeaturesGenerator __init__ accepts expected parameters."""
        from labanalysis.modelling.pytorch.modules.features_generator import FeaturesGenerator
        import inspect

        sig = inspect.signature(FeaturesGenerator.__init__)
        params = list(sig.parameters.keys())

        # Should have order, apply_log_transform, apply_inverse_transform, include_interactions parameters
        assert 'order' in params
        assert 'apply_log_transform' in params
        assert 'apply_inverse_transform' in params
        assert 'include_interactions' in params

    def test_features_generator_docstring_exists(self):
        """Test FeaturesGenerator has comprehensive docstring."""
        from labanalysis.modelling.pytorch.modules.features_generator import FeaturesGenerator

        assert FeaturesGenerator.__doc__ is not None
        assert len(FeaturesGenerator.__doc__) > 100
        assert 'polynomial' in FeaturesGenerator.__doc__.lower() or 'features' in FeaturesGenerator.__doc__.lower()
