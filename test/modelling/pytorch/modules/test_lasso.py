"""Tests for pytorch.modules.lasso module."""

import pytest


@pytest.mark.pytorch
class TestLasso:
    """Test Lasso class (L1 regularized linear layer)."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.modelling.pytorch.modules.lasso import Lasso
        assert Lasso is not None

    def test_lasso_inheritance(self):
        """Test Lasso inherits from torch.nn.Module."""
        from labanalysis.modelling.pytorch.modules.lasso import Lasso
        import torch

        assert issubclass(Lasso, torch.nn.Module)

    def test_lasso_init_signature(self):
        """Test Lasso __init__ accepts expected parameters."""
        from labanalysis.modelling.pytorch.modules.lasso import Lasso
        import inspect

        sig = inspect.signature(Lasso.__init__)
        params = list(sig.parameters.keys())

        # Should have in_features, out_features, bias parameters
        assert 'in_features' in params
        assert 'out_features' in params
        assert 'bias' in params

    def test_lasso_has_lasso_loss_method(self):
        """Test Lasso has lasso_loss method."""
        from labanalysis.modelling.pytorch.modules.lasso import Lasso

        assert hasattr(Lasso, 'lasso_loss')
        assert callable(getattr(Lasso, 'lasso_loss'))

    def test_lasso_docstring_exists(self):
        """Test Lasso has comprehensive docstring."""
        from labanalysis.modelling.pytorch.modules.lasso import Lasso

        assert Lasso.__doc__ is not None
        assert len(Lasso.__doc__) > 100
        assert 'l1' in Lasso.__doc__.lower() or 'lasso' in Lasso.__doc__.lower()
