"""Tests for pytorch.modules.sigmoid module."""

import pytest


@pytest.mark.pytorch
class TestSigmoidTransformer:
    """Test SigmoidTransformer class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.modelling.pytorch.modules.sigmoid import SigmoidTransformer
        assert SigmoidTransformer is not None

    def test_sigmoid_transformer_inheritance(self):
        """Test SigmoidTransformer inherits from torch.nn.Module."""
        from labanalysis.modelling.pytorch.modules.sigmoid import SigmoidTransformer
        import torch

        assert issubclass(SigmoidTransformer, torch.nn.Module)

    def test_sigmoid_transformer_init_signature(self):
        """Test SigmoidTransformer __init__ accepts expected parameters."""
        from labanalysis.modelling.pytorch.modules.sigmoid import SigmoidTransformer
        import inspect

        sig = inspect.signature(SigmoidTransformer.__init__)
        params = list(sig.parameters.keys())

        # Should have input_dim, output_dim, transform_dim parameters
        assert 'input_dim' in params
        assert 'output_dim' in params
        assert 'transform_dim' in params

    def test_sigmoid_transformer_docstring_exists(self):
        """Test SigmoidTransformer has comprehensive docstring."""
        from labanalysis.modelling.pytorch.modules.sigmoid import SigmoidTransformer

        assert SigmoidTransformer.__doc__ is not None
        assert len(SigmoidTransformer.__doc__) > 50
        assert 'sigmoid' in SigmoidTransformer.__doc__.lower()
