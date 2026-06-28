"""Tests for pytorch.modules.pca module."""

import pytest


@pytest.mark.pytorch
class TestPCA:
    """Test PCA class (trainable PCA layer)."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.modelling.pytorch.modules.pca import PCA
        assert PCA is not None

    def test_pca_inheritance(self):
        """Test PCA inherits from torch.nn.Module."""
        from labanalysis.modelling.pytorch.modules.pca import PCA
        import torch

        assert issubclass(PCA, torch.nn.Module)

    def test_pca_init_signature(self):
        """Test PCA __init__ accepts expected parameters."""
        from labanalysis.modelling.pytorch.modules.pca import PCA
        import inspect

        sig = inspect.signature(PCA.__init__)
        params = list(sig.parameters.keys())

        # Should have input_dim and output_dim parameters
        assert 'input_dim' in params
        assert 'output_dim' in params

    def test_pca_has_orthogonality_loss_method(self):
        """Test PCA has orthogonality_loss method."""
        from labanalysis.modelling.pytorch.modules.pca import PCA

        assert hasattr(PCA, 'orthogonality_loss')
        assert callable(getattr(PCA, 'orthogonality_loss'))

    def test_pca_docstring_exists(self):
        """Test PCA has comprehensive docstring."""
        from labanalysis.modelling.pytorch.modules.pca import PCA

        assert PCA.__doc__ is not None
        assert len(PCA.__doc__) > 100
        assert 'pca' in PCA.__doc__.lower() or 'orthogonal' in PCA.__doc__.lower()
