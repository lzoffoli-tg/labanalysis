"""Tests for pytorch.utils.losses module."""

import pytest


@pytest.mark.pytorch
class TestPinballLoss:
    """Test PinballLoss class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.modelling.pytorch.utils.losses import PinballLoss
        assert PinballLoss is not None

    def test_pinball_loss_inheritance(self):
        """Test PinballLoss inherits from torch.nn.Module."""
        from labanalysis.modelling.pytorch.utils.losses import PinballLoss
        import torch

        assert issubclass(PinballLoss, torch.nn.Module)

    def test_pinball_loss_has_forward_method(self):
        """Test PinballLoss has forward method."""
        from labanalysis.modelling.pytorch.utils.losses import PinballLoss

        assert hasattr(PinballLoss, 'forward')
        assert callable(getattr(PinballLoss, 'forward'))

    def test_pinball_loss_docstring_exists(self):
        """Test PinballLoss has comprehensive docstring."""
        from labanalysis.modelling.pytorch.utils.losses import PinballLoss

        assert PinballLoss.__doc__ is not None
        assert len(PinballLoss.__doc__) > 50
        assert 'pinball' in PinballLoss.__doc__.lower() or 'quantile' in PinballLoss.__doc__.lower()
