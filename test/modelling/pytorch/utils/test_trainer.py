"""Tests for pytorch.utils.trainer module."""

import pytest


@pytest.mark.pytorch
class TestTorchTrainer:
    """Test TorchTrainer class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.modelling.pytorch.utils.trainer import TorchTrainer
        assert TorchTrainer is not None

    def test_torch_trainer_has_fit_method(self):
        """Test TorchTrainer has fit method."""
        from labanalysis.modelling.pytorch.utils.trainer import TorchTrainer

        assert hasattr(TorchTrainer, 'fit')
        assert callable(getattr(TorchTrainer, 'fit'))

    def test_torch_trainer_init_signature(self):
        """Test TorchTrainer __init__ accepts expected parameters."""
        from labanalysis.modelling.pytorch.utils.trainer import TorchTrainer
        import inspect

        sig = inspect.signature(TorchTrainer.__init__)
        params = list(sig.parameters.keys())

        # Should have loss and optimizer_class parameters
        assert 'loss' in params
        assert 'optimizer_class' in params
