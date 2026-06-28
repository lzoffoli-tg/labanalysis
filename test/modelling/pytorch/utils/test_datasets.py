"""Tests for pytorch.utils.datasets module."""

import pytest


@pytest.mark.pytorch
class TestCustomDataset:
    """Test CustomDataset class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.modelling.pytorch.utils.datasets import CustomDataset
        assert CustomDataset is not None

    def test_custom_dataset_inheritance(self):
        """Test CustomDataset inherits from Dataset."""
        from labanalysis.modelling.pytorch.utils.datasets import CustomDataset
        from torch.utils.data import Dataset

        assert issubclass(CustomDataset, Dataset)

    def test_custom_dataset_has_getitem(self):
        """Test CustomDataset has __getitem__ method."""
        from labanalysis.modelling.pytorch.utils.datasets import CustomDataset

        assert hasattr(CustomDataset, '__getitem__')
        assert callable(getattr(CustomDataset, '__getitem__'))

    def test_custom_dataset_docstring_exists(self):
        """Test CustomDataset has comprehensive docstring."""
        from labanalysis.modelling.pytorch.utils.datasets import CustomDataset

        assert CustomDataset.__doc__ is not None
        assert len(CustomDataset.__doc__) > 100
        assert 'dataset' in CustomDataset.__doc__.lower()
