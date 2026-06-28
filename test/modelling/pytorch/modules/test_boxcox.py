"""Tests for pytorch.modules.boxcox module."""

import pytest
import torch
import numpy as np

from labanalysis.modelling.pytorch.modules.boxcox import BoxCoxTransform


@pytest.mark.pytorch
class TestBoxCoxTransform:
    """Test BoxCoxTransform class."""

    def test_init(self):
        """Test initialization."""
        transform = BoxCoxTransform(n_features=5)
        assert transform.n_features == 5

    def test_forward(self, simple_tensor_data):
        """Test forward transformation."""
        transform = BoxCoxTransform(n_features=simple_tensor_data.shape[1])
        output = transform(torch.abs(simple_tensor_data) + 1)  # Ensure positive
        assert output.shape == simple_tensor_data.shape

    def test_inverse(self, simple_tensor_data):
        """Test inverse transformation."""
        transform = BoxCoxTransform(n_features=simple_tensor_data.shape[1])
        data = torch.abs(simple_tensor_data) + 1
        forward = transform(data)
        inverse = transform.inverse(forward)
        # Round-trip should be close to original
        assert torch.allclose(inverse, data, rtol=1e-4)
