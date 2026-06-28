"""Shared fixtures for pytorch tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def simple_tensor_data():
    """Create simple tensor (10, 5) for basic module tests."""
    return torch.randn(10, 5)


@pytest.fixture
def mock_model():
    """Create simple sequential model for training tests."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )


@pytest.fixture
def sample_dataset():
    """Create sample dataset for DataLoader tests."""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    return X, y
