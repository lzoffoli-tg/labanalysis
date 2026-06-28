"""
Shared fixtures for regression tests.

Provides data fixtures used across multiple regression test files.
"""

import pytest
import numpy as np


@pytest.fixture
def linear_data():
    """Generate simple linear data for testing."""
    X = np.linspace(1, 10, 50).reshape(-1, 1)
    Y = 2 * X + 3 + np.random.randn(50, 1) * 0.1
    return X, Y


@pytest.fixture
def multivariate_data():
    """Generate multivariate data for testing."""
    X = np.column_stack(
        [np.linspace(1, 10, 50), np.linspace(2, 20, 50), np.linspace(0.5, 5, 50)]
    )
    Y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 5
    Y = Y.reshape(-1, 1)
    return X, Y


@pytest.fixture
def polynomial_data():
    """Generate polynomial data for testing."""
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    Y = 2 * X**2 + 3 * X + 1 + np.random.randn(100, 1) * 0.5
    return X, Y


@pytest.fixture
def power_data():
    """Generate power relationship data (all positive)."""
    X = np.linspace(1, 10, 50).reshape(-1, 1)
    Y = 2 * X**1.5 + np.random.randn(50, 1) * 0.1
    return X, Y


@pytest.fixture
def exponential_data():
    """Generate exponential data for testing."""
    X = np.linspace(1, 5, 50).reshape(-1, 1)
    Y = 2 + X[:, 0] ** 1.5 + np.random.randn(50) * 0.1
    Y = Y.reshape(-1, 1)
    return X, Y


@pytest.fixture
def segmented_data():
    """Generate data with clear segments."""
    X1 = np.linspace(0, 5, 30)
    Y1 = 2 * X1 + 1
    X2 = np.linspace(5, 10, 30)
    Y2 = -1 * X2 + 20
    X = np.concatenate([X1, X2])
    Y = np.concatenate([Y1, Y2])
    return X, Y
