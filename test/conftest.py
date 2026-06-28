"""Root conftest for test suite - shared fixtures across all tests."""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory with TDF files."""
    # Adjust path as needed
    data_dir = Path(__file__).parent / "data"
    if data_dir.exists():
        return data_dir
    return None


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset numpy random seed for reproducibility."""
    np.random.seed(42)
