"""
Test suite for split_data function.

Tests verify data splitting with stratified groups and proportions.
"""

import numpy as np
import pytest

from labanalysis.utils import split_data


def test_split_data_basic():
    """
    Test split_data with basic 70/30 split.

    Expected:
        - Should split data into two groups with correct proportions
        - All indices should be unique and cover full range
    """
    np.random.seed(42)
    data = np.random.randn(100)
    proportion = {"train": 0.7, "test": 0.3}

    result = split_data(data, proportion, groups=1)

    assert set(result.keys()) == {"train", "test"}
    assert len(result["train"]) == 70
    assert len(result["test"]) == 30

    # All indices should be unique
    all_indices = np.concatenate([result["train"], result["test"]])
    assert len(np.unique(all_indices)) == 100


def test_split_data_with_groups():
    """
    Test split_data with stratified groups.

    Expected:
        With groups > 1, data should be split preserving distribution
    """
    np.random.seed(42)
    data = np.random.randn(100)
    proportion = {"train": 0.8, "test": 0.2}

    result = split_data(data, proportion, groups=5)

    assert len(result["train"]) == 80
    assert len(result["test"]) == 20


def test_split_data_three_way_split():
    """
    Test split_data with three-way split (train/val/test).

    Expected:
        Should correctly split into three groups
    """
    np.random.seed(42)
    data = np.random.randn(100)
    proportion = {"train": 0.7, "val": 0.15, "test": 0.15}

    result = split_data(data, proportion, groups=1)

    assert set(result.keys()) == {"train", "val", "test"}
    assert len(result["train"]) == 70
    assert len(result["val"]) == 15
    assert len(result["test"]) == 15


def test_split_data_preserves_all_indices():
    """
    Test that split_data uses all indices exactly once.

    Expected:
        Union of all splits should equal original index range
    """
    np.random.seed(42)
    data = np.random.randn(50)
    proportion = {"a": 0.5, "b": 0.3, "c": 0.2}

    result = split_data(data, proportion, groups=1)

    all_indices = np.concatenate([result["a"], result["b"], result["c"]])
    assert len(all_indices) == 50
    assert set(all_indices) == set(range(50))


def test_split_data_stratified_distribution():
    """
    Test that stratified splitting preserves distribution.

    Expected:
        With groups > 1, each split should have similar mean/std to original
    """
    np.random.seed(42)
    # Create data with clear distribution
    data = np.concatenate([
        np.random.randn(50) - 5,  # Low values
        np.random.randn(50) + 5,  # High values
    ])
    proportion = {"train": 0.8, "test": 0.2}

    result = split_data(data, proportion, groups=5)

    # Both splits should have similar means (not perfect due to randomness)
    train_mean = np.mean(data[result["train"]])
    test_mean = np.mean(data[result["test"]])
    overall_mean = np.mean(data)

    # Both should be reasonably close to overall mean
    assert abs(train_mean - overall_mean) < 2.0
    assert abs(test_mean - overall_mean) < 2.0


def test_split_data_single_group():
    """
    Test split_data with groups=1 (pure random split).

    Expected:
        Should perform simple random split without stratification
    """
    np.random.seed(42)
    data = np.random.randn(100)
    proportion = {"a": 0.6, "b": 0.4}

    result = split_data(data, proportion, groups=1)

    assert len(result["a"]) == 60
    assert len(result["b"]) == 40


def test_split_data_small_dataset():
    """
    Test split_data with small dataset.

    Expected:
        Should handle small datasets correctly
    """
    np.random.seed(42)
    data = np.random.randn(10)
    proportion = {"train": 0.7, "test": 0.3}

    result = split_data(data, proportion, groups=1)

    assert len(result["train"]) == 7
    assert len(result["test"]) == 3


def test_split_data_returns_indices():
    """
    Test that split_data returns valid indices.

    Expected:
        All returned indices should be in range [0, len(data))
    """
    np.random.seed(42)
    data = np.random.randn(100)
    proportion = {"train": 0.8, "test": 0.2}

    result = split_data(data, proportion, groups=1)

    # All indices should be valid
    all_indices = np.concatenate([result["train"], result["test"]])
    assert np.all(all_indices >= 0)
    assert np.all(all_indices < 100)
