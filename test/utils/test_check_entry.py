"""
Test suite for check_entry function.

Tests verify DataFrame validation with MultiIndex columns.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.utils import check_entry


def test_check_entry_valid_dataframe():
    """
    Test check_entry with valid DataFrame structure.

    Expected:
        Should pass without raising for correct MultiIndex DataFrame
    """
    # Create valid DataFrame with MultiIndex columns
    data = np.random.randn(10, 4)
    columns = pd.MultiIndex.from_product([["A", "B"], ["x", "y"]])
    df = pd.DataFrame(data, columns=columns)

    # Mask is column-oriented (each row is a level > 0 value)
    mask = np.array([["x"], ["y"]])

    # Should not raise
    check_entry(df, mask)


def test_check_entry_invalid_not_dataframe():
    """
    Test check_entry with non-DataFrame input.

    Expected:
        Should raise TypeError for non-DataFrame input
    """
    with pytest.raises(TypeError, match="entry must be a pandas DataFrame"):
        check_entry([1, 2, 3], np.array([["x"], ["y"]]))


def test_check_entry_invalid_not_multiindex():
    """
    Test check_entry with DataFrame without MultiIndex columns.

    Expected:
        Should raise TypeError for non-MultiIndex columns
    """
    df = pd.DataFrame({"A": [1, 2, 3]})
    mask = np.array([["x"], ["y"]])

    with pytest.raises(TypeError, match="entry columns must be a pandas MultiIndex"):
        check_entry(df, mask)


def test_check_entry_invalid_mask_mismatch():
    """
    Test check_entry with mismatched mask.

    Expected:
        Should raise TypeError when column structure doesn't match mask
    """
    data = np.random.randn(10, 4)
    columns = pd.MultiIndex.from_product([["A", "B"], ["x", "y"]])
    df = pd.DataFrame(data, columns=columns)

    # Wrong mask
    mask = np.array([["a"], ["b"]])

    with pytest.raises(TypeError, match="entry columns must contain"):
        check_entry(df, mask)


def test_check_entry_complex_multiindex():
    """
    Test check_entry with more complex MultiIndex structure.

    Expected:
        Should validate correctly for 3-level MultiIndex
    """
    data = np.random.randn(10, 8)
    columns = pd.MultiIndex.from_product([["A", "B"], ["x", "y"], ["1", "2"]])
    df = pd.DataFrame(data, columns=columns)

    # Mask for levels > 0
    mask = np.array([["x", "1"], ["x", "2"], ["y", "1"], ["y", "2"]])

    check_entry(df, mask)


def test_check_entry_single_top_level():
    """
    Test check_entry with single top-level column group.

    Expected:
        Should work with only one top-level group
    """
    data = np.random.randn(10, 2)
    columns = pd.MultiIndex.from_product([["A"], ["x", "y"]])
    df = pd.DataFrame(data, columns=columns)

    mask = np.array([["x"], ["y"]])

    check_entry(df, mask)


def test_check_entry_with_default_index():
    """
    Test check_entry validates that index is pandas Index.

    Expected:
        Should pass with default pandas Index
    """
    data = np.random.randn(10, 4)
    columns = pd.MultiIndex.from_product([["A", "B"], ["x", "y"]])
    df = pd.DataFrame(data, columns=columns)

    mask = np.array([["x"], ["y"]])

    check_entry(df, mask)


def test_check_entry_with_custom_index():
    """
    Test check_entry with custom index.

    Expected:
        Should pass with custom pandas Index
    """
    data = np.random.randn(10, 4)
    columns = pd.MultiIndex.from_product([["A", "B"], ["x", "y"]])
    index = pd.Index(range(10, 20))
    df = pd.DataFrame(data, columns=columns, index=index)

    mask = np.array([["x"], ["y"]])

    check_entry(df, mask)
