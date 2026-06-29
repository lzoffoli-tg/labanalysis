"""
Test suite for Signal1D class.

Tests verify 1D signal time-series with single column amplitude data.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.timeseries import Signal1D


def test_signal1d_initialization_1d_array():
    """
    Test Signal1D initialization with 1D array.

    Expected:
        Should convert 1D array to 2D column vector with 'amplitude' column
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    index = np.linspace(0, 1, 5)

    signal = Signal1D(data, index, unit='V')

    assert signal.shape == (5, 1)
    assert len(signal.columns) == 1
    assert 'amplitude' in signal.columns
    assert signal.unit == 'V'


def test_signal1d_initialization_2d_array():
    """
    Test Signal1D initialization with 2D array (n, 1).

    Expected:
        Should accept 2D array with single column
    """
    data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    index = np.linspace(0, 1, 5)

    signal = Signal1D(data, index, unit='mV')

    assert signal.shape == (5, 1)
    assert signal.unit == 'mV'


def test_signal1d_initialization_invalid_columns():
    """
    Test Signal1D raises ValueError for multiple columns.

    Expected:
        Should raise ValueError when data has more than one column
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    with pytest.raises(ValueError, match="exactly one column"):
        Signal1D(data, index, unit='V')


def test_signal1d_unit_validation():
    """
    Test Signal1D validates unit type.

    Expected:
        Should raise ValueError for invalid unit type
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="unit must be"):
        Signal1D(data, index, unit=123)


def test_signal1d_copy():
    """
    Test copy method creates independent copy.

    Expected:
        Should create new Signal1D with copied data and index
    """
    data = np.array([1.0, 2.0, 3.0, 4.0])
    index = np.array([0.0, 1.0, 2.0, 3.0])

    original = Signal1D(data, index, unit='V')
    copied = original.copy()

    assert isinstance(copied, Signal1D)
    assert copied is not original
    assert copied.shape == original.shape
    assert copied.unit == original.unit
    assert np.array_equal(copied._data, original._data)
    assert np.array_equal(copied.index, original.index)


def test_signal1d_copy_independence():
    """
    Test copied Signal1D is independent from original.

    Expected:
        Modifying copy should not affect original
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])

    original = Signal1D(data, index, unit='V')
    copied = original.copy()

    # Modify copied data
    copied._data[0, 0] = 999.0

    assert original._data[0, 0] != 999.0


def test_signal1d_to_dataframe():
    """
    Test conversion to pandas DataFrame.

    Expected:
        Should return DataFrame with column name as unit (spaces removed)
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 0.5, 1.0])

    signal = Signal1D(data, index, unit='m/s')
    df = signal.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 1)
    assert df.columns[0] == 'm/s'


def test_signal1d_to_dataframe_unit_with_spaces():
    """
    Test DataFrame column name removes spaces from unit.

    Expected:
        Unit 'm / s' should become 'm/s' in column name
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])

    signal = Signal1D(data, index, unit='m / s')
    df = signal.to_dataframe()

    assert df.columns[0] == 'm/s'


def test_signal1d_shape_property():
    """
    Test shape property.

    Expected:
        Should return (n_samples, 1) for 1D signal
    """
    data = np.random.randn(100)
    index = np.arange(100)

    signal = Signal1D(data, index, unit='V')

    assert signal.shape == (100, 1)


def test_signal1d_amplitude_column_name():
    """
    Test that Signal1D always uses 'amplitude' as column name.

    Expected:
        Column name should be 'amplitude' regardless of input
    """
    data = np.array([5.0, 10.0, 15.0])
    index = np.array([0.0, 1.0, 2.0])

    signal = Signal1D(data, index, unit='V')

    assert 'amplitude' in signal.columns
    assert len(signal.columns) == 1


def test_signal1d_with_list_input():
    """
    Test Signal1D accepts list input.

    Expected:
        Should convert list to numpy array internally
    """
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    index = [0.0, 0.25, 0.5, 0.75, 1.0]

    signal = Signal1D(data, index, unit='mV')

    assert signal.shape == (5, 1)
    assert signal.unit == 'mV'


def test_signal1d_empty_array():
    """
    Test Signal1D with empty array.

    Expected:
        Should handle empty 1D array correctly
    """
    data = np.array([])
    index = np.array([])

    signal = Signal1D(data, index, unit='V')

    assert signal.shape == (0, 1)


def test_signal1d_indexing_access():
    """
    Test accessing data through indexers.

    Expected:
        Should support loc/iloc access inherited from Timeseries
    """
    data = np.array([10.0, 20.0, 30.0, 40.0])
    index = np.array([0.0, 1.0, 2.0, 3.0])

    signal = Signal1D(data, index, unit='V')

    assert hasattr(signal, 'loc')
    assert hasattr(signal, 'iloc')


def test_signal1d_loc_getter_preserves_type():
    """
    Test loc[] getter preserves Signal1D type.

    Expected:
        Sliced signal should be Signal1D instance
    """
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    index = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = Signal1D(data, index, unit='V')

    sliced = signal.loc[1.0:3.0, :]
    assert isinstance(sliced, Signal1D)
    assert len(sliced.index) == 3
    assert sliced.unit == 'V'


def test_signal1d_iloc_getter_preserves_type():
    """
    Test iloc[] getter preserves Signal1D type.

    Expected:
        Sliced signal should be Signal1D instance
    """
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    index = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = Signal1D(data, index, unit='mV')

    sliced = signal.iloc[1:4, :]
    assert isinstance(sliced, Signal1D)
    assert len(sliced.index) == 3
    assert sliced.unit == 'mV'


def test_signal1d_loc_setter():
    """
    Test loc[] setter modifies data correctly.

    Expected:
        Should update value without breaking type
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])
    signal = Signal1D(data, index, unit='V')

    signal.loc[1.0, 'amplitude'] = 999.0
    assert signal._data[1, 0] == 999.0
    assert isinstance(signal, Signal1D)


def test_signal1d_iloc_setter():
    """
    Test iloc[] setter modifies data correctly.

    Expected:
        Should update value by position
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])
    signal = Signal1D(data, index, unit='V')

    signal.iloc[2, 0] = 777.0
    assert signal._data[2, 0] == 777.0
