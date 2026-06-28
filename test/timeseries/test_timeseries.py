"""
Test suite for Timeseries base class.

Tests verify base time-series functionality with unit support.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.timeseries import Timeseries


def test_timeseries_initialization():
    """
    Test Timeseries initialization.

    Expected:
        Should create Timeseries with data, index, columns, unit
    """
    data = np.random.randn(100, 3)
    index = np.linspace(0, 10, 100)
    columns = ['X', 'Y', 'Z']

    ts = Timeseries(data, index, columns=columns, unit='m')

    assert ts.shape == (100, 3)
    assert len(ts.index) == 100
    assert len(ts.columns) == 3
    assert ts.unit == 'm'


def test_timeseries_shape_property():
    """
    Test shape property.

    Expected:
        Should return (n_timepoints, n_columns)
    """
    data = np.random.randn(50, 2)
    index = np.arange(50)

    ts = Timeseries(data, index, columns=['A', 'B'], unit='m')

    assert ts.shape == (50, 2)


def test_timeseries_unit_property():
    """
    Test unit property getter.

    Expected:
        Should return unit as string
    """
    data = np.random.randn(10, 1)
    index = np.arange(10)

    ts = Timeseries(data, index, columns=['X'], unit='mm')

    assert isinstance(ts.unit, str)
    assert ts.unit == 'mm'


def test_timeseries_set_unit():
    """
    Test set_unit method.

    Expected:
        Should update unit of measurement
    """
    data = np.random.randn(10, 1)
    index = np.arange(10)

    ts = Timeseries(data, index, columns=['X'], unit='m')
    assert ts.unit == 'm'

    ts.set_unit('cm')
    assert ts.unit == 'cm'


def test_timeseries_to_dataframe():
    """
    Test conversion to pandas DataFrame.

    Expected:
        Should return DataFrame with matching shape and columns
    """
    data = np.random.randn(20, 3)
    index = np.arange(20)
    columns = ['A', 'B', 'C']

    ts = Timeseries(data, index, columns=columns, unit='m')
    df = ts.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20, 3)
    assert list(df.columns) == columns


def test_timeseries_loc_indexer():
    """
    Test loc indexer exists.

    Expected:
        Should have loc property for label-based indexing
    """
    data = np.random.randn(10, 2)
    index = np.arange(10)

    ts = Timeseries(data, index, columns=['X', 'Y'], unit='m')

    assert hasattr(ts, 'loc')
    assert ts.loc is not None


def test_timeseries_iloc_indexer():
    """
    Test iloc indexer exists.

    Expected:
        Should have iloc property for position-based indexing
    """
    data = np.random.randn(10, 2)
    index = np.arange(10)

    ts = Timeseries(data, index, columns=['X', 'Y'], unit='m')

    assert hasattr(ts, 'iloc')
    assert ts.iloc is not None


def test_timeseries_columns_attribute():
    """
    Test columns attribute.

    Expected:
        Should store and return column labels
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)
    columns = ['col1', 'col2', 'col3']

    ts = Timeseries(data, index, columns=columns, unit='m')

    assert hasattr(ts, 'columns')
    assert len(ts.columns) == 3
    assert all(c in ts.columns for c in columns)


def test_timeseries_index_attribute():
    """
    Test index attribute.

    Expected:
        Should store and return time index
    """
    data = np.random.randn(15, 2)
    index = np.linspace(0, 5, 15)

    ts = Timeseries(data, index, columns=['A', 'B'], unit='m')

    assert hasattr(ts, 'index')
    assert len(ts.index) == 15
    assert np.allclose(ts.index[0], 0.0)
    assert np.allclose(ts.index[-1], 5.0)


def test_timeseries_invalid_unit_raises_error():
    """
    Test that invalid unit raises ValueError.

    Expected:
        Should raise ValueError for non-string, non-pint unit
    """
    data = np.random.randn(10, 1)
    index = np.arange(10)

    ts = Timeseries(data, index, columns=['X'], unit='m')

    with pytest.raises(ValueError, match="unit must be"):
        ts.set_unit(123)  # Invalid unit type
