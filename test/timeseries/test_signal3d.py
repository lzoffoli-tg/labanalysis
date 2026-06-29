"""
Test suite for Signal3D class.

Tests verify 3D signal time-series with three spatial columns (X, Y, Z).
"""

import numpy as np
import pytest

from labanalysis.timeseries import Signal1D, Signal3D


def test_signal3d_initialization_default_columns():
    """
    Test Signal3D initialization with default columns.

    Expected:
        Should create 3D signal with columns ['X', 'Y', 'Z']
    """
    data = np.random.randn(10, 3)
    index = np.linspace(0, 1, 10)

    signal = Signal3D(data, index, unit='m')

    assert signal.shape == (10, 3)
    assert len(signal.columns) == 3
    assert 'X' in signal.columns
    assert 'Y' in signal.columns
    assert 'Z' in signal.columns
    assert signal.unit == 'm'


def test_signal3d_initialization_custom_columns():
    """
    Test Signal3D initialization with custom column names.

    Expected:
        Should accept custom column names for 3D axes
    """
    data = np.random.randn(10, 3)
    index = np.linspace(0, 1, 10)
    columns = ['AP', 'ML', 'V']

    signal = Signal3D(data, index, unit='mm', columns=columns, vertical_axis='V', anteroposterior_axis='AP')

    assert signal.shape == (10, 3)
    assert list(signal.columns) == columns


def test_signal3d_initialization_invalid_columns():
    """
    Test Signal3D raises ValueError for non-3-column data.

    Expected:
        Should raise ValueError when data doesn't have exactly 3 columns
    """
    data = np.random.randn(10, 2)
    index = np.arange(10)

    with pytest.raises(ValueError):
        Signal3D(data, index, unit='m')


def test_signal3d_vertical_axis_property():
    """
    Test vertical_axis property.

    Expected:
        Should return default 'Y' or custom vertical axis
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    signal = Signal3D(data, index, unit='m', vertical_axis='Y')

    assert signal.vertical_axis == 'Y'


def test_signal3d_anteroposterior_axis_property():
    """
    Test anteroposterior_axis property.

    Expected:
        Should return default 'Z' or custom anteroposterior axis
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    signal = Signal3D(data, index, unit='m', anteroposterior_axis='Z')

    assert signal.anteroposterior_axis == 'Z'


def test_signal3d_lateral_axis_property():
    """
    Test lateral_axis property.

    Expected:
        Should infer lateral axis as the column not in vertical/anteroposterior
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    signal = Signal3D(data, index, unit='m', vertical_axis='Y', anteroposterior_axis='Z')

    assert signal.lateral_axis == 'X'


def test_signal3d_lateral_axis_custom():
    """
    Test lateral_axis with custom axes configuration.

    Expected:
        Should correctly infer lateral axis from custom column names
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)
    columns = ['ML', 'V', 'AP']

    signal = Signal3D(data, index, unit='m', columns=columns, vertical_axis='V', anteroposterior_axis='AP')

    assert signal.lateral_axis == 'ML'


def test_signal3d_set_vertical_axis():
    """
    Test set_vertical_axis method.

    Expected:
        Should update vertical axis to valid column name
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    signal = Signal3D(data, index, unit='m', vertical_axis='Y')
    signal.set_vertical_axis('X')

    assert signal.vertical_axis == 'X'


def test_signal3d_set_vertical_axis_invalid():
    """
    Test set_vertical_axis with invalid column name.

    Expected:
        Should raise ValueError for non-existent column
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    signal = Signal3D(data, index, unit='m')

    with pytest.raises(ValueError, match="vertical_axis must be any of"):
        signal.set_vertical_axis('W')


def test_signal3d_set_anteroposterior_axis():
    """
    Test set_anteroposterior_axis method.

    Expected:
        Should update anteroposterior axis to valid column name
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    signal = Signal3D(data, index, unit='m', anteroposterior_axis='Z')
    signal.set_anteroposterior_axis('Y')

    assert signal.anteroposterior_axis == 'Y'


def test_signal3d_set_anteroposterior_axis_invalid():
    """
    Test set_anteroposterior_axis with invalid column name.

    Expected:
        Should raise ValueError for non-existent column
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    signal = Signal3D(data, index, unit='m')

    with pytest.raises(ValueError, match="anteroposterior_axis must be any of"):
        signal.set_anteroposterior_axis('W')


def test_signal3d_module_property():
    """
    Test module property calculates magnitude.

    Expected:
        Should return Signal1D with L2 norm of 3D vector
    """
    data = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    index = np.array([0.0, 1.0, 2.0])

    signal = Signal3D(data, index, unit='m')
    module = signal.module

    assert isinstance(module, Signal1D)
    assert module.shape == (3, 1)
    assert np.isclose(module._data[0, 0], 5.0)  # sqrt(3^2 + 4^2 + 0^2)
    assert np.isclose(module._data[1, 0], 1.0)  # sqrt(1^2)
    assert np.isclose(module._data[2, 0], 5.0)  # sqrt(5^2)


def test_signal3d_module_preserves_unit():
    """
    Test module property preserves unit.

    Expected:
        Resulting Signal1D should have same unit as original Signal3D
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    signal = Signal3D(data, index, unit='mm')
    module = signal.module

    assert module.unit == 'mm'


def test_signal3d_copy():
    """
    Test copy method creates independent copy.

    Expected:
        Should create new Signal3D with copied data, index, axes
    """
    data = np.random.randn(10, 3)
    index = np.linspace(0, 1, 10)
    columns = ['X', 'Y', 'Z']

    original = Signal3D(data, index, unit='m', columns=columns, vertical_axis='Y', anteroposterior_axis='Z')
    copied = original.copy()

    assert isinstance(copied, Signal3D)
    assert copied is not original
    assert copied.shape == original.shape
    assert copied.unit == original.unit
    assert copied.vertical_axis == original.vertical_axis
    assert copied.anteroposterior_axis == original.anteroposterior_axis
    assert np.array_equal(copied._data, original._data)


def test_signal3d_copy_independence():
    """
    Test copied Signal3D is independent from original.

    Expected:
        Modifying copy should not affect original
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    original = Signal3D(data, index, unit='m')
    copied = original.copy()

    copied._data[0, 0] = 999.0

    assert original._data[0, 0] != 999.0


def test_signal3d_shape_property():
    """
    Test shape property.

    Expected:
        Should return (n_samples, 3) for 3D signal
    """
    data = np.random.randn(50, 3)
    index = np.arange(50)

    signal = Signal3D(data, index, unit='cm')

    assert signal.shape == (50, 3)


def test_signal3d_with_list_input():
    """
    Test Signal3D accepts list input.

    Expected:
        Should convert list to numpy array internally
    """
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    index = [0.0, 0.5, 1.0]

    signal = Signal3D(data, index, unit='m')

    assert signal.shape == (3, 3)


def test_signal3d_indexing_access():
    """
    Test accessing data through indexers.

    Expected:
        Should support loc/iloc access inherited from Timeseries
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    signal = Signal3D(data, index, unit='m')

    assert hasattr(signal, 'loc')
    assert hasattr(signal, 'iloc')


def test_signal3d_loc_preserves_axes():
    """
    Test loc[] getter preserves vertical_axis and anteroposterior_axis.

    Expected:
        Sliced Signal3D should retain axis attributes
    """
    data = np.random.randn(10, 3)
    index = np.arange(10, dtype=float)
    signal = Signal3D(data, index, unit='m', vertical_axis='Y', anteroposterior_axis='Z')

    sliced = signal.loc[2.0:6.0, :]
    assert isinstance(sliced, Signal3D)
    assert sliced.vertical_axis == signal.vertical_axis
    assert sliced.anteroposterior_axis == signal.anteroposterior_axis
    assert sliced.unit == 'm'


def test_signal3d_iloc_preserves_axes():
    """
    Test iloc[] getter preserves axis attributes.

    Expected:
        Sliced Signal3D should retain all custom attributes
    """
    data = np.random.randn(10, 3)
    index = np.arange(10, dtype=float)
    signal = Signal3D(data, index, unit='mm', vertical_axis='Y', anteroposterior_axis='Z')

    sliced = signal.iloc[2:6, :]
    assert isinstance(sliced, Signal3D)
    assert sliced.vertical_axis == signal.vertical_axis
    assert sliced.anteroposterior_axis == signal.anteroposterior_axis
    assert sliced.unit == 'mm'


def test_signal3d_loc_setter_array():
    """
    Test loc[] setter with array preserves axes.

    Expected:
        Should update data while preserving axis attributes
    """
    data = np.random.randn(10, 3)
    index = np.arange(10, dtype=float)
    signal = Signal3D(data, index, unit='m', vertical_axis='Y', anteroposterior_axis='Z')

    signal.loc[2.0, :] = [1.0, 2.0, 3.0]
    assert np.allclose(signal._data[2, :], [1.0, 2.0, 3.0])
    assert signal.vertical_axis == 'Y'


def test_signal3d_iloc_setter():
    """
    Test iloc[] setter preserves type and axes.

    Expected:
        Should update data without losing Signal3D attributes
    """
    data = np.random.randn(10, 3)
    index = np.arange(10, dtype=float)
    signal = Signal3D(data, index, unit='m', vertical_axis='Y', anteroposterior_axis='Z')

    signal.iloc[3, :] = [10.0, 20.0, 30.0]
    assert np.allclose(signal._data[3, :], [10.0, 20.0, 30.0])
    assert signal.vertical_axis == 'Y'
