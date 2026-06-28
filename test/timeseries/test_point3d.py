"""
Test suite for Point3D class.

Tests verify 3D point time-series with automatic conversion to meters.
"""

import numpy as np
import pytest

from labanalysis.timeseries import Point3D


def test_point3d_initialization_default():
    """
    Test Point3D initialization with default parameters.

    Expected:
        Should create Point3D with default columns ['X', 'Y', 'Z'] and unit='m'
    """
    data = np.random.randn(10, 3)
    index = np.linspace(0, 1, 10)

    point = Point3D(data, index)

    assert point.shape == (10, 3)
    assert list(point.columns) == ['X', 'Y', 'Z']
    assert point.unit == 'm'


def test_point3d_initialization_custom_unit():
    """
    Test Point3D with custom length unit.

    Expected:
        Should convert to meters automatically
    """
    data = np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]])
    index = np.array([0.0, 1.0])

    point = Point3D(data, index, unit='cm')

    assert point.unit == 'm'
    assert np.isclose(point._data[0, 0], 1.0)  # 100 cm = 1 m
    assert np.isclose(point._data[0, 1], 2.0)  # 200 cm = 2 m
    assert np.isclose(point._data[0, 2], 3.0)  # 300 cm = 3 m


def test_point3d_initialization_custom_columns():
    """
    Test Point3D with custom column names.

    Expected:
        Should accept custom 3D column names
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)
    columns = ['AP', 'ML', 'V']

    point = Point3D(data, index, columns=columns, vertical_axis='V', anteroposterior_axis='AP')

    assert list(point.columns) == columns


def test_point3d_invalid_unit():
    """
    Test Point3D raises ValueError for non-length unit.

    Expected:
        Should raise ValueError when unit is not length
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    with pytest.raises(ValueError, match="unit must represent length"):
        Point3D(data, index, unit='V')


def test_point3d_vertical_axis_property():
    """
    Test vertical_axis property inherited from Signal3D.

    Expected:
        Should return default 'Y' or custom vertical axis
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    point = Point3D(data, index, vertical_axis='Y')

    assert point.vertical_axis == 'Y'


def test_point3d_anteroposterior_axis_property():
    """
    Test anteroposterior_axis property inherited from Signal3D.

    Expected:
        Should return default 'Z' or custom anteroposterior axis
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    point = Point3D(data, index, anteroposterior_axis='Z')

    assert point.anteroposterior_axis == 'Z'


def test_point3d_lateral_axis_property():
    """
    Test lateral_axis property inherited from Signal3D.

    Expected:
        Should infer lateral axis correctly
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    point = Point3D(data, index, vertical_axis='Y', anteroposterior_axis='Z')

    assert point.lateral_axis == 'X'


def test_point3d_module_property():
    """
    Test module property calculates magnitude.

    Expected:
        Should return Signal1D with distance from origin
    """
    data = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
    index = np.array([0.0, 1.0])

    point = Point3D(data, index)
    module = point.module

    assert np.isclose(module._data[0, 0], 5.0)  # sqrt(3^2 + 4^2)
    assert np.isclose(module._data[1, 0], 1.0)


def test_point3d_copy():
    """
    Test copy method creates independent copy.

    Expected:
        Should create new Point3D with copied data
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    original = Point3D(data, index, unit='m')
    copied = original.copy()

    assert isinstance(copied, Point3D)
    assert copied is not original
    assert copied.shape == original.shape
    assert copied.unit == original.unit
    assert copied.vertical_axis == original.vertical_axis
    assert copied.anteroposterior_axis == original.anteroposterior_axis


def test_point3d_copy_independence():
    """
    Test copied Point3D is independent from original.

    Expected:
        Modifying copy should not affect original
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    original = Point3D(data, index)
    copied = original.copy()

    copied._data[0, 0] = 999.0

    assert original._data[0, 0] != 999.0


def test_point3d_inherits_signal3d():
    """
    Test Point3D inherits Signal3D properties.

    Expected:
        Should have Signal3D methods and properties
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    point = Point3D(data, index)

    assert hasattr(point, 'vertical_axis')
    assert hasattr(point, 'anteroposterior_axis')
    assert hasattr(point, 'lateral_axis')
    assert hasattr(point, 'module')


def test_point3d_unit_conversion_mm_to_m():
    """
    Test conversion from millimeters to meters.

    Expected:
        Should convert mm to m with correct magnitude
    """
    data = np.array([[1000.0, 2000.0, 3000.0]])
    index = np.array([0.0])

    point = Point3D(data, index, unit='mm')

    assert point.unit == 'm'
    assert np.isclose(point._data[0, 0], 1.0)  # 1000 mm = 1 m
    assert np.isclose(point._data[0, 1], 2.0)  # 2000 mm = 2 m
    assert np.isclose(point._data[0, 2], 3.0)  # 3000 mm = 3 m


def test_point3d_shape_property():
    """
    Test shape property.

    Expected:
        Should return (n_samples, 3) for 3D points
    """
    data = np.random.randn(50, 3)
    index = np.arange(50)

    point = Point3D(data, index)

    assert point.shape == (50, 3)


def test_point3d_indexing_access():
    """
    Test accessing data through indexers.

    Expected:
        Should support loc/iloc access inherited from Timeseries
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    point = Point3D(data, index)

    assert hasattr(point, 'loc')
    assert hasattr(point, 'iloc')
