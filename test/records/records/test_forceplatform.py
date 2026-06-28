"""
Test suite for ForcePlatform class.

Tests verify force platform measurement system with origin, force, and torque.
"""

import numpy as np
import pytest

from labanalysis.records import ForcePlatform
from labanalysis.timeseries import Point3D, Signal3D


def test_forceplatform_initialization():
    """
    Test ForcePlatform initialization with valid inputs.

    Expected:
        Should create ForcePlatform with origin, force, torque
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m')
    force = Signal3D(data, index, unit='N')
    torque = Signal3D(data, index, unit='Nm')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)

    assert len(fp) == 3
    assert 'origin' in fp.keys()
    assert 'force' in fp.keys()
    assert 'torque' in fp.keys()


def test_forceplatform_origin_type_validation():
    """
    Test ForcePlatform raises TypeError for invalid origin type.

    Expected:
        Should raise TypeError when origin is not Point3D
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    force = Signal3D(data, index, unit='N')
    torque = Signal3D(data, index, unit='Nm')

    with pytest.raises(TypeError, match="origin must be an instance of Point3D"):
        ForcePlatform(origin="invalid", force=force, torque=torque)


def test_forceplatform_force_type_validation():
    """
    Test ForcePlatform raises TypeError for invalid force type.

    Expected:
        Should raise TypeError when force is not Signal3D
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m')
    torque = Signal3D(data, index, unit='Nm')

    with pytest.raises(TypeError, match="force must be an instance of Signal3D"):
        ForcePlatform(origin=origin, force="invalid", torque=torque)


def test_forceplatform_torque_type_validation():
    """
    Test ForcePlatform raises TypeError for invalid torque type.

    Expected:
        Should raise TypeError when torque is not Signal3D
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m')
    force = Signal3D(data, index, unit='N')

    with pytest.raises(TypeError, match="torque must be an instance of Signal3D"):
        ForcePlatform(origin=origin, force=force, torque="invalid")


def test_forceplatform_vertical_axis_consistency():
    """
    Test ForcePlatform validates vertical axis consistency.

    Expected:
        Should raise ValueError when vertical axes don't match
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m', vertical_axis='Y')
    force = Signal3D(data, index, unit='N', vertical_axis='Z')
    torque = Signal3D(data, index, unit='Nm', vertical_axis='Y')

    with pytest.raises(ValueError, match="vertical axes must be the same"):
        ForcePlatform(origin=origin, force=force, torque=torque)


def test_forceplatform_anteroposterior_axis_consistency():
    """
    Test ForcePlatform validates anteroposterior axis consistency.

    Expected:
        Should raise ValueError when anteroposterior axes don't match
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m', anteroposterior_axis='Z')
    force = Signal3D(data, index, unit='N', anteroposterior_axis='X')
    torque = Signal3D(data, index, unit='Nm', anteroposterior_axis='Z')

    with pytest.raises(ValueError, match="anteroposterior axes must be the same"):
        ForcePlatform(origin=origin, force=force, torque=torque)


def test_forceplatform_vertical_axis_property():
    """
    Test vertical_axis property.

    Expected:
        Should return vertical axis from origin
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m', vertical_axis='Y')
    force = Signal3D(data, index, unit='N', vertical_axis='Y')
    torque = Signal3D(data, index, unit='Nm', vertical_axis='Y')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)

    assert fp.vertical_axis == 'Y'


def test_forceplatform_anteroposterior_axis_property():
    """
    Test anteroposterior_axis property.

    Expected:
        Should return anteroposterior axis from origin
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m', anteroposterior_axis='Z')
    force = Signal3D(data, index, unit='N', anteroposterior_axis='Z')
    torque = Signal3D(data, index, unit='Nm', anteroposterior_axis='Z')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)

    assert fp.anteroposterior_axis == 'Z'


def test_forceplatform_lateral_axis_property():
    """
    Test lateral_axis property.

    Expected:
        Should infer lateral axis from origin
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m', vertical_axis='Y', anteroposterior_axis='Z')
    force = Signal3D(data, index, unit='N', vertical_axis='Y', anteroposterior_axis='Z')
    torque = Signal3D(data, index, unit='Nm', vertical_axis='Y', anteroposterior_axis='Z')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)

    assert fp.lateral_axis == 'X'


def test_forceplatform_access_origin():
    """
    Test accessing origin signal.

    Expected:
        Should return Point3D instance
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m')
    force = Signal3D(data, index, unit='N')
    torque = Signal3D(data, index, unit='Nm')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)

    assert isinstance(fp['origin'], Point3D)
    assert fp['origin'] is origin


def test_forceplatform_access_force():
    """
    Test accessing force signal.

    Expected:
        Should return Signal3D instance
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m')
    force = Signal3D(data, index, unit='N')
    torque = Signal3D(data, index, unit='Nm')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)

    assert isinstance(fp['force'], Signal3D)
    assert fp['force'] is force


def test_forceplatform_access_torque():
    """
    Test accessing torque signal.

    Expected:
        Should return Signal3D instance
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m')
    force = Signal3D(data, index, unit='N')
    torque = Signal3D(data, index, unit='Nm')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)

    assert isinstance(fp['torque'], Signal3D)
    assert fp['torque'] is torque


def test_forceplatform_inherits_record():
    """
    Test ForcePlatform inherits Record methods.

    Expected:
        Should have Record properties and methods
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m')
    force = Signal3D(data, index, unit='N')
    torque = Signal3D(data, index, unit='Nm')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)

    assert hasattr(fp, 'keys')
    assert hasattr(fp, 'values')
    assert hasattr(fp, 'items')
    assert hasattr(fp, 'copy')
    assert hasattr(fp, 'to_dataframe')


def test_forceplatform_to_dataframe():
    """
    Test to_dataframe method.

    Expected:
        Should return DataFrame with all signals
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m')
    force = Signal3D(data, index, unit='N')
    torque = Signal3D(data, index, unit='Nm')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)
    df = fp.to_dataframe()

    assert df.shape == (10, 9)  # 3 signals × 3 columns each
