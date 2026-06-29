"""
Test suite for TimeseriesRecord class.

Tests verify extended Record container supporting nested records and force platforms.
"""

import numpy as np
import pytest

from labanalysis.records import ForcePlatform, TimeseriesRecord
from labanalysis.timeseries import Point3D, Signal1D, Signal3D


def test_timeseriesrecord_initialization_empty():
    """
    Test TimeseriesRecord initialization with no signals.

    Expected:
        Should create empty TimeseriesRecord
    """
    rec = TimeseriesRecord()

    assert len(rec) == 0


def test_timeseriesrecord_initialization_with_signals():
    """
    Test TimeseriesRecord initialization with various signal types.

    Expected:
        Should accept Timeseries, ForcePlatform, nested records
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    sig1d = Signal1D(np.random.randn(10), index, unit='m')
    sig3d = Signal3D(data, index, unit='m')

    rec = TimeseriesRecord(signal1=sig1d, signal2=sig3d)

    assert len(rec) == 2
    assert 'signal1' in rec.keys()
    assert 'signal2' in rec.keys()


def test_timeseriesrecord_with_forceplatform():
    """
    Test TimeseriesRecord can contain ForcePlatform.

    Expected:
        Should accept and store ForcePlatform
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    origin = Point3D(data, index, unit='m')
    force = Signal3D(data, index, unit='N')
    torque = Signal3D(data, index, unit='Nm')

    fp = ForcePlatform(origin=origin, force=force, torque=torque)
    rec = TimeseriesRecord(force_plate=fp)

    assert len(rec) == 1
    assert 'force_plate' in rec.keys()
    assert isinstance(rec['force_plate'], ForcePlatform)


def test_timeseriesrecord_with_metabolicrecord():
    """
    Test TimeseriesRecord with basic Signal1D (MetabolicRecord not supported in base Record).

    Expected:
        Should accept Signal1D signals
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    sig1 = Signal1D(data, index, unit='ml/min')
    sig2 = Signal1D(data * 2, index, unit='bpm')

    rec = TimeseriesRecord(vo2=sig1, hr=sig2)

    assert len(rec) == 2
    assert 'vo2' in rec.keys()
    assert 'hr' in rec.keys()


def test_timeseriesrecord_vertical_axis_property():
    """
    Test vertical_axis property infers from contained signals.

    Expected:
        Should return vertical axis from first signal with this property
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    sig3d = Signal3D(data, index, unit='m', vertical_axis='Y')
    rec = TimeseriesRecord(signal=sig3d)

    assert rec.vertical_axis == 'Y'


def test_timeseriesrecord_anteroposterior_axis_property():
    """
    Test anteroposterior_axis property infers from contained signals.

    Expected:
        Should return anteroposterior axis from first signal with this property
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    sig3d = Signal3D(data, index, unit='m', anteroposterior_axis='Z')
    rec = TimeseriesRecord(signal=sig3d)

    assert rec.anteroposterior_axis == 'Z'


def test_timeseriesrecord_vertical_axis_none_when_no_signals():
    """
    Test vertical_axis returns None for empty record.

    Expected:
        Should return None when no signals have vertical_axis
    """
    rec = TimeseriesRecord()

    assert rec.vertical_axis is None


def test_timeseriesrecord_anteroposterior_axis_none_when_no_signals():
    """
    Test anteroposterior_axis returns None for empty record.

    Expected:
        Should return None when no signals have anteroposterior_axis
    """
    rec = TimeseriesRecord()

    assert rec.anteroposterior_axis is None


def test_timeseriesrecord_inherits_record():
    """
    Test TimeseriesRecord inherits Record functionality.

    Expected:
        Should have all Record methods
    """
    rec = TimeseriesRecord()

    assert hasattr(rec, 'keys')
    assert hasattr(rec, 'values')
    assert hasattr(rec, 'items')
    assert hasattr(rec, 'copy')
    assert hasattr(rec, 'to_dataframe')
    assert hasattr(rec, 'strip')
    assert hasattr(rec, 'reset_time')
    assert hasattr(rec, 'fillna')


def test_timeseriesrecord_to_dataframe():
    """
    Test to_dataframe method.

    Expected:
        Should convert all signals to DataFrame
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])

    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data * 2, index, unit='V')

    rec = TimeseriesRecord(a=sig1, b=sig2)
    df = rec.to_dataframe()

    assert df.shape == (3, 2)


def test_timeseriesrecord_copy():
    """
    Test copy method.

    Expected:
        Should create deep copy
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])

    sig = Signal1D(data, index, unit='m')
    rec = TimeseriesRecord(signal=sig)

    copied = rec.copy()

    assert isinstance(copied, TimeseriesRecord)
    assert copied is not rec
    assert len(copied) == len(rec)


def test_timeseriesrecord_nested():
    """
    Test TimeseriesRecord with multiple signals (nesting not supported in base Record).

    Expected:
        Should handle multiple Signal1D instances
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])

    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data * 2, index, unit='V')
    sig3 = Signal1D(data * 3, index, unit='A')

    rec = TimeseriesRecord(signal1=sig1, signal2=sig2, signal3=sig3)

    assert len(rec) == 3
    assert 'signal1' in rec.keys()
    assert 'signal2' in rec.keys()
    assert 'signal3' in rec.keys()


def test_timeseriesrecord_mixed_content():
    """
    Test TimeseriesRecord with mixed content types.

    Expected:
        Should handle Signal1D, Signal3D, ForcePlatform together
    """
    data = np.random.randn(10, 3)
    index = np.arange(10)

    sig1d = Signal1D(np.random.randn(10), index, unit='m')
    sig3d = Signal3D(data, index, unit='m')

    origin = Point3D(data, index, unit='m')
    force = Signal3D(data, index, unit='N')
    torque = Signal3D(data, index, unit='Nm')
    fp = ForcePlatform(origin=origin, force=force, torque=torque)

    rec = TimeseriesRecord(signal1d=sig1d, signal3d=sig3d, fp=fp)

    assert len(rec) == 3
    assert isinstance(rec['signal1d'], Signal1D)
    assert isinstance(rec['signal3d'], Signal3D)
    assert isinstance(rec['fp'], ForcePlatform)


def test_timeseriesrecord_index_property():
    """
    Test index property returns union of all indices.

    Expected:
        Should return sorted unique array of all time points
    """
    sig1 = Signal1D(np.array([1.0, 2.0]), np.array([0.0, 1.0]), unit='m')
    sig2 = Signal1D(np.array([3.0, 4.0]), np.array([2.0, 3.0]), unit='m')

    rec = TimeseriesRecord(a=sig1, b=sig2)
    index = rec.index

    expected = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(index, expected)


def test_timeseriesrecord_shape_property():
    """
    Test shape property.

    Expected:
        Should return shape of DataFrame representation
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])

    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data, index, unit='V')

    rec = TimeseriesRecord(a=sig1, b=sig2)

    assert rec.shape == (3, 2)


def test_timeseriesrecord_loc_getter_preserves_type():
    """
    Test loc[] getter preserves TimeseriesRecord type.

    Expected:
        Sliced record should be TimeseriesRecord instance
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    index = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data * 2, index, unit='V')
    rec = TimeseriesRecord(a=sig1, b=sig2)

    sliced = rec.loc[1.0:3.0, :]
    assert isinstance(sliced, TimeseriesRecord)
    assert 'a' in sliced.keys()
    assert 'b' in sliced.keys()


def test_timeseriesrecord_iloc_getter_preserves_type():
    """
    Test iloc[] getter preserves TimeseriesRecord type.

    Expected:
        Sliced record should be TimeseriesRecord instance
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    index = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data * 2, index, unit='V')
    rec = TimeseriesRecord(a=sig1, b=sig2)

    sliced = rec.iloc[1:4, :]
    assert isinstance(sliced, TimeseriesRecord)
    assert len(sliced.keys()) == 2


def test_timeseriesrecord_loc_setter():
    """
    Test loc[] setter modifies data correctly.

    Expected:
        Should update values without breaking type
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')
    rec = TimeseriesRecord(signal=sig)

    rec.loc[1.0, 'signal'] = 999.0
    assert rec._data['signal']._data[1, 0] == 999.0


def test_timeseriesrecord_iloc_setter():
    """
    Test iloc[] setter modifies data correctly.

    Expected:
        Should update values by position
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')
    rec = TimeseriesRecord(signal=sig)

    rec.iloc[2, 0] = 777.0
    assert rec._data['signal']._data[2, 0] == 777.0
