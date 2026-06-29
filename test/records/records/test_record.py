"""
Test suite for Record base class.

Tests verify dictionary-like container for Timeseries objects.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.records import Record
from labanalysis.timeseries import Signal1D, Signal3D, Timeseries


def test_record_initialization_empty():
    """
    Test Record initialization with no signals.

    Expected:
        Should create empty Record
    """
    rec = Record()

    assert len(rec) == 0
    assert len(rec.keys()) == 0


def test_record_initialization_with_signals():
    """
    Test Record initialization with signals.

    Expected:
        Should store signals in internal dict
    """
    data1 = np.array([1.0, 2.0, 3.0])
    index1 = np.array([0.0, 1.0, 2.0])
    sig1 = Signal1D(data1, index1, unit='m')

    data2 = np.random.randn(3, 3)
    index2 = np.array([0.0, 1.0, 2.0])
    sig2 = Signal3D(data2, index2, unit='m')

    rec = Record(signal_a=sig1, signal_b=sig2)

    assert len(rec) == 2
    assert 'signal_a' in rec.keys()
    assert 'signal_b' in rec.keys()


def test_record_getitem_string():
    """
    Test accessing signal by string key.

    Expected:
        Should return stored signal
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')

    rec = Record(my_signal=sig)

    retrieved = rec['my_signal']
    assert isinstance(retrieved, Signal1D)
    assert np.array_equal(retrieved._data, sig._data)


def test_record_getitem_invalid_key():
    """
    Test accessing non-existent key raises KeyError.

    Expected:
        Should raise KeyError for invalid key
    """
    rec = Record()

    with pytest.raises(KeyError):
        _ = rec['nonexistent']


def test_record_setitem():
    """
    Test setting new signal.

    Expected:
        Should add signal to record
    """
    rec = Record()
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')

    rec['new_signal'] = sig

    assert 'new_signal' in rec.keys()
    assert rec['new_signal'] is sig


def test_record_setitem_invalid_key_type():
    """
    Test setitem with non-string key raises ValueError.

    Expected:
        Should raise ValueError for non-string key
    """
    rec = Record()
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')

    with pytest.raises(ValueError, match="key must be a str"):
        rec[123] = sig


def test_record_setitem_invalid_value_type():
    """
    Test setitem with invalid value type raises ValueError.

    Expected:
        Should raise ValueError for non-Timeseries value
    """
    rec = Record()

    with pytest.raises(ValueError, match="value must be a Timeseries"):
        rec['invalid'] = "not a timeseries"


def test_record_keys():
    """
    Test keys method.

    Expected:
        Should return list of signal names
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data, index, unit='V')

    rec = Record(signal_a=sig1, signal_b=sig2)

    keys = rec.keys()
    assert len(keys) == 2
    assert 'signal_a' in keys
    assert 'signal_b' in keys


def test_record_values():
    """
    Test values method.

    Expected:
        Should return list of signals
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data, index, unit='V')

    rec = Record(signal_a=sig1, signal_b=sig2)

    values = rec.values()
    assert len(values) == 2
    assert any(v is sig1 for v in values)
    assert any(v is sig2 for v in values)


def test_record_items():
    """
    Test items method.

    Expected:
        Should return list of (key, value) tuples
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')

    rec = Record(signal_a=sig)

    items = rec.items()
    assert len(items) == 1
    assert items[0][0] == 'signal_a'
    assert items[0][1] is sig


def test_record_len():
    """
    Test __len__ method.

    Expected:
        Should return number of signals
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data, index, unit='V')

    rec = Record(signal_a=sig1, signal_b=sig2)

    assert len(rec) == 2


def test_record_index_property():
    """
    Test index property returns union of all indices.

    Expected:
        Should return sorted unique array of all time points
    """
    sig1 = Signal1D(np.array([1.0, 2.0]), np.array([0.0, 1.0]), unit='m')
    sig2 = Signal1D(np.array([3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0]), unit='m')

    rec = Record(a=sig1, b=sig2)

    index = rec.index
    expected = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(index, expected)


def test_record_shape_property():
    """
    Test shape property.

    Expected:
        Should return shape of DataFrame representation
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data, index, unit='V')

    rec = Record(a=sig1, b=sig2)

    assert rec.shape == (3, 2)


def test_record_to_dataframe_empty():
    """
    Test to_dataframe on empty record.

    Expected:
        Should return empty DataFrame
    """
    rec = Record()
    df = rec.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_record_to_dataframe_with_signals():
    """
    Test to_dataframe with signals.

    Expected:
        Should return DataFrame with multi-column structure
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data * 2, index, unit='V')

    rec = Record(signal_a=sig1, signal_b=sig2)
    df = rec.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)


def test_record_copy():
    """
    Test copy method.

    Expected:
        Should create deep copy of record
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')

    rec = Record(signal=sig)
    copied = rec.copy()

    assert isinstance(copied, Record)
    assert copied is not rec
    assert len(copied) == len(rec)
    assert 'signal' in copied.keys()


def test_record_copy_independence():
    """
    Test copied record is independent.

    Expected:
        Modifying copy should not affect original
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')

    rec = Record(signal=sig)
    copied = rec.copy()

    copied['signal']._data[0, 0] = 999.0

    assert rec['signal']._data[0, 0] != 999.0


def test_record_get_existing_key():
    """
    Test get method with existing key.

    Expected:
        Should return signal
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')

    rec = Record(signal=sig)
    retrieved = rec.get('signal')

    assert retrieved is sig


def test_record_get_nonexistent_key():
    """
    Test get method with non-existent key.

    Expected:
        Should return default value (None)
    """
    rec = Record()
    retrieved = rec.get('nonexistent')

    assert retrieved is None


def test_record_get_with_default():
    """
    Test get method with custom default.

    Expected:
        Should return custom default for missing key
    """
    rec = Record()
    retrieved = rec.get('nonexistent', default='custom_default')

    assert retrieved == 'custom_default'


def test_record_drop_single_key():
    """
    Test drop method with single key.

    Expected:
        Should remove signal from record
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data, index, unit='V')

    rec = Record(a=sig1, b=sig2)
    result = rec.drop('a')

    assert len(result) == 1
    assert 'a' not in result.keys()
    assert 'b' in result.keys()


def test_record_drop_multiple_keys():
    """
    Test drop method with multiple keys.

    Expected:
        Should remove all specified signals
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data, index, unit='V')
    sig3 = Signal1D(data, index, unit='A')

    rec = Record(a=sig1, b=sig2, c=sig3)
    result = rec.drop(['a', 'b'])

    assert len(result) == 1
    assert 'c' in result.keys()


def test_record_drop_inplace():
    """
    Test drop method with inplace=True.

    Expected:
        Should modify record in place and return None
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig1 = Signal1D(data, index, unit='m')
    sig2 = Signal1D(data, index, unit='V')

    rec = Record(a=sig1, b=sig2)
    result = rec.drop('a', inplace=True)

    assert result is None
    assert len(rec) == 1
    assert 'a' not in rec.keys()


def test_record_reset_time():
    """
    Test reset_time method.

    Expected:
        Should shift all time indices to start at zero
    """
    sig = Signal1D(np.array([1.0, 2.0, 3.0]), np.array([5.0, 6.0, 7.0]), unit='m')
    rec = Record(signal=sig)

    result = rec.reset_time()

    assert result['signal'].index[0] == 0.0
    assert result['signal'].index[1] == 1.0
    assert result['signal'].index[2] == 2.0


def test_record_reset_time_inplace():
    """
    Test reset_time with inplace=True.

    Expected:
        Should modify in place and return None
    """
    sig = Signal1D(np.array([1.0, 2.0, 3.0]), np.array([5.0, 6.0, 7.0]), unit='m')
    rec = Record(signal=sig)

    result = rec.reset_time(inplace=True)

    assert result is None
    assert rec['signal'].index[0] == 0.0


def test_record_loc_indexer():
    """
    Test loc indexer exists.

    Expected:
        Should have loc property for label-based indexing
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')

    rec = Record(signal=sig)

    assert hasattr(rec, 'loc')
    assert rec.loc is not None


def test_record_iloc_indexer():
    """
    Test iloc indexer exists.

    Expected:
        Should have iloc property for position-based indexing
    """
    data = np.array([1.0, 2.0, 3.0])
    index = np.array([0.0, 1.0, 2.0])
    sig = Signal1D(data, index, unit='m')

    rec = Record(signal=sig)

    assert hasattr(rec, 'iloc')
    assert rec.iloc is not None


def test_record_loc_getter_single_item():
    """
    Test Record.loc[] gets single item.

    Expected:
        Should return Record with selected item
    """
    sig1 = Signal1D(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), unit='V')
    sig2 = Signal1D(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 2.0]), unit='A')
    rec = Record(voltage=sig1, current=sig2)

    subset = rec.loc[:, 'voltage']
    assert isinstance(subset, Record)
    assert 'voltage' in subset.keys()
    assert 'current' not in subset.keys()


def test_record_loc_setter_scalar():
    """
    Test Record.loc[] setter with scalar.

    Expected:
        Should broadcast scalar to all rows in item
    """
    sig1 = Signal1D(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), unit='V')
    sig2 = Signal1D(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 2.0]), unit='A')
    rec = Record(voltage=sig1, current=sig2)

    rec.loc[:, 'voltage'] = 99.0
    assert np.all(rec._data['voltage']._data == 99.0)


def test_record_loc_setter_dict():
    """
    Test Record.loc[] setter with dict.

    Expected:
        Should update values from dict
    """
    sig1 = Signal1D(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), unit='V')
    sig2 = Signal1D(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 2.0]), unit='A')
    rec = Record(voltage=sig1, current=sig2)

    rec.loc[1.0, :] = {'voltage': 100.0, 'current': 200.0}
    assert rec._data['voltage']._data[1, 0] == 100.0
    assert rec._data['current']._data[1, 0] == 200.0


def test_record_iloc_getter_by_position():
    """
    Test Record.iloc[] gets item by position.

    Expected:
        Should return Record with item at position
    """
    sig1 = Signal1D(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), unit='V')
    sig2 = Signal1D(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 2.0]), unit='A')
    rec = Record(voltage=sig1, current=sig2)

    subset = rec.iloc[:, 0]  # First item
    assert isinstance(subset, Record)
    assert len(subset.keys()) == 1


def test_record_iloc_setter_scalar():
    """
    Test Record.iloc[] setter with scalar.

    Expected:
        Should broadcast scalar to selected item
    """
    sig1 = Signal1D(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), unit='V')
    sig2 = Signal1D(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 2.0]), unit='A')
    rec = Record(voltage=sig1, current=sig2)

    rec.iloc[:, 0] = 42.0
    first_key = rec.keys()[0]
    assert np.all(rec._data[first_key]._data == 42.0)


def test_record_loc_broadcast_to_all_items():
    """
    Test broadcasting scalar to all items in Record.

    Expected:
        Should broadcast to all items at selected time
    """
    sig1 = Signal1D(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), unit='V')
    sig2 = Signal1D(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 2.0]), unit='V')
    rec = Record(sig1=sig1, sig2=sig2)

    rec.loc[1.0, :] = 42.0  # Broadcast to all items at time 1.0
    assert rec._data['sig1']._data[1, 0] == 42.0
    assert rec._data['sig2']._data[1, 0] == 42.0
