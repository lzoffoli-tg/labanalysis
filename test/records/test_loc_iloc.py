"""
Test loc/iloc implementation for Timeseries, Signal3D, and Record classes.

Tests both getter and setter functionality for label-based (loc) and
position-based (iloc) indexing, plus backward compatibility with legacy
__getitem__/__setitem__ methods.
"""

import sys
from pathlib import Path

# Add src directory to path to import from source code, not installed package
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import labanalysis as laban


def test_timeseries_loc_get():
    """Test Timeseries.loc getter."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    index = np.array([0.0, 0.1, 0.2])
    ts = laban.Timeseries(data, index, columns=['X', 'Y', 'Z'], unit='m')

    # Test single column access
    col_x = ts.loc[:, 'X']
    assert isinstance(col_x, laban.Timeseries)
    assert col_x.shape == (3, 1)

    # Test row range
    subset = ts.loc[0:0.15, :]
    assert subset.shape[0] == 2  # Should include 0.0 and 0.1

    print("✓ Timeseries.loc getter works")


def test_timeseries_loc_set():
    """Test Timeseries.loc setter."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    index = np.array([0.0, 0.1, 0.2])
    ts = laban.Timeseries(data, index, columns=['X', 'Y', 'Z'], unit='m')

    # Test setting single value
    ts.loc[0.0, 'X'] = 99
    assert ts._data[0, 0] == 99

    # Test setting column
    ts.loc[:, 'Y'] = 0
    assert np.all(ts._data[:, 1] == 0)

    print("✓ Timeseries.loc setter works")


def test_timeseries_iloc_get():
    """Test Timeseries.iloc getter."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    index = np.array([0.0, 0.1, 0.2])
    ts = laban.Timeseries(data, index, columns=['X', 'Y', 'Z'], unit='m')

    # Test position-based access
    first_row = ts.iloc[0, :]
    assert first_row.shape == (1, 3)
    assert np.allclose(first_row._data, [[1, 2, 3]])

    # Test slice
    subset = ts.iloc[:2, :]
    assert subset.shape == (2, 3)

    print("✓ Timeseries.iloc getter works")


def test_timeseries_iloc_set():
    """Test Timeseries.iloc setter."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    index = np.array([0.0, 0.1, 0.2])
    ts = laban.Timeseries(data, index, columns=['X', 'Y', 'Z'], unit='m')

    # Test setting by position
    ts.iloc[0, 0] = 99
    assert ts._data[0, 0] == 99

    # Test setting range
    ts.iloc[:2, :] = 0
    assert np.all(ts._data[:2, :] == 0)

    print("✓ Timeseries.iloc setter works")


def test_timeseries_backward_compat():
    """Test backward compatibility with __getitem__/__setitem__."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    index = np.array([0.0, 0.1, 0.2])
    ts = laban.Timeseries(data, index, columns=['X', 'Y', 'Z'], unit='m')

    # Test old-style access still works
    col_x = ts['X']
    assert isinstance(col_x, laban.Timeseries)

    # Test old-style assignment still works
    ts['Y'] = 0
    assert np.all(ts._data[:, 1] == 0)

    # Test tuple access
    val = ts[0.0, 'X']
    assert val.shape == (1, 1)

    # Test tuple assignment
    ts[0.1, 'Z'] = 99
    assert ts._data[1, 2] == 99

    print("✓ Backward compatibility works")


def test_signal3d_type_preservation():
    """Test that Signal3D preserves type when appropriate."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    index = np.array([0.0, 0.1, 0.2])
    sig = laban.Signal3D(data, index, unit='m')

    # All columns -> preserve Signal3D type
    subset = sig.loc[0:0.1, :]
    assert isinstance(subset, laban.Signal3D)

    # Single column -> should return Signal1D or Timeseries
    col_x = sig.loc[:, 'X']
    assert col_x.shape == (3, 1)

    print("✓ Signal3D type preservation works")


def test_record_loc_basic():
    """Test Record.loc basic functionality."""
    sig1 = laban.Signal1D(np.array([1, 2, 3]), np.array([0.0, 0.1, 0.2]), unit='V')
    sig2 = laban.Signal1D(np.array([4, 5, 6]), np.array([0.0, 0.1, 0.2]), unit='V')
    rec = laban.Record(signal1=sig1, signal2=sig2)

    # Test getting single item
    subset = rec.loc[:, 'signal1']
    assert 'signal1' in subset.keys()
    assert 'signal2' not in subset.keys()

    # Test setting value
    rec.loc[:, 'signal1'] = 0
    assert np.all(rec._data['signal1']._data == 0)

    print("✓ Record.loc basic functionality works")


def test_record_iloc_basic():
    """Test Record.iloc basic functionality."""
    sig1 = laban.Signal1D(np.array([1, 2, 3]), np.array([0.0, 0.1, 0.2]), unit='V')
    sig2 = laban.Signal1D(np.array([4, 5, 6]), np.array([0.0, 0.1, 0.2]), unit='V')
    rec = laban.Record(signal1=sig1, signal2=sig2)

    # Test position-based access
    subset = rec.iloc[:, 0]  # First item
    assert len(subset.keys()) == 1

    # Test setting
    rec.iloc[:, 0] = 99
    first_key = rec.keys()[0]
    assert np.all(rec._data[first_key]._data == 99)

    print("✓ Record.iloc basic functionality works")


if __name__ == '__main__':
    test_timeseries_loc_get()
    test_timeseries_loc_set()
    test_timeseries_iloc_get()
    test_timeseries_iloc_set()
    test_timeseries_backward_compat()
    test_signal3d_type_preservation()
    test_record_loc_basic()
    test_record_iloc_basic()

    print("\n✅ All quick tests passed!")
