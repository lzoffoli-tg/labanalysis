"""
Test for Timeseries.strip() optimization and correctness.
"""
import numpy as np
import pytest

from labanalysis.timeseries import Timeseries, EMGSignal, Point3D


class TestTimeseriesStrip:
    """Test Timeseries.strip() optimization."""

    def test_strip_preserves_data_correctness(self):
        """Verify strip produces correct results."""
        n_rows = 1000
        n_cols = 10

        data = np.random.randn(n_rows, n_cols)
        data[:100, :] = np.nan  # Leading NaNs
        data[-100:, :] = np.nan  # Trailing NaNs
        data[:, :2] = np.nan  # Leading column NaNs
        data[:, -2:] = np.nan  # Trailing column NaNs

        index = np.arange(n_rows, dtype=float)
        columns = [f"col_{i}" for i in range(n_cols)]

        ts = Timeseries(data, index, columns, "m")
        ts_stripped = ts.strip(inplace=False)

        # Check shape
        assert ts_stripped.shape == (800, 6)

        # Check index range
        assert ts_stripped.index[0] == 100.0
        assert ts_stripped.index[-1] == 899.0

        # Check no NaN at borders
        assert not np.all(np.isnan(ts_stripped._data[0, :]))
        assert not np.all(np.isnan(ts_stripped._data[-1, :]))
        assert not np.all(np.isnan(ts_stripped._data[:, 0]))
        assert not np.all(np.isnan(ts_stripped._data[:, -1]))

    def test_strip_axis_0_only(self):
        """Test strip with axis=0 only."""
        data = np.random.randn(100, 5)
        data[:10, :] = np.nan
        data[-10:, :] = np.nan

        ts = Timeseries(data, np.arange(100.0), [f"c{i}" for i in range(5)], "m")
        ts_stripped = ts.strip(axis=0, inplace=False)

        assert ts_stripped.shape == (80, 5)
        assert ts_stripped.index[0] == 10.0

    def test_strip_axis_1_only(self):
        """Test strip with axis=1 only."""
        data = np.random.randn(100, 10)
        data[:, :2] = np.nan
        data[:, -2:] = np.nan

        ts = Timeseries(data, np.arange(100.0), [f"c{i}" for i in range(10)], "m")
        ts_stripped = ts.strip(axis=1, inplace=False)

        assert ts_stripped.shape == (100, 6)

    def test_strip_inplace(self):
        """Test strip with inplace=True."""
        data = np.random.randn(100, 5)
        data[:10, :] = np.nan
        data[-10:, :] = np.nan

        ts = Timeseries(data, np.arange(100.0), [f"c{i}" for i in range(5)], "m")
        original_id = id(ts)

        result = ts.strip(inplace=True)

        assert result is None
        assert ts.shape == (80, 5)
        assert id(ts) == original_id

    def test_strip_performance(self):
        """Verify strip is reasonably fast on large data."""
        import time

        n_rows = 10000
        n_cols = 50

        data = np.random.randn(n_rows, n_cols)
        data[:100, :] = np.nan
        data[-100:, :] = np.nan

        ts = Timeseries(data, np.arange(n_rows, dtype=float),
                       [f"c{i}" for i in range(n_cols)], "m")

        start = time.perf_counter()
        ts.strip(inplace=False)
        elapsed = time.perf_counter() - start

        # Should complete in under 100ms (was ~1.6s before optimization)
        assert elapsed < 0.1, f"strip() too slow: {elapsed*1000:.0f}ms"


class TestEMGSignalStripCopy:
    """Test EMGSignal attributes are preserved during strip and copy."""

    def test_emgsignal_copy_preserves_attributes(self):
        """Test that EMGSignal.copy() preserves muscle_name and side."""
        data = np.random.randn(100, 1)
        index = np.linspace(0, 10, 100)
        emg = EMGSignal(data, index, muscle_name="Biceps", side="left", unit="uV")

        emg_copy = emg.copy()

        assert emg_copy.muscle_name == "Biceps"
        assert emg_copy.side == "left"

    def test_emgsignal_attributes_after_slicing(self):
        """Test that EMGSignal attributes are preserved after slicing."""
        data = np.random.randn(100, 1)
        index = np.linspace(0, 10, 100)
        emg = EMGSignal(data, index, muscle_name="Biceps", side="left", unit="uV")

        emg_slice = emg[10:90]

        assert hasattr(emg_slice, '_name')
        assert hasattr(emg_slice, '_side')
        assert emg_slice._name == "Biceps"
        assert emg_slice._side == "left"

    def test_emgsignal_copy_after_slicing(self):
        """
        Test that EMGSignal can be copied after slicing.

        NOTE: This test requires the updated EMGSignal.copy() that uses
        self._name and self._side instead of properties.
        """
        data = np.random.randn(100, 1)
        index = np.linspace(0, 10, 100)
        emg = EMGSignal(data, index, muscle_name="Biceps", side="left", unit="uV")

        # Use .loc to get proper slice with 2D data
        emg_slice = emg.loc[1.0:9.0, :]

        # Verify attributes are preserved after slicing
        assert hasattr(emg_slice, '_name')
        assert hasattr(emg_slice, '_side')
        assert emg_slice._name == "Biceps"
        assert emg_slice._side == "left"

        # Copy should work (requires updated copy() method)
        emg_copy = emg_slice.copy()

        assert emg_copy.muscle_name == "Biceps"
        assert emg_copy.side == "left"


class TestPoint3DStripCopy:
    """Test Point3D attributes are preserved during strip and copy."""

    def test_point3d_copy_preserves_axes(self):
        """Test that Point3D.copy() preserves axis attributes."""
        data = np.random.randn(100, 3)
        index = np.linspace(0, 10, 100)
        point = Point3D(data, index, unit="m", vertical_axis="Y",
                       anteroposterior_axis="Z")

        point_copy = point.copy()

        assert point_copy.vertical_axis == "Y"
        assert point_copy.anteroposterior_axis == "Z"

    def test_point3d_attributes_after_slicing(self):
        """Test that Point3D attributes are preserved after slicing."""
        data = np.random.randn(100, 3)
        index = np.linspace(0, 10, 100)
        point = Point3D(data, index, unit="m", vertical_axis="Y",
                       anteroposterior_axis="Z")

        point_slice = point[10:90]

        assert hasattr(point_slice, '_vertical_axis')
        assert hasattr(point_slice, '_anteroposterior_axis')
        assert point_slice._vertical_axis == "Y"
        assert point_slice._anteroposterior_axis == "Z"
