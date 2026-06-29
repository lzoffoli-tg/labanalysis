"""
Test for Record.strip() optimization and attribute preservation.
"""
import numpy as np
import pytest

from labanalysis.timeseries import Point3D, EMGSignal, Signal3D
from labanalysis.records import TimeseriesRecord, ForcePlatform


class TestRecordStripOptimization:
    """Test Record.strip() optimization and correctness."""

    def test_strip_independent_false_preserves_attributes(self):
        """
        Critical test: strip(independent=False, inplace=True) must preserve
        subclass attributes like EMGSignal._name and Point3D._vertical_axis.

        This was the main bug: strip was replacing objects with views,
        losing subclass-specific attributes.
        """
        n_samples = 1000
        index = np.linspace(0, 10, n_samples)

        # Create EMGSignal with attributes
        emg_data = np.random.randn(n_samples, 1)
        emg_data[:100, :] = np.nan
        emg_data[-100:, :] = np.nan
        emg = EMGSignal(emg_data, index, muscle_name="Biceps", side="left", unit="uV")

        # Create Point3D with attributes
        point_data = np.random.randn(n_samples, 3)
        point_data[:50, :] = np.nan
        point_data[-50:, :] = np.nan
        point = Point3D(point_data, index, unit="m", vertical_axis="Y",
                       anteroposterior_axis="Z")

        rec = TimeseriesRecord(emg=emg, marker=point)

        # Apply strip(independent=False, inplace=True)
        rec.strip(independent=False, inplace=True)

        # Check EMG attributes are preserved
        assert hasattr(rec['emg'], '_name'), "EMG lost _name attribute"
        assert hasattr(rec['emg'], '_side'), "EMG lost _side attribute"
        assert rec['emg']._name == "Biceps"
        assert rec['emg']._side == "left"

        # Check Point3D attributes are preserved
        assert hasattr(rec['marker'], '_vertical_axis')
        assert hasattr(rec['marker'], '_anteroposterior_axis')
        assert rec['marker']._vertical_axis == "Y"
        assert rec['marker']._anteroposterior_axis == "Z"

    def test_copy_after_strip_works(self):
        """
        Test that copy() works after strip(independent=False, inplace=True).

        This was failing before because EMGSignal.copy() tried to access
        self.muscle_name property, which requires self._name attribute that
        was lost during strip.
        """
        n_samples = 1000
        index = np.linspace(0, 10, n_samples)

        emg_data = np.random.randn(n_samples, 1)
        emg_data[:100, :] = np.nan
        emg = EMGSignal(emg_data, index, muscle_name="Biceps", side="left", unit="uV")

        rec = TimeseriesRecord(emg=emg)
        rec.strip(independent=False, inplace=True)

        # This should not raise AttributeError
        rec_copy = rec.copy()

        assert rec_copy['emg'].muscle_name == "Biceps"
        assert rec_copy['emg'].side == "left"

    def test_strip_with_forceplatform(self):
        """
        Test strip with nested Record (ForcePlatform).

        ForcePlatform is a Record containing Timeseries objects,
        so it needs recursive handling.
        """
        n_samples = 1000
        index = np.linspace(0, 10, n_samples)

        fp_data = np.random.randn(n_samples, 9)
        fp_data[:100, :] = np.nan
        fp_data[-100:, :] = np.nan

        force = Signal3D(fp_data[:, 0:3], index, unit="N")
        torque = Signal3D(fp_data[:, 3:6], index, unit="Nm")
        origin = Point3D(fp_data[:, 6:9], index, unit="m")

        fp = ForcePlatform(origin=origin, force=force, torque=torque)

        # Create record with ForcePlatform
        rec = TimeseriesRecord(forceplatform=fp)

        # Strip should work with nested Record
        rec.strip(independent=False, inplace=True)

        # Should be able to copy
        rec_copy = rec.copy()

        assert 'forceplatform' in rec_copy.keys()

    def test_strip_performance(self):
        """Verify strip(independent=False) is reasonably fast."""
        import time

        n_samples = 5000
        index = np.linspace(0, 10, n_samples)

        # Create multiple timeseries
        data1 = np.random.randn(n_samples, 3)
        data1[:100, :] = np.nan
        data1[-100:, :] = np.nan

        data2 = np.random.randn(n_samples, 1)
        data2[:150, :] = np.nan
        data2[-150:, :] = np.nan

        data3 = np.random.randn(n_samples, 3)
        data3[:120, :] = np.nan
        data3[-120:, :] = np.nan

        rec = TimeseriesRecord(
            marker=Point3D(data1, index, unit="m"),
            emg=EMGSignal(data2, index, muscle_name="Test", side="left", unit="uV"),
            force=Signal3D(data3, index, unit="N"),
        )

        start = time.perf_counter()
        rec.strip(independent=False, inplace=False)
        elapsed = time.perf_counter() - start

        # Should complete in under 50ms (was ~5s before optimization)
        assert elapsed < 0.05, f"strip() too slow: {elapsed*1000:.0f}ms"

    def test_strip_independent_true_still_works(self):
        """Verify independent=True mode still works correctly."""
        n_samples = 1000
        index = np.linspace(0, 10, n_samples)

        # Data with different NaN patterns
        data1 = np.random.randn(n_samples, 1)
        data1[:100, :] = np.nan  # only leading

        data2 = np.random.randn(n_samples, 1)
        data2[-200:, :] = np.nan  # only trailing

        emg1 = EMGSignal(data1, index, muscle_name="A", side="left", unit="uV")
        emg2 = EMGSignal(data2, index, muscle_name="B", side="right", unit="uV")

        rec = TimeseriesRecord(emg1=emg1, emg2=emg2)
        rec.strip(independent=True, inplace=True)

        # With independent=True, each should have different lengths
        assert len(rec['emg1'].index) != len(rec['emg2'].index)

        # Attributes should still be preserved
        assert rec['emg1'].muscle_name == "A"
        assert rec['emg2'].muscle_name == "B"

    def test_strip_shared_timeframe(self):
        """
        Verify strip(independent=False) creates shared timeframe.

        All elements should have the same index range after stripping.
        """
        n_samples = 1000
        index = np.arange(n_samples, dtype=float)

        # sig1 has values only in [100, 300]
        data1 = np.full((n_samples, 1), np.nan)
        data1[100:301, :] = np.random.randn(201, 1)

        # sig2 has values only in [200, 400]
        data2 = np.full((n_samples, 1), np.nan)
        data2[200:401, :] = np.random.randn(201, 1)

        emg1 = EMGSignal(data1, index, muscle_name="A", side="left", unit="uV")
        emg2 = EMGSignal(data2, index, muscle_name="B", side="right", unit="uV")

        rec = TimeseriesRecord(emg1=emg1, emg2=emg2)
        rec.strip(independent=False, inplace=True)

        # Both should now have range [100, 400] (union of valid ranges)
        assert np.array_equal(rec['emg1'].index, rec['emg2'].index)
        assert rec['emg1'].index[0] == 100.0
        assert rec['emg1'].index[-1] == 400.0
