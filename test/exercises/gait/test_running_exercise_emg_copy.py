"""
Test suite for RunningExercise copy behavior with EMGSignal.

This test focuses on the EMGSignal copy behavior in the context of
RunningExercise._get_cycle, which slices and copies signals.
"""

import numpy as np
import pytest

from labanalysis.timeseries import EMGSignal
from labanalysis.records import TimeseriesRecord


class TestEMGSignalCopyInExerciseContext:
    """Test EMGSignal copy behavior as it would occur in RunningExercise."""

    def test_emg_slice_and_copy_preserves_attributes(self):
        """Test the exact pattern used in RunningExercise._get_cycle."""
        # Create an EMG signal
        n_samples = 1000
        time = np.linspace(0, 10, n_samples)

        emg = EMGSignal(
            data=np.random.randn(n_samples) * 100 + 200,
            index=time,
            muscle_name="tibialis_anterior",
            side="left",
            unit="uV"
        )

        # Simulate what happens in RunningExercise._get_cycle:
        # args.update(**{i: v.copy()[start:stop] for i, v in self.items()})
        start = 2.0
        stop = 4.0

        # This is the exact pattern that was failing
        emg_sliced_and_copied = emg.copy()[start:stop]

        # Verify attributes are preserved
        assert isinstance(emg_sliced_and_copied, EMGSignal)
        assert emg_sliced_and_copied.muscle_name == "tibialis_anterior"
        assert emg_sliced_and_copied.side == "left"
        assert emg_sliced_and_copied.index[0] >= start
        assert emg_sliced_and_copied.index[-1] <= stop

    def test_emg_in_timeseriesrecord_items_pattern(self):
        """Test EMG signals in dict iteration pattern used by RunningExercise."""
        # Create a TimeseriesRecord with EMG signals
        time = np.linspace(0, 5, 500)

        record = TimeseriesRecord(
            emg_quad=EMGSignal(
                data=np.random.randn(500) * 100,
                index=time,
                muscle_name="quadriceps",
                side="left",
                unit="uV"
            ),
            emg_ham=EMGSignal(
                data=np.random.randn(500) * 150,
                index=time,
                muscle_name="hamstrings",
                side="right",
                unit="uV"
            ),
        )

        # Simulate the pattern in _get_cycle
        start = 1.0
        stop = 2.0
        sliced_data = {i: v.copy()[start:stop] for i, v in record.items()}

        # Verify both EMG signals preserve attributes
        assert "emg_quad" in sliced_data
        assert "emg_ham" in sliced_data

        emg_quad = sliced_data["emg_quad"]
        assert isinstance(emg_quad, EMGSignal)
        assert emg_quad.muscle_name == "quadriceps"
        assert emg_quad.side == "left"

        emg_ham = sliced_data["emg_ham"]
        assert isinstance(emg_ham, EMGSignal)
        assert emg_ham.muscle_name == "hamstrings"
        assert emg_ham.side == "right"

    def test_multiple_copy_slice_operations(self):
        """Test multiple sequential copy and slice operations."""
        time = np.linspace(0, 10, 1000)
        emg = EMGSignal(
            data=np.random.randn(1000) * 100,
            index=time,
            muscle_name="gastrocnemius_medialis",
            side="right",
            unit="uV"
        )

        # First copy and slice
        emg1 = emg.copy()[2.0:8.0]
        assert emg1.muscle_name == "gastrocnemius_medialis"
        assert emg1.side == "right"

        # Second copy and slice
        emg2 = emg1.copy()[4.0:6.0]
        assert emg2.muscle_name == "gastrocnemius_medialis"
        assert emg2.side == "right"

        # Direct double operation
        emg3 = emg.copy()[2.0:8.0].copy()[4.0:6.0]
        assert emg3.muscle_name == "gastrocnemius_medialis"
        assert emg3.side == "right"

    def test_emg_copy_then_slice_vs_slice_then_copy(self):
        """Test both operation orders preserve attributes."""
        time = np.linspace(0, 5, 500)
        emg = EMGSignal(
            data=np.random.randn(500) * 100,
            index=time,
            muscle_name="soleus",
            side="bilateral",
            unit="uV"
        )

        # Pattern A: copy then slice (used in _get_cycle)
        emg_a = emg.copy()[1.0:3.0]

        # Pattern B: slice then copy
        emg_b = emg[1.0:3.0].copy()

        # Both should preserve attributes
        assert emg_a.muscle_name == "soleus"
        assert emg_a.side == "bilateral"

        assert emg_b.muscle_name == "soleus"
        assert emg_b.side == "bilateral"

        # Data should be equivalent
        np.testing.assert_array_almost_equal(
            emg_a.to_numpy(),
            emg_b.to_numpy()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
