"""
Test suite for EMGSignal copy behavior and muscle_name attribute.
"""

import numpy as np
import pytest

from labanalysis.timeseries import EMGSignal


class TestEMGSignalCopy:
    """Test EMGSignal copy behavior."""

    def test_basic_copy(self):
        """Test basic copy operation."""
        data = np.random.randn(100)
        index = np.linspace(0, 1, 100)
        muscle_name = "biceps_brachii"
        side = "left"

        emg = EMGSignal(data, index, muscle_name, side)
        emg_copy = emg.copy()

        assert emg_copy.muscle_name == muscle_name
        assert emg_copy.side == side
        np.testing.assert_array_equal(emg_copy.to_numpy(), emg.to_numpy())
        np.testing.assert_array_equal(emg_copy.index, emg.index)

    def test_copy_preserves_muscle_name(self):
        """Test that copy preserves muscle_name attribute."""
        data = np.random.randn(50)
        index = np.linspace(0, 0.5, 50)
        muscle_name = "gastrocnemius_medialis"
        side = "right"

        emg = EMGSignal(data, index, muscle_name, side)

        # Verify original has muscle_name
        assert hasattr(emg, '_name')
        assert emg.muscle_name == muscle_name

        # Copy and verify
        emg_copy = emg.copy()
        assert hasattr(emg_copy, '_name')
        assert emg_copy.muscle_name == muscle_name

    def test_copy_preserves_side(self):
        """Test that copy preserves side attribute."""
        data = np.random.randn(50)
        index = np.linspace(0, 0.5, 50)
        muscle_name = "tibialis_anterior"

        for side in ["left", "right", "bilateral"]:
            emg = EMGSignal(data, index, muscle_name, side)
            emg_copy = emg.copy()
            assert emg_copy.side == side

    def test_copy_independence(self):
        """Test that copy creates independent object."""
        data = np.random.randn(100)
        index = np.linspace(0, 1, 100)
        muscle_name = "vastus_lateralis"
        side = "left"

        emg = EMGSignal(data, index, muscle_name, side)
        emg_copy = emg.copy()

        # Modify copy's data
        emg_copy._data[0] = 999.0

        # Original should be unchanged
        assert emg._data[0] != 999.0

    def test_sliced_copy(self):
        """Test copy of sliced EMGSignal."""
        data = np.random.randn(100)
        index = np.linspace(0, 1, 100)
        muscle_name = "soleus"
        side = "right"

        emg = EMGSignal(data, index, muscle_name, side)
        emg_sliced = emg[0.2:0.8]
        emg_copy = emg_sliced.copy()

        assert emg_copy.muscle_name == muscle_name
        assert emg_copy.side == side
        assert emg_copy.shape[0] < emg.shape[0]

    def test_copy_with_different_units(self):
        """Test copy with different unit types."""
        data = np.random.randn(50)
        index = np.linspace(0, 0.5, 50)
        muscle_name = "rectus_femoris"
        side = "left"

        # Test with uV
        emg_uv = EMGSignal(data, index, muscle_name, side, unit="uV")
        copy_uv = emg_uv.copy()
        assert copy_uv.muscle_name == muscle_name
        # Accept various representations of micro: ASCII 'u', unicode micro (μ), and Latin-1 micro (µ)
        assert copy_uv.unit in ["uV", "μV", "µV"]

        # Test with mV - should be converted to uV
        emg_mv = EMGSignal(data * 1000, index, muscle_name, side, unit="mV")
        copy_mv = emg_mv.copy()
        assert copy_mv.muscle_name == muscle_name
        assert copy_mv.unit in ["uV", "μV", "µV"]

        # Test with percentage
        emg_pct = EMGSignal(data, index, muscle_name, side, unit="%")
        copy_pct = emg_pct.copy()
        assert copy_pct.muscle_name == muscle_name
        assert copy_pct.unit == "%"

    def test_attribute_consistency(self):
        """Test that _name and _muscle_name are consistent."""
        data = np.random.randn(50)
        index = np.linspace(0, 0.5, 50)
        muscle_name = "gluteus_maximus"
        side = "bilateral"

        emg = EMGSignal(data, index, muscle_name, side)

        # Check that internal attribute exists
        assert hasattr(emg, '_name')

        # Check that property returns the same value
        assert emg.muscle_name == muscle_name

        # Copy and verify consistency
        emg_copy = emg.copy()
        assert hasattr(emg_copy, '_name')
        assert emg_copy.muscle_name == muscle_name


class TestEMGSignalInRunningExercise:
    """Test EMGSignal behavior when used in RunningExercise copy operations."""

    def test_emg_in_dict_copy(self):
        """Test EMGSignal when copied as part of a dictionary."""
        data = np.random.randn(100)
        index = np.linspace(0, 1, 100)
        muscle_name = "biceps_femoris"
        side = "left"

        emg = EMGSignal(data, index, muscle_name, side)

        # Simulate what happens in RunningExercise._get_cycle
        signals_dict = {"emg_bf": emg}
        copied_dict = {k: v.copy() for k, v in signals_dict.items()}

        assert "emg_bf" in copied_dict
        copied_emg = copied_dict["emg_bf"]
        assert isinstance(copied_emg, EMGSignal)
        assert copied_emg.muscle_name == muscle_name
        assert copied_emg.side == side

    def test_emg_time_slicing_and_copy(self):
        """Test EMGSignal with time slicing (as in RunningExercise._get_cycle)."""
        data = np.random.randn(1000)
        index = np.linspace(0, 10, 1000)
        muscle_name = "gastrocnemius_lateralis"
        side = "right"

        emg = EMGSignal(data, index, muscle_name, side)

        # Simulate what happens in RunningExercise._get_cycle: slice then copy
        start_time = 2.0
        stop_time = 4.0
        emg_sliced = emg[start_time:stop_time].copy()

        assert isinstance(emg_sliced, EMGSignal)
        assert emg_sliced.muscle_name == muscle_name
        assert emg_sliced.side == side
        assert emg_sliced.index[0] >= start_time
        assert emg_sliced.index[-1] <= stop_time

    def test_multiple_emg_signals_copy(self):
        """Test multiple EMGSignal instances in a dict copy."""
        index = np.linspace(0, 5, 500)

        muscles = {
            "emg_quad": EMGSignal(np.random.randn(500), index, "quadriceps", "left"),
            "emg_ham": EMGSignal(np.random.randn(500), index, "hamstrings", "right"),
            "emg_gast": EMGSignal(np.random.randn(500), index, "gastrocnemius", "left"),
        }

        # Copy all signals
        copied_muscles = {k: v.copy() for k, v in muscles.items()}

        # Verify all copies
        assert copied_muscles["emg_quad"].muscle_name == "quadriceps"
        assert copied_muscles["emg_quad"].side == "left"

        assert copied_muscles["emg_ham"].muscle_name == "hamstrings"
        assert copied_muscles["emg_ham"].side == "right"

        assert copied_muscles["emg_gast"].muscle_name == "gastrocnemius"
        assert copied_muscles["emg_gast"].side == "left"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
