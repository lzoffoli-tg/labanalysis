"""
Test suite for Timeseries._copy_view_attributes extensibility.

This test demonstrates that the hook-based system allows
subclasses to easily preserve custom attributes during slicing.

The _copy_view_attributes() hook method allows subclasses to specify
which attributes should be preserved when creating views/slices.

Example Usage:
    class CustomSignal(Signal1D):
        def __init__(self, data, index, unit, sensor_id):
            super().__init__(data, index, unit)
            self._sensor_id = sensor_id

        def _copy_view_attributes(self, view_obj):
            super()._copy_view_attributes(view_obj)
            if hasattr(self, '_sensor_id'):
                view_obj._sensor_id = self._sensor_id

This ensures that custom attributes survive slicing operations like:
    sliced = signal[2.0:5.0]
    result = signal.copy()[start:stop]
"""

import numpy as np
import pytest

from labanalysis.timeseries import Signal1D, Signal3D, EMGSignal, Point3D


class CustomSignal1D(Signal1D):
    """Example custom signal with extra attributes."""

    def __init__(self, data, index, unit, sensor_id=None, calibration_factor=1.0):
        super().__init__(data, index, unit)
        self._sensor_id = sensor_id
        self._calibration_factor = calibration_factor

    @property
    def sensor_id(self):
        return self._sensor_id

    @property
    def calibration_factor(self):
        return self._calibration_factor

    def copy(self):
        result = CustomSignal1D(
            self._data.copy(),
            self.index.copy(),
            self.unit,
            self._sensor_id,
            self._calibration_factor,
        )
        return result

    def _copy_view_attributes(self, view_obj):
        """Override to preserve custom attributes."""
        super()._copy_view_attributes(view_obj)
        if hasattr(self, '_sensor_id'):
            view_obj._sensor_id = self._sensor_id
        if hasattr(self, '_calibration_factor'):
            view_obj._calibration_factor = self._calibration_factor


class TestViewAttributeExtensibility:
    """Test the extensibility of _copy_view_attributes."""

    def test_custom_signal_preserves_attributes_on_slice(self):
        """Test that custom signal preserves attributes during slicing."""
        data = np.random.randn(100)
        time = np.linspace(0, 10, 100)

        signal = CustomSignal1D(
            data, time, "V",
            sensor_id="SENSOR_001",
            calibration_factor=2.5
        )

        # Slice the signal
        sliced = signal[2.0:5.0]

        # Verify attributes are preserved
        assert isinstance(sliced, CustomSignal1D)
        assert sliced.sensor_id == "SENSOR_001"
        assert sliced.calibration_factor == 2.5

    def test_custom_signal_copy_then_slice(self):
        """Test copy then slice pattern."""
        data = np.random.randn(100)
        time = np.linspace(0, 10, 100)

        signal = CustomSignal1D(
            data, time, "A",
            sensor_id="SENSOR_042",
            calibration_factor=1.5
        )

        # Copy then slice (RunningExercise pattern)
        result = signal.copy()[3.0:7.0]

        assert isinstance(result, CustomSignal1D)
        assert result.sensor_id == "SENSOR_042"
        assert result.calibration_factor == 1.5

    def test_emgsignal_preserves_attributes_via_hook(self):
        """Test that EMGSignal uses the hook system correctly."""
        data = np.random.randn(100) * 100
        time = np.linspace(0, 1, 100)

        emg = EMGSignal(data, time, "biceps_brachii", "left", "uV")

        # Slice
        emg_sliced = emg[0.2:0.8]

        # Verify EMG-specific attributes
        assert emg_sliced.muscle_name == "biceps_brachii"
        assert emg_sliced.side == "left"

    def test_signal3d_preserves_axes_via_hook(self):
        """Test that Signal3D preserves axes through hook system."""
        data = np.random.randn(100, 3)
        time = np.linspace(0, 10, 100)

        signal = Signal3D(
            data, time, "m/s",
            columns=["X", "Y", "Z"],
            vertical_axis="Z",
            anteroposterior_axis="X"
        )

        # Slice
        sliced = signal[2.0:5.0]

        # Verify axes are preserved
        assert sliced.vertical_axis == "Z"
        assert sliced.anteroposterior_axis == "X"
        assert sliced.lateral_axis == "Y"

    def test_point3d_inherits_hook_system(self):
        """Test that Point3D inherits the hook system from Signal3D."""
        data = np.random.randn(100, 3)
        time = np.linspace(0, 10, 100)

        point = Point3D(
            data, time, "mm",
            columns=["X", "Y", "Z"],
            vertical_axis="Y",
            anteroposterior_axis="Z"
        )

        # Slice
        sliced = point[3.0:7.0]

        # Verify axes and unit conversion
        assert sliced.vertical_axis == "Y"
        assert sliced.anteroposterior_axis == "Z"
        assert sliced.unit == "m"  # Point3D converts to meters

    def test_multiple_inheritance_levels(self):
        """Test that the hook system works through multiple inheritance levels."""

        class ExtendedEMGSignal(EMGSignal):
            """Example of extending EMGSignal with additional attributes."""

            def __init__(self, data, index, muscle_name, side, unit="uV", electrode_type=None):
                super().__init__(data, index, muscle_name, side, unit)
                self._electrode_type = electrode_type

            @property
            def electrode_type(self):
                return self._electrode_type

            def copy(self):
                return ExtendedEMGSignal(
                    self._data.copy(),
                    self.index.copy(),
                    self.muscle_name,
                    self.side,
                    self.unit,
                    self._electrode_type,
                )

            def _copy_view_attributes(self, view_obj):
                """Override to add electrode_type preservation."""
                super()._copy_view_attributes(view_obj)
                if hasattr(self, '_electrode_type'):
                    view_obj._electrode_type = self._electrode_type

        data = np.random.randn(100) * 100
        time = np.linspace(0, 1, 100)

        emg = ExtendedEMGSignal(
            data, time,
            "gastrocnemius",
            "right",
            "uV",
            electrode_type="surface"
        )

        # Slice
        sliced = emg[0.3:0.7]

        # Verify all levels of attributes are preserved
        assert sliced.muscle_name == "gastrocnemius"  # EMGSignal level
        assert sliced.side == "right"  # EMGSignal level
        assert sliced.electrode_type == "surface"  # ExtendedEMGSignal level


class TestHookSystemDocumentation:
    """Test that the hook system is properly documented and works as advertised."""

    def test_hook_method_exists(self):
        """Verify _copy_view_attributes exists in base class."""
        from labanalysis.timeseries._base import Timeseries

        assert hasattr(Timeseries, '_copy_view_attributes')
        assert callable(getattr(Timeseries, '_copy_view_attributes'))

    def test_hook_method_signature(self):
        """Verify hook method has correct signature."""
        from labanalysis.timeseries._base import Timeseries
        import inspect

        sig = inspect.signature(Timeseries._copy_view_attributes)
        params = list(sig.parameters.keys())

        assert params == ['self', 'view_obj']

    def test_hook_documentation_exists(self):
        """Verify hook method is documented."""
        from labanalysis.timeseries._base import Timeseries

        assert Timeseries._copy_view_attributes.__doc__ is not None
        assert "subclass" in Timeseries._copy_view_attributes.__doc__.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
