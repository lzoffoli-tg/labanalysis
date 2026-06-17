# Custom Signals and Signal Extensions

Advanced guide for extending Signal1D, Signal3D, and creating custom signal types for specialized biomechanical data.

## Overview

The `labanalysis` signal classes (`Signal1D`, `Signal3D`, `Point3D`, `EMGSignal`) provide a robust foundation for time-series data. This guide shows how to:

- **Subclass existing signals** for domain-specific functionality
- **Add custom properties** and computed metrics
- **Implement signal-specific processing** methods
- **Integrate with existing workflows** seamlessly

**When to create custom signals**:
- Specialized sensor data (IMU, pressure insoles, wearables)
- Domain-specific metrics (sport-specific angles, custom force metrics)
- Automated processing pipelines for repeated analyses

## Quick Reference

```python
from labanalysis.records.timeseries import Signal1D
import numpy as np

class CustomForceSignal(Signal1D):
    """Custom force signal with domain-specific methods."""
    
    @property
    def peak_force(self):
        """Calculate peak force magnitude."""
        return np.max(np.abs(self.to_numpy()))
    
    @property
    def impulse(self):
        """Calculate impulse (area under curve)."""
        dt = np.mean(np.diff(self.index))
        return np.sum(self.to_numpy()) * dt
```

## Extending Signal1D

### Basic Subclass Pattern

```python
from labanalysis.records.timeseries import Signal1D
import numpy as np

class PressureSignal(Signal1D):
    """
    Pressure signal with additional metrics.
    
    Extends Signal1D to add pressure-specific calculations
    for insole pressure sensors.
    """
    
    def __init__(self, data, index, unit='kPa', **kwargs):
        # Ensure pressure unit
        if unit not in ['Pa', 'kPa', 'MPa', 'psi']:
            raise ValueError(f"Invalid pressure unit: {unit}")
        
        super().__init__(
            data=data,
            index=index,
            columns=['pressure'],
            unit=unit,
            **kwargs
        )
    
    @property
    def peak_pressure(self):
        """Peak pressure value."""
        return np.max(self.to_numpy())
    
    @property
    def mean_pressure(self):
        """Mean pressure over time."""
        return np.mean(self.to_numpy())
    
    @property
    def pressure_time_integral(self):
        """Pressure-time integral (PTI)."""
        dt = np.mean(np.diff(self.index))
        return np.sum(self.to_numpy()) * dt
    
    def contact_duration(self, threshold=10):
        """
        Calculate contact duration above threshold.
        
        Parameters
        ----------
        threshold : float
            Pressure threshold for contact detection (in signal units)
        
        Returns
        -------
        float
            Contact duration in seconds
        """
        above_threshold = self.to_numpy() > threshold
        contact_samples = np.sum(above_threshold)
        dt = np.mean(np.diff(self.index))
        return contact_samples * dt
```

**Usage:**

```python
import numpy as np

# Create pressure signal
time = np.linspace(0, 2, 200)
pressure = 50 * np.sin(2 * np.pi * time) + 60  # Simulated pressure

signal = PressureSignal(
    data=pressure,
    index=time,
    unit='kPa'
)

# Use custom properties
print(f"Peak pressure: {signal.peak_pressure:.1f} kPa")
print(f"PTI: {signal.pressure_time_integral:.2f} kPa·s")
print(f"Contact duration: {signal.contact_duration(threshold=30):.2f} s")
```

### Adding Computed Metrics

```python
class IMUSignal(Signal3D):
    """
    Inertial Measurement Unit (IMU) signal.
    
    Extends Signal3D for accelerometer/gyroscope data
    with orientation and magnitude calculations.
    """
    
    @property
    def magnitude(self):
        """Calculate signal magnitude (L2 norm)."""
        data = self.to_numpy()
        return Signal1D(
            data=np.sqrt(np.sum(data**2, axis=1)),
            index=self.index,
            columns=['magnitude'],
            unit=self.unit
        )
    
    @property
    def inclination_angle(self):
        """
        Calculate inclination angle from vertical (Z-axis).
        
        Returns angle in degrees.
        """
        data = self.to_numpy()
        z_component = data[:, 2]
        magnitude = np.sqrt(np.sum(data**2, axis=1))
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_theta = z_component / magnitude
            cos_theta = np.clip(cos_theta, -1, 1)
        
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.rad2deg(angle_rad)
        
        return Signal1D(
            data=angle_deg,
            index=self.index,
            columns=['inclination'],
            unit='deg'
        )
    
    def rotate(self, rotation_matrix):
        """
        Apply rotation matrix to IMU data.
        
        Parameters
        ----------
        rotation_matrix : np.ndarray
            3x3 rotation matrix
        
        Returns
        -------
        IMUSignal
            Rotated signal
        """
        data = self.to_numpy()
        rotated = data @ rotation_matrix.T
        
        return IMUSignal(
            data=rotated,
            index=self.index,
            columns=self.columns,
            unit=self.unit
        )
```

## Extending Signal3D for Specialized Markers

### Custom Marker with Auto-Computed Metrics

```python
class FootMarker(Point3D):
    """
    Foot marker with gait-specific metrics.
    
    Extends Point3D to automatically compute
    foot clearance, progression angle, etc.
    """
    
    @property
    def vertical_clearance(self):
        """Vertical clearance (max Z during swing)."""
        z = self["Z"].to_numpy()
        
        # Find swing phases (Z > threshold)
        threshold = np.percentile(z, 25)  # Bottom 25% = stance
        swing_mask = z > threshold
        
        if np.any(swing_mask):
            return Signal1D(
                data=z - threshold,
                index=self.index,
                columns=['clearance'],
                unit='m'
            )
        else:
            return None
    
    @property
    def progression_velocity(self):
        """Forward (X-axis) velocity."""
        from labanalysis.signalprocessing import derivative
        
        x_position = self["X"]
        velocity = derivative(x_position, order=1, method='winter')
        
        return velocity
    
    @property
    def step_length(self):
        """
        Estimate step length from forward displacement.
        
        Returns displacement between consecutive heel strikes.
        """
        x = self["X"].to_numpy()
        z = self["Z"].to_numpy()
        
        # Detect heel strikes (local minima in Z)
        from labanalysis.signalprocessing import find_peaks
        
        peaks_idx, _ = find_peaks(-z, threshold=None, distance=50)
        
        if len(peaks_idx) < 2:
            return None
        
        # Calculate distances between strikes
        step_lengths = np.diff(x[peaks_idx])
        
        return step_lengths
```

## Creating Completely New Signal Types

### Electromyography (EMG) Extension

The library includes `EMGSignal`, but here's how to extend it further:

```python
from labanalysis.records.timeseries import EMGSignal

class ProcessedEMG(EMGSignal):
    """
    EMG signal with additional processing methods.
    
    Adds muscle activation analysis, fatigue detection,
    and co-contraction metrics.
    """
    
    @property
    def activation_threshold(self, percentile=10):
        """
        Calculate activation threshold (baseline + noise).
        
        Uses percentile method for robust threshold.
        """
        baseline = np.percentile(self.to_numpy(), percentile)
        noise_std = np.std(self.to_numpy()[self.to_numpy() < baseline])
        
        return baseline + 3 * noise_std
    
    @property
    def activation_periods(self):
        """
        Detect activation periods above threshold.
        
        Returns list of (start_time, end_time, duration) tuples.
        """
        threshold = self.activation_threshold
        data = self.to_numpy()
        
        # Find threshold crossings
        above = data > threshold
        crossings = np.diff(above.astype(int))
        
        starts = np.where(crossings == 1)[0] + 1
        ends = np.where(crossings == -1)[0] + 1
        
        # Handle edge cases
        if len(starts) == 0 or len(ends) == 0:
            return []
        
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            ends = np.append(ends, len(data) - 1)
        
        periods = []
        for start, end in zip(starts, ends):
            start_time = self.index[start]
            end_time = self.index[end]
            duration = end_time - start_time
            periods.append((start_time, end_time, duration))
        
        return periods
    
    def median_frequency(self, window_size=0.5):
        """
        Calculate median frequency over sliding windows.
        
        Useful for fatigue detection (decreases with fatigue).
        
        Parameters
        ----------
        window_size : float
            Window size in seconds
        
        Returns
        -------
        Signal1D
            Median frequency time series
        """
        from labanalysis.signalprocessing import power_spectral_density
        
        # Implementation would use sliding window PSD
        # and find median frequency for each window
        pass  # Simplified for brevity
```

## Integration with Existing Classes

### Adding Custom Signals to Records

```python
from labanalysis.records.records import TimeseriesRecord

class CustomGaitRecord(TimeseriesRecord):
    """
    Gait record with custom foot markers.
    
    Combines standard markers with custom FootMarker
    for automatic gait metric computation.
    """
    
    def __init__(self):
        super().__init__()
        self._foot_markers = {}
    
    def add_foot_marker(self, label, marker_data):
        """
        Add foot marker with auto-computed metrics.
        
        Parameters
        ----------
        label : str
            Marker label (e.g., 'left_heel', 'right_toe')
        marker_data : Point3D or array-like
            Marker position data
        """
        if not isinstance(marker_data, FootMarker):
            # Convert to FootMarker
            marker_data = FootMarker(
                data=marker_data.to_numpy(),
                index=marker_data.index,
                columns=['X', 'Y', 'Z'],
                unit=marker_data.unit
            )
        
        self._foot_markers[label] = marker_data
        self.set(label, marker_data)
    
    @property
    def left_foot_clearance(self):
        """Get left foot clearance metric."""
        if 'left_heel' in self._foot_markers:
            return self._foot_markers['left_heel'].vertical_clearance
        return None
    
    @property
    def step_symmetry(self):
        """
        Calculate step length symmetry.
        
        Returns symmetry index (%)
        """
        left_steps = self._foot_markers['left_heel'].step_length
        right_steps = self._foot_markers['right_heel'].step_length
        
        if left_steps is None or right_steps is None:
            return None
        
        # Average step lengths
        left_avg = np.mean(left_steps)
        right_avg = np.mean(right_steps)
        
        # Symmetry index
        si = (left_avg - right_avg) / (0.5 * (left_avg + right_avg)) * 100
        
        return si
```

## Best Practices

### 1. Preserve Base Class Functionality

Always call `super().__init__()` and preserve base class methods:

```python
class MySignal(Signal1D):
    def __init__(self, data, index, custom_param=None, **kwargs):
        # Call parent constructor
        super().__init__(data=data, index=index, **kwargs)
        
        # Add custom attributes
        self.custom_param = custom_param
```

### 2. Use Type Hints

```python
from typing import Optional
import numpy as np

class TypedSignal(Signal1D):
    def process(self, threshold: float) -> Optional[np.ndarray]:
        """
        Process signal with type hints.
        
        Parameters
        ----------
        threshold : float
            Processing threshold
        
        Returns
        -------
        Optional[np.ndarray]
            Processed data or None if invalid
        ```
        if threshold < 0:
            return None
        
        return self.to_numpy() * threshold
```

### 3. Document Custom Methods

Use NumPy docstring format:

```python
def calculate_metric(self, param1, param2=10):
    """
    Calculate custom metric from signal.
    
    Parameters
    ----------
    param1 : float
        First parameter description
    param2 : int, optional
        Second parameter (default: 10)
    
    Returns
    -------
    float
        Computed metric value
    
    Notes
    -----
    Additional notes about calculation method,
    assumptions, or limitations.
    
    Examples
    --------
    >>> signal = CustomSignal(data, index)
    >>> metric = signal.calculate_metric(5.0, param2=20)
    ```
    pass
```

### 4. Handle Edge Cases

```python
@property
def safe_metric(self):
    """Metric with edge case handling."""
    data = self.to_numpy()
    
    # Check for empty data
    if len(data) == 0:
        return np.nan
    
    # Check for all-NaN data
    if np.all(np.isnan(data)):
        return np.nan
    
    # Check for invalid values
    valid_data = data[~np.isnan(data)]
    if len(valid_data) < 3:
        return np.nan
    
    # Compute metric
    return np.mean(valid_data)
```

## See Also

- [Signal Processing](../user-guide/signal-processing/filtering.md) - Standard signal operations
- [API Reference - Timeseries](../api-reference/records/timeseries.md) - Base signal classes
- [Extending Protocols](extending-protocols.md) - Custom protocol creation
- [Unit Handling](unit-handling.md) - Pint integration for custom units

---

**Extend Signal1D/Signal3D classes** to add domain-specific metrics and processing for specialized biomechanical sensors and analyses. Preserve base class functionality and use proper typing and documentation.
