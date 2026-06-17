# Extending Test Protocols

Guide for creating custom test protocols by extending `TestProtocol` and `TestResults` base classes.

## Overview

Custom protocols allow you to:
- **Define sport-specific tests** (agility drills, sport skills)
- **Automate repeated analyses** with consistent processing
- **Package domain knowledge** into reusable components
- **Integrate with existing workflows** seamlessly

**When to create custom protocols**: Repeated tests with consistent structure, sport-specific assessments, automated reporting needs.

## Quick Reference

```python
from labanalysis.protocols import TestProtocol, TestResults
import labanalysis as laban

class CustomTest(TestProtocol):
    """Custom test protocol."""
    
    @classmethod
    def from_files(cls, marker_file, force_file):
        """Load test from files."""
        body = laban.WholeBody.from_tdf_file(marker_file, labels="LABEL")
        fp = laban.ForcePlatform.from_tdf_file(force_file, fp_label="FP1")
        return cls(body=body, force_platform=fp)
    
    def process(self):
        """Process test and return results."""
        # Custom processing logic
        metrics = self._calculate_metrics()
        return CustomTestResults(metrics=metrics)
```

## Complete Example: T-Test Protocol

```python
import labanalysis as laban
from labanalysis.protocols import TestProtocol, TestResults
from labanalysis.signalprocessing import butterworth_filter, find_peaks
import numpy as np

class TTestProtocol(TestProtocol):
    """
    T-Test agility protocol.
    
    Measures change-of-direction performance using
    force platforms and motion capture.
    """
    
    def __init__(self, body, force_platform):
        """
        Initialize T-Test.
        
        Parameters
        ----------
        body : WholeBody
            Full-body kinematics
        force_platform : ForcePlatform
            Force platform data
        """
        self.body = body
        self.force_platform = force_platform
        self._results = None
    
    @classmethod
    def from_tdf_file(cls, file_path, labels="LABEL", fp_label="FP1"):
        """Load from BTS TDF file."""
        body = laban.WholeBody.from_tdf_file(file_path, labels=labels)
        fp = laban.ForcePlatform.from_tdf_file(file_path, fp_label=fp_label)
        return cls(body=body, force_platform=fp)
    
    def detect_turns(self, pelvis_threshold=0.1):
        """
        Detect direction changes from pelvis velocity.
        
        Parameters
        ----------
        pelvis_threshold : float
            Velocity threshold for turn detection (m/s)
        
        Returns
        -------
        list
            Turn time points
        """
        pelvis = self.body.pelvis_center
        
        # Calculate velocity
        from labanalysis.signalprocessing import derivative
        velocity = derivative(pelvis, order=1, method='winter')
        
        # Velocity magnitude
        vel_mag = np.sqrt(np.sum(velocity.to_numpy()**2, axis=1))
        
        # Find local minima (direction changes)
        peaks_idx, _ = find_peaks(-vel_mag, threshold=-pelvis_threshold, distance=50)
        
        return self.body.index[peaks_idx]
    
    def calculate_segment_times(self, turn_times):
        """Calculate time between direction changes."""
        if len(turn_times) < 2:
            return []
        
        return np.diff(turn_times)
    
    def process(self):
        """
        Process T-Test and extract metrics.
        
        Returns
        -------
        TTestResults
            Test results with all metrics
        """
        # Detect turns
        turn_times = self.detect_turns()
        
        # Calculate metrics
        segment_times = self.calculate_segment_times(turn_times)
        total_time = turn_times[-1] - turn_times[0] if len(turn_times) > 1 else np.nan
        
        # Peak GRF during each segment
        grf_z = butterworth_filter(
            self.force_platform["FORCE", "Z"],
            frequency=10, order=4
        )
        
        segment_peak_forces = []
        for i in range(len(turn_times) - 1):
            t_start = turn_times[i]
            t_end = turn_times[i + 1]
            segment_grf = grf_z[t_start:t_end]
            segment_peak_forces.append(abs(segment_grf.to_numpy().min()))
        
        # Create results
        self._results = TTestResults(
            total_time=total_time,
            turn_times=turn_times,
            segment_times=segment_times,
            segment_peak_forces=segment_peak_forces,
            body=self.body,
            force_platform=self.force_platform
        )
        
        return self._results


class TTestResults(TestResults):
    """Results container for T-Test."""
    
    def __init__(self, total_time, turn_times, segment_times, 
                 segment_peak_forces, body, force_platform):
        """
        Initialize results.
        
        Parameters
        ----------
        total_time : float
            Total test duration (s)
        turn_times : np.ndarray
            Time points of direction changes
        segment_times : np.ndarray
            Duration of each segment
        segment_peak_forces : list
            Peak GRF during each segment
        body : WholeBody
            Kinematics data
        force_platform : ForcePlatform
            Force data
        """
        self.total_time = total_time
        self.turn_times = turn_times
        self.segment_times = segment_times
        self.segment_peak_forces = segment_peak_forces
        self.body = body
        self.force_platform = force_platform
    
    @property
    def average_segment_time(self):
        """Average time per segment."""
        return np.mean(self.segment_times)
    
    @property
    def consistency_cv(self):
        """Coefficient of variation for segments (%)."""
        return (np.std(self.segment_times) / np.mean(self.segment_times)) * 100
    
    def to_dataframe(self):
        """Export results to DataFrame."""
        import pandas as pd
        
        return pd.DataFrame({
            'Metric': ['Total Time', 'Avg Segment Time', 'Consistency (CV%)'],
            'Value': [
                f"{self.total_time:.2f}",
                f"{self.average_segment_time:.2f}",
                f"{self.consistency_cv:.1f}"
            ],
            'Unit': ['s', 's', '%']
        })
```

**Usage:**

```python
# Load and process
test = TTestProtocol.from_tdf_file("t_test.tdf")
results = test.process()

# Access metrics
print(f"Total time: {results.total_time:.2f} s")
print(f"Consistency: {results.consistency_cv:.1f}%")

# Export
df = results.to_dataframe()
df.to_csv("t_test_results.csv", index=False)
```

## Best Practices

### Required Methods

```python
class MyProtocol(TestProtocol):
    @classmethod
    def from_files(cls, *args, **kwargs):
        """REQUIRED: Load test from file(s)."""
        pass
    
    def process(self):
        """REQUIRED: Process and return results."""
        return MyResults(...)
```

### Validation

```python
def _validate_inputs(self):
    """Validate test data before processing."""
    if len(self.body.labels) < 10:
        raise ValueError("Insufficient markers for test")
    
    if self.force_platform.index[-1] < 5.0:
        raise ValueError("Test duration too short (< 5s)")
```

### Progressive Processing

```python
def process(self, force_filter_freq=10, min_turn_velocity=0.1):
    """
    Process with configurable parameters.
    
    Parameters
    ----------
    force_filter_freq : float
        Filter cutoff for GRF (Hz)
    min_turn_velocity : float
        Velocity threshold for turns (m/s)
    """
    # Filter data
    self._preprocess(force_filter_freq)
    
    # Detect events
    self._detect_events(min_turn_velocity)
    
    # Calculate metrics
    self._calculate_metrics()
    
    return self._results
```

## See Also

- [Custom Signals](custom-signals.md) - Extending signal classes
- [Test Protocols Guide](../user-guide/test-protocols/) - Using built-in protocols
- [Tutorial - Custom Protocol](../tutorials/06-custom-protocol.md) - Complete walkthrough

---

**Create custom test protocols** by extending `TestProtocol` and `TestResults`. Implement `from_files()` and `process()` methods for consistent, reusable test analyses.
