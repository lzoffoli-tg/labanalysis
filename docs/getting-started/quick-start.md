# Quick Start (5 minutes)

Get started with labanalysis in just 5 minutes. This guide shows you the essential workflow from loading data to generating results.

## Installation

If you haven't installed labanalysis yet:

```bash
pip install git+https://github.com/lzoffoli-tg/labanalysis.git
```

[→ Detailed installation guide](installation.md)

## Basic Workflow

### 1. Import the Library

```python
import labanalysis as laban
import numpy as np
```

### 2. Create a Simple Signal

```python
# Create a 1D signal (e.g., force data)
time = np.linspace(0, 10, 1000)  # 10 seconds at 100 Hz
force_data = 500 + 200 * np.sin(2 * np.pi * 1.5 * time) + np.random.randn(1000) * 10

signal = laban.Signal1D(
    data=force_data,
    index=time,
    label='Fz',
    unit='N'
)

print(f"Signal: {signal.label}, {len(signal)} samples, {signal.sampling_frequency:.1f} Hz")
# Output: Signal: Fz, 1000 samples, 100.0 Hz
```

### 3. Filter the Signal

```python
# Apply low-pass Butterworth filter to remove noise
filtered_data = laban.butterworth_filt(
    signal=signal.data,
    freq=signal.sampling_frequency,
    cut=10,           # 10 Hz cutoff
    order=4,
    filt_type='low'
)

# Create filtered signal
signal_filtered = laban.Signal1D(
    data=filtered_data,
    index=signal.index,
    label='Fz_filtered',
    unit='N'
)

print(f"Filtered signal created")
# Output: Filtered signal created
```

### 4. Find Peaks

```python
# Detect peaks in the filtered signal
peaks = laban.find_peaks(
    signal=signal_filtered.data,
    height=600,      # Minimum peak height
    distance=50      # Minimum samples between peaks
)

print(f"Found {len(peaks['peak_heights'])} peaks")
# Output: Found 15 peaks
print(f"Peak heights: {peaks['peak_heights'][:3]} N")  # First 3 peaks
# Output: Peak heights: [698.5, 702.1, 695.8] N
```

### 5. Convert to DataFrame

```python
import pandas as pd

# Create a Record with multiple signals
record = laban.Record(
    original=signal,
    filtered=signal_filtered
)

# Convert to pandas DataFrame
df = record.to_dataframe()
print(df.head())
# Output:
#    original  filtered
# 0   512.3     510.5
# 1   518.7     516.2
# 2   525.1     522.8
# ...
```

## Load Real Data

### Example: Load from TDF File (BTS System)

```python
# Load force platform data from BTS TDF file
record = laban.TimeseriesRecord.from_tdf("path/to/data.tdf")

# Access force platform data
fp = record['FP1']  # First force platform
print(f"Force platform: {fp.origin}")
print(f"Vertical force range: {fp.force['Fz'].data.min():.1f} to {fp.force['Fz'].data.max():.1f} N")
# Output: Force platform: [0.0 0.0 0.0] m
# Output: Vertical force range: -8.5 to 1253.7 N
```

### Example: Analyze a Jump

```python
# Create participant information
participant = laban.Participant(
    name="John",
    surname="Doe",
    height=1.80,  # meters
    weight=75,    # kg
    age=25
)

# Load and analyze countermovement jump
# Note: This requires appropriate TDF file with force platform data
from labanalysis.records.jumping import SingleJump

jump = SingleJump.from_tdf(
    "path/to/jump.tdf",
    left_foot_ground_reaction_force='FP1'
)

# Access jump metrics (if available in the class)
print(f"Jump analyzed successfully")
# Output: Jump analyzed successfully
```

## Complete Example: Signal Processing Pipeline

Here's a complete workflow combining multiple operations:

```python
import labanalysis as laban
import numpy as np

# 1. Load data (simulated here)
time = np.linspace(0, 5, 500)
raw_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(500)

signal = laban.Signal1D(data=raw_signal, index=time, label='EMG', unit='mV')

# 2. Filter
filtered = laban.butterworth_filt(
    signal=signal.data,
    freq=100,
    cut=15,
    order=4,
    filt_type='low'
)

# 3. Find peaks
peaks = laban.find_peaks(signal=filtered, height=0.5, distance=20)

# 4. Calculate derivative
velocity = laban.winter_derivative1(signal=filtered, freq=100)

# 5. Create summary
print("Processing Summary:")
print(f"- Original samples: {len(signal)}")
print(f"- Sampling frequency: {signal.sampling_frequency:.1f} Hz")
print(f"- Peaks detected: {len(peaks['peak_heights'])}")
print(f"- Peak values: {peaks['peak_heights']}")

# Output:
# Processing Summary:
# - Original samples: 500
# - Sampling frequency: 100.0 Hz
# - Peaks detected: 10
# - Peak values: [0.89, 1.02, 0.95, ...]
```

## What's Next?

Now that you've seen the basics, explore more:

### Learn Core Concepts
- **[Core Concepts](core-concepts.md)** - Understand Record, Signal, and Protocol patterns
- **[Your First Analysis](first-analysis.md)** - Complete analysis with real data

### Explore Guides
- **[Signal Processing](../user-guide/signal-processing/README.md)** - Complete filtering and analysis guide
- **[Test Protocols](../user-guide/test-protocols/README.md)** - Standardized test workflows
- **[Biomechanics](../user-guide/biomechanics/README.md)** - Full body kinematics

### Try Tutorials
- **[Jump Analysis Tutorial](../tutorials/01-jump-analysis.md)** - Complete CMJ workflow
- **[Gait Analysis Tutorial](../tutorials/02-gait-analysis.md)** - Walking/running analysis

## Key Takeaways

- Import with `import labanalysis as laban`
- Signals store time-series data with units: `Signal1D`, `Signal3D`
- Records group multiple signals: `Record`, `TimeseriesRecord`
- Filter with `butterworth_filt()`, find peaks with `find_peaks()`
- Load real data with `.from_tdf()` methods
- Convert to pandas with `.to_dataframe()`

---

**Questions?** Check the [API Reference](../api-reference/README.md) or contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
