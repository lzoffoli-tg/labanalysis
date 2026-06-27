# Signal Processing

Comprehensive guide to signal processing tools in labanalysis for filtering, peak detection, derivatives, frequency analysis, and transformations.

## Overview

labanalysis provides 30+ signal processing functions optimized for biomechanical signals including force, EMG, acceleration, and position data.

## Quick Reference

| Operation | Function | Use Case |
|-----------|----------|----------|
| **[Filtering](filtering.md)** | `butterworth_filt()`, `fir_filt()` | Remove noise |
| **[Peak Detection](peak-detection.md)** | `find_peaks()` | Find local maxima/minima |
| **[Derivatives](derivatives.md)** | `winter_derivative1()`, `winter_derivative2()` | Calculate velocity, acceleration |
| **[Missing Data](missing-data.md)** | `fillna()`, `cubicspline_interp()` | Handle gaps |
| **[Frequency Analysis](frequency-analysis.md)** | `psd()`, `residual_analysis()` | Spectral content |
| **[Transformations](transformations.md)** | `to_reference_frame()`, `gram_schmidt()` | Coordinate systems |

## Quick Start

### Filter a Signal

```python
import labanalysis as laban
import numpy as np

# Load or create signal
signal = laban.Signal1D.from_tdf("data.tdf", column="Fz")

# Apply low-pass Butterworth filter
filtered = laban.butterworth_filt(
    signal=signal.data,
    freq=signal.sampling_frequency,
    cut=10,           # 10 Hz cutoff
    order=4,
    filt_type='low'
)

# Create filtered signal object
signal_filtered = laban.Signal1D(
    data=filtered,
    index=signal.index,
    label=f"{signal.label}_filtered",
    unit=signal.unit
)
```

### Find Peaks

```python
# Find peaks in signal
peaks = laban.find_peaks(
    signal=filtered,
    height=500,       # Minimum peak height
    distance=100,     # Minimum samples between peaks
    prominence=50     # Minimum prominence
)

print(f"Found {len(peaks['peak_heights'])} peaks")
print(f"Peak indices: {peaks['peaks']}")
print(f"Peak heights: {peaks['peak_heights']}")
```

### Calculate Derivative

```python
# Calculate velocity (1st derivative) using Winter 2009 method
velocity = laban.winter_derivative1(
    signal=signal.data,
    freq=signal.sampling_frequency
)

# Calculate acceleration (2nd derivative)
acceleration = laban.winter_derivative2(
    signal=signal.data,
    freq=signal.sampling_frequency
)
```

## Common Workflows

### Complete Processing Pipeline

```python
import labanalysis as laban

# 1. Load signal
signal = laban.Signal1D.from_tdf("data.tdf", column="Fz")

# 2. Remove leading/trailing NaNs
signal = signal.strip()

# 3. Fill missing data
signal = signal.fillna(method='spline')

# 4. Filter
filtered_data = laban.butterworth_filt(
    signal=signal.data,
    freq=signal.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# 5. Find peaks
peaks = laban.find_peaks(filtered_data, height=500, distance=100)

# 6. Calculate derivative
velocity = laban.winter_derivative1(filtered_data, freq=signal.sampling_frequency)

print(f"Processing complete: {len(peaks['peaks'])} peaks detected")
```

### EMG Signal Processing

```python
# Load EMG signal
emg = laban.EMGSignal.from_tdf("data.tdf", column="Biceps")

# 1. High-pass filter (remove baseline)
hp_filtered = laban.butterworth_filt(
    signal=emg.data,
    freq=emg.sampling_frequency,
    cut=20,
    order=4,
    filt_type='high'
)

# 2. Full-wave rectification
rectified = np.abs(hp_filtered)

# 3. Low-pass filter (linear envelope)
envelope = laban.butterworth_filt(
    signal=rectified,
    freq=emg.sampling_frequency,
    cut=6,
    order=4,
    filt_type='low'
)

# 4. RMS filter (alternative envelope)
rms = laban.rms_filt(signal=hp_filtered, window_size=100)
```

### Force Platform Data Processing

```python
# Load force platform
fp = record['FP1']
fz = fp.force['Fz']

# 1. Filter vertical force
fz_filtered = laban.butterworth_filt(
    signal=fz.data,
    freq=fz.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# 2. Detect contact phases (force > threshold)
threshold = 20  # N
contacts = laban.crossings(fz_filtered - threshold, direction='both')

# 3. Calculate rate of force development
force_rate = laban.winter_derivative1(fz_filtered, freq=fz.sampling_frequency)
```

## Available Functions

### Filtering Functions

- **`butterworth_filt()`** - Butterworth IIR filter (low, high, band, stop)
- **`fir_filt()`** - FIR filter design
- **`mean_filt()`** - Moving average filter
- **`median_filt()`** - Median filter (removes spikes)
- **`rms_filt()`** - RMS filter (for EMG envelope)

[→ Complete filtering guide](filtering.md)

### Peak Detection

- **`find_peaks()`** - Find local maxima with height, distance, prominence criteria

[→ Complete peak detection guide](peak-detection.md)

### Derivatives

- **`winter_derivative1()`** - 1st derivative (Winter 2009 method)
- **`winter_derivative2()`** - 2nd derivative (Winter 2009 method)

[→ Complete derivatives guide](derivatives.md)

### Interpolation & Missing Data

- **`cubicspline_interp()`** - Cubic spline interpolation
- **`fillna()`** - Fill missing data (constant, spline, regression)

[→ Complete missing data guide](missing-data.md)

### Frequency Analysis

- **`psd()`** - Power spectral density
- **`residual_analysis()`** - Optimal cutoff frequency determination
- **`xcorr()`** - Cross-correlation

[→ Complete frequency analysis guide](frequency-analysis.md)

### Transformations

- **`to_reference_frame()`** - Transform to new coordinate system
- **`gram_schmidt()`** - Orthogonalize vectors
- **`tkeo()`** - Teager-Kaiser energy operator

[→ Complete transformations guide](transformations.md)

### Utilities

- **`crossings()`** - Zero-crossing detection
- **`normalize()`** - Signal normalization
- **`threshold()`** - Threshold detection

## Best Practices

### Choosing Filter Parameters

**Sampling Frequency Considerations:**
- Force plates (1000 Hz): cutoff 10-15 Hz
- Motion capture (100-200 Hz): cutoff 6-12 Hz
- EMG (2000 Hz): high-pass 20 Hz, low-pass 450 Hz

**Filter Order:**
- 2nd order: gentle roll-off, less phase distortion
- 4th order: standard choice (good balance)
- 6th+ order: sharp roll-off, more phase distortion

### Avoiding Common Pitfalls

**1. Filtering Before Differentiation:**
```python
# Good: Filter first
filtered = laban.butterworth_filt(signal, freq=1000, cut=10, order=4)
velocity = laban.winter_derivative1(filtered, freq=1000)

# Bad: Differentiate noisy signal
velocity = laban.winter_derivative1(signal, freq=1000)  # Amplifies noise!
```

**2. Appropriate Cutoff Frequency:**
```python
# Use residual analysis to find optimal cutoff
cutoff = laban.residual_analysis(signal, freq=1000)
print(f"Suggested cutoff: {cutoff:.1f} Hz")
```

**3. Handle Missing Data Before Processing:**
```python
# Good: Fill gaps first
signal = signal.fillna(method='spline')
filtered = laban.butterworth_filt(signal.data, freq=1000, cut=10)

# Bad: Filter with NaNs (produces more NaNs)
filtered = laban.butterworth_filt(signal_with_gaps.data, freq=1000, cut=10)
```

## Topic Guides

Detailed guides for each category:

1. **[Filtering](filtering.md)** - Butterworth, FIR, moving average, median, RMS
2. **[Peak Detection](peak-detection.md)** - Find peaks with various criteria
3. **[Derivatives](derivatives.md)** - Winter 2009 methods for biomechanics
4. **[Missing Data](missing-data.md)** - Interpolation strategies
5. **[Frequency Analysis](frequency-analysis.md)** - PSD, residual analysis
6. **[Transformations](transformations.md)** - Coordinate systems, rotations

## See Also

- **[API Reference: Signal Processing](../../api/signalprocessing.md)** - Complete function reference
- **[Biomechanics Guide](../biomechanics/README.md)** - Apply to biomechanical data
- **[Examples](../../examples/basic/filter-signal.py)** - Signal processing examples

---

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
