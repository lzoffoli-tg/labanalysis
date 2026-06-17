# Peak Detection

Complete guide to detecting peaks and valleys in biomechanical signals using labanalysis.

## Overview

Peak detection is essential for identifying critical events in biomechanical data:
- **Force platforms**: Detecting steps, jumps, impacts
- **EMG signals**: Finding activation bursts
- **Marker trajectories**: Identifying movement extremes
- **Metabolic data**: Finding VO2 peaks

labanalysis provides `find_peaks()` and `find_valleys()` with flexible filtering based on height, distance, prominence, and width.

## Quick Start

```python
import labanalysis as laban
import numpy as np

# Load force signal
record = laban.TimeseriesRecord.from_tdf("jump.tdf")
fz = record['FP1'].force['Fz'].data

# Find peaks above 500 N, at least 100 samples apart
peaks = laban.find_peaks(fz, height=500, distance=100)

print(f"Found {len(peaks['peak_indices'])} peaks")
print(f"Peak locations: {peaks['peak_indices']}")
print(f"Peak values: {peaks['peak_heights']}")
```

**Output:**
```
Found 3 peaks
Peak locations: [245, 1823, 3401]
Peak values: [892.3, 1245.7, 978.4]
```

## find_peaks() Function

### Signature

```python
def find_peaks(
    signal: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    width: Optional[int] = None
) -> dict
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `signal` | ndarray | Input signal (1D array) |
| `height` | float, optional | Minimum peak height |
| `distance` | int, optional | Minimum distance between peaks (samples) |
| `prominence` | float, optional | Minimum peak prominence |
| `width` | int, optional | Minimum peak width (samples) |

### Returns

Dictionary containing:
- `'peak_indices'`: Peak locations (sample indices)
- `'peak_heights'`: Peak values at each index
- `'prominences'`: Peak prominences (height above surrounding baseline)
- `'widths'`: Peak widths at half prominence height (samples)
- `'left_bases'`: Left edge of peak base
- `'right_bases'`: Right edge of peak base

### Example: Complete Peak Information

```python
import labanalysis as laban

# Load signal
fz = record['FP1'].force['Fz'].data

# Find peaks with all properties
peaks = laban.find_peaks(
    fz,
    height=500,      # At least 500 N
    distance=100,    # At least 100 samples apart
    prominence=200,  # At least 200 N above baseline
    width=10         # At least 10 samples wide
)

# Access all properties
for i, idx in enumerate(peaks['peak_indices']):
    print(f"Peak {i+1}:")
    print(f"  Location: {idx} samples ({idx/1000:.3f} s)")
    print(f"  Height: {peaks['peak_heights'][i]:.1f} N")
    print(f"  Prominence: {peaks['prominences'][i]:.1f} N")
    print(f"  Width: {peaks['widths'][i]:.1f} samples ({peaks['widths'][i]/1000:.4f} s)")
    print(f"  Base: samples {peaks['left_bases'][i]} to {peaks['right_bases'][i]}")
```

**Output:**
```
Peak 1:
  Location: 245 samples (0.245 s)
  Height: 892.3 N
  Prominence: 654.2 N
  Width: 23.5 samples (0.0235 s)
  Base: samples 220 to 268

Peak 2:
  Location: 1823 samples (1.823 s)
  Height: 1245.7 N
  Prominence: 987.3 N
  Width: 45.2 samples (0.0452 s)
  Base: samples 1789 to 1856
```

## Parameter Selection Guide

### Height

Minimum peak value. Use when you know the absolute threshold:

```python
# Find all peaks above bodyweight
bodyweight = 75 * 9.81  # 75 kg * g
peaks = laban.find_peaks(fz, height=bodyweight)

# Find peaks above 10% of bodyweight
threshold = bodyweight * 0.10
peaks = laban.find_peaks(fz, height=threshold)
```

**When to use:**
- Force platforms: Threshold for contact detection
- EMG: Activation threshold above baseline
- Known physiological minimums

### Distance

Minimum samples between consecutive peaks. Prevents detecting noise as multiple peaks:

```python
# Find steps in gait (assuming ~1 Hz cadence at 1000 Hz sampling)
min_step_interval = int(0.8 * 1000)  # 0.8 seconds minimum
peaks = laban.find_peaks(fz, height=100, distance=min_step_interval)

# Find jumps (at least 2 seconds apart)
min_jump_interval = int(2.0 * 1000)
peaks = laban.find_peaks(fz, height=500, distance=min_jump_interval)
```

**When to use:**
- Repeated movements with known minimum frequency
- Prevent double-detection of same event
- Filter out noise spikes

**Rule of thumb:**
- `distance = sampling_freq / expected_max_frequency`
- Example: 1000 Hz sampling, max 2 Hz events → distance = 500

### Prominence

Height of peak above surrounding baseline. Better than absolute height for varying baselines:

```python
# Find prominent peaks regardless of absolute height
peaks = laban.find_peaks(fz, prominence=200)

# Combine with height for robust detection
peaks = laban.find_peaks(
    fz,
    height=300,       # Must be at least 300 N
    prominence=150    # And at least 150 N above local baseline
)
```

**When to use:**
- Signal with varying baseline (drift, trend)
- Want peaks that "stand out" from surroundings
- Don't know absolute threshold

**Visual example:**
```
Signal:  /\    /\      /\
         |  \  /  \    /  \
   _____|    \/    \__/    \____
         ^    ^     ^
         |    |     |
    Prominence measures height above local baseline, not absolute value
```

### Width

Minimum peak width at half-prominence height:

```python
# Find peaks at least 20 ms wide (removes narrow spikes)
min_width_samples = int(0.020 * 1000)  # 20 ms at 1000 Hz
peaks = laban.find_peaks(fz, width=min_width_samples)

# Find only broad peaks (>100 ms)
broad_only = int(0.100 * 1000)
peaks = laban.find_peaks(fz, width=broad_only)
```

**When to use:**
- Filter out narrow spikes (electrical noise)
- Distinguish sharp impacts from gradual loading
- Ensure peak duration matches expected event

## Common Workflows

### Jump Detection (Force Platform)

```python
import labanalysis as laban
import numpy as np

# Load force data
record = laban.TimeseriesRecord.from_tdf("jumps.tdf")
fz = record['FP1'].force['Fz']
freq = fz.sampling_frequency

# Filter signal first
fz_filtered = laban.butterworth_filt(
    fz.data,
    freq=freq,
    cut=10,
    order=4,
    filt_type='low'
)

# Calculate bodyweight (mean during quiet standing)
bodyweight = fz_filtered[:int(2*freq)].mean()  # First 2 seconds

# Find peaks above 1.5x bodyweight, at least 2s apart
peaks = laban.find_peaks(
    fz_filtered,
    height=bodyweight * 1.5,
    distance=int(2.0 * freq),
    prominence=bodyweight * 0.5
)

print(f"Detected {len(peaks['peak_indices'])} jumps")
for i, idx in enumerate(peaks['peak_indices']):
    time_s = idx / freq
    force_N = peaks['peak_heights'][i]
    print(f"Jump {i+1}: t={time_s:.2f}s, F={force_N:.1f}N ({force_N/bodyweight:.2f}x BW)")
```

**Output:**
```
Detected 5 jumps
Jump 1: t=3.24s, F=1345.2N (1.83x BW)
Jump 2: t=8.12s, F=1289.7N (1.75x BW)
Jump 3: t=13.45s, F=1402.8N (1.91x BW)
Jump 4: t=18.67s, F=1356.3N (1.84x BW)
Jump 5: t=23.89s, F=1378.1N (1.87x BW)
```

### Gait Analysis (Step Detection)

```python
# Load gait data
record = laban.TimeseriesRecord.from_tdf("walking.tdf")
fp1 = record['FP1'].force['Fz'].data
fp2 = record['FP2'].force['Fz'].data
freq = record['FP1'].sampling_frequency

# Filter
fp1_filt = laban.butterworth_filt(fp1, freq=freq, cut=15, order=4)
fp2_filt = laban.butterworth_filt(fp2, freq=freq, cut=15, order=4)

# Detect heel strikes (minimum 0.6s between steps)
min_step_time = 0.6  # seconds
steps_fp1 = laban.find_peaks(
    fp1_filt,
    height=50,  # 50 N threshold
    distance=int(min_step_time * freq),
    prominence=200
)

steps_fp2 = laban.find_peaks(
    fp2_filt,
    height=50,
    distance=int(min_step_time * freq),
    prominence=200
)

print(f"Platform 1: {len(steps_fp1['peak_indices'])} steps")
print(f"Platform 2: {len(steps_fp2['peak_indices'])} steps")

# Calculate cadence (steps per minute)
duration_s = len(fp1_filt) / freq
total_steps = len(steps_fp1['peak_indices']) + len(steps_fp2['peak_indices'])
cadence = (total_steps / duration_s) * 60

print(f"Cadence: {cadence:.1f} steps/min")
```

### EMG Burst Detection

```python
# Load EMG data
record = laban.TimeseriesRecord.from_tdf("emg.tdf")
emg = record['EMG']['biceps']
freq = emg.sampling_frequency

# Process EMG: band-pass → rectify → envelope
emg_bp = laban.butterworth_filt(emg.data, freq=freq, cut=(20, 450), filt_type='band')
emg_rect = np.abs(emg_bp)
emg_env = laban.butterworth_filt(emg_rect, freq=freq, cut=3, filt_type='low')

# Calculate baseline and threshold
baseline = emg_env[:int(1.0 * freq)].mean()  # First 1 second
threshold = baseline + 3 * emg_env[:int(1.0 * freq)].std()

# Find activation bursts (at least 50 ms wide, 200 ms apart)
bursts = laban.find_peaks(
    emg_env,
    height=threshold,
    distance=int(0.200 * freq),  # 200 ms
    width=int(0.050 * freq)      # 50 ms
)

print(f"Detected {len(bursts['peak_indices'])} activation bursts")
print(f"Baseline: {baseline*1000:.2f} mV")
print(f"Threshold: {threshold*1000:.2f} mV")

# Calculate activation duration
total_active_samples = sum(bursts['widths'])
total_active_time = total_active_samples / freq
print(f"Total active time: {total_active_time:.2f} s")
```

### Marker Trajectory Peaks (Range of Motion)

```python
# Load motion capture
body = laban.WholeBody.from_tdf(
    "squat.tdf",
    left_psis="LPSI", right_psis="RPSI",
    left_asis="LASI", right_asis="RASI"
)

# Get knee angle
knee_angle = body.left_knee_flexionextension.data
freq = body.left_knee_flexionextension.sampling_frequency

# Find peak flexion (maximum angles, at least 1s apart)
peak_flexion = laban.find_peaks(
    knee_angle,
    distance=int(1.0 * freq),
    prominence=20  # At least 20° ROM
)

print(f"Detected {len(peak_flexion['peak_indices'])} squat reps")
for i, idx in enumerate(peak_flexion['peak_indices']):
    angle = peak_flexion['peak_heights'][i]
    time_s = idx / freq
    print(f"Rep {i+1}: Peak flexion = {angle:.1f}° at t={time_s:.2f}s")
```

## find_valleys() Function

Find valleys (negative peaks) in signal.

### Signature

```python
def find_valleys(
    signal: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    width: Optional[int] = None
) -> dict
```

**Note**: `find_valleys()` is equivalent to `find_peaks(-signal)`. Parameters work the same way.

### Example: Detect Unweighting Phase

```python
# Load CMJ data
fz = record['FP1'].force['Fz'].data
freq = record['FP1'].sampling_frequency

# Filter
fz_filt = laban.butterworth_filt(fz, freq=freq, cut=10, order=4)

# Calculate bodyweight
bodyweight = fz_filt[:int(2*freq)].mean()

# Find valleys (unweighting) below 90% bodyweight
valleys = laban.find_valleys(
    fz_filt,
    height=-bodyweight * 0.90,  # Below 90% BW (negative because valley)
    distance=int(2.0 * freq),
    prominence=bodyweight * 0.1
)

print(f"Detected {len(valleys['peak_indices'])} unweighting phases")
for i, idx in enumerate(valleys['peak_indices']):
    time_s = idx / freq
    force_N = -valleys['peak_heights'][i]  # Negate back to positive
    print(f"Valley {i+1}: t={time_s:.2f}s, F={force_N:.1f}N ({force_N/bodyweight:.2f}x BW)")
```

## Visualization

### Plot Peaks on Signal

```python
import plotly.graph_objects as go

# Find peaks
peaks = laban.find_peaks(fz_filtered, height=500, distance=100)

# Create time axis
time = np.arange(len(fz_filtered)) / freq

# Plot
fig = go.Figure()

# Signal
fig.add_trace(go.Scatter(
    x=time,
    y=fz_filtered,
    mode='lines',
    name='Signal',
    line=dict(color='blue')
))

# Peaks
fig.add_trace(go.Scatter(
    x=time[peaks['peak_indices']],
    y=peaks['peak_heights'],
    mode='markers',
    name='Peaks',
    marker=dict(color='red', size=10, symbol='x')
))

fig.update_layout(
    title='Peak Detection',
    xaxis_title='Time (s)',
    yaxis_title='Force (N)',
    hovermode='x unified'
)

fig.show()
```

### Plot with Prominence Visualization

```python
# Find peaks with prominence
peaks = laban.find_peaks(fz_filtered, prominence=200)

fig = go.Figure()

# Signal
fig.add_trace(go.Scatter(
    x=time,
    y=fz_filtered,
    mode='lines',
    name='Signal'
))

# Peaks and prominence lines
for i, idx in enumerate(peaks['peak_indices']):
    peak_time = time[idx]
    peak_height = peaks['peak_heights'][i]
    prominence = peaks['prominences'][i]
    base_height = peak_height - prominence
    
    # Peak marker
    fig.add_trace(go.Scatter(
        x=[peak_time],
        y=[peak_height],
        mode='markers',
        marker=dict(color='red', size=10),
        showlegend=False
    ))
    
    # Prominence line
    fig.add_trace(go.Scatter(
        x=[peak_time, peak_time],
        y=[base_height, peak_height],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        showlegend=False
    ))

fig.update_layout(
    title='Peak Prominence',
    xaxis_title='Time (s)',
    yaxis_title='Force (N)'
)

fig.show()
```

## Troubleshooting

### Too Many Peaks Detected

**Problem**: Find hundreds of peaks due to noise

**Solution 1**: Increase `distance` parameter
```python
# Instead of:
peaks = laban.find_peaks(fz, height=100)  # Finds 347 peaks

# Use:
peaks = laban.find_peaks(fz, height=100, distance=500)  # Finds 8 peaks
```

**Solution 2**: Filter signal first
```python
# Remove noise before peak detection
fz_smooth = laban.butterworth_filt(fz, freq=1000, cut=10, order=4)
peaks = laban.find_peaks(fz_smooth, height=100)
```

**Solution 3**: Increase prominence
```python
# Only peaks that stand out significantly
peaks = laban.find_peaks(fz, prominence=200)
```

### No Peaks Detected

**Problem**: `find_peaks()` returns empty array

**Solution 1**: Check signal range
```python
print(f"Signal range: {fz.min():.1f} to {fz.max():.1f}")
# If max is 450 but height=500, no peaks will be found
```

**Solution 2**: Reduce height threshold
```python
# Try without height constraint first
peaks_all = laban.find_peaks(fz)
print(f"Found {len(peaks_all['peak_indices'])} peaks without constraints")
print(f"Peak heights: {peaks_all['peak_heights']}")

# Then set appropriate threshold
peaks = laban.find_peaks(fz, height=peaks_all['peak_heights'].mean())
```

**Solution 3**: Check for inverted signal
```python
# If force is negative (common with some platforms)
fz_corrected = -fz  # Flip sign
peaks = laban.find_peaks(fz_corrected, height=100)
```

### Peaks Slightly Offset from Visual Maximum

**Problem**: Peak indices don't match visual peak locations

**Cause**: Filtering introduces phase shift OR low sampling rate

**Solution**: Use zero-phase filtering
```python
# butterworth_filt() already uses filtfilt (zero-phase)
# But double-check you're not applying filter twice

# If you need to filter:
fz_filt = laban.butterworth_filt(fz, freq=1000, cut=10)

# Then detect peaks on filtered signal:
peaks = laban.find_peaks(fz_filt, height=500)
```

### Width Parameter Not Working as Expected

**Problem**: Width filtering doesn't seem to work

**Cause**: Width is measured at half-prominence height, not base

**Solution**: Understand width measurement
```python
# Width is measured at FWHM (Full Width at Half Maximum prominence)
# Not at the peak base

# For narrow spike filtering, use smaller width values:
peaks = laban.find_peaks(fz, width=5)  # Remove spikes < 5 samples wide

# For broad peak detection:
peaks = laban.find_peaks(fz, width=50)  # Only peaks > 50 samples wide
```

## Best Practices

### 1. Always Filter First

```python
# Bad: Detect peaks on raw signal
peaks = laban.find_peaks(fz_raw, height=500)  # Detects noise

# Good: Filter then detect
fz_filt = laban.butterworth_filt(fz_raw, freq=1000, cut=10, order=4)
peaks = laban.find_peaks(fz_filt, height=500)
```

### 2. Validate Results Visually

```python
# Always plot to verify
import plotly.graph_objects as go

peaks = laban.find_peaks(fz_filt, height=500, distance=100)

fig = go.Figure()
fig.add_trace(go.Scatter(y=fz_filt, mode='lines', name='Signal'))
fig.add_trace(go.Scatter(
    x=peaks['peak_indices'],
    y=peaks['peak_heights'],
    mode='markers',
    marker=dict(size=10, color='red'),
    name='Peaks'
))
fig.show()

# Check if results make sense
print(f"Detected {len(peaks['peak_indices'])} peaks")
print("Does this match what you expect?")
```

### 3. Use Prominence for Varying Baselines

```python
# Bad: Absolute height with drifting baseline
peaks = laban.find_peaks(emg_env, height=0.5)  # Misses peaks when baseline drifts

# Good: Prominence adapts to local baseline
peaks = laban.find_peaks(emg_env, prominence=0.2)
```

### 4. Combine Multiple Criteria

```python
# Most robust: Combine height, distance, prominence, width
peaks = laban.find_peaks(
    fz_filtered,
    height=bodyweight * 1.2,      # Absolute minimum
    distance=int(1.5 * freq),      # Temporal separation
    prominence=bodyweight * 0.3,   # Stands out from baseline
    width=int(0.020 * freq)        # At least 20 ms wide
)
```

## See Also

- **[Filtering](filtering.md)** - Pre-process signals before peak detection
- **[Derivatives](derivatives.md)** - Calculate velocity and acceleration from peaks
- **[API Reference: find_peaks()](../../api-reference/signalprocessing.md#find_peaks)** - Complete API documentation
- **[Tutorial: Jump Analysis](../../tutorials/01-jump-analysis.md)** - Complete workflow using peak detection

---

**Reference**: SciPy signal.find_peaks() documentation
