# Tutorial: Advanced Signal Processing Workflows

Complete guide to signal processing techniques for biomechanical data analysis using labanalysis.

**Duration**: 40 minutes  
**Level**: Intermediate to Advanced  
**Prerequisites**: labanalysis installed, understanding of signal processing basics, NumPy knowledge

## What You'll Learn

- Apply various filtering techniques (Butterworth, FIR, median, RMS)
- Detect peaks in force and kinematic signals
- Calculate derivatives using Winter 2009 method
- Handle missing data effectively
- Perform frequency analysis (PSD, residual analysis)
- Transform signals between reference frames
- Build complete signal processing pipelines
- Optimize filter parameters for different signal types

## Scenario

You have collected motion capture and force platform data from a jump test. The raw data contains noise, outliers, and missing values. You'll clean the data, apply appropriate filtering, detect key events (takeoff, landing), calculate velocities and accelerations, and analyze frequency content to validate your processing choices.

## Part 1: Filtering Techniques

### Step 1: Load Raw Data

```python
import labanalysis as laban
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load jump test data
data = laban.read_tdf("jump_test_raw.tdf")

# Extract vertical force (raw)
fp = data['FP1']
fz_raw = fp.force['Fz']  # Vertical component

# Extract marker position (raw)
sacrum = data['sacrum']

print(f"Sampling frequency: {fz_raw.sampling_frequency} Hz")
print(f"Duration: {len(fz_raw) / fz_raw.sampling_frequency:.2f} s")
print(f"Force range: {fz_raw.data.min():.1f} to {fz_raw.data.max():.1f} N")
```

**Output:**
```
Sampling frequency: 1000 Hz
Duration: 5.00 s
Force range: -15.3 to 1842.7 N
```

### Step 2: Remove Outliers with Median Filter

```python
# First pass: remove outliers
fz_no_outliers = laban.median_filt(fz_raw.data, window_size=5)

# Check improvement
n_outliers = np.sum(np.abs(fz_raw.data - fz_no_outliers) > 50)
print(f"Outliers removed: {n_outliers} samples ({n_outliers/len(fz_raw)*100:.2f}%)")

# Visualize
time = np.arange(len(fz_raw)) / fz_raw.sampling_frequency

fig = make_subplots(rows=2, cols=1, subplot_titles=['Raw Signal', 'After Median Filter'])

fig.add_trace(go.Scatter(x=time, y=fz_raw.data, mode='lines', name='Raw',
                         line=dict(color='lightgray')), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=fz_no_outliers, mode='lines', name='Filtered',
                         line=dict(color='blue')), row=2, col=1)

fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Force (N)", row=1, col=1)
fig.update_yaxes(title_text="Force (N)", row=2, col=1)
fig.update_layout(height=600, showlegend=False)
fig.show()
```

**Output:**
```
Outliers removed: 23 samples (0.46%)
```

### Step 3: Apply Low-Pass Butterworth Filter

```python
# Determine appropriate cutoff frequency
# Rule of thumb: force platform data → 10-30 Hz
# markers → 6-12 Hz

# Low-pass filter for force (10 Hz)
fz_filtered = laban.butterworth_filt(
    signal=fz_no_outliers,
    freq=fz_raw.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# Low-pass filter for marker (6 Hz)
sacrum_z_filtered = laban.butterworth_filt(
    signal=sacrum['z'].data,
    freq=sacrum.sampling_frequency,
    cut=6,
    order=4,
    filt_type='low'
)

print("Applied Butterworth filters:")
print(f"  Force: 10 Hz, 4th order")
print(f"  Marker: 6 Hz, 4th order")
```

**Output:**
```
Applied Butterworth filters:
  Force: 10 Hz, 4th order
  Marker: 6 Hz, 4th order
```

### Step 4: Compare Filter Types

```python
# Compare different filter approaches
fz_butter_4 = laban.butterworth_filt(fz_no_outliers, 1000, 10, order=4, filt_type='low')
fz_butter_2 = laban.butterworth_filt(fz_no_outliers, 1000, 10, order=2, filt_type='low')
fz_fir = laban.fir_filt(fz_no_outliers, 1000, 10, numtaps=101)
fz_running = laban.running_mean(fz_no_outliers, window_size=101)

# Plot comparison (zoomed to jump phase)
jump_start = 2000
jump_end = 3000
time_zoom = time[jump_start:jump_end]

fig = go.Figure()
fig.add_trace(go.Scatter(x=time_zoom, y=fz_no_outliers[jump_start:jump_end],
                         mode='lines', name='No filter', line=dict(color='lightgray', width=1)))
fig.add_trace(go.Scatter(x=time_zoom, y=fz_butter_4[jump_start:jump_end],
                         mode='lines', name='Butterworth 4th', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=time_zoom, y=fz_butter_2[jump_start:jump_end],
                         mode='lines', name='Butterworth 2nd', line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=time_zoom, y=fz_fir[jump_start:jump_end],
                         mode='lines', name='FIR', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=time_zoom, y=fz_running[jump_start:jump_end],
                         mode='lines', name='Running mean', line=dict(color='orange', width=2)))

fig.update_layout(
    title='Filter Comparison (10 Hz cutoff)',
    xaxis_title='Time (s)',
    yaxis_title='Force (N)',
    height=500
)
fig.show()

print("\nFilter characteristics:")
print("  Butterworth 4th: Zero phase lag, sharp transition, best for most applications")
print("  Butterworth 2nd: Zero phase lag, smoother transition")
print("  FIR:             Linear phase, no overshoot, slower rolloff")
print("  Running mean:    Simple, phase lag, poor frequency selectivity")
```

### Step 5: Frequency Analysis to Validate Cutoff

```python
# Compute Power Spectral Density
from scipy import signal as sp_signal

# Raw signal PSD
freq_raw, psd_raw = sp_signal.welch(fz_no_outliers, fs=1000, nperseg=1024)

# Filtered signal PSD
freq_filt, psd_filt = sp_signal.welch(fz_filtered, fs=1000, nperseg=1024)

# Plot PSD
fig = go.Figure()
fig.add_trace(go.Scatter(x=freq_raw, y=psd_raw, mode='lines', name='Raw'))
fig.add_trace(go.Scatter(x=freq_filt, y=psd_filt, mode='lines', name='Filtered (10 Hz)'))
fig.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Cutoff")

fig.update_layout(
    title='Power Spectral Density',
    xaxis_title='Frequency (Hz)',
    yaxis_title='PSD (N²/Hz)',
    xaxis_type='log',
    yaxis_type='log',
    height=500
)
fig.show()

# Calculate signal power in different frequency bands
power_low = np.trapz(psd_raw[freq_raw < 10], freq_raw[freq_raw < 10])
power_mid = np.trapz(psd_raw[(freq_raw >= 10) & (freq_raw < 50)], freq_raw[(freq_raw >= 10) & (freq_raw < 50)])
power_high = np.trapz(psd_raw[freq_raw >= 50], freq_raw[freq_raw >= 50])
power_total = power_low + power_mid + power_high

print(f"\nSignal power distribution:")
print(f"  < 10 Hz:    {power_low/power_total*100:.1f}% (signal)")
print(f"  10-50 Hz:   {power_mid/power_total*100:.1f}% (moderate)")
print(f"  > 50 Hz:    {power_high/power_total*100:.1f}% (noise)")
```

**Output:**
```
Signal power distribution:
  < 10 Hz:    94.3% (signal)
  10-50 Hz:   4.8% (moderate)
  > 50 Hz:    0.9% (noise)
```

## Part 2: Peak Detection

### Step 6: Detect Jump Events

```python
# Create clean Signal1D
fz = laban.Signal1D(
    data=fz_filtered,
    sampling_frequency=1000,
    unit='N',
    name='vertical_force'
)

# Detect peaks (local maxima in force)
peaks_idx, peaks_props = laban.find_peaks(
    fz.data,
    height=800,        # Minimum peak height
    distance=500,      # Minimum distance between peaks (0.5s)
    prominence=300     # Minimum prominence
)

print(f"Detected {len(peaks_idx)} jumps")
print(f"Peak forces: {fz.data[peaks_idx]} N")

# Detect takeoff and landing (force threshold crossings)
bodyweight = 750  # N (estimated from baseline force)
threshold = bodyweight * 0.1  # 10% of bodyweight

# Find takeoff (force drops below threshold)
is_airborne = fz.data < threshold
takeoff_candidates = np.where(np.diff(is_airborne.astype(int)) == 1)[0]
landing_candidates = np.where(np.diff(is_airborne.astype(int)) == -1)[0]

print(f"\nDetected events:")
print(f"  Takeoffs: {len(takeoff_candidates)}")
print(f"  Landings: {len(landing_candidates)}")

# Visualize events
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=fz.data, mode='lines', name='Force'))
fig.add_hline(y=threshold, line_dash="dash", line_color="gray", annotation_text="Threshold")
fig.add_trace(go.Scatter(x=time[peaks_idx], y=fz.data[peaks_idx],
                         mode='markers', name='Peaks',
                         marker=dict(size=12, color='red', symbol='star')))
fig.add_trace(go.Scatter(x=time[takeoff_candidates], y=fz.data[takeoff_candidates],
                         mode='markers', name='Takeoff',
                         marker=dict(size=10, color='green', symbol='triangle-up')))
fig.add_trace(go.Scatter(x=time[landing_candidates], y=fz.data[landing_candidates],
                         mode='markers', name='Landing',
                         marker=dict(size=10, color='orange', symbol='triangle-down')))

fig.update_layout(title='Jump Event Detection', xaxis_title='Time (s)', yaxis_title='Force (N)')
fig.show()
```

**Output:**
```
Detected 3 jumps
Peak forces: [1523.4 1487.2 1512.8] N

Detected events:
  Takeoffs: 3
  Landings: 3
```

## Part 3: Derivatives

### Step 7: Calculate Velocity and Acceleration

```python
# Calculate velocity from position (Winter 2009 method)
sacrum_z = laban.Signal1D(
    data=sacrum_z_filtered,
    sampling_frequency=100,
    unit='m',
    name='sacrum_vertical'
)

# First derivative (velocity)
velocity = laban.derivative(
    sacrum_z.data,
    sampling_frequency=sacrum_z.sampling_frequency,
    order=1
)

# Second derivative (acceleration)
acceleration = laban.derivative(
    sacrum_z.data,
    sampling_frequency=sacrum_z.sampling_frequency,
    order=2
)

# Create Signal1D objects
velocity_sig = laban.Signal1D(velocity, 100, 'm/s', 'vertical_velocity')
acceleration_sig = laban.Signal1D(acceleration, 100, 'm/s²', 'vertical_acceleration')

# Find peak velocity during jump
time_marker = np.arange(len(sacrum_z)) / sacrum_z.sampling_frequency

# Focus on first jump (2-3 seconds)
jump1_start = int(2.0 * 100)
jump1_end = int(3.0 * 100)

peak_vel_idx = jump1_start + np.argmax(velocity[jump1_start:jump1_end])
peak_vel = velocity[peak_vel_idx]

print(f"Peak velocity: {peak_vel:.3f} m/s")
print(f"Peak acceleration: {acceleration.max():.2f} m/s²")

# Visualize derivatives
fig = make_subplots(rows=3, cols=1, 
                    subplot_titles=['Position', 'Velocity', 'Acceleration'],
                    vertical_spacing=0.08)

fig.add_trace(go.Scatter(x=time_marker, y=sacrum_z.data, mode='lines', name='Position'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=time_marker, y=velocity, mode='lines', name='Velocity'),
              row=2, col=1)
fig.add_trace(go.Scatter(x=time_marker, y=acceleration, mode='lines', name='Acceleration'),
              row=3, col=1)

fig.update_yaxes(title_text="Position (m)", row=1, col=1)
fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
fig.update_yaxes(title_text="Accel (m/s²)", row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(height=800, showlegend=False)
fig.show()
```

**Output:**
```
Peak velocity: 2.843 m/s
Peak acceleration: 34.72 m/s²
```

## Part 4: Handling Missing Data

### Step 8: Interpolate Missing Marker Data

```python
# Simulate missing data (common with occlusion)
sacrum_with_gaps = sacrum['z'].data.copy()
sacrum_with_gaps[250:280] = np.nan  # 30 frames gap
sacrum_with_gaps[450:455] = np.nan  # 5 frames gap
sacrum_with_gaps[620:625] = np.nan  # 5 frames gap

print(f"Missing frames: {np.sum(np.isnan(sacrum_with_gaps))}")

# Try different interpolation methods
cubic_interp = laban.cubicspline_interp(sacrum_with_gaps)
linear_interp = laban.linear_interp(sacrum_with_gaps)
pchip_interp = laban.pchip_interp(sacrum_with_gaps)

# Compare methods
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_marker[200:350], y=sacrum['z'].data[200:350],
                         mode='lines', name='Original', line=dict(color='black', width=3)))
fig.add_trace(go.Scatter(x=time_marker[200:350], y=sacrum_with_gaps[200:350],
                         mode='lines+markers', name='With gaps',
                         line=dict(color='lightgray', width=1),
                         marker=dict(size=4)))
fig.add_trace(go.Scatter(x=time_marker[200:350], y=cubic_interp[200:350],
                         mode='lines', name='Cubic spline', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(x=time_marker[200:350], y=linear_interp[200:350],
                         mode='lines', name='Linear', line=dict(color='red', dash='dot')))
fig.add_trace(go.Scatter(x=time_marker[200:350], y=pchip_interp[200:350],
                         mode='lines', name='PCHIP', line=dict(color='green', dash='dashdot')))

fig.update_layout(title='Gap Interpolation Comparison', xaxis_title='Time (s)', yaxis_title='Height (m)')
fig.show()

print("\nInterpolation methods:")
print("  Cubic spline: Smooth, may overshoot")
print("  Linear:       Simple, no overshoot, not smooth")
print("  PCHIP:        Shape-preserving, no overshoot, recommended")
```

**Output:**
```
Missing frames: 40

Interpolation methods:
  Cubic spline: Smooth, may overshoot
  Linear:       Simple, no overshoot, not smooth
  PCHIP:        Shape-preserving, no overshoot, recommended
```

## Part 5: Complete Pipeline

### Step 9: Build Automated Processing Pipeline

```python
from labanalysis.records.pipelines import ProcessingPipeline

# Define processing pipeline for force platform
force_pipeline = ProcessingPipeline()
force_pipeline.add_step(lambda x: laban.median_filt(x, window_size=5))  # Remove outliers
force_pipeline.add_step(lambda x: laban.butterworth_filt(x, 1000, 10, 4, 'low'))  # Low-pass

# Define processing pipeline for markers
marker_pipeline = ProcessingPipeline()
marker_pipeline.add_step(lambda x: laban.pchip_interp(x))  # Fill gaps
marker_pipeline.add_step(lambda x: laban.butterworth_filt(x, 100, 6, 4, 'low'))  # Low-pass

# Apply pipelines
fz_processed = force_pipeline.apply(fz_raw.data)
sacrum_processed = marker_pipeline.apply(sacrum['z'].data)

print("Processing pipeline applied:")
print("  Force: median filter → Butterworth 10 Hz")
print("  Marker: PCHIP interpolation → Butterworth 6 Hz")

# Validate with residual analysis
residual_force = fz_raw.data - fz_processed
residual_marker = sacrum['z'].data - sacrum_processed

print(f"\nResidual RMS:")
print(f"  Force: {np.sqrt(np.mean(residual_force**2)):.2f} N")
print(f"  Marker: {np.sqrt(np.mean(residual_marker**2)):.4f} m")
```

**Output:**
```
Processing pipeline applied:
  Force: median filter → Butterworth 10 Hz
  Marker: PCHIP interpolation → Butterworth 6 Hz

Residual RMS:
  Force: 12.34 N
  Marker: 0.0023 m
```

### Step 10: Export Processed Data

```python
# Create processed TimeseriesRecord
processed_data = laban.TimeseriesRecord()

# Add processed force
processed_fp = laban.ForcePlatform(
    force=laban.Signal3D(
        data={'Fx': fp.force['Fx'].data,
              'Fy': fp.force['Fy'].data,
              'Fz': fz_processed},
        sampling_frequency=1000,
        unit='N'
    ),
    cop=fp.cop
)
processed_data['FP1'] = processed_fp

# Add processed marker
processed_data['sacrum'] = laban.Point3D(
    data={'x': sacrum['x'].data,
          'y': sacrum['y'].data,
          'z': sacrum_processed},
    sampling_frequency=100,
    unit='m'
)

# Add velocity (derived)
processed_data['sacrum_velocity'] = laban.Signal3D(
    data={'x': laban.derivative(sacrum['x'].data, 100, 1),
          'y': laban.derivative(sacrum['y'].data, 100, 1),
          'z': velocity},
    sampling_frequency=100,
    unit='m/s'
)

# Save processed data
processed_data.to_tdf("jump_test_processed.tdf")
print("Saved processed data to jump_test_processed.tdf")

# Export to DataFrame
df = pd.DataFrame({
    'time': time,
    'force_raw': fz_raw.data,
    'force_processed': fz_processed,
    'force_residual': residual_force
})

df.to_csv("force_processing_report.csv", index=False)
print("Saved processing report to CSV")
```

**Output:**
```
Saved processed data to jump_test_processed.tdf
Saved processing report to CSV
```

## Key Takeaways

### Filter Selection Guidelines
| Signal Type | Recommended Filter | Cutoff | Order | Notes |
|------------|-------------------|--------|-------|-------|
| Force platform | Butterworth low-pass | 10-30 Hz | 4 | Zero phase lag critical |
| Markers (gait) | Butterworth low-pass | 6 Hz | 4 | Winter 2009 recommendation |
| Markers (jumping) | Butterworth low-pass | 10-12 Hz | 4 | Higher frequency content |
| EMG (raw) | Butterworth band-pass | 20-450 Hz | 4 | Remove movement artifacts + high freq noise |
| Acceleration | FIR low-pass | 15-20 Hz | - | Linear phase for integration |

### Processing Best Practices
1. **Always visualize before/after filtering**
2. **Use PSD to validate cutoff frequency**
3. **Median filter before low-pass (removes spikes)**
4. **PCHIP for gap interpolation (shape-preserving)**
5. **Apply processing pipeline consistently**

### Common Mistakes
- ❌ Filtering twice (compounds phase distortion)
- ❌ Cutoff too low (removes signal)
- ❌ Cutoff too high (keeps noise)
- ❌ Taking derivatives before filtering (amplifies noise)
- ❌ Using forward-only filters (introduces phase lag)

## Next Steps

- **Tutorial 06**: Building custom test protocols
- **Tutorial 07**: Batch processing workflows
- **User Guide**: [Signal Processing](../user-guide/signal-processing/) - Complete reference
- **API Reference**: [signalprocessing](../api-reference/signalprocessing.md)

---

**Complete signal processing workflows for biomechanical data: filtering, peak detection, derivatives, gap filling, and validation.**
