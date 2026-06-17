# IRCAM Pressure Mat Data Loading

Guide to loading pressure mat image streams from IRCAM system NPZ files.

## Overview

IRCAM is a pressure mapping system that captures spatial pressure distribution data as time-stamped image streams. labanalysis provides a dedicated reader for IRCAM NPZ (NumPy compressed) files, returning pressure maps indexed by timestamp.

**Data Format:**
- **File type**: `.npz` (NumPy compressed archive)
- **Structure**: Dictionary mapping timestamps to 2D pressure arrays
- **Timestamps**: `datetime.datetime` objects
- **Pressure maps**: `numpy.ndarray` (typically 2D grayscale images)

**Applications:**
- Center of Pressure (COP) tracking
- Pressure distribution analysis
- Balance assessment
- Gait analysis (plantar pressure)
- Postural sway analysis

## Quick Reference

```python
import labanalysis as laban
from datetime import datetime
import numpy as np

# Load NPZ file
pressure_stream = laban.read_npz("pressure_mat_trial.npz")

# Structure: dict[datetime, np.ndarray]
print(f"Total frames: {len(pressure_stream)}")
print(f"Frame shape: {list(pressure_stream.values())[0].shape}")

# Access frames by timestamp
for timestamp, pressure_map in pressure_stream.items():
    print(f"Time: {timestamp}, Pressure shape: {pressure_map.shape}")
    # pressure_map is a 2D numpy array (e.g., 64x64 sensor grid)

# Extract all timestamps (sorted)
timestamps = sorted(pressure_stream.keys())
start_time = timestamps[0]
end_time = timestamps[-1]
duration = (end_time - start_time).total_seconds()

print(f"Recording duration: {duration:.2f} seconds")
```

## NPZ File Format

NPZ files are NumPy archives containing multiple arrays with named keys. IRCAM uses datetime timestamps as keys:

```python
# Structure of NPZ file
{
    datetime(2024, 6, 16, 10, 30, 0, 0): array([[0, 2, 5, ...], [1, 3, 8, ...], ...]),
    datetime(2024, 6, 16, 10, 30, 0, 10000): array([[0, 2, 6, ...], [1, 3, 9, ...], ...]),
    ...
}
```

## Loading IRCAM Data

### Basic Loading

```python
import labanalysis as laban

# Load pressure mat data
stream = laban.read_npz("balance_test.npz")

# Verify data structure
assert all(isinstance(k, datetime) for k in stream.keys()), "Keys must be datetime"
assert all(isinstance(v, np.ndarray) for v in stream.values()), "Values must be arrays"

# Get first frame
first_timestamp = min(stream.keys())
first_frame = stream[first_timestamp]

print(f"Sensor grid: {first_frame.shape}")
print(f"Pressure range: {first_frame.min()} to {first_frame.max()}")
```

### Extract Temporal Sequence

```python
# Sort frames by timestamp
timestamps = sorted(stream.keys())

# Create time vector (seconds from start)
start_time = timestamps[0]
time_s = [(t - start_time).total_seconds() for t in timestamps]

# Stack frames into 3D array (time, height, width)
frames = np.array([stream[t] for t in timestamps])

print(f"Sequence shape: {frames.shape}")  # (n_frames, height, width)
print(f"Sampling rate: {1 / np.mean(np.diff(time_s)):.1f} Hz")
```

## Center of Pressure (COP) Calculation

### 2D COP Trajectory

```python
import labanalysis as laban
import numpy as np

# Load data
stream = laban.read_npz("standing_trial.npz")
timestamps = sorted(stream.keys())
start_time = timestamps[0]

# Calculate COP for each frame
cop_x = []
cop_y = []
time_s = []

for t in timestamps:
    pressure = stream[t]
    
    # Create spatial grid (assuming uniform sensor spacing)
    height, width = pressure.shape
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    
    # Total pressure (sum of all sensors)
    total_pressure = pressure.sum()
    
    if total_pressure > 0:  # Avoid division by zero
        # Weighted average positions
        cop_x_frame = np.sum(pressure.sum(axis=0) * x_coords) / total_pressure
        cop_y_frame = np.sum(pressure.sum(axis=1) * y_coords) / total_pressure
    else:
        cop_x_frame = np.nan
        cop_y_frame = np.nan
    
    cop_x.append(cop_x_frame)
    cop_y.append(cop_y_frame)
    time_s.append((t - start_time).total_seconds())

# Convert to Signal1D
cop_x_signal = laban.Signal1D(
    data=np.array(cop_x),
    index=np.array(time_s),
    unit='sensor units'
)

cop_y_signal = laban.Signal1D(
    data=np.array(cop_y),
    index=np.array(time_s),
    unit='sensor units'
)
```

### COP Sway Analysis

```python
# Calculate sway area (95% confidence ellipse)
std_x = np.nanstd(cop_x)
std_y = np.nanstd(cop_y)
sway_area_95 = np.pi * 2.447 * std_x * std_y  # mm² (if calibrated)

# COP path length (total excursion)
cop_diff_x = np.diff(cop_x)
cop_diff_y = np.diff(cop_y)
path_length = np.sum(np.sqrt(cop_diff_x**2 + cop_diff_y**2))

# Mean velocity
duration = time_s[-1] - time_s[0]
mean_velocity = path_length / duration

print(f"Sway area (95%): {sway_area_95:.1f} sensor units²")
print(f"COP path length: {path_length:.1f} sensor units")
print(f"Mean COP velocity: {mean_velocity:.2f} sensor units/s")
```

## Visualization

### Animate Pressure Distribution

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load data
stream = laban.read_npz("gait_trial.npz")
timestamps = sorted(stream.keys())

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))

# Initialize with first frame
im = ax.imshow(stream[timestamps[0]], cmap='hot', interpolation='nearest')
ax.set_title(f'Frame 0: {timestamps[0]}')
plt.colorbar(im, ax=ax, label='Pressure')

# Animation update function
def update(frame_idx):
    t = timestamps[frame_idx]
    im.set_data(stream[t])
    ax.set_title(f'Frame {frame_idx}: {t}')
    return [im]

# Create animation
anim = animation.FuncAnimation(
    fig, update, frames=len(timestamps),
    interval=50, blit=True  # 50ms between frames
)

plt.show()
```

### Plot COP Trajectory

```python
import matplotlib.pyplot as plt

# Calculate COP (as shown above)
# cop_x, cop_y, time_s

plt.figure(figsize=(12, 5))

# COP trajectory in 2D
plt.subplot(1, 2, 1)
plt.plot(cop_x, cop_y, 'b-', alpha=0.5, linewidth=0.5)
plt.scatter(cop_x[0], cop_y[0], c='green', s=100, label='Start', zorder=3)
plt.scatter(cop_x[-1], cop_y[-1], c='red', s=100, label='End', zorder=3)
plt.xlabel('COP X (sensor units)')
plt.ylabel('COP Y (sensor units)')
plt.title('COP Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# COP time series
plt.subplot(1, 2, 2)
plt.plot(time_s, cop_x, label='COP X')
plt.plot(time_s, cop_y, label='COP Y')
plt.xlabel('Time (s)')
plt.ylabel('COP Position (sensor units)')
plt.title('COP Time Series')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Heatmap of Mean Pressure Distribution

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data
stream = laban.read_npz("standing_trial.npz")

# Calculate mean pressure across all frames
all_frames = np.array(list(stream.values()))
mean_pressure = all_frames.mean(axis=0)

# Plot
plt.figure(figsize=(8, 8))
plt.imshow(mean_pressure, cmap='hot', interpolation='bilinear')
plt.colorbar(label='Mean Pressure')
plt.title('Mean Pressure Distribution')
plt.xlabel('X (sensor)')
plt.ylabel('Y (sensor)')
plt.show()
```

## Advanced Analysis

### Temporal Filtering

```python
import labanalysis as laban

# Load and calculate COP (as shown above)
# cop_x_signal, cop_y_signal

# Filter COP signals (remove high-frequency noise)
freq = 50  # Hz (estimated from timestamps)

cop_x_filtered = laban.butterworth_filt(
    cop_x_signal.data,
    freq=freq,
    cut=5,  # 5 Hz cutoff
    order=4,
    filt_type='low'
)

cop_y_filtered = laban.butterworth_filt(
    cop_y_signal.data,
    freq=freq,
    cut=5,
    order=4,
    filt_type='low'
)

# Update signal data
cop_x_signal.data = cop_x_filtered
cop_y_signal.data = cop_y_filtered
```

### Pressure Thresholding

```python
# Apply threshold to remove noise
threshold = 5  # Pressure units

stream_thresholded = {}
for t, frame in stream.items():
    frame_thresh = frame.copy()
    frame_thresh[frame_thresh < threshold] = 0
    stream_thresholded[t] = frame_thresh

# Recalculate COP with thresholded data
```

### Extract Region of Interest (ROI)

```python
# Define ROI (forefoot region)
roi_y_start = 0
roi_y_end = 30  # Front 30 rows
roi_x_start = 10
roi_x_end = 54  # Centered columns

roi_stream = {}
for t, frame in stream.items():
    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    roi_stream[t] = roi

# Analyze forefoot pressure separately
```

## Calibration

Convert sensor units to physical units (if calibration data available):

```python
# Example calibration (adjust for your system)
sensor_spacing_mm = 10  # 10mm between sensors
pressure_calibration_kPa = 0.1  # 0.1 kPa per sensor unit

# Apply spatial calibration
cop_x_mm = np.array(cop_x) * sensor_spacing_mm
cop_y_mm = np.array(cop_y) * sensor_spacing_mm

# Apply pressure calibration
calibrated_stream = {}
for t, frame in stream.items():
    pressure_kPa = frame * pressure_calibration_kPa
    calibrated_stream[t] = pressure_kPa
```

## Troubleshooting

### Issue: "File must have .npz extension"

```python
# Ensure file has correct extension
file_path = "data.npz"
assert file_path.endswith('.npz'), "File must be .npz format"

stream = laban.read_npz(file_path)
```

### Issue: "Keys must be datetime objects"

If NPZ file has non-datetime keys, manual conversion needed:

```python
import numpy as np
from datetime import datetime, timedelta

# Load raw NPZ
raw_data = np.load("data.npz", allow_pickle=True)

# Convert to datetime-keyed dict
start_time = datetime.now()
stream = {}

for i, key in enumerate(raw_data.files):
    # Assume keys are frame indices
    timestamp = start_time + timedelta(milliseconds=i*20)  # 50 Hz
    stream[timestamp] = raw_data[key]
```

### Issue: Memory Error with Large Files

Process frames sequentially instead of loading all at once:

```python
import numpy as np

# Load NPZ lazily
npz_file = np.load("large_file.npz", mmap_mode='r')

# Process one frame at a time
for key in npz_file.files:
    frame = npz_file[key]
    # Process frame
    # ...
    
npz_file.close()
```

### Issue: Inconsistent Frame Dimensions

Check if all frames have same shape:

```python
stream = laban.read_npz("data.npz")

shapes = [v.shape for v in stream.values()]
unique_shapes = set(shapes)

if len(unique_shapes) > 1:
    print(f"Warning: Multiple frame shapes found: {unique_shapes}")
    
    # Crop to minimum common size
    min_height = min(s[0] for s in shapes)
    min_width = min(s[1] for s in shapes)
    
    stream_cropped = {
        t: frame[:min_height, :min_width]
        for t, frame in stream.items()
    }
```

## Performance Tips

### Batch Processing

```python
# Extract all frames as 3D array for vectorized operations
timestamps = sorted(stream.keys())
frames_3d = np.array([stream[t] for t in timestamps])

# Vectorized threshold
frames_3d[frames_3d < threshold] = 0

# Vectorized COP calculation
height, width = frames_3d.shape[1:]
x_coords = np.arange(width)
y_coords = np.arange(height)

total_pressure = frames_3d.sum(axis=(1, 2))  # Sum per frame
cop_x_batch = np.sum(frames_3d.sum(axis=1) * x_coords, axis=1) / total_pressure
cop_y_batch = np.sum(frames_3d.sum(axis=2) * y_coords, axis=1) / total_pressure
```

## See Also

- [Balance Tests](../test-protocols/balance-tests.md) - COP analysis for balance assessment
- [Signal Processing: Filtering](../signal-processing/filtering.md) - Filtering COP trajectories
- [API Reference: I/O](../../api-reference/io/read.md#ircam) - Complete read_npz() API

---

**IRCAM Pressure Mat**: High-resolution pressure mapping system for biomechanical analysis of standing balance, gait, and posture.
