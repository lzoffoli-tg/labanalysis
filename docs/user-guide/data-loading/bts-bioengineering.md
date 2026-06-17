# BTS Bioengineering (TDF Files)

Complete guide to loading and working with BTS TDF files in labanalysis.

## Overview

BTS Bioengineering systems (Smart-DX, SMART Clinic, ELITE) export data in TDF (Tab-Delimited File) format. labanalysis provides comprehensive support for loading TDF files containing:

- Motion capture markers (3D positions)
- Force platform data (forces, moments, COP)
- EMG signals
- Synchronized multi-device data

## Quick Start

```python
import labanalysis as laban

# Load TDF file
record = laban.TimeseriesRecord.from_tdf("C:/Data/trial001.tdf")

# Access force platforms
fp1 = record['FP1']
fz = fp1.force['Fz']

# Access markers
markers = record['MKRS']
c7 = markers['C7']

# Access EMG (if present)
if 'EMG' in record:
    emg = record['EMG']
    biceps = emg['biceps']
```

## TDF File Structure

### Header Section

TDF files start with metadata:

```
TYPE=ASCII
VERSION=1.0
FREQUENCY=1000
LABELS=TIME,FP1_Fx,FP1_Fy,FP1_Fz,...
UNITS=s,N,N,N,...
```

**Key metadata:**
- `FREQUENCY`: Sampling rate in Hz
- `LABELS`: Column names
- `UNITS`: Physical units for each column

### Data Section

Tab-delimited numerical data:

```
0.000	-12.3	-5.6	842.1	...
0.001	-11.8	-5.4	845.3	...
...
```

## Loading TDF Files

### Basic Loading

```python
# Load complete file
record = laban.TimeseriesRecord.from_tdf("trial.tdf")

# Check what's available
print(record.keys())
# Output: dict_keys(['FP1', 'FP2', 'MKRS', 'EMG'])
```

### Loading Specific Columns

For large files, load only needed columns:

```python
# Load only force platform 1
fp1_only = laban.ForcePlatform.from_tdf(
    "trial.tdf",
    force_x='FP1_Fx',
    force_y='FP1_Fy',
    force_z='FP1_Fz',
    torque_x='FP1_Mx',
    torque_y='FP1_My',
    torque_z='FP1_Mz'
)
```

### Loading with WholeBody

```python
# Load motion capture for full body model
body = laban.WholeBody.from_tdf(
    "mocap.tdf",
    # Map TDF labels to anatomical markers
    left_psis="LPSI",
    right_psis="RPSI",
    left_asis="LASI",
    right_asis="RASI",
    left_knee_medial="LKNEM",
    left_knee_lateral="LKNEL",
    # ... etc
)

# Access joint angles
knee_angle = body.left_knee_flexionextension
```

## Common BTS Label Conventions

### Force Platform Labels

BTS typically uses these patterns:

| Device | Force | Torque | COP |
|--------|-------|--------|-----|
| FP1 | `FP1_Fx`, `FP1_Fy`, `FP1_Fz` | `FP1_Mx`, `FP1_My`, `FP1_Mz` | `FP1_COPx`, `FP1_COPy` |
| FP2 | `FP2_Fx`, `FP2_Fy`, `FP2_Fz` | `FP2_Mx`, `FP2_My`, `FP2_Mz` | `FP2_COPx`, `FP2_COPy` |

### Marker Labels

BTS markers are typically in the `MKRS` group:

```python
markers = record['MKRS']

# Common marker names
c7 = markers['C7']          # C7 vertebra
lasi = markers['LASI']      # Left ASIS
rasi = markers['RASI']      # Right ASIS
lknem = markers['LKNEM']    # Left knee medial
```

Marker data structure:
```python
# Each marker is a Point3D with x, y, z components
print(c7.data.shape)  # (n_samples, 3)
print(c7.labels)      # ['x', 'y', 'z']
print(c7.units)       # ['mm', 'mm', 'mm'] or ['m', 'm', 'm']
```

### EMG Labels

EMG channels are in the `EMG` group:

```python
emg = record['EMG']

# Access specific muscles
biceps = emg['biceps']
triceps = emg['triceps']
vastus_lateralis = emg['vastus_lateralis']
```

## Multi-Platform Setup

### Dual Force Platforms

```python
# Load trial with two force platforms
record = laban.TimeseriesRecord.from_tdf("gait.tdf")

# Access both platforms
fp1 = record['FP1']
fp2 = record['FP2']

# Compare left and right forces
fz_left = fp1.force['Fz'].data
fz_right = fp2.force['Fz'].data

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(y=fz_left, name='Left (FP1)'))
fig.add_trace(go.Scatter(y=fz_right, name='Right (FP2)'))
fig.update_layout(title='Bilateral GRF', yaxis_title='Force (N)')
fig.show()
```

### Four-Platform Setup

```python
# Load all four platforms
record = laban.TimeseriesRecord.from_tdf("trial.tdf")

platforms = [record[f'FP{i}'] for i in range(1, 5)]

# Calculate total vertical force
total_fz = sum(fp.force['Fz'].data for fp in platforms)

print(f"Total vertical force: {total_fz.max():.1f} N")
```

## Coordinate Systems

### BTS Global Coordinate System

BTS uses a right-handed coordinate system:

- **X-axis**: Forward (direction of progression)
- **Y-axis**: Upward (vertical)
- **Z-axis**: Right (mediolateral)

**Important**: This differs from ISB recommendations. Use `change_reference_frame()` to convert if needed:

```python
# Convert marker from BTS to ISB coordinate system
c7_bts = body.c7_vertebra

# BTS: X=forward, Y=up, Z=right
# ISB: X=forward, Y=up, Z=right (same in this case)

# Example: if your BTS has Z=forward instead
c7_isb = laban.change_reference_frame(
    c7_bts,
    from_system='bts',
    to_system='isb'
)
```

### Force Platform Orientation

Force platforms have local coordinate systems. Check your BTS setup:

```python
fp = record['FP1']

# Check force directions
print(f"Max Fx: {fp.force['Fx'].data.max():.1f} N (forward/backward)")
print(f"Max Fy: {fp.force['Fy'].data.max():.1f} N (up/down)")
print(f"Max Fz: {fp.force['Fz'].data.max():.1f} N (left/right)")

# Typical: Fy is vertical force during standing
# If Fz is vertical, platform may be rotated
```

## Unit Handling

BTS files may use different units. labanalysis automatically handles unit conversion:

```python
# Check units
fp = record['FP1']
print(f"Force unit: {fp.force['Fz'].unit}")  # Usually 'N'

markers = record['MKRS']
c7 = markers['C7']
print(f"Position unit: {c7.unit}")  # Could be 'mm' or 'm'

# Convert units if needed
from pint import UnitRegistry
ureg = UnitRegistry()

# Convert mm to m
if c7.unit == 'mm':
    c7_meters = c7.data * ureg.mm
    c7_meters = c7_meters.to('m').magnitude
```

## Common Workflows

### Jump Analysis from TDF

```python
import labanalysis as laban
import numpy as np

# Load jump data
record = laban.TimeseriesRecord.from_tdf("cmj.tdf")
fp = record['FP1']

# Get vertical force
fz = fp.force['Fz']

# Filter
fz_filtered = laban.butterworth_filt(
    fz.data,
    freq=fz.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# Find takeoff and landing
threshold = 20  # N above bodyweight
contacts = fz_filtered > threshold

# Find transitions
takeoff_idx = np.where(np.diff(contacts.astype(int)) == -1)[0][0]
landing_idx = np.where(np.diff(contacts.astype(int)) == 1)[0][0]

# Calculate flight time
flight_time = (landing_idx - takeoff_idx) / fz.sampling_frequency

# Jump height from flight time
jump_height = 0.5 * 9.81 * (flight_time / 2) ** 2

print(f"Flight time: {flight_time*1000:.1f} ms")
print(f"Jump height: {jump_height*100:.1f} cm")
```

### Gait Analysis from TDF

```python
# Load gait trial with markers and force platforms
record = laban.TimeseriesRecord.from_tdf("gait.tdf")

# Create WholeBody for kinematics
body = laban.WholeBody.from_tdf(
    "gait.tdf",
    # Pelvis markers
    left_psis="LPSI", right_psis="RPSI",
    left_asis="LASI", right_asis="RASI",
    # Leg markers
    left_heel="LHEE", left_toe="LTOE",
    right_heel="RHEE", right_toe="RTOE",
    # ... other markers
)

# Get force platforms
fp1 = record['FP1']
fp2 = record['FP2']

# Detect heel strikes from force
fz1 = fp1.force['Fz'].data
heel_strikes_left = laban.find_peaks(fz1, height=50, distance=500)

# Extract stride
stride_start = heel_strikes_left['peak_indices'][0]
stride_end = heel_strikes_left['peak_indices'][1]

# Get joint angles for this stride
hip_angle = body.left_hip_flexionextension.data[stride_start:stride_end]
knee_angle = body.left_knee_flexionextension.data[stride_start:stride_end]

# Normalize to 0-100% gait cycle
gait_percent = np.linspace(0, 100, len(hip_angle))

print(f"Stride length: {stride_end - stride_start} samples")
```

### EMG Analysis from TDF

```python
# Load TDF with EMG
record = laban.TimeseriesRecord.from_tdf("emg_trial.tdf")

# Access EMG channels
emg = record['EMG']
biceps = emg['biceps']

# Standard EMG processing pipeline
# 1. Band-pass filter (20-450 Hz)
biceps_bp = laban.butterworth_filt(
    biceps.data,
    freq=biceps.sampling_frequency,
    cut=(20, 450),
    order=4,
    filt_type='band'
)

# 2. Full-wave rectification
biceps_rect = np.abs(biceps_bp)

# 3. Linear envelope (low-pass at 3 Hz)
biceps_env = laban.butterworth_filt(
    biceps_rect,
    freq=biceps.sampling_frequency,
    cut=3,
    order=4,
    filt_type='low'
)

# Find activation threshold
baseline = biceps_env[:int(1.0 * biceps.sampling_frequency)].mean()  # First 1 second
threshold = baseline + 3 * biceps_env[:int(1.0 * biceps.sampling_frequency)].std()

# Detect activations
activations = biceps_env > threshold

print(f"Baseline EMG: {baseline:.3f} mV")
print(f"Activation threshold: {threshold:.3f} mV")
print(f"Active: {(activations.sum() / len(activations)) * 100:.1f}% of trial")
```

## Troubleshooting

### File Not Found

```python
from pathlib import Path

# Use absolute paths
file_path = Path("C:/Data/trial.tdf").resolve()

if not file_path.exists():
    print(f"File not found: {file_path}")
else:
    record = laban.TimeseriesRecord.from_tdf(str(file_path))
```

### Missing Columns

```python
try:
    record = laban.TimeseriesRecord.from_tdf("trial.tdf")
    fp1 = record['FP1']
except KeyError as e:
    print(f"Missing device: {e}")
    print(f"Available devices: {list(record.keys())}")
    
# Check available labels in raw file
import pandas as pd
df = pd.read_csv("trial.tdf", sep='\t', skiprows=5, nrows=1)
print(f"Available columns: {list(df.columns)}")
```

### Incorrect Sampling Frequency

```python
# Verify sampling frequency
record = laban.TimeseriesRecord.from_tdf("trial.tdf")
fp = record['FP1']

print(f"Detected sampling frequency: {fp.sampling_frequency} Hz")

# If incorrect, check TDF header
with open("trial.tdf", 'r') as f:
    for _ in range(10):
        line = f.readline()
        if 'FREQUENCY' in line:
            print(line)
```

### Unit Mismatch

```python
# Force platforms in N, markers in mm - need consistent units
fp = record['FP1']
markers = record['MKRS']
c7 = markers['C7']

print(f"Force units: {fp.force['Fz'].unit}")     # N
print(f"Marker units: {c7.unit}")                 # mm

# Convert if needed
if c7.unit == 'mm':
    # Convert to meters for consistency
    c7_data_m = c7.data / 1000
```

### Large File Performance

```python
# For very large TDF files (>100 MB), load only needed columns

# Option 1: Load specific time range
df = pd.read_csv(
    "large_trial.tdf",
    sep='\t',
    skiprows=6,
    usecols=['TIME', 'FP1_Fz'],  # Only needed columns
    nrows=10000  # Only first 10000 samples
)

# Option 2: Downsample
record = laban.TimeseriesRecord.from_tdf("large_trial.tdf")
fp = record['FP1']

# Downsample from 1000 Hz to 100 Hz
from scipy import signal
downsampled = signal.decimate(fp.force['Fz'].data, q=10)
```

## Best Practices

### 1. File Organization

```python
# Organize TDF files systematically
from pathlib import Path

data_dir = Path("C:/BTS_Data")
participant_dir = data_dir / "P001"
session_dir = participant_dir / "session_01"

trial_files = list(session_dir.glob("*.tdf"))
print(f"Found {len(trial_files)} trials")

# Process batch
for trial_file in trial_files:
    record = laban.TimeseriesRecord.from_tdf(trial_file)
    # ... analysis
```

### 2. Validation

```python
# Validate loaded data
record = laban.TimeseriesRecord.from_tdf("trial.tdf")

# Check force platform data quality
fp = record['FP1']
fz = fp.force['Fz'].data

# Check for zeros (platform not in contact)
zero_samples = (fz < 10).sum()
print(f"Zero force samples: {zero_samples} ({zero_samples/len(fz)*100:.1f}%)")

# Check for outliers
mean_fz = fz.mean()
std_fz = fz.std()
outliers = np.abs(fz - mean_fz) > 5 * std_fz
print(f"Outlier samples: {outliers.sum()}")
```

### 3. Metadata Documentation

```python
# Document trial metadata
trial_info = {
    'file': "trial001.tdf",
    'participant': "P001",
    'date': "2026-06-15",
    'condition': "barefoot_running",
    'sampling_freq': 1000,
    'platforms': ['FP1', 'FP2'],
    'markers': 42,
    'duration_s': len(record['FP1'].force['Fz']) / 1000
}

import json
with open("trial001_metadata.json", 'w') as f:
    json.dump(trial_info, f, indent=2)
```

## See Also

- **[OpenSim Files](opensim.md)** - Load C3D/MOT/STO files
- **[Signal Processing](../signal-processing/README.md)** - Process loaded signals
- **[WholeBody Model](../biomechanics/whole-body-model.md)** - Full body kinematics
- **[API Reference: I/O](../../api-reference/io/read.md)** - Complete loading functions

---

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
