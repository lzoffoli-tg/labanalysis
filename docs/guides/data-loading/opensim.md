# OpenSim Data Loading

Guide to loading OpenSim motion capture and simulation data files (.trc, .mot, .sto) in labanalysis.

## Overview

OpenSim is a widely-used biomechanical simulation software that exports motion capture and simulation data in standardized text formats. labanalysis provides dedicated readers for:

- **TRC files** (Track Row Column): Marker trajectories from motion capture systems
- **MOT files** (Motion): Joint angles, forces, or generic time-series data
- **STO files** (Storage): Simulation results (forces, muscle activations, joint angles)

All readers return pandas DataFrames with multi-level column indices for easy filtering and analysis.

## Quick Reference

```python
import labanalysis as laban

# Load marker trajectories (TRC)
markers = laban.read_trc("mocap_data.trc")
# Returns DataFrame with MultiIndex: (OBJECT, DIMENSION, UNIT)

# Load motion data (MOT)
motion = laban.read_mot("inverse_kinematics.mot")
# Returns DataFrame with MultiIndex: (OBJECT, QUANTITY, DIMENSION)

# Access specific marker
marker_x = markers['HeadTop', 'X', 'mm']
marker_xyz = markers['HeadTop']  # All 3 dimensions

# Convert to labanalysis objects
head = laban.Point3D(
    data=markers['HeadTop'].values,
    index=markers.index.values,
    unit='mm'
)
```

## TRC Files (Marker Trajectories)

### File Format

TRC files contain 3D marker positions from motion capture systems:

```
PathFileType	4	(X/Y/Z)	mocap_data.trc
DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
100.00	100.00	1000	15	mm	100.00	1	1000
Frame#	Time	HeadTop		LASI		RASI		...
		X	Y	Z	X	Y	Z	X	Y	Z	...
1	0.01	1245.3	1678.9	234.5	345.6	789.1	123.4	...
2	0.02	1246.1	1679.2	234.8	345.8	789.3	123.6	...
```

### Loading TRC Files

```python
import labanalysis as laban

# Load TRC file
markers = laban.read_trc("walking_trial.trc")

# Inspect structure
print(markers.head())
print(f"Markers: {markers.columns.get_level_values('OBJECT').unique()}")
print(f"Sampling rate: {1 / (markers.index[1] - markers.index[0]):.0f} Hz")

# Output:
# Markers: Index(['HeadTop', 'LASI', 'RASI', 'C7', 'LKNEM', ...], dtype='object')
# Sampling rate: 100 Hz
```

### Working with TRC Data

#### Access Individual Markers

```python
# Get all coordinates for one marker
head_top = markers['HeadTop']
print(head_top.head())
#          X        Y        Z
# Time                        
# 0.01  1245.3  1678.9   234.5
# 0.02  1246.1  1679.2   234.8

# Get single dimension
head_x = markers['HeadTop', 'X', 'mm']

# Get multiple markers
pelvis_markers = markers[['LASI', 'RASI', 'LPSI', 'RPSI']]
```

#### Convert to Point3D Objects

```python
# Create Point3D from TRC marker
head = laban.Point3D(
    data=markers['HeadTop'].values,
    index=markers.index.values,
    columns=['X', 'Y', 'Z'],
    unit='mm'
)

# Now you can use all Point3D methods
head_filtered = laban.butterworth_filt(head.data, freq=100, cut=6, order=4)
head.data = head_filtered
```

#### Create WholeBody from TRC

```python
# Extract marker names from TRC
marker_names = markers.columns.get_level_values('OBJECT').unique()

# Create Point3D for each marker
marker_dict = {}
for name in marker_names:
    marker_dict[name] = laban.Point3D(
        data=markers[name].values,
        index=markers.index.values,
        columns=['X', 'Y', 'Z'],
        unit='mm'
    )

# Create WholeBody (map TRC names to WholeBody parameters)
body = laban.WholeBody(
    left_asis=marker_dict['LASI'],
    right_asis=marker_dict['RASI'],
    left_psis=marker_dict['LPSI'],
    right_psis=marker_dict['RPSI'],
    left_knee_lateral=marker_dict['LKNEL'],
    left_knee_medial=marker_dict['LKNEM'],
    # ... map other markers
)

# Access joint angles
knee_angle = body.left_knee_flexionextension
```

## MOT Files (Motion Data)

### File Format

MOT files contain joint angles, forces, or other time-varying data:

```
Coordinates
version=1
nRows=1000
nColumns=37
inDegrees=yes
endheader
time	pelvis_tilt	pelvis_list	pelvis_rotation	hip_flexion_r	...
0.00	-5.234	2.456	1.234	23.456	...
0.01	-5.189	2.478	1.245	23.512	...
```

### Loading MOT Files

```python
import labanalysis as laban

# Load MOT file (inverse kinematics results)
ik_results = laban.read_mot("inverse_kinematics.mot")

# Inspect structure
print(ik_results.columns.get_level_values('OBJECT').unique())
# Output: Index(['pelvis', 'hip', 'knee', 'ankle', ...], dtype='object')

print(ik_results.columns.get_level_values('QUANTITY').unique())
# Output: Index(['ORIGIN', 'TILT', 'LIST', 'ROTATION', 'FLEXION', ...], dtype='object')
```

### Working with MOT Data

#### Access Joint Angles

```python
# Get specific joint angle
hip_flexion_r = ik_results['hip', 'FLEXION', 'R']
knee_angle_r = ik_results['knee', 'ANGLE', 'R']

# Get all coordinates for one joint
hip_angles = ik_results['hip']
print(hip_angles.columns)
# Output: MultiIndex([('FLEXION', 'R'), ('ADDUCTION', 'R'), ('ROTATION', 'R')], ...)

# Convert to Signal1D
knee_flex = laban.Signal1D(
    data=knee_angle_r.values,
    index=knee_angle_r.index.values,
    unit='°'
)
```

#### Extract Pelvis Kinematics

```python
# Pelvis position and orientation
pelvis_tx = ik_results['pelvis', 'ORIGIN', 'X']  # Anteroposterior position
pelvis_ty = ik_results['pelvis', 'ORIGIN', 'Y']  # Vertical position
pelvis_tz = ik_results['pelvis', 'ORIGIN', 'Z']  # Mediolateral position

pelvis_tilt = ik_results['pelvis', 'TILT', 'X']      # Sagittal tilt
pelvis_obliquity = ik_results['pelvis', 'LIST', 'Y']  # Frontal tilt
pelvis_rotation = ik_results['pelvis', 'ROTATION', 'Z']  # Transverse rotation

# Combine into Signal3D
pelvis_position = laban.Signal3D(
    data=np.column_stack([pelvis_tx, pelvis_ty, pelvis_tz]),
    index=pelvis_tx.index.values,
    columns=['X', 'Y', 'Z'],
    unit='m'
)
```

## STO Files (Storage/Simulation Results)

STO files have the same format as MOT files and can be loaded with `read_mot()`:

```python
# Load muscle forces from simulation
forces = laban.read_mot("muscle_forces.sto")

# Access specific muscle
gastrocnemius_force = forces['gastroc', 'FORCE', '']

# Load muscle activations
activations = laban.read_mot("muscle_activations.sto")
gastroc_activation = activations['gastroc', 'ACTIVATION', '']
```

## Complete Workflow Example

### Inverse Kinematics Analysis

```python
import labanalysis as laban
import numpy as np

# 1. Load marker data
markers = laban.read_trc("walking.trc")

# 2. Load IK results
ik = laban.read_mot("walking_ik.mot")

# 3. Extract gait cycle events
# (Assuming you have GRF data or use a different method)
heel_strikes = [100, 250, 400, 550, 700]  # Frame indices

# 4. Extract joint angles for one gait cycle
start_frame = heel_strikes[0]
end_frame = heel_strikes[1]

hip_flex = ik.loc[start_frame:end_frame, ('hip', 'FLEXION', 'R')]
knee_flex = ik.loc[start_frame:end_frame, ('knee', 'ANGLE', 'R')]
ankle_flex = ik.loc[start_frame:end_frame, ('ankle', 'ANGLE', 'R')]

# 5. Calculate range of motion
rom = {
    'hip': hip_flex.max() - hip_flex.min(),
    'knee': knee_flex.max() - knee_flex.min(),
    'ankle': ankle_flex.max() - ankle_flex.min()
}

print("Gait Cycle ROM:")
for joint, value in rom.items():
    print(f"  {joint.capitalize()}: {value:.1f}°")

# Output:
# Gait Cycle ROM:
#   Hip: 42.3°
#   Knee: 58.7°
#   Ankle: 28.4°
```

### Marker Trajectory Smoothing

```python
import labanalysis as laban

# Load markers
markers = laban.read_trc("noisy_trial.trc")
freq = 100  # Hz

# Convert to Point3D and filter
marker_names = markers.columns.get_level_values('OBJECT').unique()

smoothed_markers = {}
for name in marker_names:
    # Create Point3D
    marker = laban.Point3D(
        data=markers[name].values,
        index=markers.index.values,
        columns=['X', 'Y', 'Z'],
        unit='mm'
    )
    
    # Apply low-pass filter (6 Hz cutoff)
    filtered_data = laban.butterworth_filt(
        marker.data,
        freq=freq,
        cut=6,
        order=4,
        filt_type='low'
    )
    
    marker.data = filtered_data
    smoothed_markers[name] = marker

# Export smoothed markers back to TRC
# (requires write_trc function)
```

## File Specification Reference

### TRC Format Details

- **Line 1**: File type and version
- **Line 2**: Metadata (DataRate, CameraRate, NumFrames, NumMarkers, Units)
- **Line 3**: Data rate values
- **Line 4**: Marker names (header row 1)
- **Line 5**: Dimension labels (X, Y, Z repeated)
- **Line 6+**: Data rows (Frame#, Time, marker coordinates)

**Units**: Typically `mm` for positions, but can be `m` or `cm`

### MOT/STO Format Details

- **Header section**: Metadata ending with `endheader`
- **Column headers**: Time + variable names (format: `object_quantity_dimension`)
- **Data rows**: Numeric values

**Common quantities**:
- `origin`: Position coordinates (x, y, z)
- `tilt`, `list`, `rotation`: Euler angles
- `flexion`, `adduction`, `rotation`: Joint angles
- `force`, `moment`: Kinetic quantities
- `activation`: Muscle activation level (0-1)

## Troubleshooting

### Issue: "FileNotFoundError"

```python
# Ensure file path is correct
import os
file_path = "data/walking.trc"
assert os.path.exists(file_path), f"File not found: {file_path}"

markers = laban.read_trc(file_path)
```

### Issue: "Incorrect file extension"

```python
# Ensure file has correct extension
assert file_path.endswith('.trc'), "File must have .trc extension"
```

### Issue: "MultiIndex column access error"

```python
# Access MultiIndex columns correctly
# WRONG: markers['HeadTop']['X']
# RIGHT: markers['HeadTop', 'X']  or  markers['HeadTop']['X', 'mm']

# Get marker with all dimensions
head = markers['HeadTop']  # Returns all X, Y, Z columns

# Get specific dimension
head_x = markers['HeadTop', 'X', 'mm']  # Specific dimension
```

### Issue: "MOT file missing 'endheader'"

Old MOT files may not have the `endheader` line. Update the file or modify the reader:

```python
# If file doesn't have 'endheader', manually parse
import pandas as pd

with open("old_file.mot", "r") as f:
    lines = f.readlines()

# Skip header lines manually
data_start = 7  # Adjust based on your file
df = pd.read_csv("old_file.mot", sep='\t', skiprows=data_start)
```

### Issue: "Misaligned columns in TRC"

Ensure the TRC file uses tabs (`\t`) as delimiters, not spaces. OpenSim exports should use tabs by default.

## Performance Tips

### Large TRC Files

For very large TRC files (>100 MB):

```python
# Load only specific time range
markers = laban.read_trc("large_file.trc")
markers_subset = markers.loc[10.0:20.0]  # 10-20 seconds

# Or downsample
markers_downsampled = markers.iloc[::10]  # Every 10th frame
```

### Memory-Efficient Processing

Process markers one at a time for large datasets:

```python
markers = laban.read_trc("huge_file.trc")
marker_names = markers.columns.get_level_values('OBJECT').unique()

results = {}
for name in marker_names:
    # Process one marker at a time
    marker_data = markers[name].values
    filtered = laban.butterworth_filt(marker_data, freq=100, cut=6, order=4)
    results[name] = filtered
    
    # Clear from memory
    del marker_data, filtered
```

## See Also

- [BTS Data Loading](bts-bioengineering.md) - Loading BTS TDF files
- [Data Export: OpenSim](../data-export/opensim-export.md) - Exporting to OpenSim formats
- [WholeBody Model](../biomechanics/whole-body-model.md) - Creating full-body models
- [Signal Processing: Filtering](../signal-processing/filtering.md) - Filtering marker data
- [API Reference: I/O Functions](../../api/io/read.md) - Complete I/O API

---

**Format Specifications:**
- OpenSim Documentation: https://simtk-confluence.stanford.edu/display/OpenSim/Marker+Data
- TRC Format: https://simtk-confluence.stanford.edu/display/OpenSim/TRC+(.trc)+Files
- MOT/STO Format: https://simtk-confluence.stanford.edu/display/OpenSim/Storage+Files
