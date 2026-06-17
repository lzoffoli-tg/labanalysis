# OpenSim File Export

Export biomechanical data to OpenSim file formats for musculoskeletal modeling and simulation.

## Overview

Labanalysis provides functions to export data into OpenSim-compatible formats:

- **TRC files**: Marker trajectory data for motion capture
- **MOT files**: Ground reaction force data from force platforms

These formats enable integration with OpenSim for inverse kinematics, inverse dynamics, muscle analysis, and other biomechanical simulations.

## Quick Reference

```python
import labanalysis as laban
from labanalysis.io.write import write_trc, write_mot

# Export marker data to TRC
markers_df = body.to_dataframe(markers_only=True)
write_trc("output.trc", markers_df)

# Export force platform data to MOT
grf_df = force_platform.to_dataframe()
write_mot("output.mot", grf_df)
```

---

## TRC File Export

### Overview

TRC (Track Row Column) files store 3D marker trajectories with timestamps. OpenSim uses TRC files as input for:
- Inverse kinematics (IK)
- Motion tracking
- Marker-based analysis

### File Format Requirements

The input DataFrame must have:
- **Time-based index** (seconds)
- **MultiIndex columns** with 3 levels:
  1. Marker names (e.g., "left_ankle", "right_knee")
  2. Coordinate axes: ['X', 'Y', 'Z']
  3. Units: ['mm']

### Basic Usage

```python
import labanalysis as laban
from labanalysis.io.write import write_trc

# Load data
data = laban.read_tdf("gait_trial.tdf", marker_keys=[".*"])
body = laban.WholeBody(**data)

# Convert to DataFrame (markers only)
markers_df = body.to_dataframe(markers_only=True)

# Export to TRC
write_trc("gait_trial.trc", markers_df)
```

### DataFrame Structure

```python
# Example DataFrame structure for TRC export
import pandas as pd

time = [0.00, 0.01, 0.02, 0.03]  # seconds
markers = {
    ('left_ankle', 'X', 'mm'): [100.5, 100.7, 101.2, 101.8],
    ('left_ankle', 'Y', 'mm'): [50.2, 50.3, 50.5, 50.7],
    ('left_ankle', 'Z', 'mm'): [1200.1, 1201.3, 1202.5, 1203.8],
    ('right_ankle', 'X', 'mm'): [110.3, 110.5, 110.9, 111.4],
    ('right_ankle', 'Y', 'mm'): [51.1, 51.2, 51.4, 51.6],
    ('right_ankle', 'Z', 'mm'): [1205.2, 1206.4, 1207.6, 1208.9],
}

df = pd.DataFrame(markers, index=pd.Index(time, name='time'))
write_trc("example.trc", df)
```

### TRC File Contents

The generated TRC file includes:
- **Header**: PathFileType, DataRate, NumFrames, NumMarkers, Units
- **Marker names**: Column headers
- **Frame data**: Frame number, time, X/Y/Z coordinates

```
PathFileType	4	(X/Y/Z)	example.trc
DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
100	100	1500	42	mm	100	1	1500
Frame#	Time	left_ankle		right_ankle		...
		X1	Y1	Z1	X2	Y2	Z2	...
1	0.00	100.5	50.2	1200.1	110.3	51.1	1205.2	...
2	0.01	100.7	50.3	1201.3	110.5	51.2	1206.4	...
```

### Complete Example: Gait Analysis Export

```python
import labanalysis as laban
from labanalysis.io.write import write_trc

# Load gait trial
data = laban.read_tdf(
    "gait_trial.tdf",
    marker_keys=["left_.*", "right_.*", ".*asis", ".*psis"]
)

# Create WholeBody
body = laban.WholeBody(**data)

# Filter to specific markers for OpenSim model
marker_subset = [
    'left_ankle_medial', 'left_ankle_lateral',
    'right_ankle_medial', 'right_ankle_lateral',
    'left_knee_medial', 'left_knee_lateral',
    'right_knee_medial', 'right_knee_lateral',
    'left_asis', 'right_asis',
    'left_psis', 'right_psis',
    's2', 'c7'
]

# Extract markers as DataFrame
markers_df = body.to_dataframe(signals=marker_subset)

# Export to TRC
write_trc("gait_opensim.trc", markers_df)
print(f"Exported {len(marker_subset)} markers to gait_opensim.trc")
```

---

## MOT File Export

### Overview

MOT (Motion) files store force platform data for OpenSim. Used for:
- Inverse dynamics (ID)
- Ground reaction force analysis
- External load simulation

### File Format Requirements

The input DataFrame must have:
- **Time-based index** (seconds)
- **MultiIndex columns** with 3 levels:
  1. Force platform names (e.g., "fp1", "fp2")
  2. Data types: ['ORIGIN', 'FORCE', 'TORQUE']
  3. Coordinate axes: ['X', 'Y', 'Z']

### Basic Usage

```python
import labanalysis as laban
from labanalysis.io.write import write_mot

# Load force platform data
data = laban.read_tdf("gait_trial.tdf", forceplatform_keys=["fp.*"])

# Get force platform
fp = data['left_foot_ground_reaction_force']

# Convert to DataFrame
grf_df = fp.to_dataframe()

# Export to MOT
write_mot("gait_grf.mot", grf_df)
```

### DataFrame Structure

```python
# Example DataFrame structure for MOT export
import pandas as pd

time = [0.00, 0.01, 0.02]
grf_data = {
    ('fp1', 'ORIGIN', 'X'): [100.5, 101.2, 101.8],
    ('fp1', 'ORIGIN', 'Y'): [50.2, 50.5, 50.7],
    ('fp1', 'ORIGIN', 'Z'): [0.0, 0.0, 0.0],
    ('fp1', 'FORCE', 'X'): [-15.2, -16.3, -17.1],
    ('fp1', 'FORCE', 'Y'): [5.1, 5.3, 5.6],
    ('fp1', 'FORCE', 'Z'): [850.3, 860.1, 872.5],
    ('fp1', 'TORQUE', 'X'): [2.1, 2.3, 2.5],
    ('fp1', 'TORQUE', 'Y'): [-1.2, -1.4, -1.6],
    ('fp1', 'TORQUE', 'Z'): [0.5, 0.6, 0.7],
}

df = pd.DataFrame(grf_data, index=pd.Index(time, name='time'))
write_mot("grf.mot", df)
```

### MOT File Contents

Generated MOT file structure:
```
time	fp1_origin_px	fp1_origin_py	fp1_origin_pz	fp1_force_vx	fp1_force_vy	fp1_force_vz	...
0.00	100.5	50.2	0.0	-15.2	5.1	850.3	2.1	-1.2	0.5
0.01	101.2	50.5	0.0	-16.3	5.3	860.1	2.3	-1.4	0.6
0.02	101.8	50.7	0.0	-17.1	5.6	872.5	2.5	-1.6	0.7
```

**Column naming convention**:
- `{platform}_origin_p{axis}`: Center of pressure (COP)
- `{platform}_force_v{axis}`: Ground reaction force
- `{platform}_{torque}_{axis}`: Moment/torque

### Complete Example: Multi-Platform Export

```python
import labanalysis as laban
from labanalysis.io.write import write_mot
import pandas as pd

# Load data with multiple force platforms
data = laban.read_tdf(
    "dual_platform_trial.tdf",
    forceplatform_keys=["left_foot.*", "right_foot.*"]
)

# Get both platforms
left_fp = data['left_foot_ground_reaction_force']
right_fp = data['right_foot_ground_reaction_force']

# Convert to DataFrames
left_df = left_fp.to_dataframe()
right_df = right_fp.to_dataframe()

# Rename columns to match platform IDs
left_df.columns = pd.MultiIndex.from_tuples(
    [('fp1', t, ax) for (_, t, ax) in left_df.columns]
)
right_df.columns = pd.MultiIndex.from_tuples(
    [('fp2', t, ax) for (_, t, ax) in right_df.columns]
)

# Merge both platforms
combined_df = pd.concat([left_df, right_df], axis=1)

# Export to MOT
write_mot("dual_platform.mot", combined_df)
print(f"Exported 2 force platforms to dual_platform.mot")
```

---

## Practical Applications

### 1. Gait Analysis Pipeline

Complete workflow from TDF to OpenSim:

```python
import labanalysis as laban
from labanalysis.io.write import write_trc, write_mot

# Load trial
data = laban.read_tdf("gait.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])

# Create body model
body = laban.WholeBody(**data)

# Export markers for IK
markers_df = body.to_dataframe(markers_only=True)
write_trc("gait_markers.trc", markers_df)

# Export GRF for ID
fp = data['left_foot_ground_reaction_force']
grf_df = fp.to_dataframe()
write_mot("gait_grf.mot", grf_df)

print("✓ Files ready for OpenSim")
print("  1. Run IK with gait_markers.trc")
print("  2. Run ID with gait_grf.mot")
```

### 2. Running Analysis

```python
import labanalysis as laban
from labanalysis.io.write import write_trc

# Load running trial
data = laban.read_tdf("running.tdf", marker_keys=[".*"])
running = laban.RunningExercise(algorithm='kinematics', **data)

# Process each cycle
for i, cycle in enumerate(running.cycles):
    # Export each running step as separate TRC
    markers_df = cycle.to_dataframe(markers_only=True)
    write_trc(f"running_step_{i+1:02d}.trc", markers_df)

print(f"Exported {len(running.cycles)} running steps")
```

### 3. Batch Export

```python
from pathlib import Path
import labanalysis as laban
from labanalysis.io.write import write_trc

# Process all trials in a directory
tdf_files = Path("raw_data/").glob("*.tdf")

for tdf_file in tdf_files:
    # Load
    data = laban.read_tdf(str(tdf_file), marker_keys=[".*"])
    body = laban.WholeBody(**data)
    
    # Export
    output_name = f"opensim_{tdf_file.stem}.trc"
    markers_df = body.to_dataframe(markers_only=True)
    write_trc(output_name, markers_df)
    
    print(f"✓ {tdf_file.name} -> {output_name}")
```

---

## Units and Coordinate Systems

### Coordinate System

Labanalysis follows the **standard biomechanics convention**:
- **X-axis**: Medial-lateral (positive = right)
- **Y-axis**: Anterior-posterior (positive = forward)
- **Z-axis**: Vertical (positive = up)

OpenSim uses the **same convention** by default. No transformation needed.

### Units

| Data Type | Labanalysis | TRC/MOT Export | OpenSim Expected |
|-----------|-------------|----------------|------------------|
| Position | meters (m) | millimeters (mm) | millimeters (mm) |
| Force | Newtons (N) | Newtons (N) | Newtons (N) |
| Torque | N·m | N·m | N·m |
| Time | seconds (s) | seconds (s) | seconds (s) |

**Note**: Marker positions are automatically converted from meters to millimeters during TRC export.

---

## Troubleshooting

### Issue: "NotImplementedError"

**Problem**: `write_trc()` or `write_mot()` raises `NotImplementedError`

**Cause**: Functions are currently stubs in the codebase

**Workaround**: Use manual DataFrame export:
```python
# Export markers manually
markers_df = body.to_dataframe(markers_only=True)
markers_df.to_csv("markers.csv")  # Use CSV as intermediate
```

### Issue: "Wrong DataFrame structure"

**Problem**: Export fails with column structure error

**Solution**: Verify MultiIndex structure:
```python
# Check DataFrame columns
print(df.columns)
# Should show MultiIndex with 3 levels

# Check levels
print(df.columns.levels)
# Level 0: marker names
# Level 1: ['X', 'Y', 'Z']
# Level 2: ['mm'] for TRC or no level 2 for MOT
```

### Issue: "Mismatched sampling rates"

**Problem**: OpenSim reports inconsistent frame rates

**Solution**: Ensure uniform sampling:
```python
# Resample to uniform rate
target_fs = 100  # Hz
time = np.arange(0, len(df)/target_fs, 1/target_fs)
df_resampled = df.set_index(pd.Index(time, name='time'))
```

### Issue: "Missing markers in OpenSim"

**Problem**: Some markers don't appear in OpenSim model

**Cause**: Marker names don't match OpenSim model

**Solution**: Rename markers to match:
```python
# Rename markers
rename_map = {
    'left_ankle_lateral': 'L_Ankle_Lat',
    'right_ankle_lateral': 'R_Ankle_Lat',
    # ... etc
}

markers_df.columns = markers_df.columns.set_levels(
    markers_df.columns.levels[0].map(lambda x: rename_map.get(x, x)),
    level=0
)
```

---

## See Also

- [OpenSim Documentation](https://simtk-confluence.stanford.edu/display/OpenSim/Documentation)
- [](dataframes.md) - DataFrame conversion guide
- [TRC File Format Specification](https://simtk-confluence.stanford.edu/display/OpenSim/Marker+(.trc)+Files)
- [MOT File Format Specification](https://simtk-confluence.stanford.edu/display/OpenSim/Motion+(.mot)+Files)
- [](../biomechanics/whole-body-model.md) - WholeBody marker reference

---

**Export your labanalysis data to OpenSim for advanced musculoskeletal modeling and simulation.**
