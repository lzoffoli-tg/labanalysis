# File Format Specifications

Technical specifications for biomechanical data file formats supported by labanalysis.

## Overview

labanalysis supports multiple file formats from motion capture systems, force platforms, and biomechanical software. This document provides technical specifications for each format.

## BTS Bioengineering (.tdf)

### Format Description

**TDF** (Time Data File) is a proprietary binary format used by BTS Bioengineering systems.

**System**: BTS Smart, BTS Elite, BTS Vicon MX

**Data types**:
- 3D marker positions
- Force platform data (forces, moments, COP)
- EMG signals
- Analog channels

### File Structure

**Binary format** with multiple data blocks:

```
Header Block
├── File version
├── Sampling frequency
├── Number of frames
├── Number of markers
├── Marker labels
└── Channel configuration

Data Block (repeated per frame)
├── Frame number
├── Marker positions (X, Y, Z per marker)
├── Force platform data
└── Analog channels
```

### Reading with labanalysis

```python
import labanalysis as laban

# Read markers
body = laban.WholeBody.from_tdf_file(
    "trial.tdf",
    labels="LABEL"  # Label format (LABEL or DAVIS)
)

# Read force platform
fp = laban.ForcePlatform.from_tdf_file(
    "trial.tdf",
    fp_label="FP1"
)

# Read raw TDF data
from labanalysis.io import read_tdf
data_dict = read_tdf(
    "trial.tdf",
    labels=["marker1", "marker2"],
    force_platforms=["FP1"],
    emg_channels=["EMG1", "EMG2"]
)
```

### Coordinate System

**BTS Default**:
- X: Forward (direction of progression)
- Y: Vertical (upward)
- Z: Lateral (right)

**Units**:
- Position: millimeters (mm)
- Force: Newtons (N)
- Moment: Newton-meters (N·m)

**Note**: labanalysis automatically converts to SI units (meters).

### Label Formats

**LABEL format**:
```
LAnkLat  (Left Ankle Lateral)
RKneMed  (Right Knee Medial)
C7       (7th Cervical Vertebra)
```

**DAVIS format** (different naming convention):
```
LANK     (Left Ankle)
RKNE     (Right Knee)
```

## OpenSim Formats

### .trc (Track Row Column)

**ASCII text format** for marker trajectories.

**Structure**:

```
PathFileType	4	(X/Y/Z)	trial.trc
DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
100.00	100.00	500	20	m	100.00	1	500
Frame#	Time	marker1	marker2	...
		X1	Y1	Z1	X2	Y2	Z2	...
1	0.000	0.100	1.200	0.050	0.150	1.250	0.060
2	0.010	0.102	1.201	0.051	0.152	1.252	0.061
...
```

**Reading**:

```python
from labanalysis.io import read_opensim

# Read TRC file
markers = read_opensim("trial.trc", file_type="trc")

# Access specific marker
ankle = markers["left_ankle"]
```

**Writing**:

```python
from labanalysis.io import write_opensim

# Export to TRC
write_opensim(
    "output.trc",
    markers_dict,
    file_type="trc",
    data_rate=100
)
```

### .mot (Motion)

**ASCII text format** for generalized coordinates, forces, or generic time-series data.

**Structure**:

```
output.mot
version=1
nRows=500
nColumns=15
inDegrees=yes

endheader
time	hip_flexion_r	knee_angle_r	ankle_angle_r	...
0.000	20.5	5.2	-10.3	...
0.010	20.6	5.3	-10.2	...
...
```

**Reading**:

```python
from labanalysis.io import read_opensim

# Read MOT file
data = read_opensim("inverse_kinematics.mot", file_type="mot")

# Access columns
hip_angle = data["hip_flexion_r"]
```

### .sto (Storage)

**Similar to .mot** but with different header format.

**Used for**:
- Inverse kinematics results
- Inverse dynamics results
- Muscle forces
- Joint reactions

## C3D (Coordinate 3D)

**Binary format** widely used in biomechanics (standard format).

**System**: Vicon, Qualisys, Motion Analysis, OptiTrack

**Data types**:
- 3D marker trajectories
- Analog data (force platforms, EMG)
- Event markers
- Metadata

### File Structure

```
Header Section (512 bytes)
├── Parameter pointer
├── Data pointer
└── Analog samples per frame

Parameter Section
├── Groups (POINT, ANALOG, FORCE_PLATFORM, etc.)
├── Parameters within groups
└── Metadata

Data Section (per frame)
├── 3D point data (X, Y, Z, residual, camera mask)
└── Analog data (EMG, force, etc.)
```

### Reading with Python

```python
# Using ezc3d library
import ezc3d

c3d = ezc3d.c3d("trial.c3d")

# Get marker data
markers = c3d['data']['points']
labels = c3d['parameters']['POINT']['LABELS']['value']

# Get analog data
analog = c3d['data']['analogs']
analog_labels = c3d['parameters']['ANALOG']['LABELS']['value']

# Sampling rates
point_rate = c3d['parameters']['POINT']['RATE']['value'][0]
analog_rate = c3d['parameters']['ANALOG']['RATE']['value'][0]
```

**Note**: labanalysis does not currently have native C3D support. Use `ezc3d` to load, then convert to labanalysis objects.

### Coordinate System

**Standard C3D** (varies by manufacturer):
- Vicon: X forward, Y up, Z right
- Qualisys: X forward, Y left, Z up

**Units**: Usually millimeters (check POINT:UNITS parameter)

## EMG Formats

### BTS .emg

**Binary format** for EMG signals from BTS systems.

**Contains**:
- Raw EMG signals
- Sampling frequency
- Channel labels
- Calibration factors

**Reading**:

```python
from labanalysis.io import read_tdf

# EMG often embedded in TDF files
emg_signals = read_tdf(
    "trial.tdf",
    emg_channels=["biceps", "triceps", "quadriceps"]
)

biceps = emg_signals["biceps"]  # EMGSignal object
```

### .csv (Generic)

**ASCII text format** for EMG or any time-series data.

**Structure**:

```csv
time,emg_biceps,emg_triceps
0.000,0.0025,0.0018
0.001,0.0027,0.0019
0.002,0.0026,0.0020
...
```

**Reading**:

```python
import pandas as pd
import labanalysis as laban

df = pd.read_csv("emg_data.csv")

# Convert to EMGSignal
biceps = laban.EMGSignal(
    data=df['emg_biceps'].values,
    index=df['time'].values,
    columns=['biceps'],
    unit='V'
)
```

## Force Platform Formats

### AMTI (.txt)

**ASCII text format** from AMTI force platforms.

**Structure**:

```
Frame	Fx	Fy	Fz	Mx	My	Mz	COPx	COPy
1	-5.2	10.3	852.6	-2.1	3.4	0.5	0.102	0.215
2	-5.1	10.4	854.2	-2.0	3.5	0.4	0.103	0.216
...
```

**Reading**:

```python
import pandas as pd
import labanalysis as laban

df = pd.read_csv("force_data.txt", sep="\t")

# Create ForcePlatform
fp = laban.ForcePlatform(
    data=df[['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']].values,
    index=df.index / sampling_rate,  # Convert frame to time
    columns=[
        ("FORCE", "X"),
        ("FORCE", "Y"),
        ("FORCE", "Z"),
        ("MOMENT", "X"),
        ("MOMENT", "Y"),
        ("MOMENT", "Z"),
    ],
    unit='N'
)
```

### Kistler (.txt/.csv)

**ASCII text format** from Kistler force platforms.

**Similar to AMTI** but may have different column order or header format.

## Custom Formats

### Creating Custom Readers

```python
import numpy as np
import pandas as pd
import labanalysis as laban

def read_custom_format(filepath):
    """
    Read custom biomechanical data format.
    
    Parameters
    ----------
    filepath : str or Path
        Path to custom file
    
    Returns
    -------
    dict
        Dictionary of Signal objects
    """
    # Example: Read custom CSV with specific structure
    df = pd.read_csv(filepath)
    
    # Extract time vector
    time = df['time'].values
    
    # Create signals
    signals = {}
    
    for column in df.columns:
        if column == 'time':
            continue
        
        signal = laban.Signal1D(
            data=df[column].values,
            index=time,
            columns=[column],
            unit='m'  # Adjust based on data type
        )
        
        signals[column] = signal
    
    return signals

# Usage
data = read_custom_format("custom_data.csv")
ankle_signal = data['ankle_height']
```

## Data Export Formats

### Excel (.xlsx)

```python
import pandas as pd

# Create summary DataFrame
summary = pd.DataFrame({
    'Metric': ['Jump Height', 'Peak Force', 'Flight Time'],
    'Value': [0.42, 2450, 0.52],
    'Unit': ['m', 'N', 's']
})

# Export to Excel
with pd.ExcelWriter('results.xlsx') as writer:
    summary.to_excel(writer, sheet_name='Summary', index=False)
    timeseries_df.to_excel(writer, sheet_name='Time Series')
```

### CSV (.csv)

```python
# Export signal to CSV
ankle_angle.to_dataframe().to_csv('ankle_angle.csv', index=True)

# Export test results
results.to_dataframe().to_csv('jump_results.csv', index=False)
```

### JSON (.json)

```python
import json

# Export metrics as JSON
metrics = {
    'jump_height': 0.42,
    'peak_force': 2450,
    'flight_time': 0.52,
    'athlete_id': 'ATH001',
    'test_date': '2024-03-15'
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

## Format Conversion

### TDF to TRC

```python
import labanalysis as laban
from labanalysis.io import write_opensim

# Read BTS TDF
body = laban.WholeBody.from_tdf_file("trial.tdf")

# Extract all markers
markers = {
    label: body.get_point(label) 
    for label in body.labels
}

# Write to OpenSim TRC
write_opensim(
    "trial.trc",
    markers,
    file_type="trc",
    data_rate=100
)
```

### Force Platform to CSV

```python
import labanalysis as laban

# Read force platform
fp = laban.ForcePlatform.from_tdf_file("trial.tdf", fp_label="FP1")

# Export to CSV
fp.to_dataframe().to_csv("force_data.csv")
```

## Best Practices

### File Naming Conventions

```
# Recommended format
ATHLETE_TEST_CONDITION_DATE.ext

# Examples
JSmith_CMJ_Baseline_20240315.tdf
MJones_Gait_PostTreatment_20240316.tdf
```

### Directory Structure

```
project/
├── raw_data/
│   ├── athlete1/
│   │   ├── session1/
│   │   │   ├── trial001.tdf
│   │   │   ├── trial002.tdf
│   │   │   └── trial003.tdf
│   │   └── session2/
│   └── athlete2/
├── processed_data/
│   ├── athlete1/
│   │   └── session1/
│   │       ├── jump_results.csv
│   │       └── gait_analysis.csv
│   └── athlete2/
└── reports/
    ├── athlete1_summary.xlsx
    └── athlete2_summary.xlsx
```

### Metadata Management

```python
import json
from datetime import datetime

# Create metadata file
metadata = {
    'athlete_id': 'ATH001',
    'test_date': datetime.now().isoformat(),
    'test_type': 'CMJ',
    'body_mass': 75.5,
    'height': 1.82,
    'dominant_leg': 'right',
    'equipment': {
        'motion_capture': 'BTS Smart DX',
        'force_platform': 'BTS P-6000',
        'sampling_rate': 1000
    },
    'notes': 'Baseline assessment'
}

# Save alongside data
with open('trial_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

## Troubleshooting

### Common Issues

**Issue**: File encoding errors when reading text files.

**Solution**: Specify encoding explicitly.
```python
df = pd.read_csv("data.csv", encoding='utf-8')
```

**Issue**: Missing markers or channels in TDF files.

**Solution**: Check available labels first.
```python
from labanalysis.io import read_tdf

# List available labels
data = read_tdf("trial.tdf")
print("Available markers:", list(data.keys()))
```

**Issue**: Unit conversion errors.

**Solution**: Always verify units in file headers or documentation.
```python
# Check units
print(f"Marker unit: {marker.unit}")

# Convert if needed
marker_m = marker.to_unit('m')
```

## See Also

- [Data Loading Guide](../user-guide/data-loading/) - Practical loading examples
- [I/O API Reference](../api/io/) - Function documentation
- [OpenSim Export](../user-guide/data-export/opensim-export.md) - Export workflows

---

**Understand file formats for successful data import.** Check coordinate systems, units, and label conventions when working with new data sources.
