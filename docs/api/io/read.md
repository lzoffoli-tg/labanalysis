# labanalysis.io.read

Data import functions for various file formats.

**Source**: `src/labanalysis/io/read/`

## Overview

The `io.read` module provides functions for loading biomechanical data from various formats:

- **read_tdf()** - BTS Bioengineering TDF files (markers, forces, EMG, events)
- **read_trc()** - OpenSim TRC files (marker trajectories)
- **read_mot()** - OpenSim MOT files (forces, joint angles)
- **read_emt()** - BTS EMG files
- **read_npz()** - IRCAM pressure mat NPZ files
- **BiostrengthProduct** - Biostrength device data readers

## Functions

### read_tdf()

Load BTS Bioengineering TDF files.

```python
def read_tdf(
    path: str,
    marker_keys: list[str] | None = None,
    forceplatform_keys: list[str] | None = None,
    emg_keys: list[str] | None = None
) -> dict
```

**Parameters:**
- `path` (str): Path to TDF file
- `marker_keys` (list, optional): Regex patterns for marker names to load (e.g., `["left_.*", "right_ankle"]`)
- `forceplatform_keys` (list, optional): Regex patterns for force platform names
- `emg_keys` (list, optional): Regex patterns for EMG channel names

**Returns:**
- `dict`: Dictionary with loaded data
  - Keys: Signal names (marker names, force platform names, EMG channels)
  - Values: `Point3D`, `ForcePlatform`, or `EMGSignal` objects

**Example:**
```python
import labanalysis as laban

# Load all markers and force platforms
data = laban.read_tdf("trial.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])

# Load specific markers
data = laban.read_tdf(
    "trial.tdf",
    marker_keys=["left_ankle", "right_ankle", ".*knee.*"],
    forceplatform_keys=["left_foot.*", "right_foot.*"]
)

# Load with EMG
data = laban.read_tdf(
    "trial.tdf",
    marker_keys=[".*"],
    emg_keys=[".*gastrocnemius.*", ".*tibialis.*"]
)

# Access loaded data
left_ankle = data['left_ankle']  # Point3D
fp = data['left_foot_ground_reaction_force']  # ForcePlatform
emg = data['left_gastrocnemius']  # EMGSignal
```

---

### read_trc()

Load OpenSim TRC marker trajectory files.

```python
def read_trc(file_path: str) -> pd.DataFrame
```

**Parameters:**
- `file_path` (str): Path to TRC file

**Returns:**
- `pd.DataFrame`: Marker trajectories
  - Index: Time (seconds)
  - Columns: MultiIndex (marker_name, axis, unit)

**Example:**
```python
import labanalysis as laban

# Load TRC file
markers = laban.read_trc("trial.trc")

# Access marker data
left_ankle_x = markers[('left_ankle', 'X', 'mm')]
```

---

### read_mot()

Load OpenSim MOT files (forces, joint angles, etc.).

```python
def read_mot(file_path: str) -> pd.DataFrame
```

**Parameters:**
- `file_path` (str): Path to MOT file

**Returns:**
- `pd.DataFrame`: Time-series data
  - Index: Time (seconds)
  - Columns: Data channels

**Example:**
```python
import labanalysis as laban

# Load MOT file
mot_data = laban.read_mot("grf.mot")

# Access columns
vertical_force = mot_data['ground_force_vy']
```

---

### read_emt()

Load BTS EMG files.

```python
def read_emt(file_path: str) -> dict
```

**Parameters:**
- `file_path` (str): Path to EMT file

**Returns:**
- `dict`: EMG channels
  - Keys: Channel names
  - Values: `EMGSignal` objects

---

### read_npz()

Load IRCAM pressure mat NPZ files.

```python
def read_npz(filename: str) -> dict
```

**Parameters:**
- `filename` (str): Path to NPZ file

**Returns:**
- `dict`: Pressure mat data

---

### BiostrengthProduct

Reader for Biostrength device data.

```python
from labanalysis.io.read.biostrength import BiostrengthProduct

reader = BiostrengthProduct()
data = reader.read(file_path)
```

---

## Common Workflows

### Load Complete Dataset

```python
import labanalysis as laban

# Load everything
data = laban.read_tdf(
    "trial.tdf",
    marker_keys=[".*"],
    forceplatform_keys=[".*"],
    emg_keys=[".*"]
)

# Check what was loaded
print(f"Loaded {len(data)} signals:")
for key in data.keys():
    print(f"  - {key}: {type(data[key]).__name__}")
```

### Selective Loading (Faster)

```python
import labanalysis as laban

# Load only what's needed (faster for large files)
data = laban.read_tdf(
    "gait.tdf",
    marker_keys=[".*ankle.*", ".*knee.*", ".*hip.*", "s2"],
    forceplatform_keys=["left_foot.*", "right_foot.*"]
)
```

---

## See Also

- [Write Functions](write.md) - Data export
- [Records](../records/records.md) - Data structures
- [User Guide: Data Loading](../../guides/data-loading/overview.md) - Complete guide

---

**Load biomechanical data from various file formats.**
