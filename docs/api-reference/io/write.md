# labanalysis.io.write

Data export functions for OpenSim and other formats.

**Source**: `src/labanalysis/io/write/`

## Overview

The `io.write` module provides functions for exporting labanalysis data to external formats:

- **write_trc()** - Export marker trajectories to OpenSim TRC format
- **write_mot()** - Export forces/moments to OpenSim MOT format

## Functions

### write_trc()

Export marker data to OpenSim TRC format.

```python
def write_trc(
    file_path: str,
    markers_df: pd.DataFrame
) -> None
```

**Parameters:**
- `file_path` (str): Output TRC file path
- `markers_df` (pd.DataFrame): Marker trajectories
  - Index: Time (seconds)
  - Columns: MultiIndex (marker_name, axis=['X','Y','Z'], unit='mm')

**Example:**
```python
import labanalysis as laban
from labanalysis.io.write import write_trc

# Load data
data = laban.read_tdf("gait.tdf", marker_keys=[".*"])
body = laban.WholeBody(**data)

# Convert to DataFrame
markers_df = body.to_dataframe(markers_only=True)

# Export to TRC
write_trc("gait_markers.trc", markers_df)
```

**Status:** Currently raises `NotImplementedError` (planned feature)

---

### write_mot()

Export force/moment data to OpenSim MOT format.

```python
def write_mot(
    file_path: str,
    data_df: pd.DataFrame
) -> None
```

**Parameters:**
- `file_path` (str): Output MOT file path
- `data_df` (pd.DataFrame): Force platform data
  - Index: Time (seconds)
  - Columns: MultiIndex (platform_name, data_type=['ORIGIN','FORCE','TORQUE'], axis=['X','Y','Z'])

**Example:**
```python
import labanalysis as laban
from labanalysis.io.write import write_mot

# Load data
data = laban.read_tdf("gait.tdf", forceplatform_keys=[".*"])
fp = data['left_foot_ground_reaction_force']

# Convert to DataFrame
grf_df = fp.to_dataframe()

# Export to MOT
write_mot("grf.mot", grf_df)
```

**Status:** Currently raises `NotImplementedError` (planned feature)

---

## Workarounds

Until `write_trc()` and `write_mot()` are implemented, use DataFrame export:

```python
import labanalysis as laban

# Load and convert
data = laban.read_tdf("trial.tdf", marker_keys=[".*"])
body = laban.WholeBody(**data)
markers_df = body.to_dataframe(markers_only=True)

# Export to CSV (intermediate format)
markers_df.to_csv("markers.csv")

# Or use pandas Excel export
markers_df.to_excel("markers.xlsx")
```

---

## See Also

- [Read Functions](read.md) - Data import
- [Data Export Guide](../../user-guide/data-export/opensim-export.md) - Complete workflow
- [DataFrame Export](../../user-guide/data-export/dataframes.md) - Alternative export methods

---

**Export labanalysis data to OpenSim and other formats.**
