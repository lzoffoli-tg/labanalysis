# Data Loading

Guide to loading data from various laboratory equipment and file formats into labanalysis.

## Supported Formats

labanalysis supports multiple data formats from common biomechanical equipment:

| Format | Equipment | File Extension | Reader Function |
|--------|-----------|----------------|-----------------|
| **[BTS Bioengineering](bts-bioengineering.md)** | BTS motion capture, force platforms | `.tdf` | `from_tdf()` |
| **[OpenSim](opensim.md)** | OpenSim motion files | `.mot`, `.sto`, `.c3d` | `read_opensim()` |
| **[Biostrength](biostrength.md)** | Biodex strength testing | Various | `BiostrengthProduct` classes |
| **[IRCAM](ircam.md)** | IRCAM pressure mats | Custom | `read_ircam()` |
| **[Cosmed](cosmed.md)** | Cosmed metabolic systems | Custom | Metabolic readers |

## Quick Start

### Load BTS TDF File

Most common use case - loading BTS Bioengineering TDF files:

```python
import labanalysis as laban

# Load complete TDF file
record = laban.TimeseriesRecord.from_tdf("path/to/file.tdf")

# Access force platforms
fp1 = record['FP1']
print(f"Force platform 1: {fp1.force['Fz'].data.shape}")

# Access marker data
markers = record['MKRS']
print(f"Markers loaded: {len(markers)}")
```

[→ Complete BTS guide](bts-bioengineering.md)

### Load OpenSim Files

```python
from labanalysis.io import read_opensim

# Read MOT file
data = read_opensim("motion.mot")
print(f"Loaded {data.shape[1]} columns")
```

[→ Complete OpenSim guide](opensim.md)

### Load Biostrength Data

```python
from labanalysis.io.read import ChestPress

# Read chest press data
exercise = ChestPress.from_file("chest_press_test.txt")
print(f"Repetitions: {len(exercise.repetitions)}")
```

[→ Complete Biostrength guide](biostrength.md)

## General Loading Workflow

### 1. Import labanalysis

```python
import labanalysis as laban
```

### 2. Load Data

Use the appropriate reader for your file format:

```python
# For TDF files (most common)
record = laban.TimeseriesRecord.from_tdf("file.tdf")

# For OpenSim
from labanalysis.io import read_opensim
data = read_opensim("file.mot")

# For Biostrength
from labanalysis.io.read import LegPress
exercise = LegPress.from_file("file.txt")
```

### 3. Inspect Loaded Data

```python
# Check what was loaded
print(f"Loaded {len(record)} objects")
for key, value in record.items():
    print(f"  {key}: {type(value).__name__}")
```

### 4. Extract Specific Data

```python
# Extract force platform
fp = record['FP1']

# Extract markers
markers = record['MKRS']

# Extract specific signal
signal = markers['C7']  # C7 vertebra marker
```

## Working with Loaded Data

### Convert to pandas DataFrame

All loaded data can be converted to pandas DataFrames:

```python
# Convert entire record
df = record.to_dataframe()

# Convert specific signal
signal_df = fp.force['Fz'].to_dataframe()

# Save to Excel
df.to_excel("data.xlsx", index=False)
```

### Filter by Type

```python
# Get all force platforms
force_platforms = record.filter(laban.ForcePlatform)

# Get all 1D signals
signals_1d = record.filter(laban.Signal1D)

# Get all 3D points
points = record.filter(laban.Point3D)
```

### Access Signal Properties

```python
signal = fp.force['Fz']

print(f"Label: {signal.label}")
print(f"Unit: {signal.unit}")
print(f"Samples: {len(signal)}")
print(f"Sampling frequency: {signal.sampling_frequency:.1f} Hz")
print(f"Duration: {signal.index[-1] - signal.index[0]:.2f} s")
print(f"Range: {signal.data.min():.2f} to {signal.data.max():.2f} {signal.unit}")
```

## Common Loading Patterns

### Load and Immediately Filter

```python
# Load data
record = laban.TimeseriesRecord.from_tdf("file.tdf")

# Apply filtering to all signals
filtered_record = record.apply(
    lambda sig: laban.butterworth_filt(
        sig.data, 
        freq=1000, 
        cut=10, 
        order=4, 
        filt_type='low'
    )
)
```

### Load Specific Channels Only

For TDF files, you can load specific channels:

```python
# Load only force platform 1
record = laban.TimeseriesRecord.from_tdf(
    "file.tdf",
    channels=['FP1']  # Only load FP1
)
```

### Batch Loading

Load multiple files:

```python
import os
from pathlib import Path

# Get all TDF files in directory
tdf_files = Path("data/").glob("*.tdf")

# Load all files
records = {}
for file in tdf_files:
    records[file.stem] = laban.TimeseriesRecord.from_tdf(file)

print(f"Loaded {len(records)} files")
```

## Data Validation

### Check for Missing Data

```python
import numpy as np

signal = record['FP1'].force['Fz']

# Check for NaN values
has_missing = np.any(np.isnan(signal.data))
n_missing = np.sum(np.isnan(signal.data))

print(f"Missing data: {has_missing}")
print(f"Missing samples: {n_missing} / {len(signal)}")
```

### Verify Sampling Frequency

```python
# Check sampling frequency
fs = signal.sampling_frequency
print(f"Sampling frequency: {fs:.1f} Hz")

# Verify time vector is uniform
dt = np.diff(signal.index)
is_uniform = np.allclose(dt, dt[0])
print(f"Uniform sampling: {is_uniform}")
```

### Check Signal Range

```python
# Check if signal is in expected range
force_z = fp.force['Fz']

if force_z.data.min() < -50:
    print("Warning: Negative forces detected (check calibration)")

if force_z.data.max() > 5000:
    print("Warning: Very high forces detected")
```

## Troubleshooting

### File Not Found

```python
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution**: Use absolute paths or verify file location:

```python
import os
file_path = os.path.abspath("data/file.tdf")
print(f"Loading from: {file_path}")
record = laban.TimeseriesRecord.from_tdf(file_path)
```

### Unsupported File Format

```python
ValueError: Unsupported file format: .xyz
```

**Solution**: Check supported formats in this guide or convert your data to a supported format.

### Missing Channels

```python
KeyError: 'FP2'
```

**Solution**: Check what channels are available:

```python
print(f"Available channels: {list(record.keys())}")
```

[→ More troubleshooting](../../troubleshooting/data-loading-issues.md)

## Format-Specific Guides

Detailed guides for each supported format:

- **[BTS Bioengineering (TDF)](bts-bioengineering.md)** - Most commonly used format
- **[OpenSim (MOT/STO/C3D)](opensim.md)** - Motion analysis and simulation
- **[Biostrength](biostrength.md)** - Strength testing equipment
- **[IRCAM](ircam.md)** - Pressure mat data
- **[Cosmed](cosmed.md)** - Metabolic measurements

## See Also

- **[Signal Processing Guide](../signal-processing/README.md)** - Process loaded data
- **[API Reference: I/O Module](../../api/io/README.md)** - Complete I/O API
- **[Examples](../../examples/basic/load-and-plot.py)** - Loading examples

---

**Questions?** Check [troubleshooting](../../troubleshooting/data-loading-issues.md) or contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
