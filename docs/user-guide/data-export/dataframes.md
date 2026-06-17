# DataFrame Export

Convert labanalysis objects to pandas DataFrames for analysis, export, and integration with other tools.

## Overview

All labanalysis Record objects provide a `.to_dataframe()` method for converting data to pandas DataFrame format. This enables:

- **Data export** to CSV, Excel, Parquet, or other formats
- **Statistical analysis** with pandas/numpy
- **Integration** with other Python data tools
- **Custom processing** beyond labanalysis built-ins

## Quick Reference

```python
import labanalysis as laban

# Load data
data = laban.read_tdf("trial.tdf", marker_keys=[".*"])
body = laban.WholeBody(**data)

# Convert to DataFrame
df = body.to_dataframe()

# Export to CSV
df.to_csv("trial_data.csv")

# Export to Excel
df.to_excel("trial_data.xlsx")
```

---

## Basic Usage

### Single Signal

```python
import labanalysis as laban

# Create a signal
time = [0.0, 0.01, 0.02, 0.03]
data = [10.5, 11.2, 12.1, 11.8]
signal = laban.Signal1D(data=data, index=time, unit="N")

# Convert to DataFrame
df = signal.to_dataframe()
print(df)
```

**Output:**
```
        value
time         
0.00     10.5
0.01     11.2
0.02     12.1
0.03     11.8
```

### Multiple Signals

```python
import labanalysis as laban

# Create record with multiple signals
data = laban.read_tdf("trial.tdf", marker_keys=["left_ankle"])
body = laban.WholeBody(**data)

# Get ankle flexion angle
ankle_angle = body.left_ankle_flexionextension

# Convert to DataFrame
df = ankle_angle.to_dataframe()
print(df)
```

**Output:**
```
        left_ankle_flexionextension
time                               
0.00                          -5.2
0.01                          -5.1
0.02                          -4.9
...
```

---

## WholeBody DataFrame Export

### Export All Signals

```python
import labanalysis as laban

data = laban.read_tdf("gait.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])
body = laban.WholeBody(**data)

# Export everything
df = body.to_dataframe()

# DataFrame contains:
# - All marker positions (X, Y, Z for each)
# - All calculated angles
# - All anthropometric measurements
# - Force platform data
```

### Export Markers Only

```python
# Get only 3D marker positions
markers_df = body.to_dataframe(markers_only=True)

# MultiIndex columns:
# Level 0: marker names
# Level 1: ['X', 'Y', 'Z']
# Level 2: units ['m']
```

### Export Specific Signals

```python
# Select specific signals
signals = [
    'left_ankle_flexionextension',
    'right_ankle_flexionextension',
    'left_knee_flexionextension',
    'right_knee_flexionextension'
]

df = body.to_dataframe(signals=signals)
```

### Export with Custom Index Name

```python
# Rename time index
df = body.to_dataframe()
df.index.name = 'timestamp'
```

---

## Force Platform DataFrame

### Basic Export

```python
import labanalysis as laban

# Load force platform data
data = laban.read_tdf("jump.tdf", forceplatform_keys=[".*"])
fp = data['left_foot_ground_reaction_force']

# Convert to DataFrame
df = fp.to_dataframe()
```

**DataFrame structure:**
```
time	origin_X	origin_Y	origin_Z	force_X	force_Y	force_Z	torque_X	torque_Y	torque_Z
0.00	100.5		50.2		0.0			-15.2	5.1		850.3	2.1			-1.2		0.5
0.01	101.2		50.5		0.0			-16.3	5.3		860.1	2.3			-1.4		0.6
```

### Extract Specific Components

```python
# Get only vertical force
vertical_force = fp.force['Z'].to_numpy()

# Or using DataFrame
df = fp.to_dataframe()
vertical_force = df['force_Z'].values
```

---

## EMG Signal DataFrame

### Single EMG Channel

```python
import labanalysis as laban

# Load EMG data
data = laban.read_tdf("emg_trial.tdf", emg_keys=[".*gastrocnemius.*"])
emg = data['left_gastrocnemius']

# Convert to DataFrame
df = emg.to_dataframe()

# Columns: raw, processed (if available)
```

### Multiple EMG Channels

```python
# Load multiple EMG channels
data = laban.read_tdf(
    "emg_trial.tdf",
    emg_keys=[".*gastrocnemius.*", ".*soleus.*", ".*tibialis.*"]
)

# Combine into one DataFrame
import pandas as pd

dfs = []
for muscle, signal in data.items():
    if isinstance(signal, laban.EMGSignal):
        df = signal.to_dataframe()
        df.columns = [f"{muscle}_{col}" for col in df.columns]
        dfs.append(df)

combined_df = pd.concat(dfs, axis=1)
```

---

## Test Protocol DataFrames

### Jump Test Results

```python
import labanalysis as laban

# Load jump test
data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])
jump = laban.SingleJump(**data)

# Get summary metrics
summary_df = jump.output_metrics

print(summary_df)
```

**Output:**
```
   jump_height_m  flight_time_s  peak_force_N  ...
0          0.45          0.538          1850
```

### Gait Cycle Results

```python
import labanalysis as laban

# Load running trial
data = laban.read_tdf("running.tdf", marker_keys=[".*"])
running = laban.RunningExercise(algorithm='kinematics', **data)

# Get metrics for all cycles
all_cycles = []
for cycle in running.cycles:
    metrics = cycle.output_metrics
    all_cycles.append(metrics)

# Combine into one DataFrame
import pandas as pd
results_df = pd.concat(all_cycles, ignore_index=True)

print(results_df)
```

**Output:**
```
   type          side  init_s  end_s  footstrike_s  ...
0  RunningStep   left    0.00   0.75         0.35  ...
1  RunningStep   right   0.75   1.50         1.15  ...
2  RunningStep   left    1.50   2.25         1.95  ...
```

---

## Export Formats

### CSV Export

```python
# Basic CSV export
df = body.to_dataframe()
df.to_csv("data.csv")

# With custom options
df.to_csv(
    "data.csv",
    sep=';',              # Semicolon separator
    decimal=',',          # Comma as decimal
    index_label='time',   # Name for index column
    float_format='%.3f'   # 3 decimal places
)
```

### Excel Export

```python
# Single sheet
df = body.to_dataframe()
df.to_excel("data.xlsx", sheet_name='Trial1')

# Multiple sheets
with pd.ExcelWriter("results.xlsx") as writer:
    markers_df.to_excel(writer, sheet_name='Markers')
    angles_df.to_excel(writer, sheet_name='Angles')
    summary_df.to_excel(writer, sheet_name='Summary')
```

### Parquet Export (Efficient for Large Data)

```python
# Parquet: faster, smaller files
df = body.to_dataframe()
df.to_parquet("data.parquet", compression='snappy')

# Read back
import pandas as pd
df_loaded = pd.read_parquet("data.parquet")
```

### HDF5 Export

```python
# HDF5: good for hierarchical data
df = body.to_dataframe()
df.to_hdf("data.h5", key='trial1', mode='w')

# Read back
df_loaded = pd.read_hdf("data.h5", key='trial1')
```

---

## Practical Applications

### 1. Statistical Analysis

```python
import labanalysis as laban
import pandas as pd

# Load multiple trials
trials = []
for i in range(1, 11):
    data = laban.read_tdf(f"trial_{i:02d}.tdf", marker_keys=[".*"])
    jump = laban.SingleJump(**data)
    metrics = jump.output_metrics
    metrics['trial'] = i
    trials.append(metrics)

# Combine all trials
all_trials_df = pd.concat(trials, ignore_index=True)

# Statistical analysis
print("Mean jump height:", all_trials_df['jump_height_m'].mean())
print("Std jump height:", all_trials_df['jump_height_m'].std())
print("CV%:", all_trials_df['jump_height_m'].std() / all_trials_df['jump_height_m'].mean() * 100)

# Export statistics
stats = all_trials_df.describe()
stats.to_csv("jump_statistics.csv")
```

### 2. Time-Series Analysis

```python
import labanalysis as laban
import pandas as pd

# Load gait trial
data = laban.read_tdf("gait.tdf", marker_keys=[".*"])
body = laban.WholeBody(**data)

# Get ankle angle time series
ankle_df = body.left_ankle_flexionextension.to_dataframe()

# Pandas time-series operations
ankle_df_resampled = ankle_df.resample('10ms').mean()  # Downsample to 100 Hz
ankle_df_rolling = ankle_df.rolling(window=10).mean()  # Moving average

# Export
ankle_df_resampled.to_csv("ankle_100hz.csv")
```

### 3. Data Cleaning and Filtering

```python
import labanalysis as laban
import pandas as pd
import numpy as np

# Load data
data = laban.read_tdf("trial.tdf", marker_keys=[".*"])
body = laban.WholeBody(**data)

# Convert to DataFrame
df = body.to_dataframe(markers_only=True)

# Remove outliers (markers with Z < 0)
for col in df.columns:
    if col[1] == 'Z':  # Z coordinate
        df.loc[df[col] < 0, col] = np.nan

# Fill missing values
df_filled = df.fillna(method='linear')

# Export cleaned data
df_filled.to_csv("cleaned_data.csv")
```

### 4. Batch Processing

```python
from pathlib import Path
import labanalysis as laban
import pandas as pd

# Process all TDF files in directory
tdf_files = Path("raw_data/").glob("*.tdf")

all_results = []
for tdf_file in tdf_files:
    # Load
    data = laban.read_tdf(str(tdf_file), forceplatform_keys=[".*"])
    jump = laban.SingleJump(**data)
    
    # Get metrics
    metrics = jump.output_metrics
    metrics['filename'] = tdf_file.name
    all_results.append(metrics)

# Combine and export
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_excel("all_jumps_summary.xlsx", index=False)
print(f"Processed {len(all_results)} files")
```

### 5. Comparison Analysis

```python
import labanalysis as laban
import pandas as pd

# Load pre and post intervention trials
pre_data = laban.read_tdf("pre_intervention.tdf", forceplatform_keys=[".*"])
post_data = laban.read_tdf("post_intervention.tdf", forceplatform_keys=[".*"])

pre_jump = laban.SingleJump(**pre_data)
post_jump = laban.SingleJump(**post_data)

# Get metrics
pre_metrics = pre_jump.output_metrics
post_metrics = post_jump.output_metrics

# Add condition column
pre_metrics['condition'] = 'pre'
post_metrics['condition'] = 'post'

# Combine
comparison_df = pd.concat([pre_metrics, post_metrics], ignore_index=True)

# Calculate change
jump_height_change = (
    post_metrics['jump_height_m'].values[0] - 
    pre_metrics['jump_height_m'].values[0]
)
print(f"Jump height change: {jump_height_change:.3f} m")

# Export
comparison_df.to_csv("pre_post_comparison.csv", index=False)
```

---

## DataFrame Structure Details

### Signal1D DataFrame

```python
# Signal1D produces simple DataFrame
signal = laban.Signal1D(data=[1, 2, 3], index=[0, 1, 2], unit="m/s")
df = signal.to_dataframe()

# Structure:
#        value
# time        
# 0          1
# 1          2
# 2          3
```

### Signal3D DataFrame

```python
# Signal3D produces MultiIndex columns
signal = laban.Signal3D(
    data=[[1, 2, 3], [4, 5, 6]],
    index=[0, 1],
    unit="m"
)
df = signal.to_dataframe()

# Structure:
#        X    Y    Z
# time              
# 0      1    2    3
# 1      4    5    6
```

### Point3D DataFrame

```python
# Point3D (marker) produces MultiIndex
marker = laban.Point3D(
    data=[[100, 50, 1200], [101, 51, 1201]],
    index=[0.0, 0.01],
    unit="mm"
)
df = marker.to_dataframe()

# Structure (MultiIndex):
#        ('marker', 'X', 'mm')  ('marker', 'Y', 'mm')  ('marker', 'Z', 'mm')
# time                                                                       
# 0.00                    100                     50                   1200
# 0.01                    101                     51                   1201
```

---

## Performance Tips

### 1. Memory Efficiency

```python
# For large datasets, export in chunks
chunk_size = 10000
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk.to_csv(f"data_chunk_{i//chunk_size:03d}.csv")
```

### 2. Selective Export

```python
# Only export what you need
# Instead of:
df_all = body.to_dataframe()  # All 90 properties!

# Do:
df_angles = body.to_dataframe(signals=[
    'left_ankle_flexionextension',
    'left_knee_flexionextension',
    'left_hip_flexionextension'
])  # Only 3 signals
```

### 3. Compression

```python
# Use compression for large files
df.to_csv("data.csv.gz", compression='gzip')

# Or Parquet (built-in compression)
df.to_parquet("data.parquet", compression='snappy')
```

---

## Troubleshooting

### Issue: "MultiIndex columns are confusing"

**Problem**: DataFrame has complex MultiIndex columns

**Solution**: Flatten columns
```python
df = body.to_dataframe(markers_only=True)

# Flatten MultiIndex to single level
df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Now columns are: 'left_ankle_X_mm', 'left_ankle_Y_mm', etc.
```

### Issue: "Missing values (NaN) in export"

**Problem**: DataFrame contains NaN values

**Solution**: Handle missing data before export
```python
# Fill with interpolation
df_filled = df.fillna(method='linear')

# Or drop rows with NaN
df_clean = df.dropna()

# Or replace with specific value
df_filled = df.fillna(0)
```

### Issue: "Excel file too large"

**Problem**: Excel has row limit (1,048,576 rows)

**Solution**: Use CSV or Parquet instead
```python
# CSV (no row limit)
df.to_csv("large_data.csv")

# Or Parquet (efficient for large data)
df.to_parquet("large_data.parquet")
```

### Issue: "Index not preserved in CSV"

**Problem**: Time index lost when exporting to CSV

**Solution**: Explicitly include index
```python
# Include index in export
df.to_csv("data.csv", index=True)

# Or reset index to make it a column
df_reset = df.reset_index()
df_reset.to_csv("data.csv", index=False)
```

---

## See Also

- [pandas DataFrame Documentation](https://pandas.pydata.org/docs/reference/frame.html)
- [](opensim-export.md) - OpenSim TRC/MOT export
- [](reports.md) - Report generation
- [](../biomechanics/whole-body-model.md) - WholeBody properties

---

**Convert labanalysis data to pandas DataFrames for flexible analysis and export.**
