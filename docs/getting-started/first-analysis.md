# Your First Analysis

Complete walkthrough of analyzing real biomechanical data using labanalysis. This tutorial takes you from raw data to final results step-by-step.

## What You'll Learn

By the end of this tutorial, you'll be able to:

- Load data from TDF files (BTS Bioengineering format)
- Process signals with filtering
- Detect events in force data
- Visualize results
- Export data for further analysis

## Prerequisites

- labanalysis installed ([installation guide](installation.md))
- Basic Python knowledge
- A TDF file with force platform data (or use the example below)

## Scenario

You've collected force platform data during a balance test. The participant stood on the force platform for 30 seconds, and you want to:

1. Load the vertical ground reaction force (GRF)
2. Filter the signal to remove noise
3. Calculate center of pressure (COP) movement
4. Compute balance metrics
5. Export results

## Step 1: Import and Setup

```python
import labanalysis as laban
import numpy as np
import matplotlib.pyplot as plt

# Create participant information
participant = laban.Participant(
    name="John",
    surname="Doe",
    gender="M",
    height=1.75,  # meters
    weight=70,    # kg
    age=30
)

print(f"Participant: {participant.surname}, {participant.name}")
print(f"BMI: {participant.bmi:.1f} kg/m²")
print(f"Max HR: {participant.max_heart_rate:.0f} bpm")
```

**Output:**
```
Participant: Doe, John
BMI: 22.9 kg/m²
Max HR: 190 bpm
```

## Step 2: Load Data from TDF File

```python
# Load force platform data
record = laban.TimeseriesRecord.from_tdf("path/to/balance_test.tdf")

# Inspect what was loaded
print(f"Loaded {len(record)} objects:")
for key, value in record.items():
    print(f"  - {key}: {type(value).__name__}")
```

**Expected output:**
```
Loaded 2 objects:
  - FP1: ForcePlatform
  - MKRS: Record
```

## Step 3: Extract Force Platform Data

```python
# Get first force platform
fp = record['FP1']

# Access force components
force_x = fp.force['Fx']  # Anterior-posterior
force_y = fp.force['Fy']  # Medial-lateral  
force_z = fp.force['Fz']  # Vertical

print(f"Force platform at: {fp.origin} m")
print(f"Sampling frequency: {force_z.sampling_frequency:.1f} Hz")
print(f"Duration: {force_z.index[-1]:.1f} seconds")
print(f"Vertical force range: {force_z.data.min():.1f} to {force_z.data.max():.1f} N")
```

**Expected output:**
```
Force platform at: [0. 0. 0.] m
Sampling frequency: 1000.0 Hz
Duration: 30.0 seconds
Vertical force range: 650.3 to 720.5 N
```

## Step 4: Filter the Signal

Remove high-frequency noise using a low-pass Butterworth filter:

```python
# Filter vertical force (10 Hz cutoff)
fz_filtered = laban.butterworth_filt(
    signal=force_z.data,
    freq=force_z.sampling_frequency,
    cut=10,           # 10 Hz cutoff frequency
    order=4,          # 4th order filter
    filt_type='low'   # Low-pass
)

# Create filtered Signal1D object
force_z_filtered = laban.Signal1D(
    data=fz_filtered,
    index=force_z.index,
    label='Fz_filtered',
    unit='N'
)

print(f"Signal filtered: {len(force_z_filtered)} samples")
```

**Output:**
```
Signal filtered: 30000 samples
```

## Step 5: Calculate Center of Pressure (COP)

```python
# Extract moment components (needed for COP calculation)
moment_x = fp.torque['Mx']
moment_y = fp.torque['My']

# Calculate COP position
# COPx = -My / Fz
# COPy = Mx / Fz

cop_x = -moment_y.data / force_z_filtered
cop_y = moment_x.data / force_z_filtered

# Create COP signals
cop_x_signal = laban.Signal1D(
    data=cop_x,
    index=force_z.index,
    label='COPx',
    unit='m'
)

cop_y_signal = laban.Signal1D(
    data=cop_y,
    index=force_z.index,
    label='COPy',
    unit='m'
)

print(f"COP range X: {cop_x.min()*100:.1f} to {cop_x.max()*100:.1f} cm")
print(f"COP range Y: {cop_y.min()*100:.1f} to {cop_y.max()*100:.1f} cm")
```

**Expected output:**
```
COP range X: -2.3 to 3.1 cm
COP range Y: -1.8 to 2.5 cm
```

## Step 6: Calculate Balance Metrics

```python
# Calculate standard deviation (measure of sway)
cop_x_std = np.std(cop_x) * 100  # Convert to cm
cop_y_std = np.std(cop_y) * 100

# Calculate total COP path length
cop_velocity_x = laban.winter_derivative1(cop_x, freq=force_z.sampling_frequency)
cop_velocity_y = laban.winter_derivative1(cop_y, freq=force_z.sampling_frequency)
cop_speed = np.sqrt(cop_velocity_x**2 + cop_velocity_y**2)
total_path_length = np.sum(cop_speed) * (1 / force_z.sampling_frequency)

# Calculate 95% confidence ellipse area
from labanalysis.modelling.ols import Ellipse
ellipse = Ellipse()
ellipse.fit(cop_x, cop_y)
ellipse_area = ellipse.area * (100**2)  # Convert m² to cm²

# Summary
print("Balance Metrics:")
print(f"  COP sway (SD):")
print(f"    - X direction: {cop_x_std:.2f} cm")
print(f"    - Y direction: {cop_y_std:.2f} cm")
print(f"  Total path length: {total_path_length:.2f} m")
print(f"  95% ellipse area: {ellipse_area:.1f} cm²")
```

**Expected output:**
```
Balance Metrics:
  COP sway (SD):
    - X direction: 0.85 cm
    - Y direction: 0.62 cm
  Total path length: 45.32 m
  95% ellipse area: 12.5 cm²
```

## Step 7: Visualize Results

```python
import plotly.graph_objects as go

# Create figure with subplots
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Vertical Force', 'COP Trajectory', 'COP X over Time', 'COP Y over Time')
)

# Plot 1: Vertical force (original vs filtered)
fig.add_trace(
    go.Scatter(x=force_z.index, y=force_z.data, name='Original', opacity=0.5),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=force_z.index, y=fz_filtered, name='Filtered'),
    row=1, col=1
)

# Plot 2: COP trajectory
fig.add_trace(
    go.Scatter(x=cop_x*100, y=cop_y*100, mode='lines', name='COP Path'),
    row=1, col=2
)

# Plot 3: COP X over time
fig.add_trace(
    go.Scatter(x=cop_x_signal.index, y=cop_x*100, name='COP X'),
    row=2, col=1
)

# Plot 4: COP Y over time
fig.add_trace(
    go.Scatter(x=cop_y_signal.index, y=cop_y*100, name='COP Y'),
    row=2, col=2
)

# Update layout
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="X (cm)", row=1, col=2)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=2)

fig.update_yaxes(title_text="Force (N)", row=1, col=1)
fig.update_yaxes(title_text="Y (cm)", row=1, col=2)
fig.update_yaxes(title_text="COP X (cm)", row=2, col=1)
fig.update_yaxes(title_text="COP Y (cm)", row=2, col=2)

fig.update_layout(height=800, showlegend=True, title_text="Balance Test Analysis")
fig.show()
```

This creates an interactive visualization with four panels showing force and COP data.

## Step 8: Export Results

### Export to pandas DataFrame

```python
# Create results DataFrame
import pandas as pd

results_df = pd.DataFrame({
    'Time': force_z.index,
    'Fz_original': force_z.data,
    'Fz_filtered': fz_filtered,
    'COPx_m': cop_x,
    'COPy_m': cop_y,
    'COPx_cm': cop_x * 100,
    'COPy_cm': cop_y * 100
})

# Save to Excel
results_df.to_excel('balance_test_results.xlsx', index=False)
print("Results exported to balance_test_results.xlsx")
```

### Export to CSV

```python
results_df.to_csv('balance_test_results.csv', index=False)
print("Results exported to balance_test_results.csv")
```

### Create Summary Report

```python
# Create summary dictionary
summary = {
    'participant': f"{participant.surname}, {participant.name}",
    'age': participant.age,
    'weight_kg': participant.weight,
    'height_m': participant.height,
    'bmi': participant.bmi,
    'test_duration_s': force_z.index[-1],
    'sampling_freq_hz': force_z.sampling_frequency,
    'mean_vertical_force_n': np.mean(fz_filtered),
    'cop_sway_x_cm': cop_x_std,
    'cop_sway_y_cm': cop_y_std,
    'total_path_length_m': total_path_length,
    'ellipse_area_cm2': ellipse_area
}

# Save as JSON
import json
with open('balance_test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary saved to balance_test_summary.json")
```

## Complete Script

Here's the complete analysis in one script:

```python
import labanalysis as laban
import numpy as np
import pandas as pd

# 1. Setup
participant = laban.Participant(
    name="John", surname="Doe", gender="M", 
    height=1.75, weight=70, age=30
)

# 2. Load data
record = laban.TimeseriesRecord.from_tdf("balance_test.tdf")
fp = record['FP1']
force_z = fp.force['Fz']

# 3. Filter
fz_filtered = laban.butterworth_filt(
    signal=force_z.data,
    freq=force_z.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# 4. Calculate COP
cop_x = -fp.torque['My'].data / fz_filtered
cop_y = fp.torque['Mx'].data / fz_filtered

# 5. Metrics
cop_x_std = np.std(cop_x) * 100
cop_y_std = np.std(cop_y) * 100

cop_velocity_x = laban.winter_derivative1(cop_x, freq=force_z.sampling_frequency)
cop_velocity_y = laban.winter_derivative1(cop_y, freq=force_z.sampling_frequency)
cop_speed = np.sqrt(cop_velocity_x**2 + cop_velocity_y**2)
total_path_length = np.sum(cop_speed) / force_z.sampling_frequency

# 6. Report
print("Balance Test Results:")
print(f"  Participant: {participant.surname}, {participant.name}")
print(f"  COP Sway: {cop_x_std:.2f} cm (X), {cop_y_std:.2f} cm (Y)")
print(f"  Path Length: {total_path_length:.2f} m")

# 7. Export
results_df = pd.DataFrame({
    'Time': force_z.index,
    'Fz': fz_filtered,
    'COPx_cm': cop_x * 100,
    'COPy_cm': cop_y * 100
})
results_df.to_csv('results.csv', index=False)
```

## Next Steps

Congratulations! You've completed your first analysis with labanalysis. 

### Learn More

- **[Signal Processing Guide](../user-guide/signal-processing/README.md)** - Advanced filtering and analysis
- **[Balance Tests Guide](../user-guide/test-protocols/balance-tests.md)** - Standardized balance protocols
- **[Visualization Guide](../user-guide/visualization/README.md)** - Create publication-quality figures

### Try More Tutorials

- **[Jump Analysis](../tutorials/01-jump-analysis.md)** - Analyze countermovement jumps
- **[Gait Analysis](../tutorials/02-gait-analysis.md)** - Walking and running analysis
- **[Full Body Kinematics](../tutorials/03-full-body-kinematics.md)** - Complete motion capture workflow

### Explore API

- **[API Reference](../api-reference/README.md)** - Complete API documentation
- **[Examples](../examples/README.md)** - More code examples

## Troubleshooting

**Problem**: File not found error

```python
FileNotFoundError: [Errno 2] No such file or directory: 'balance_test.tdf'
```

**Solution**: Use absolute path or verify the file location:
```python
import os
file_path = os.path.abspath("path/to/balance_test.tdf")
record = laban.TimeseriesRecord.from_tdf(file_path)
```

**Problem**: Missing force platform data

```python
KeyError: 'FP1'
```

**Solution**: Check what keys are available:
```python
print(list(record.keys()))
```

[→ More troubleshooting](../troubleshooting/common-errors.md)

---

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
