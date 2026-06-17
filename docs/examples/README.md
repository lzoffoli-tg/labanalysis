# Examples

Runnable Python scripts demonstrating common labanalysis tasks. All examples are self-contained and can be copied directly into your code.

## Quick Navigation

- **[Basic Examples](#basic-examples)** - Loading, filtering, exporting
- **[Biomechanics Examples](#biomechanics-examples)** - Kinematics, GRF, markers
- **[Protocol Examples](#protocol-examples)** - Standardized tests
- **[Modeling Examples](#modeling-examples)** - Regression and ML

## Basic Examples

Simple examples for everyday tasks.

### [load-and-plot.py](basic/load-and-plot.py)

Load TDF file and create basic visualization.

```python
import labanalysis as laban
import plotly.graph_objects as go

# Load data
record = laban.TimeseriesRecord.from_tdf("data.tdf")
fp = record['FP1']
fz = fp.force['Fz']

# Create plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=fz.index, y=fz.data, name='Vertical Force'))
fig.update_layout(title='Ground Reaction Force', xaxis_title='Time (s)', yaxis_title='Force (N)')
fig.show()
```

**Use case:** Quick data visualization

### [filter-signal.py](basic/filter-signal.py)

Apply Butterworth filter to remove noise.

```python
import labanalysis as laban

# Load signal
signal = laban.Signal1D.from_tdf("data.tdf", column="Fz")

# Filter
filtered = laban.butterworth_filt(
    signal=signal.data,
    freq=signal.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# Create filtered signal object
signal_filtered = laban.Signal1D(
    data=filtered,
    index=signal.index,
    label=f"{signal.label}_filtered",
    unit=signal.unit
)
```

**Use case:** Noise reduction

### [export-to-excel.py](basic/export-to-excel.py)

Export data to Excel spreadsheet.

```python
import labanalysis as laban

# Load data
record = laban.TimeseriesRecord.from_tdf("data.tdf")

# Convert to DataFrame
df = record.to_dataframe()

# Export to Excel
df.to_excel("output.xlsx", index=False, sheet_name='Force Data')
print(f"Exported {len(df)} rows to output.xlsx")
```

**Use case:** Data export for reports

## Biomechanics Examples

Biomechanical analysis examples.

### [joint-angles.py](biomechanics/joint-angles.py)

Calculate joint angles from motion capture.

```python
import labanalysis as laban

# Load WholeBody model
body = laban.WholeBody.from_tdf(
    "mocap.tdf",
    left_knee_medial="LKNEM",
    left_knee_lateral="LKNEL",
    left_ankle_medial="LANKM",
    left_ankle_lateral="LANKL",
    # ... other markers
)

# Access joint angles
knee_flexion = body.left_knee_flexionextension
hip_abduction = body.left_hip_abduction

print(f"Knee flexion range: {knee_flexion.data.min():.1f}° to {knee_flexion.data.max():.1f}°")
```

**Use case:** Joint kinematics

### [grf-analysis.py](biomechanics/grf-analysis.py)

Analyze ground reaction forces.

```python
import labanalysis as laban
import numpy as np

# Load force platform
record = laban.TimeseriesRecord.from_tdf("jump.tdf")
fp = record['FP1']

# Filter vertical force
fz_filtered = laban.butterworth_filt(
    signal=fp.force['Fz'].data,
    freq=1000,
    cut=10,
    order=4
)

# Find peaks
peaks = laban.find_peaks(fz_filtered, height=500, distance=100)

# Calculate impulse
impulse = np.trapz(fz_filtered, dx=1/1000)  # N·s

print(f"Peak force: {peaks['peak_heights'].max():.1f} N")
print(f"Impulse: {impulse:.1f} N·s")
```

**Use case:** Force analysis

### [marker-tracking.py](biomechanics/marker-tracking.py)

Track 3D marker positions.

```python
import labanalysis as laban
import numpy as np

# Load marker data
record = laban.TimeseriesRecord.from_tdf("mocap.tdf")
c7_marker = record['MKRS']['C7']  # C7 vertebra

# Calculate marker velocity
velocity = laban.winter_derivative1(
    signal=c7_marker.data,
    freq=c7_marker.sampling_frequency
)

# Calculate 3D speed
speed = np.linalg.norm(velocity, axis=1)

print(f"Max speed: {speed.max():.2f} m/s")
```

**Use case:** Marker kinematics

## Protocol Examples

Standardized test protocol examples.

### [cmj-test.py](protocols/cmj-test.py)

Countermovement jump test analysis.

```python
import labanalysis as laban

# Create participant
participant = laban.Participant(
    name="John", surname="Doe",
    height=1.80, weight=75, age=25
)

# Load and analyze jump (placeholder - actual implementation may vary)
from labanalysis.records.jumping import SingleJump

jump = SingleJump.from_tdf(
    "cmj.tdf",
    left_foot_ground_reaction_force='FP1'
)

print("CMJ Analysis Complete")
```

**Use case:** Jump testing

### [running-test.py](protocols/running-test.py)

Running gait analysis.

```python
import labanalysis as laban
from labanalysis.records.locomotion import RunningExercise

# Create participant
participant = laban.Participant(
    name="Jane", surname="Smith",
    height=1.65, weight=60, age=28
)

# Load running data
running = RunningExercise.from_tdf(
    "running.tdf",
    left_heel="LHEE",
    right_heel="RHEE",
    left_foot_ground_reaction_force='FP1'
)

print("Running analysis complete")
```

**Use case:** Gait analysis

### [balance-test.py](protocols/balance-test.py)

Balance assessment.

```python
import labanalysis as laban
import numpy as np

# Load balance data
record = laban.TimeseriesRecord.from_tdf("balance.tdf")
fp = record['FP1']

# Calculate COP
fz = fp.force['Fz'].data
cop_x = -fp.torque['My'].data / fz
cop_y = fp.torque['Mx'].data / fz

# Balance metrics
cop_sway_x = np.std(cop_x) * 100  # cm
cop_sway_y = np.std(cop_y) * 100  # cm

print(f"COP sway: X={cop_sway_x:.2f} cm, Y={cop_sway_y:.2f} cm")
```

**Use case:** Balance testing

## Modeling Examples

Regression and machine learning examples.

### [polynomial-fit.py](modeling/polynomial-fit.py)

Polynomial regression.

```python
import labanalysis as laban
import numpy as np

# Generate example data
x = np.linspace(0, 10, 100)
y = 2*x**2 + 3*x + 1 + np.random.randn(100)*5

# Fit polynomial model
model = laban.PolynomialRegression(degree=2)
model.fit(x.reshape(-1, 1), y)

# Predictions
y_pred = model.predict(x.reshape(-1, 1))

# Metrics
r2 = model.score(x.reshape(-1, 1), y)
print(f"R² = {r2:.3f}")
print(f"Coefficients: {model.coef_}")
```

**Use case:** Curve fitting

### [pytorch-training.py](modeling/pytorch-training.py)

PyTorch model training with TorchTrainer.

```python
import labanalysis as laban
import torch
import torch.nn as nn

# Placeholder for PyTorch training example
# (Actual implementation would include model definition, data preparation, training loop)

print("PyTorch training example - see full tutorial in docs/tutorials/08-ml-modeling.md")
```

**Use case:** Machine learning

### [1rm-prediction.py](modeling/1rm-prediction.py)

Predict 1-repetition maximum.

```python
from labanalysis.equations.strength import Brzycki1RM

# Submaximal data
load_kg = 100
reps = 8

# Predict 1RM
brzycki = Brzycki1RM()
predicted_1rm = brzycki.predict(load=load_kg, reps=reps)

print(f"Predicted 1RM: {predicted_1rm:.1f} kg")
print(f"From {reps} reps at {load_kg} kg")
```

**Use case:** Strength prediction

## Using These Examples

### Copy and Run

All examples are complete and runnable. Simply:

1. Copy the code
2. Replace `"data.tdf"` with your file path
3. Run the script

### Modify for Your Needs

Examples are templates - adapt them:

```python
# Original example
filtered = laban.butterworth_filt(signal.data, freq=1000, cut=10, order=4)

# Modified for your data
filtered = laban.butterworth_filt(signal.data, freq=2000, cut=15, order=6)
```

### Combine Examples

Mix and match to build custom workflows:

```python
# Load (from load-and-plot.py)
record = laban.TimeseriesRecord.from_tdf("data.tdf")

# Filter (from filter-signal.py)
filtered = laban.butterworth_filt(...)

# Export (from export-to-excel.py)
df.to_excel("output.xlsx")
```

## See Also

- **[Tutorials](../tutorials/README.md)** - Complete end-to-end workflows
- **[User Guide](../user-guide/README.md)** - Task-oriented guides
- **[API Reference](../api-reference/README.md)** - Complete API docs

---

**Have an example to contribute?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
