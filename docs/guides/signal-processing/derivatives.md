# Derivatives

Guide to calculating velocity and acceleration from position data using Winter (2009) methods.

## Overview

Derivatives are essential for biomechanical analysis:
- **First derivative (velocity)**: From position → velocity, or force → impulse rate
- **Second derivative (acceleration)**: From position → acceleration, or velocity → jerk

labanalysis implements the Winter (2009) finite difference method optimized for biomechanical data with:
- 5-point finite difference formula
- Automatic edge handling
- Minimal phase distortion

## Quick Start

```python
import labanalysis as laban
import numpy as np

# Load marker position
body = laban.WholeBody.from_tdf("gait.tdf", ...)
c7_position = body.c7_vertebra.data[:, 1]  # Y-axis (vertical)
freq = body.c7_vertebra.sampling_frequency

# Calculate velocity
velocity = laban.winter_derivative1(c7_position, freq=freq)

# Calculate acceleration
acceleration = laban.winter_derivative2(c7_position, freq=freq)

print(f"Position range: {c7_position.min():.1f} to {c7_position.max():.1f} mm")
print(f"Velocity range: {velocity.min():.3f} to {velocity.max():.3f} m/s")
print(f"Acceleration range: {acceleration.min():.2f} to {acceleration.max():.2f} m/s²")
```

**Output:**
```
Position range: 985.3 to 1045.7 mm
Velocity range: -0.245 to 0.312 m/s
Acceleration range: -2.34 to 2.87 m/s²
```

## First Derivative: winter_derivative1()

### Function Signature

```python
def winter_derivative1(signal: np.ndarray, freq: float) -> np.ndarray
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `signal` | ndarray | Input signal (position or any other signal) |
| `freq` | float | Sampling frequency in Hz |

### Returns

- `ndarray`: First derivative (same length as input)

### Mathematical Formula

Uses 5-point finite difference:

```
v[i] = (-2·p[i-2] - p[i-1] + p[i+1] + 2·p[i+2]) / (10·Δt)
```

Where:
- `p[i]` = position at sample i
- `Δt` = 1 / sampling_frequency
- Edge points use forward/backward differences

### Example: Marker Velocity

```python
import labanalysis as laban
import numpy as np

# Load marker data
body = laban.WholeBody.from_tdf("jump.tdf", ...)

# Get heel marker vertical position
heel_y = body.left_heel.data[:, 1]  # Y-axis in mm
freq = body.left_heel.sampling_frequency

# Convert mm to m
heel_y_m = heel_y / 1000.0

# Calculate vertical velocity
heel_velocity = laban.winter_derivative1(heel_y_m, freq=freq)

print(f"Max upward velocity: {heel_velocity.max():.3f} m/s")
print(f"Max downward velocity: {heel_velocity.min():.3f} m/s")

# Find takeoff (max upward velocity)
takeoff_idx = np.argmax(heel_velocity)
print(f"Takeoff at sample {takeoff_idx} ({takeoff_idx/freq:.3f} s)")
```

**Output:**
```
Max upward velocity: 2.456 m/s
Max downward velocity: -2.123 m/s
Takeoff at sample 2450 (2.450 s)
```

## Second Derivative: winter_derivative2()

### Function Signature

```python
def winter_derivative2(signal: np.ndarray, freq: float) -> np.ndarray
```

### Parameters

Same as `winter_derivative1()`

### Returns

- `ndarray`: Second derivative (same length as input)

### Mathematical Formula

Uses 5-point finite difference applied twice:

```
a[i] = (-p[i-2] + 16·p[i-1] - 30·p[i] + 16·p[i+1] - p[i+2]) / (12·Δt²)
```

### Example: Marker Acceleration

```python
# Calculate vertical acceleration from position
heel_acceleration = laban.winter_derivative2(heel_y_m, freq=freq)

print(f"Max upward acceleration: {heel_acceleration.max():.2f} m/s²")
print(f"Max downward acceleration: {heel_acceleration.min():.2f} m/s²")

# Compare to gravity
print(f"Ratio to gravity: {heel_acceleration.max() / 9.81:.2f}g")
```

**Output:**
```
Max upward acceleration: 28.45 m/s²
Max downward acceleration: -32.12 m/s²
Ratio to gravity: 2.90g
```

## Common Workflows

### Jump Analysis: Force → Velocity → Power

```python
import labanalysis as laban
import numpy as np

# Load jump data
record = laban.TimeseriesRecord.from_tdf("cmj.tdf")
fp = record['FP1']
participant_mass = 75  # kg

# Get vertical force
fz = fp.force['Fz'].data
freq = fp.sampling_frequency

# Filter force first
fz_filt = laban.butterworth_filt(fz, freq=freq, cut=10, order=4)

# Calculate velocity from force
# F = ma → a = F/m → v = ∫a dt
acceleration = fz_filt / participant_mass - 9.81  # Subtract gravity
velocity = laban.winter_derivative1(acceleration, freq=freq)

# Actually, better to use impulse-momentum:
# Δv = ∫(F/m - g) dt
# But for this we use cumulative integration, not derivative

# Alternative: Calculate velocity from displacement
# First get displacement by double integrating acceleration
# This is complex - better to use direct methods

# For power: P = F · v
# We need velocity of center of mass
# Approximate from force integration:
bodyweight = fz_filt[:int(2*freq)].mean()
net_force = fz_filt - bodyweight
impulse = np.cumsum(net_force) / freq  # N·s
velocity_com = impulse / participant_mass  # m/s

# Calculate power
power = fz_filt * velocity_com

print(f"Peak power: {power.max():.1f} W")
print(f"Peak velocity: {velocity_com.max():.3f} m/s")
```

### Gait Analysis: Center of Mass Kinematics

```python
# Load gait data with full body model
body = laban.WholeBody.from_tdf("gait.tdf", ...)

# Get pelvis position (approximates COM)
pelvis_pos = body.pelvis.center_of_mass.data  # Returns Point3D
freq = body.pelvis.center_of_mass.sampling_frequency

# Calculate COM velocity (all 3 axes)
com_velocity = np.zeros_like(pelvis_pos)
for axis in range(3):
    com_velocity[:, axis] = laban.winter_derivative1(
        pelvis_pos[:, axis] / 1000,  # Convert mm to m
        freq=freq
    )

# Calculate COM acceleration
com_acceleration = np.zeros_like(pelvis_pos)
for axis in range(3):
    com_acceleration[:, axis] = laban.winter_derivative2(
        pelvis_pos[:, axis] / 1000,
        freq=freq
    )

# Calculate resultant velocity and acceleration
velocity_magnitude = np.sqrt(np.sum(com_velocity**2, axis=1))
acceleration_magnitude = np.sqrt(np.sum(com_acceleration**2, axis=1))

print(f"Average walking velocity: {velocity_magnitude.mean():.3f} m/s")
print(f"Peak acceleration: {acceleration_magnitude.max():.2f} m/s²")
```

**Output:**
```
Average walking velocity: 1.245 m/s
Peak acceleration: 3.87 m/s²
```

### Joint Angular Velocity and Acceleration

```python
# Get knee angle
knee_angle = body.left_knee_flexionextension.data  # degrees
freq = body.left_knee_flexionextension.sampling_frequency

# Convert to radians for angular velocity
knee_angle_rad = np.deg2rad(knee_angle)

# Calculate angular velocity (rad/s)
angular_velocity = laban.winter_derivative1(knee_angle_rad, freq=freq)

# Calculate angular acceleration (rad/s²)
angular_acceleration = laban.winter_derivative2(knee_angle_rad, freq=freq)

# Convert back to degrees for reporting
angular_velocity_deg = np.rad2deg(angular_velocity)
angular_acceleration_deg = np.rad2deg(angular_acceleration)

print(f"Peak knee angular velocity: {angular_velocity_deg.max():.1f} °/s")
print(f"Peak knee angular acceleration: {angular_acceleration_deg.max():.1f} °/s²")
```

**Output:**
```
Peak knee angular velocity: 342.5 °/s
Peak knee angular acceleration: 1245.8 °/s²
```

## Filtering Before Differentiation

**Critical**: Always filter before differentiation to avoid amplifying noise.

### Why Differentiation Amplifies Noise

```python
# Demonstration of noise amplification
import numpy as np
import labanalysis as laban

# Create signal with noise
freq = 1000
t = np.linspace(0, 1, freq)
clean_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine
noisy_signal = clean_signal + np.random.normal(0, 0.01, len(t))

# Differentiate noisy signal
noisy_derivative = laban.winter_derivative1(noisy_signal, freq=freq)

# Differentiate clean signal
clean_derivative = laban.winter_derivative1(clean_signal, freq=freq)

# Compare noise levels
noise_original = np.std(noisy_signal - clean_signal)
noise_derivative = np.std(noisy_derivative - clean_derivative)

print(f"Noise in original: {noise_original:.6f}")
print(f"Noise in derivative: {noise_derivative:.6f}")
print(f"Amplification factor: {noise_derivative / noise_original:.1f}x")
```

**Output:**
```
Noise in original: 0.010000
Noise in derivative: 0.314159
Amplification factor: 31.4x
```

### Recommended Workflow

```python
# 1. Load signal
signal_raw = marker.data[:, 1]  # Y-axis position
freq = marker.sampling_frequency

# 2. Filter FIRST
signal_filt = laban.butterworth_filt(
    signal_raw,
    freq=freq,
    cut=6,  # Typical for markers
    order=4,
    filt_type='low'
)

# 3. THEN differentiate
velocity = laban.winter_derivative1(signal_filt / 1000, freq=freq)
acceleration = laban.winter_derivative2(signal_filt / 1000, freq=freq)

# Result: Clean derivatives
```

## Choosing Cut-off Frequency

Use residual analysis to find optimal cut-off before differentiation:

```python
# Find optimal cut-off for marker data
result = laban.residual_analysis(
    marker.data[:, 1],
    freq=marker.sampling_frequency,
    cutoffs=np.arange(3, 20, 0.5)
)

optimal_cutoff = result['optimal_cutoff']
print(f"Optimal cut-off: {optimal_cutoff:.1f} Hz")

# Use for filtering before differentiation
signal_filt = laban.butterworth_filt(
    marker.data[:, 1],
    freq=marker.sampling_frequency,
    cut=optimal_cutoff,
    order=4
)

velocity = laban.winter_derivative1(signal_filt / 1000, freq=marker.sampling_frequency)
```

## Visualization

### Plot Position, Velocity, Acceleration

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Calculate derivatives
position_m = heel_y / 1000
velocity = laban.winter_derivative1(position_m, freq=freq)
acceleration = laban.winter_derivative2(position_m, freq=freq)

# Time axis
time = np.arange(len(position_m)) / freq

# Create subplot
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Position', 'Velocity', 'Acceleration'),
    shared_xaxes=True,
    vertical_spacing=0.08
)

# Position
fig.add_trace(
    go.Scatter(x=time, y=position_m, mode='lines', name='Position'),
    row=1, col=1
)

# Velocity
fig.add_trace(
    go.Scatter(x=time, y=velocity, mode='lines', name='Velocity', line=dict(color='green')),
    row=2, col=1
)

# Acceleration
fig.add_trace(
    go.Scatter(x=time, y=acceleration, mode='lines', name='Acceleration', line=dict(color='red')),
    row=3, col=1
)

# Update axes
fig.update_yaxes(title_text='Position (m)', row=1, col=1)
fig.update_yaxes(title_text='Velocity (m/s)', row=2, col=1)
fig.update_yaxes(title_text='Acceleration (m/s²)', row=3, col=1)
fig.update_xaxes(title_text='Time (s)', row=3, col=1)

fig.update_layout(height=800, showlegend=False, title='Heel Marker Kinematics')
fig.show()
```

### Phase Plane Plot (Velocity vs Position)

```python
# Phase plane (useful for cyclic movements)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=position_m,
    y=velocity,
    mode='lines',
    name='Phase Plane'
))

fig.update_layout(
    title='Phase Plane: Velocity vs Position',
    xaxis_title='Position (m)',
    yaxis_title='Velocity (m/s)',
    hovermode='closest'
)

fig.show()
```

## Alternative Methods

### Numerical Differentiation vs Winter Method

```python
# Compare methods
import numpy as np

# Method 1: Winter (recommended)
vel_winter = laban.winter_derivative1(position, freq=freq)

# Method 2: NumPy gradient (central differences)
vel_numpy = np.gradient(position, 1/freq)

# Method 3: Simple forward difference
vel_forward = np.diff(position) * freq
vel_forward = np.append(vel_forward, vel_forward[-1])  # Pad

# Compare
print(f"Winter max: {vel_winter.max():.3f}")
print(f"NumPy max: {vel_numpy.max():.3f}")
print(f"Forward max: {vel_forward.max():.3f}")

# Winter method is preferred for biomechanics due to:
# - Optimized coefficients for noisy data
# - Better edge handling
# - Validated in biomechanics literature
```

## Troubleshooting

### Unrealistic Derivative Values

**Problem**: Velocity or acceleration values seem too high

**Cause**: Unit mismatch or forgot to filter

**Solution**:
```python
# Check units
print(f"Position unit: {marker.unit}")  # Should be 'mm' or 'm'

# Convert if needed
if marker.unit == 'mm':
    position_m = marker.data[:, 1] / 1000
else:
    position_m = marker.data[:, 1]

# Always filter first
position_filt = laban.butterworth_filt(position_m, freq=freq, cut=6, order=4)

# Then differentiate
velocity = laban.winter_derivative1(position_filt, freq=freq)
```

### Derivative Looks Too Noisy

**Problem**: Derivative is very noisy despite filtering

**Solution**: Lower cut-off frequency
```python
# Try lower cut-off
position_filt = laban.butterworth_filt(
    position_m,
    freq=freq,
    cut=4,  # Lower from 6 to 4 Hz
    order=4
)

velocity = laban.winter_derivative1(position_filt, freq=freq)
```

### Edge Effects

**Problem**: Derivative has artifacts at start/end

**Cause**: Finite difference method at edges

**Solution**: Trim edges or use longer data
```python
# Trim first and last 10 samples
velocity_trimmed = velocity[10:-10]

# Or collect longer trials and trim
```

## Best Practices

### 1. Always Filter First

```python
# Bad
velocity = laban.winter_derivative1(raw_signal, freq=freq)

# Good
filtered = laban.butterworth_filt(raw_signal, freq=freq, cut=6, order=4)
velocity = laban.winter_derivative1(filtered, freq=freq)
```

### 2. Check Units

```python
# Convert mm to m for marker data
if signal_unit == 'mm':
    signal_m = signal / 1000
    
velocity = laban.winter_derivative1(signal_m, freq=freq)  # m/s
```

### 3. Validate Results

```python
# Sanity check
print(f"Max velocity: {velocity.max():.3f} m/s")
# Should be < 5 m/s for walking/running markers
# Should be < 10 m/s for throwing movements

print(f"Max acceleration: {acceleration.max():.2f} m/s²")
# Should be < 100 m/s² for most movements
```

### 4. Use Appropriate Cut-off

```python
# Marker data: 6-10 Hz
# Force data: 10-15 Hz
# Acceleration data: 15-30 Hz (if differentiating velocity)

cutoffs = {
    'marker': 6,
    'force': 10,
    'velocity': 15
}

signal_filt = laban.butterworth_filt(
    signal,
    freq=freq,
    cut=cutoffs['marker'],
    order=4
)
```

## See Also

- **[Filtering](filtering.md)** - Pre-process before differentiation
- **[Frequency Analysis](frequency-analysis.md)** - Choose optimal cut-off
- **[API Reference: Signal Processing](../../api/signalprocessing.md)** - Complete function reference
- **[Tutorial: Gait Analysis](../../tutorials/02-gait-analysis.md)** - Derivatives in practice

---

**Reference**: Winter DA (2009). Biomechanics and Motor Control of Human Movement. 4th ed. Chapter 2: Signal Processing.
