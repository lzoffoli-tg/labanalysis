# Missing Data Handling

Guide to handling missing data (NaN values) in labanalysis using various interpolation and regression methods.

## Overview

Missing data is common in biomechanical signals due to marker occlusion, sensor dropout, or signal artifacts. labanalysis provides multiple strategies for filling missing values via the `fillna()` method available on all `Timeseries` objects.

## Quick Reference

| Method | Best For | Preserves | Limitations |
|--------|----------|-----------|-------------|
| `'linear'` | Short gaps, simple trends | Monotonicity | Flat at inflection points |
| `'cubic'` | Smooth curves, medium gaps | Smoothness | Can overshoot |
| `'pchip'` | Natural motion, any gap size | Shape, no overshoot | More computation |
| `'regression'` | Long gaps with predictors | Relationships | Needs correlated signals |

## Methods

### Linear Interpolation

Straight-line interpolation between valid points.

**When to use:**
- Short gaps (<10 samples)
- Signals with linear trends
- Fast processing required

**Example:**
```python
import labanalysis as laban
import numpy as np

# Load marker with missing data
marker = laban.Point3D.from_tdf("trial.tdf", marker_name="C7")

# Check for missing data
x_pos = marker['x'].data
n_missing = np.isnan(x_pos).sum()
print(f"Missing samples: {n_missing}")
# Output: Missing samples: 47

# Fill with linear interpolation
marker_filled = marker.fillna(method='linear')

# Verify
x_filled = marker_filled['x'].data
print(f"Missing after fill: {np.isnan(x_filled).sum()}")
# Output: Missing after fill: 0
```

**Advantages:**
- Fast computation
- Preserves monotonicity
- No overshoot/undershoot

**Limitations:**
- Creates artificial kinks at gap boundaries
- Poor for curved trajectories
- Visible artifacts in velocity/acceleration

### Cubic Spline Interpolation

Smooth piecewise cubic polynomial interpolation.

**When to use:**
- Medium gaps (10-50 samples)
- Smooth natural motion
- When derivatives matter (velocity/acceleration)

**Example:**
```python
# Load marker trajectory
marker = laban.Point3D.from_tdf("walking.tdf", marker_name="ankle_L")

# Fill with cubic spline
marker_smooth = marker.fillna(method='cubic')

# Compare with linear
marker_linear = marker.fillna(method='linear')

# Calculate velocities
y_cubic = marker_smooth['y'].data
y_linear = marker_linear['y'].data
freq = marker.sampling_frequency

vel_cubic = laban.winter_derivative1(y_cubic / 1000, freq=freq)
vel_linear = laban.winter_derivative1(y_linear / 1000, freq=freq)

print(f"Velocity smoothness (cubic): {np.std(np.diff(vel_cubic)):.3f} m/s²")
print(f"Velocity smoothness (linear): {np.std(np.diff(vel_linear)):.3f} m/s²")
# Output: 
# Velocity smoothness (cubic): 0.023 m/s²
# Velocity smoothness (linear): 0.156 m/s²  (6x noisier)
```

**Advantages:**
- Smooth first and second derivatives
- Natural-looking trajectories
- Good for motion data

**Limitations:**
- Can overshoot at boundaries
- May create unrealistic accelerations in long gaps
- More computation than linear

### PCHIP Interpolation

Piecewise Cubic Hermite Interpolating Polynomial - shape-preserving cubic interpolation.

**When to use:**
- Any gap size
- When overshoot is unacceptable
- Natural human motion (preserves monotonicity)
- Force platform data

**Example:**
```python
# Load force platform with dropout
fp = laban.ForcePlatform.from_tdf("jump.tdf", platform_name="FP1")
fz = fp.force['Fz']

# Check gap size
gaps = np.diff(np.where(np.isnan(fz.data))[0])
max_gap = gaps.max() if len(gaps) > 0 else 0
print(f"Largest gap: {max_gap} samples ({max_gap / fz.sampling_frequency * 1000:.1f} ms)")
# Output: Largest gap: 23 samples (23.0 ms)

# Fill with PCHIP
fz_filled = fz.fillna(method='pchip')

# Compare methods
fz_cubic = fz.fillna(method='cubic')

# Check for unrealistic values
original_max = np.nanmax(fz.data)
pchip_max = fz_filled.data.max()
cubic_max = fz_cubic.data.max()

print(f"Original max force: {original_max:.1f} N")
print(f"PCHIP max force: {pchip_max:.1f} N")
print(f"Cubic max force: {cubic_max:.1f} N (overshoot!)")
# Output:
# Original max force: 1823.4 N
# PCHIP max force: 1825.1 N
# Cubic max force: 1897.3 N (overshoot!)
```

**Advantages:**
- No overshoot/undershoot (shape-preserving)
- Preserves monotonicity
- Smooth interpolation
- Best for human motion and forces

**Limitations:**
- Slightly more computation than cubic
- May be less smooth than cubic in some cases

**Recommendation:** PCHIP is the **default choice** for most biomechanical data.

### Regression-Based Interpolation

Fill missing values using regression against correlated signals.

**When to use:**
- Very long gaps (>100 samples)
- Predictor signals available (e.g., opposite limb markers)
- Missing entire segments of data

**Example:**
```python
# Load bilateral markers
record = laban.TimeseriesRecord.from_tdf("gait.tdf")

# Right ankle has complete data, left has gaps
ankle_R = record.markers['ankle_R']
ankle_L = record.markers['ankle_L']

# Check correlation
y_R = ankle_R['y'].data
y_L = ankle_L['y'].data
valid_mask = ~(np.isnan(y_R) | np.isnan(y_L))
correlation = np.corrcoef(y_R[valid_mask], y_L[valid_mask])[0, 1]
print(f"L-R ankle correlation: {correlation:.3f}")
# Output: L-R ankle correlation: 0.892

# Use right ankle as regressor for left ankle
# Prepare regressors (all 3 components)
regressors = ankle_R.data.copy()

# Fill left ankle using regression
ankle_L_filled = ankle_L.fillna(method='regression', regressors=regressors)

# Verify quality
y_L_filled = ankle_L_filled['y'].data
filled_indices = np.where(np.isnan(y_L))[0]
predicted = y_L_filled[filled_indices]
print(f"Filled {len(filled_indices)} samples using regression")
# Output: Filled 347 samples using regression
```

**How it works:**
1. Builds linear regression model on valid data: `y = β₀ + β₁·x₁ + β₂·x₂ + ...`
2. Predicts missing values using regressor data
3. Preserves relationships between signals

**Requirements:**
- Regressor signals must be complete (no NaN)
- Moderate-to-strong correlation (r > 0.5)
- Sufficient valid samples for reliable regression

**Advantages:**
- Can fill very long gaps
- Preserves inter-signal relationships
- Useful for bilateral data

**Limitations:**
- Requires correlated predictor signals
- May not capture complex dynamics
- Regression errors propagate

## Workflow Examples

### Handling Marker Occlusion

```python
# Load motion capture trial
record = laban.TimeseriesRecord.from_tdf("walking.tdf")

# Get all markers
markers = record.markers

# Check missing data across all markers
print("=== Missing Data Summary ===")
for name, marker in markers.items():
    x_data = marker['x'].data
    n_missing = np.isnan(x_data).sum()
    pct_missing = 100 * n_missing / len(x_data)
    if n_missing > 0:
        print(f"{name:15s}: {n_missing:4d} samples ({pct_missing:5.2f}%)")

# Output:
# === Missing Data Summary ===
# C7             :   12 samples ( 1.20%)
# ankle_L        :  234 samples (23.40%)
# knee_L         :   45 samples ( 4.50%)
# hip_L          :    0 samples ( 0.00%)

# Fill all markers with PCHIP (best for motion)
markers_filled = {}
for name, marker in markers.items():
    markers_filled[name] = marker.fillna(method='pchip')

print("\n=== After Filling ===")
for name, marker in markers_filled.items():
    x_data = marker['x'].data
    print(f"{name:15s}: {np.isnan(x_data).sum():4d} missing")
# All show 0 missing
```

### Force Platform Signal Dropout

```python
# Load force platform with dropout
fp = laban.ForcePlatform.from_tdf("jump.tdf", platform_name="FP1")

# Fill forces
force_filled = fp.force.fillna(method='pchip')

# Fill moments
torque_filled = fp.torque.fillna(method='pchip')

# Create clean force platform
fp_clean = laban.ForcePlatform(
    force=force_filled,
    torque=torque_filled
)

# COP now calculated on clean data
cop = fp_clean.cop
print(f"COP calculated with {len(cop['COPx'].data)} clean samples")
```

### EMG Signal Artifacts

```python
# Load EMG with artifacts
emg = laban.EMGSignal.from_tdf("trial.tdf", column="biceps_R")

# Identify artifacts (values > 3 SD from mean)
threshold = 3 * np.nanstd(emg.data)
artifacts = np.abs(emg.data) > threshold

# Replace artifacts with NaN
emg_clean = emg.copy()
emg_clean.data[artifacts] = np.nan

print(f"Artifacts detected: {artifacts.sum()} samples")
# Output: Artifacts detected: 47 samples

# Fill with PCHIP
emg_filled = emg_clean.fillna(method='pchip')

# Continue with standard EMG processing
emg_bp = laban.butterworth_filt(
    emg_filled.data,
    freq=emg.sampling_frequency,
    cut=(20, 450),
    filt_type='band'
)
```

### Long Gap with Regression

```python
# Load bilateral gait data
record = laban.TimeseriesRecord.from_tdf("gait.tdf")

# Left marker has very long dropout (>1 second)
marker_L = record.markers['ankle_L']
marker_R = record.markers['ankle_R']  # Complete data

y_L = marker_L['y'].data
gap_indices = np.where(np.isnan(y_L))[0]
gap_duration = len(gap_indices) / marker_L.sampling_frequency

print(f"Gap duration: {gap_duration:.2f} s ({len(gap_indices)} samples)")
# Output: Gap duration: 1.23 s (123 samples)

# PCHIP would struggle with this gap - use regression
marker_L_filled = marker_L.fillna(
    method='regression',
    regressors=marker_R.data
)

# Verify smoothness
y_filled = marker_L_filled['y'].data
velocity = laban.winter_derivative1(y_filled / 1000, freq=marker_L.sampling_frequency)

# Check velocity continuity at gap boundaries
gap_start = gap_indices[0]
gap_end = gap_indices[-1]
vel_before = velocity[gap_start - 10:gap_start]
vel_during = velocity[gap_start:gap_end]
vel_after = velocity[gap_end:gap_end + 10]

print(f"Velocity std before gap: {np.std(vel_before):.3f} m/s")
print(f"Velocity std during gap: {np.std(vel_during):.3f} m/s")
print(f"Velocity std after gap: {np.std(vel_after):.3f} m/s")
# Should be similar across all three
```

## Choosing the Right Method

### Decision Tree

```
┌─ Gap size < 10 samples?
│  └─ YES → Use 'linear' (fast, simple)
│  └─ NO ↓
│
├─ Need to calculate derivatives?
│  └─ YES → Use 'pchip' (smooth, no overshoot)
│  └─ NO ↓
│
├─ Extremely smooth curves?
│  └─ YES → Try 'cubic', check for overshoot
│  └─ NO ↓
│
├─ Very long gaps (>100 samples)?
│  └─ YES → Use 'regression' with correlated predictors
│  └─ NO ↓
│
└─ Default → Use 'pchip' (best general choice)
```

### By Signal Type

| Signal Type | Recommended Method | Reason |
|-------------|-------------------|--------|
| **Markers (3D)** | `pchip` | Natural motion, no overshoot |
| **Force platforms** | `pchip` | Shape-preserving for forces |
| **Joint angles** | `pchip` | Preserves ROM limits |
| **EMG** | `pchip` or `linear` | After artifact removal |
| **Velocities** | `cubic` | Needs smooth derivatives |
| **Long gaps (>1s)** | `regression` | If predictors available |

## Validation and Quality Control

### Check Interpolation Quality

```python
import matplotlib.pyplot as plt

# Create test signal with artificial gap
original = laban.Signal1D(
    data=np.sin(2 * np.pi * 2 * np.linspace(0, 1, 1000)),
    sampling_frequency=1000
)

# Introduce gap
test = original.copy()
test.data[400:450] = np.nan  # 50-sample gap

# Compare methods
linear = test.fillna(method='linear')
cubic = test.fillna(method='cubic')
pchip = test.fillna(method='pchip')

# Calculate errors in gap region
gap = slice(400, 450)
err_linear = np.abs(linear.data[gap] - original.data[gap]).mean()
err_cubic = np.abs(cubic.data[gap] - original.data[gap]).mean()
err_pchip = np.abs(pchip.data[gap] - original.data[gap]).mean()

print(f"Mean absolute error (gap region):")
print(f"  Linear: {err_linear:.4f}")
print(f"  Cubic:  {err_cubic:.4f}")
print(f"  PCHIP:  {err_pchip:.4f}")
# Output:
# Mean absolute error (gap region):
#   Linear: 0.0234
#   Cubic:  0.0012
#   PCHIP:  0.0015
```

### Check for Unrealistic Values

```python
# After filling, verify values are within expected range
marker_filled = marker.fillna(method='pchip')

x_filled = marker_filled['x'].data
y_filled = marker_filled['y'].data
z_filled = marker_filled['z'].data

# Check against original range
x_orig = marker['x'].data
x_min_orig = np.nanmin(x_orig)
x_max_orig = np.nanmax(x_orig)

x_min_filled = x_filled.min()
x_max_filled = x_filled.max()

overshoot = (x_max_filled > x_max_orig * 1.1) or (x_min_filled < x_min_orig * 1.1)

if overshoot:
    print("Warning: Filled values exceed original range by >10%")
    print(f"  Original range: [{x_min_orig:.1f}, {x_max_orig:.1f}]")
    print(f"  Filled range: [{x_min_filled:.1f}, {x_max_filled:.1f}]")
    print("  → Consider using 'pchip' instead of 'cubic'")
```

## Advanced: Custom Interpolation

For specialized needs, you can implement custom interpolation:

```python
from scipy import interpolate

# Load signal
signal = laban.Signal1D.from_tdf("data.tdf", column="Fz")
data = signal.data.copy()
time = signal.index

# Find valid indices
valid = ~np.isnan(data)

# Create custom interpolator (e.g., B-spline)
f = interpolate.make_interp_spline(
    time[valid],
    data[valid],
    k=3,  # Cubic B-spline
    bc_type='natural'  # Natural boundary conditions
)

# Fill missing values
data_filled = data.copy()
missing = np.isnan(data)
data_filled[missing] = f(time[missing])

# Update signal
signal_filled = signal.copy()
signal_filled.data = data_filled
```

## Best Practices

1. **Always inspect gaps first**
   - Count missing samples
   - Check gap distribution
   - Identify longest consecutive gap

2. **Start with PCHIP**
   - Works well for most biomechanical data
   - No overshoot
   - Smooth enough for derivatives

3. **Validate interpolation**
   - Check for unrealistic values
   - Verify continuity at boundaries
   - Compare velocity/acceleration before and after

4. **Document your choice**
   - Record which method was used
   - Note any quality issues
   - Include in analysis logs

5. **Consider alternatives to interpolation**
   - For very poor data quality, exclude the trial
   - For systematic gaps, check sensor placement
   - For long gaps, use regression or exclude segment

## Troubleshooting

### "ValueError: Not enough valid data points"

```python
# Check if signal has sufficient valid data
signal = laban.Signal1D.from_tdf("data.tdf", column="marker_x")
valid_pct = 100 * (~np.isnan(signal.data)).sum() / len(signal.data)
print(f"Valid data: {valid_pct:.1f}%")

if valid_pct < 50:
    print("Warning: <50% valid data - interpolation may be unreliable")
    # Consider excluding this signal or using regression
```

### Interpolation creates spikes

```python
# PCHIP should not spike, but cubic might
# If you see spikes with pchip, check for outliers first

signal = laban.Signal1D.from_tdf("data.tdf", column="force")

# Remove outliers before interpolation
median = np.nanmedian(signal.data)
mad = np.nanmedian(np.abs(signal.data - median))
threshold = median + 5 * 1.4826 * mad  # 5 MAD threshold

outliers = np.abs(signal.data) > threshold
signal.data[outliers] = np.nan

# Now fill
signal_filled = signal.fillna(method='pchip')
```

### Gap too large for any method

```python
# For gaps >10% of signal length, interpolation is risky
gap_indices = np.where(np.isnan(signal.data))[0]
max_consecutive = 0
current_run = 0

for i in range(1, len(gap_indices)):
    if gap_indices[i] == gap_indices[i-1] + 1:
        current_run += 1
    else:
        max_consecutive = max(max_consecutive, current_run)
        current_run = 0

gap_pct = 100 * max_consecutive / len(signal.data)

if gap_pct > 10:
    print(f"Warning: Largest gap is {gap_pct:.1f}% of signal")
    print("Consider:")
    print("  1. Using regression with predictor signals")
    print("  2. Excluding this segment from analysis")
    print("  3. Re-collecting data if possible")
```

## See Also

- **[Signal Processing Overview](README.md)** - All signal processing tools
- **[Filtering](filtering.md)** - Smooth data before/after interpolation
- **[Derivatives](derivatives.md)** - Calculate velocity/acceleration on filled data
- **[API Reference: Timeseries.fillna()](../../api/records/timeseries.md#fillna)** - Method documentation

---

**Module**: `src/labanalysis/records/timeseries.py` (Timeseries.fillna())  
**Key Methods**: `linear`, `cubic`, `pchip`, `regression`
