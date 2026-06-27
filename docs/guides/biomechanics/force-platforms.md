# Force Platforms

Guide to working with force platform data using the ForcePlatform class in labanalysis.

## Overview

Force platforms measure ground reaction forces (GRF), center of pressure (COP), and torques during human movement. The `ForcePlatform` class in labanalysis provides a unified interface for:

- **3D Ground Reaction Force** (`force`): Vertical, anteroposterior, and mediolateral forces
- **Center of Pressure** (`origin`): 3D COP trajectory over time  
- **Torque** (`torque`): 3D torque vector about the COP

All components are automatically synchronized and can be filtered, transformed, and analyzed together.

## Quick Reference

```python
import labanalysis as laban

# Load force platform data from TDF file
record = laban.TimeseriesRecord.from_tdf("jump_trial.tdf")
force_platform = record['FP1']  # Access force platform by name

# Access components
cop = force_platform['origin']          # Point3D: COP trajectory (x, y, z)
grf = force_platform['force']           # Signal3D: Force (Fx, Fy, Fz)
torque_vec = force_platform['torque']   # Signal3D: Torque (Tx, Ty, Tz)

# Get vertical force
fz = grf[force_platform.vertical_axis]  # Usually 'Y' axis
print(f"Peak vertical force: {fz.data.max():.1f} N")

# Analyze COP excursion
cop_ml = cop[force_platform.lateral_axis]         # Mediolateral COP
cop_ap = cop[force_platform.anteroposterior_axis]  # Anteroposterior COP
```

## ForcePlatform Structure

The `ForcePlatform` class is a specialized `Record` containing exactly three components:

| Property | Type | Description | Units |
|----------|------|-------------|-------|
| `origin` | Point3D | Center of Pressure (COP) position | mm or m |
| `force` | Signal3D | Ground Reaction Force (GRF) vector | N |
| `torque` | Signal3D | Torque vector about COP | Nm |

**Axis Convention:**
- `vertical_axis`: Usually `'Y'` (upward positive)
- `anteroposterior_axis`: Usually `'Z'` (forward positive)
- `lateral_axis`: Usually `'X'` (left positive)

All three components share the same axis convention.

## Loading Force Platform Data

### From TDF Files

```python
import labanalysis as laban

# Load TDF file containing force platform data
record = laban.TimeseriesRecord.from_tdf("trial.tdf")

# Check available force platforms
print(record.keys())
# Output: ['FP1', 'FP2', 'marker1', 'marker2', ...]

# Extract force platform
fp = record['FP1']

# Verify it's a ForcePlatform
print(type(fp))
# Output: <class 'labanalysis.records.records.ForcePlatform'>
```

### Creating Manually

```python
import labanalysis as laban
import numpy as np

# Generate synthetic data
n_samples = 1000
time = np.linspace(0, 10, n_samples)

# COP trajectory (sinusoidal sway)
cop_data = np.column_stack([
    np.sin(time * 2) * 10,  # X: mediolateral
    np.zeros(n_samples),     # Y: vertical (on platform surface)
    np.cos(time * 2) * 15    # Z: anteroposterior
])

# GRF (vertical force with AP/ML components)
force_data = np.column_stack([
    np.random.randn(n_samples) * 5,      # Fx: mediolateral
    800 + np.random.randn(n_samples) * 20, # Fy: vertical (~80 kg person)
    np.random.randn(n_samples) * 10      # Fz: anteroposterior
])

# Torque (primarily about vertical axis during standing)
torque_data = np.column_stack([
    np.random.randn(n_samples) * 2,   # Tx
    np.random.randn(n_samples) * 50,  # Ty: dominant component
    np.random.randn(n_samples) * 2    # Tz
])

# Create components
cop = laban.Point3D(cop_data, time, columns=['X', 'Y', 'Z'], unit='mm')
grf = laban.Signal3D(force_data, time, columns=['X', 'Y', 'Z'], unit='N')
torque = laban.Signal3D(torque_data, time, columns=['X', 'Y', 'Z'], unit='Nm')

# Create ForcePlatform
fp = laban.ForcePlatform(origin=cop, force=grf, torque=torque)
```

## Accessing Force Platform Data

### Individual Components

```python
# Access by key
cop = fp['origin']
grf = fp['force']
torque = fp['torque']

# Access by attribute
cop = fp.origin
grf = fp.force  
torque = fp.torque
```

### Axis-Specific Data

```python
# Vertical force (most common in biomechanics)
fz = grf[fp.vertical_axis]  # Signal1D

# Anteroposterior force
fx = grf[fp.anteroposterior_axis]

# Mediolateral force  
fy = grf[fp.lateral_axis]

# COP coordinates
cop_x = cop[fp.lateral_axis]
cop_y = cop[fp.vertical_axis]
cop_z = cop[fp.anteroposterior_axis]
```

## Common Analyses

### Ground Contact Detection

```python
import labanalysis as laban

# Load data
record = laban.TimeseriesRecord.from_tdf("gait.tdf")
fp = record['FP1']

# Get vertical force
fz = fp.force[fp.vertical_axis].data
time = fp.force.index
freq = 1 / np.mean(np.diff(time))

# Detect ground contacts (vertical force > threshold)
threshold = 20  # N
contact_mask = fz > threshold

# Find contact periods
from labanalysis import find_peaks

# Smooth force first
fz_smooth = laban.butterworth_filt(fz, freq=freq, cut=10, order=4)

# Detect peaks (mid-stance)
peaks, _ = find_peaks(
    fz_smooth,
    height=100,  # Minimum force
    distance=int(0.5 * freq)  # Minimum 0.5s between steps
)

print(f"Detected {len(peaks)} ground contacts")
print(f"Contact times: {time[peaks]}")
```

### COP Trajectory Analysis

```python
# Extract COP 2D trajectory (mediolateral vs anteroposterior)
cop_ml = fp.origin[fp.lateral_axis].data
cop_ap = fp.origin[fp.anteroposterior_axis].data

# Remove NaN (when not in contact)
valid = ~(np.isnan(cop_ml) | np.isnan(cop_ap))
cop_ml_valid = cop_ml[valid]
cop_ap_valid = cop_ap[valid]

# Calculate COP path length
cop_diff_ml = np.diff(cop_ml_valid)
cop_diff_ap = np.diff(cop_ap_valid)
path_length = np.sum(np.sqrt(cop_diff_ml**2 + cop_diff_ap**2))

# Calculate 95% confidence ellipse area
std_ml = np.std(cop_ml_valid)
std_ap = np.std(cop_ap_valid)
sway_area_95 = np.pi * 2.447 * std_ml * std_ap

print(f"COP path length: {path_length:.1f} mm")
print(f"Sway area (95%): {sway_area_95:.1f} mm²")
```

### Force-Time Curve Analysis

```python
# Get vertical force during a jump
fz = fp.force[fp.vertical_axis].data
time = fp.force.index
freq = 1 / np.mean(np.diff(time))

# Find takeoff and landing
# (Assuming jump in the middle of recording)
bodyweight = 700  # N (estimate from quiet standing)
threshold = 0.1 * bodyweight

# Takeoff: force drops below threshold
takeoff_idx = np.where(fz < threshold)[0][0]
takeoff_time = time[takeoff_idx]

# Landing: force rises above threshold after takeoff
landing_candidates = np.where(fz[takeoff_idx:] > threshold)[0]
if len(landing_candidates) > 0:
    landing_idx = takeoff_idx + landing_candidates[0]
    landing_time = time[landing_idx]
    
    # Flight time
    flight_time = landing_time - takeoff_time
    
    # Jump height (from flight time)
    g = 9.81  # m/s²
    jump_height = 0.125 * g * flight_time**2  # meters
    
    print(f"Takeoff: {takeoff_time:.3f} s")
    print(f"Landing: {landing_time:.3f} s")
    print(f"Flight time: {flight_time:.3f} s")
    print(f"Jump height: {jump_height*1000:.1f} mm")
```

### Impulse and Momentum

```python
# Calculate vertical impulse during push-off phase
# (Assuming takeoff_idx and landing_idx from above)

push_off_start = 0  # Start of recording
push_off_end = takeoff_idx

fz_push = fz[push_off_start:push_off_end]
time_push = time[push_off_start:push_off_end]

# Integrate force to get impulse
# Impulse = ∫F dt
impulse = np.trapz(fz_push, time_push)  # N·s

# Takeoff velocity
# v = impulse / mass
mass = bodyweight / 9.81  # kg
takeoff_velocity = impulse / mass - 9.81 * (time_push[-1] - time_push[0])

print(f"Push-off impulse: {impulse:.1f} N·s")
print(f"Takeoff velocity: {takeoff_velocity:.2f} m/s")
```

## Filtering Force Platform Data

```python
import labanalysis as laban

# Apply butterworth filter to entire ForcePlatform
fp_filtered = fp.apply(
    laban.butterworth_filt,
    axis=0,
    inplace=False,
    freq=1000,  # Hz
    cut=10,     # Hz cutoff
    order=4
)

# Now all components (force, origin, torque) are filtered
fz_filtered = fp_filtered.force[fp.vertical_axis]
```

## Coordinate Transformations

### Change Reference Frame

```python
# Transform force to pelvis coordinate system
# (Assuming you have pelvis orientation matrix R_pelvis)

# Get force in global frame
force_global = fp.force

# Transform to pelvis frame
force_pelvis = force_global.change_reference_frame(
    new_x=R_pelvis[:, :, 0],  # Pelvis X axis
    new_y=R_pelvis[:, :, 1],  # Pelvis Y axis  
    new_z=R_pelvis[:, :, 2],  # Pelvis Z axis
    new_origin=pelvis_center   # Pelvis origin
)
```

## Multiple Force Platforms

### Bilateral Analysis

```python
# Load data with two force platforms
record = laban.TimeseriesRecord.from_tdf("bilateral_jump.tdf")

fp_left = record['FP1']
fp_right = record['FP2']

# Get vertical forces
fz_left = fp_left.force[fp_left.vertical_axis].data
fz_right = fp_right.force[fp_right.vertical_axis].data

# Calculate asymmetry
# (During peak force)
peak_left = np.max(fz_left)
peak_right = np.max(fz_right)

asymmetry = (peak_left - peak_right) / (peak_left + peak_right) * 100

print(f"Peak force left: {peak_left:.1f} N")
print(f"Peak force right: {peak_right:.1f} N")  
print(f"Asymmetry: {asymmetry:.1f}%")

# Asymmetry > 10% may indicate bilateral imbalance
```

### Combined COP

```python
# Calculate combined COP from two force platforms
fz_left = fp_left.force[fp_left.vertical_axis].data
fz_right = fp_right.force[fp_right.vertical_axis].data

cop_ml_left = fp_left.origin[fp_left.lateral_axis].data
cop_ml_right = fp_right.origin[fp_right.lateral_axis].data

# Weighted average COP
total_fz = fz_left + fz_right
cop_ml_combined = (cop_ml_left * fz_left + cop_ml_right * fz_right) / total_fz

# Handle division by zero
cop_ml_combined[total_fz < 10] = np.nan
```

## Troubleshooting

### Issue: "origin must be an instance of Point3D"

```python
# WRONG: Passing Signal3D as origin
fp = laban.ForcePlatform(origin=some_signal, force=grf, torque=torque)

# RIGHT: origin must be Point3D
cop = laban.Point3D(cop_data, time, columns=['X', 'Y', 'Z'], unit='mm')
fp = laban.ForcePlatform(origin=cop, force=grf, torque=torque)
```

### Issue: Axis Mismatch Error

All three components must use the same axis convention:

```python
# Create components with consistent axes
cop = laban.Point3D(
    cop_data, time, 
    vertical_axis='Y',
    anteroposterior_axis='Z'
)

grf = laban.Signal3D(
    force_data, time,
    vertical_axis='Y',  # Must match cop
    anteroposterior_axis='Z'  # Must match cop
)

torque = laban.Signal3D(
    torque_data, time,
    vertical_axis='Y',  # Must match cop
    anteroposterior_axis='Z'  # Must match cop
)
```

### Issue: NaN Values in COP

COP is undefined when no force is applied (e.g., during flight phase):

```python
cop_ml = fp.origin[fp.lateral_axis].data

# Remove NaN before analysis
cop_ml_clean = cop_ml[~np.isnan(cop_ml)]

# Or fill NaN using interpolation
cop_filled = fp.origin.fillna(method='linear')
```

## Export to DataFrame

```python
# Convert to pandas DataFrame
df = fp.to_dataframe()

print(df.columns)
# Output: ['origin X_mm', 'origin Y_mm', 'origin Z_mm', 
#          'force X_N', 'force Y_N', 'force Z_N',
#          'torque X_Nm', 'torque Y_Nm', 'torque Z_Nm']

# Export to CSV
df.to_csv("force_platform_data.csv")

# Export to Excel
df.to_excel("force_platform_data.xlsx")
```

## See Also

- [Test Protocols: Jump Tests](../test-protocols/jump-tests.md) - Jump analysis using force platforms
- [Test Protocols: Gait Analysis](../test-protocols/gait-analysis.md) - Gait analysis with force platforms
- [Test Protocols: Balance Tests](../test-protocols/balance-tests.md) - Balance assessment using COP
- [Signal Processing: Filtering](../signal-processing/filtering.md) - Filtering force platform data
- [API Reference: Records](../../api/records/records.md#forceplatform) - Complete ForcePlatform API

---

**Force Platforms**: Essential tool for measuring ground reaction forces, center of pressure, and torques during human movement analysis.
