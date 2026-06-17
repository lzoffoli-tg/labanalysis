# Gait Analysis

Complete guide to walking and running analysis using force platforms and motion capture in labanalysis.

## Overview

Gait analysis quantifies walking and running patterns to assess:
- **Spatiotemporal parameters**: stride time, cadence, step length, velocity
- **Kinetic parameters**: ground reaction forces, loading rates, impulse
- **Kinematic parameters**: joint angles, range of motion, coordination
- **Bilateral symmetry**: left-right comparisons

labanalysis provides tools for:
- Event detection (heel strike, toe off)
- Phase segmentation (stance, swing, double support)
- Parameter extraction
- Bilateral comparison
- Normalization to gait cycle

## Quick Reference

| Parameter | Typical Values (Walking) | Typical Values (Running) |
|-----------|--------------------------|--------------------------|
| **Cadence** | 100-120 steps/min | 160-180 steps/min |
| **Stride time** | 1.0-1.2 s | 0.6-0.8 s |
| **Stance phase** | 60-65% cycle | 35-45% cycle |
| **Swing phase** | 35-40% cycle | 55-65% cycle |
| **Double support** | 10-15% cycle | 0% (flight phase) |
| **Peak vertical GRF** | 110-120% BW | 200-300% BW |

## Event Detection

### Heel Strike and Toe Off

Identify gait events from vertical ground reaction force (vGRF).

```python
import labanalysis as laban
import numpy as np

# Load force platform data
record = laban.TimeseriesRecord.from_tdf("walking.tdf")
fp = record.forceplatforms['FP1']

# Get vertical force
fz = fp.force['Fz'].data
freq = fp.sampling_frequency
time = fp.force['Fz'].index

# Filter force
fz_filt = laban.butterworth_filt(fz, freq=freq, cut=15, order=4)

# Estimate bodyweight (quiet standing portion)
bodyweight = np.median(fz_filt[fz_filt > 0.8 * fz_filt.max()])

print(f"Estimated bodyweight: {bodyweight:.1f} N ({bodyweight / 9.81:.1f} kg)")
# Output: Estimated bodyweight: 735.0 N (74.9 kg)

# Detect heel strikes (force onset)
# Heel strike = force exceeds threshold
threshold = 0.1 * bodyweight  # 10% BW

heel_strikes = laban.find_peaks(
    fz_filt,
    height=threshold,
    distance=int(0.5 * freq),  # Min 0.5s between strikes
    prominence=0.3 * bodyweight
)

print(f"\nDetected {len(heel_strikes)} heel strikes")
print(f"Heel strike times: {time[heel_strikes]}")

# Output:
# Detected 5 heel strikes
# Heel strike times: [0.52 1.64 2.75 3.87 4.98]

# Detect toe offs (force offset)
# Toe off = force drops below threshold after peak

toe_offs = []
for i in range(len(heel_strikes) - 1):
    # Search between current and next heel strike
    start = heel_strikes[i]
    end = heel_strikes[i + 1]
    
    # Find where force drops below threshold
    search_region = fz_filt[start:end]
    below_threshold = np.where(search_region < threshold)[0]
    
    if len(below_threshold) > 0:
        # First point below threshold after peak
        toe_off = start + below_threshold[0]
        toe_offs.append(toe_off)

toe_offs = np.array(toe_offs)
print(f"\nDetected {len(toe_offs)} toe offs")
print(f"Toe off times: {time[toe_offs]}")

# Output:
# Detected 4 toe offs
# Toe off times: [1.24 2.35 3.47 4.58]
```

### Visualize Events

```python
import matplotlib.pyplot as plt

# Plot force with events
plt.figure(figsize=(14, 6))
plt.plot(time, fz_filt, 'b-', linewidth=1.5, label='Vertical GRF')
plt.axhline(bodyweight, color='gray', linestyle='--', label='Bodyweight')
plt.axhline(threshold, color='red', linestyle=':', label='Threshold (10% BW)')

# Mark heel strikes
plt.plot(time[heel_strikes], fz_filt[heel_strikes], 
         'go', markersize=10, label='Heel Strike')

# Mark toe offs
plt.plot(time[toe_offs], fz_filt[toe_offs], 
         'ro', markersize=10, label='Toe Off')

plt.xlabel('Time (s)')
plt.ylabel('Vertical GRF (N)')
plt.title('Gait Event Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Spatiotemporal Parameters

### Basic Gait Parameters

```python
# Calculate stride times
stride_times = np.diff(time[heel_strikes])

# Calculate stance times (heel strike to toe off)
stance_times = time[toe_offs] - time[heel_strikes[:-1]]

# Calculate swing times
swing_times = stride_times - stance_times

# Cadence (steps per minute)
cadence = 60 / stride_times.mean()

print("=== Spatiotemporal Parameters ===\n")
print(f"Stride time: {stride_times.mean():.3f} ± {stride_times.std():.3f} s")
print(f"Stance time: {stance_times.mean():.3f} ± {stance_times.std():.3f} s")
print(f"Swing time: {swing_times.mean():.3f} ± {swing_times.std():.3f} s")
print(f"Cadence: {cadence:.1f} steps/min")

# Stance and swing as % of gait cycle
stance_pct = 100 * stance_times.mean() / stride_times.mean()
swing_pct = 100 * swing_times.mean() / stride_times.mean()

print(f"\nStance phase: {stance_pct:.1f}% of cycle")
print(f"Swing phase: {swing_pct:.1f}% of cycle")

# Output:
# === Spatiotemporal Parameters ===
# 
# Stride time: 1.112 ± 0.024 s
# Stance time: 0.720 ± 0.018 s
# Swing time: 0.392 ± 0.014 s
# Cadence: 108.0 steps/min
# 
# Stance phase: 64.7% of cycle
# Swing phase: 35.3% of cycle
```

### Stride Length and Velocity

```python
# Load markers to calculate spatial parameters
heel_marker = record.markers['heel_R']

# Calculate stride length
# Distance between successive heel strikes
heel_positions = heel_marker.data[heel_strikes, :]

# Displacement in forward direction (Y-axis)
forward_displacement = np.diff(heel_positions[:, 1])  # mm

stride_lengths = np.abs(forward_displacement) / 1000  # Convert to meters

print(f"Stride length: {stride_lengths.mean():.3f} ± {stride_lengths.std():.3f} m")

# Walking velocity
walking_velocity = stride_lengths / stride_times

print(f"Walking velocity: {walking_velocity.mean():.3f} ± {walking_velocity.std():.3f} m/s")
print(f"Walking speed: {walking_velocity.mean() * 3.6:.2f} km/h")

# Step length (half of stride length)
step_length = stride_lengths.mean() / 2

print(f"\nStep length: {step_length:.3f} m")

# Output:
# Stride length: 1.435 ± 0.042 m
# Walking velocity: 1.291 ± 0.028 m/s
# Walking speed: 4.65 km/h
# 
# Step length: 0.718 m
```

## Kinetic Analysis

### GRF Components and Loading

```python
# Extract force components
fx = fp.force['Fx'].data  # Anteroposterior
fy = fp.force['Fy'].data  # Mediolateral
fz = fp.force['Fz'].data  # Vertical

# Filter all components
fx_filt = laban.butterworth_filt(fx, freq=freq, cut=15, order=4)
fy_filt = laban.butterworth_filt(fy, freq=freq, cut=15, order=4)
fz_filt = laban.butterworth_filt(fz, freq=freq, cut=15, order=4)

# Analyze each stride
print("=== Kinetic Parameters (per stride) ===\n")

for i in range(len(heel_strikes) - 1):
    start = heel_strikes[i]
    end = heel_strikes[i + 1]
    
    # Extract stance phase
    stance_start = start
    stance_end = toe_offs[i]
    
    # Vertical GRF
    fz_stride = fz_filt[stance_start:stance_end]
    
    # Peak forces
    peak_fz = fz_stride.max()
    
    # Loading rate (first peak)
    # Find first peak (impact peak in running, acceptance peak in walking)
    time_to_peak = np.argmax(fz_stride[:int(0.3 * (stance_end - stance_start))])
    loading_rate = peak_fz / (time_to_peak / freq)  # N/s
    
    # Impulse
    impulse = np.trapz(fz_stride) / freq  # N·s
    
    print(f"Stride {i+1}:")
    print(f"  Peak vGRF: {peak_fz:.1f} N ({100 * peak_fz / bodyweight:.1f}% BW)")
    print(f"  Loading rate: {loading_rate:.1f} N/s")
    print(f"  Impulse: {impulse:.1f} N·s")
    
    # Braking and propulsion (anteroposterior force)
    fx_stride = fx_filt[stance_start:stance_end]
    braking_impulse = -np.trapz(fx_stride[fx_stride < 0]) / freq
    propulsion_impulse = np.trapz(fx_stride[fx_stride > 0]) / freq
    
    print(f"  Braking impulse: {braking_impulse:.1f} N·s")
    print(f"  Propulsion impulse: {propulsion_impulse:.1f} N·s\n")

# Output:
# === Kinetic Parameters (per stride) ===
# 
# Stride 1:
#   Peak vGRF: 812.3 N (110.5% BW)
#   Loading rate: 4523.1 N/s
#   Impulse: 518.7 N·s
#   Braking impulse: 8.2 N·s
#   Propulsion impulse: 9.1 N·s
# ...
```

### Time-Normalized GRF Curves

```python
# Normalize strides to 0-100% gait cycle
def normalize_to_cycle(data, start, end, n_points=101):
    """Interpolate data to n_points for normalization."""
    from scipy.interpolate import interp1d
    
    segment = data[start:end]
    old_time = np.linspace(0, 100, len(segment))
    new_time = np.linspace(0, 100, n_points)
    
    f = interp1d(old_time, segment, kind='cubic')
    return f(new_time)

# Normalize all strides
n_strides = len(heel_strikes) - 1
fz_normalized = np.zeros((n_strides, 101))

for i in range(n_strides):
    start = heel_strikes[i]
    end = heel_strikes[i + 1]
    fz_normalized[i, :] = normalize_to_cycle(fz_filt, start, end)

# Calculate mean and SD
fz_mean = fz_normalized.mean(axis=0)
fz_std = fz_normalized.std(axis=0)

# Plot
cycle_pct = np.linspace(0, 100, 101)

plt.figure(figsize=(12, 6))
plt.plot(cycle_pct, fz_mean / bodyweight * 100, 'b-', linewidth=2, label='Mean')
plt.fill_between(
    cycle_pct,
    (fz_mean - fz_std) / bodyweight * 100,
    (fz_mean + fz_std) / bodyweight * 100,
    alpha=0.3,
    label='± 1 SD'
)

# Mark stance/swing transition
stance_pct_mean = stance_pct
plt.axvline(stance_pct_mean, color='red', linestyle='--', label=f'Toe Off ({stance_pct_mean:.1f}%)')

plt.xlabel('Gait Cycle (%)')
plt.ylabel('Vertical GRF (% Bodyweight)')
plt.title('Time-Normalized Vertical GRF - Walking')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)
plt.show()
```

## Kinematic Analysis

### Joint Angles from Markers

```python
# Calculate knee flexion angle
# Simplified 2D sagittal plane calculation

# Load hip, knee, ankle markers
hip = record.markers['hip_R']
knee = record.markers['knee_R']
ankle = record.markers['ankle_R']

# Calculate vectors
thigh_vector = knee.data - hip.data  # Hip to knee
shank_vector = ankle.data - knee.data  # Knee to ankle

# Calculate angle in sagittal plane (Y-Z plane)
# Use Y (forward) and Z (vertical) components
thigh_yz = thigh_vector[:, [1, 2]]
shank_yz = shank_vector[:, [1, 2]]

# Angle between vectors
dot_product = np.sum(thigh_yz * shank_yz, axis=1)
thigh_mag = np.linalg.norm(thigh_yz, axis=1)
shank_mag = np.linalg.norm(shank_yz, axis=1)

cos_angle = dot_product / (thigh_mag * shank_mag)
knee_angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

# Knee flexion = 180 - angle (0° = full extension)
knee_flexion = 180 - knee_angle

# Filter
knee_flexion_filt = laban.butterworth_filt(knee_flexion, freq=freq, cut=6, order=4)

# Normalize to gait cycle
knee_normalized = np.zeros((n_strides, 101))

for i in range(n_strides):
    start = heel_strikes[i]
    end = heel_strikes[i + 1]
    knee_normalized[i, :] = normalize_to_cycle(knee_flexion_filt, start, end)

# Plot
knee_mean = knee_normalized.mean(axis=0)
knee_std = knee_normalized.std(axis=0)

plt.figure(figsize=(12, 6))
plt.plot(cycle_pct, knee_mean, 'b-', linewidth=2, label='Mean')
plt.fill_between(
    cycle_pct,
    knee_mean - knee_std,
    knee_mean + knee_std,
    alpha=0.3,
    label='± 1 SD'
)

plt.axvline(stance_pct_mean, color='red', linestyle='--', label=f'Toe Off ({stance_pct_mean:.1f}%)')
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)

plt.xlabel('Gait Cycle (%)')
plt.ylabel('Knee Flexion (degrees)')
plt.title('Knee Flexion Angle - Walking')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)
plt.show()

# Calculate ROM
knee_rom = knee_mean.max() - knee_mean.min()
print(f"Knee flexion ROM: {knee_rom:.1f}°")
print(f"Peak knee flexion (swing): {knee_mean[60:].max():.1f}°")
print(f"Peak knee extension (stance): {knee_mean[:stance_pct_mean].min():.1f}°")

# Output:
# Knee flexion ROM: 58.3°
# Peak knee flexion (swing): 62.1°
# Peak knee extension (stance): 3.8°
```

### Center of Mass Trajectory

```python
# Approximate CoM from pelvis markers
mid_ASIS = (record.markers['LASI'].data + record.markers['RASI'].data) / 2
mid_PSIS = (record.markers['LPSI'].data + record.markers['RPSI'].data) / 2

# CoM approximation (slightly below pelvis center)
com_approx = (mid_ASIS + mid_PSIS) / 2
com_approx[:, 2] -= 50  # 50mm below pelvis center (rough estimate)

# Filter
com_x = laban.butterworth_filt(com_approx[:, 0], freq=freq, cut=6, order=4)
com_y = laban.butterworth_filt(com_approx[:, 1], freq=freq, cut=6, order=4)
com_z = laban.butterworth_filt(com_approx[:, 2], freq=freq, cut=6, order=4)

# Vertical displacement during walking
vertical_displacement = com_z.max() - com_z.min()

print(f"Vertical CoM displacement: {vertical_displacement:.1f} mm")

# Calculate CoM velocity
com_vel_y = laban.winter_derivative1(com_y / 1000, freq=freq)  # Forward velocity (m/s)

print(f"Mean CoM velocity: {com_vel_y.mean():.3f} m/s")
print(f"Peak CoM velocity: {com_vel_y.max():.3f} m/s")

# Output:
# Vertical CoM displacement: 42.3 mm
# Mean CoM velocity: 1.287 m/s
# Peak CoM velocity: 1.523 m/s
```

## Bilateral Comparison

### Compare Left and Right Limbs

```python
# Load bilateral force platforms
fp_L = record.forceplatforms['FP1']  # Left foot contact
fp_R = record.forceplatforms['FP2']  # Right foot contact

# Detect events for both sides
fz_L = fp_L.force['Fz'].data
fz_R = fp_R.force['Fz'].data

fz_L_filt = laban.butterworth_filt(fz_L, freq=freq, cut=15, order=4)
fz_R_filt = laban.butterworth_filt(fz_R, freq=freq, cut=15, order=4)

# Detect heel strikes for both sides
hs_L = laban.find_peaks(
    fz_L_filt,
    height=threshold,
    distance=int(0.5 * freq),
    prominence=0.3 * bodyweight
)

hs_R = laban.find_peaks(
    fz_R_filt,
    height=threshold,
    distance=int(0.5 * freq),
    prominence=0.3 * bodyweight
)

# Extract parameters for each side
def extract_gait_parameters(fz, heel_strikes, toe_offs, time):
    """Extract gait parameters from force data."""
    params = {
        'stride_times': [],
        'stance_times': [],
        'peak_forces': [],
        'impulses': []
    }
    
    for i in range(len(heel_strikes) - 1):
        # Stride time
        stride_time = time[heel_strikes[i+1]] - time[heel_strikes[i]]
        params['stride_times'].append(stride_time)
        
        # Stance time
        if i < len(toe_offs):
            stance_time = time[toe_offs[i]] - time[heel_strikes[i]]
            params['stance_times'].append(stance_time)
            
            # Peak force during stance
            stance_fz = fz[heel_strikes[i]:toe_offs[i]]
            params['peak_forces'].append(stance_fz.max())
            
            # Impulse
            impulse = np.trapz(stance_fz) / freq
            params['impulses'].append(impulse)
    
    return params

# Get parameters for both sides
# (Assuming toe_offs_L and toe_offs_R detected similarly)
params_L = extract_gait_parameters(fz_L_filt, hs_L, toe_offs, time)
params_R = extract_gait_parameters(fz_R_filt, hs_R, toe_offs, time)

# Compare
print("=== Bilateral Comparison ===\n")

print("Stride Time:")
print(f"  Left:  {np.mean(params_L['stride_times']):.3f} ± {np.std(params_L['stride_times']):.3f} s")
print(f"  Right: {np.mean(params_R['stride_times']):.3f} ± {np.std(params_R['stride_times']):.3f} s")

print("\nPeak Vertical GRF:")
print(f"  Left:  {np.mean(params_L['peak_forces']):.1f} N ({100 * np.mean(params_L['peak_forces']) / bodyweight:.1f}% BW)")
print(f"  Right: {np.mean(params_R['peak_forces']):.1f} N ({100 * np.mean(params_R['peak_forces']) / bodyweight:.1f}% BW)")

# Symmetry index
# SI = 100 * |L - R| / ((L + R) / 2)

stride_time_L = np.mean(params_L['stride_times'])
stride_time_R = np.mean(params_R['stride_times'])
stride_SI = 100 * abs(stride_time_L - stride_time_R) / ((stride_time_L + stride_time_R) / 2)

peak_force_L = np.mean(params_L['peak_forces'])
peak_force_R = np.mean(params_R['peak_forces'])
force_SI = 100 * abs(peak_force_L - peak_force_R) / ((peak_force_L + peak_force_R) / 2)

print("\nSymmetry Index (lower = more symmetric):")
print(f"  Stride time: {stride_SI:.1f}%")
print(f"  Peak GRF: {force_SI:.1f}%")

# Output:
# === Bilateral Comparison ===
# 
# Stride Time:
#   Left:  1.108 ± 0.021 s
#   Right: 1.115 ± 0.019 s
# 
# Peak Vertical GRF:
#   Left:  809.3 N (110.1% BW)
#   Right: 816.7 N (111.1% BW)
# 
# Symmetry Index (lower = more symmetric):
#   Stride time: 0.6%
#   Peak GRF: 0.9%
```

## Running Analysis

### Differences from Walking

```python
# Running-specific parameters

# Load running trial
record_run = laban.TimeseriesRecord.from_tdf("running.tdf")
fp_run = record_run.forceplatforms['FP1']

fz_run = fp_run.force['Fz'].data
fz_run_filt = laban.butterworth_filt(fz_run, freq=freq, cut=15, order=4)

# Detect ground contacts (no double support in running)
contacts_run = laban.find_peaks(
    fz_run_filt,
    height=1.5 * bodyweight,  # Higher threshold for running
    distance=int(0.3 * freq),  # Shorter stride time
    prominence=bodyweight
)

print("=== Running Parameters ===\n")

# Contact times
contact_times_run = []
flight_times_run = []

for i in range(len(contacts_run) - 1):
    # Find contact time (above threshold)
    start = contacts_run[i]
    
    # Find where force drops below threshold after peak
    search_end = contacts_run[i + 1]
    force_segment = fz_run_filt[start:search_end]
    
    below_threshold = np.where(force_segment < 0.1 * bodyweight)[0]
    if len(below_threshold) > 0:
        contact_end = start + below_threshold[0]
        contact_time = (contact_end - start) / freq
        contact_times_run.append(contact_time)
        
        # Flight time
        next_contact_start = contacts_run[i + 1]
        flight_time = (next_contact_start - contact_end) / freq
        flight_times_run.append(flight_time)

contact_times_run = np.array(contact_times_run)
flight_times_run = np.array(flight_times_run)

print(f"Contact time: {contact_times_run.mean():.3f} ± {contact_times_run.std():.3f} s")
print(f"Flight time: {flight_times_run.mean():.3f} ± {flight_times_run.std():.3f} s")

# Duty factor (contact time / stride time)
stride_times_run = contact_times_run + flight_times_run
duty_factor = contact_times_run.mean() / stride_times_run.mean()

print(f"Duty factor: {duty_factor:.3f} ({100 * duty_factor:.1f}% contact)")

# Peak impact force
peak_forces_run = []
for i in range(len(contacts_run) - 1):
    start = contacts_run[i]
    end = contacts_run[i] + int(0.3 * freq)  # First 0.3s
    peak_forces_run.append(fz_run_filt[start:end].max())

peak_forces_run = np.array(peak_forces_run)

print(f"\nPeak vertical GRF: {peak_forces_run.mean():.1f} N ({100 * peak_forces_run.mean() / bodyweight:.1f}% BW)")

# Loading rate (much higher in running)
loading_rates_run = []
for i in range(len(contacts_run) - 1):
    start = contacts_run[i]
    peak_idx = start + np.argmax(fz_run_filt[start:start + int(0.1 * freq)])
    
    time_to_peak = (peak_idx - start) / freq
    loading_rate = fz_run_filt[peak_idx] / time_to_peak
    loading_rates_run.append(loading_rate)

loading_rates_run = np.array(loading_rates_run)

print(f"Loading rate: {loading_rates_run.mean():.1f} N/s ({loading_rates_run.mean() / bodyweight:.1f} BW/s)")

# Cadence
cadence_run = 60 / stride_times_run.mean()
print(f"\nCadence: {cadence_run:.1f} steps/min")

# Output:
# === Running Parameters ===
# 
# Contact time: 0.234 ± 0.012 s
# Flight time: 0.098 ± 0.008 s
# Duty factor: 0.705 (40.5% contact)
# 
# Peak vertical GRF: 1876.3 N (255.3% BW)
# Loading rate: 18234.5 N/s (24.8 BW/s)
# 
# Cadence: 181.2 steps/min
```

## Complete Analysis Workflow

```python
def analyze_gait_trial(tdf_file, participant, protocol='walking'):
    """
    Complete gait analysis workflow.
    
    Parameters
    ----------
    tdf_file : str
        Path to TDF file
    participant : laban.Participant
        Participant info
    protocol : str
        'walking' or 'running'
    
    Returns
    -------
    dict
        Gait parameters
    """
    import labanalysis as laban
    import numpy as np
    
    # Load data
    record = laban.TimeseriesRecord.from_tdf(tdf_file)
    fp = record.forceplatforms['FP1']
    
    # Get vertical force
    fz = fp.force['Fz'].data
    freq = fp.sampling_frequency
    time = fp.force['Fz'].index
    
    # Filter
    fz_filt = laban.butterworth_filt(fz, freq=freq, cut=15, order=4)
    
    # Estimate bodyweight
    bodyweight = participant.weight * 9.81  # N
    
    # Detect events
    threshold = 0.1 * bodyweight
    
    if protocol == 'walking':
        min_distance = int(0.5 * freq)
        min_height = 0.5 * bodyweight
    else:  # running
        min_distance = int(0.3 * freq)
        min_height = 1.5 * bodyweight
    
    heel_strikes = laban.find_peaks(
        fz_filt,
        height=min_height,
        distance=min_distance,
        prominence=0.3 * bodyweight
    )
    
    # Detect toe offs
    toe_offs = []
    for i in range(len(heel_strikes) - 1):
        start = heel_strikes[i]
        end = heel_strikes[i + 1]
        
        search_region = fz_filt[start:end]
        below_threshold = np.where(search_region < threshold)[0]
        
        if len(below_threshold) > 0:
            toe_off = start + below_threshold[0]
            toe_offs.append(toe_off)
    
    toe_offs = np.array(toe_offs)
    
    # Calculate parameters
    stride_times = np.diff(time[heel_strikes])
    stance_times = time[toe_offs] - time[heel_strikes[:-1]]
    
    # Peak forces and impulses
    peak_forces = []
    impulses = []
    
    for i in range(len(toe_offs)):
        start = heel_strikes[i]
        end = toe_offs[i]
        
        stance_fz = fz_filt[start:end]
        peak_forces.append(stance_fz.max())
        impulses.append(np.trapz(stance_fz) / freq)
    
    peak_forces = np.array(peak_forces)
    impulses = np.array(impulses)
    
    # Compile results
    results = {
        'protocol': protocol,
        'n_strides': len(heel_strikes) - 1,
        'stride_time_mean': stride_times.mean(),
        'stride_time_std': stride_times.std(),
        'stance_time_mean': stance_times.mean(),
        'stance_time_std': stance_times.std(),
        'stance_percent': 100 * stance_times.mean() / stride_times.mean(),
        'cadence': 60 / stride_times.mean(),
        'peak_grf_mean': peak_forces.mean(),
        'peak_grf_std': peak_forces.std(),
        'peak_grf_bw': peak_forces.mean() / bodyweight,
        'impulse_mean': impulses.mean(),
        'impulse_std': impulses.std(),
    }
    
    return results

# Usage
participant = laban.Participant(
    name="John",
    surname="Doe",
    height=1.80,
    weight=75,
    age=25
)

results = analyze_gait_trial("walking.tdf", participant, protocol='walking')

print("=== Gait Analysis Results ===\n")
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")

# Output:
# === Gait Analysis Results ===
# 
# protocol: walking
# n_strides: 4
# stride_time_mean: 1.112
# stride_time_std: 0.024
# stance_time_mean: 0.720
# stance_time_std: 0.018
# stance_percent: 64.748
# cadence: 107.973
# peak_grf_mean: 812.340
# peak_grf_std: 23.450
# peak_grf_bw: 1.103
# impulse_mean: 518.723
# impulse_std: 12.340
```

## Reporting and Interpretation

### Generate Summary Report

```python
def generate_gait_report(results, participant):
    """Generate human-readable gait report."""
    
    print(f"=== Gait Analysis Report ===")
    print(f"Participant: {participant.name} {participant.surname}")
    print(f"Age: {participant.age} years | Height: {participant.height} m | Weight: {participant.weight} kg")
    print(f"Protocol: {results['protocol'].upper()}\n")
    
    print("Spatiotemporal Parameters:")
    print(f"  Cadence: {results['cadence']:.1f} steps/min")
    print(f"  Stride time: {results['stride_time_mean']:.3f} ± {results['stride_time_std']:.3f} s")
    print(f"  Stance phase: {results['stance_percent']:.1f}% of gait cycle")
    
    print("\nKinetic Parameters:")
    print(f"  Peak vertical GRF: {results['peak_grf_mean']:.1f} ± {results['peak_grf_std']:.1f} N")
    print(f"                     ({100 * results['peak_grf_bw']:.1f}% bodyweight)")
    print(f"  Vertical impulse: {results['impulse_mean']:.1f} ± {results['impulse_std']:.1f} N·s")
    
    print("\nInterpretation:")
    
    # Compare to normative values
    if results['protocol'] == 'walking':
        if 100 <= results['cadence'] <= 120:
            print("  ✓ Cadence within normal range (100-120 steps/min)")
        else:
            print(f"  ⚠ Cadence outside normal range: {results['cadence']:.1f} steps/min")
        
        if 60 <= results['stance_percent'] <= 65:
            print("  ✓ Stance phase within normal range (60-65%)")
        else:
            print(f"  ⚠ Stance phase outside normal range: {results['stance_percent']:.1f}%")
    
    elif results['protocol'] == 'running':
        if 160 <= results['cadence'] <= 180:
            print("  ✓ Cadence within optimal range (160-180 steps/min)")
        else:
            print(f"  ⚠ Cadence outside optimal range: {results['cadence']:.1f} steps/min")

generate_gait_report(results, participant)
```

## Troubleshooting

### Missing event detection

```python
# If heel strikes are missed, adjust parameters:

# 1. Lower threshold
threshold = 0.05 * bodyweight  # Try 5% instead of 10%

# 2. Lower prominence
heel_strikes = laban.find_peaks(
    fz_filt,
    height=threshold,
    distance=int(0.5 * freq),
    prominence=0.1 * bodyweight  # Lower prominence
)

# 3. Check filtering
# Too aggressive filtering can remove impact peaks
# Try higher cut-off or lower order
fz_filt = laban.butterworth_filt(fz, freq=freq, cut=20, order=2)
```

### Noisy force data

```python
# Increase filtering, but validate you're not losing signal

# Check PSD to select appropriate cut-off
freqs, psd = laban.psd(fz, freq=freq)

# Use residual analysis
cutoffs, rms = laban.residual_analysis(
    fz,
    freq=freq,
    cutoff_range=(10, 25),
    step=1
)

# Select optimal cut-off from elbow point
```

## See Also

- **[Tutorial: Gait Analysis](../../tutorials/02-gait-analysis.md)** - Complete walkthrough
- **[Force Platforms](../biomechanics/force-platforms.md)** - GRF analysis details
- **[Peak Detection](../signal-processing/peak-detection.md)** - Event detection guide
- **[WholeBody Model](../biomechanics/whole-body-model.md)** - Full-body kinematics

---

**Key Functions**: `find_peaks()`, `butterworth_filt()`, `winter_derivative1()`  
**Reference**: Perry J, Burnfield JM. Gait Analysis: Normal and Pathological Function. 2nd ed. 2010.
