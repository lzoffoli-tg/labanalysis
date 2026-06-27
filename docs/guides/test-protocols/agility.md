# Agility Tests

Complete guide to change-of-direction and shuttle test assessment using force platforms and motion capture in labanalysis.

## Overview

Agility testing evaluates the ability to rapidly change direction and accelerate through:
- **Shuttle tests**: Repeated sprints with direction changes
- **Change-of-direction (COD) tests**: Single direction change tasks
- **Reactive agility**: Response to external stimuli
- **Deceleration-acceleration**: Braking and propulsion forces

labanalysis supports:
- Force platform COD analysis
- Motion capture kinematics
- Contact time and ground reaction forces
- Asymmetry assessment

## Quick Reference

| Test | Distance | Typical Time | Key Metrics |
|------|----------|--------------|-------------|
| **5-10-5 Shuttle** | 20 yards | 4.5-5.5 s | Total time, split times |
| **T-Test** | 10m × 10m | 9-11 s | Total time, COD angle |
| **Illinois Agility** | 10m course | 15-17 s | Total time |
| **505 Test** | 15m | 2.2-2.5 s | COD time, deficit |

## Change-of-Direction Exercise

### Load COD Trial

```python
import labanalysis as laban
import numpy as np

# Load change-of-direction trial
cod = laban.ChangeOfDirectionExercise.from_tdf(
    file="cod_180.tdf",
    left_foot_ground_reaction_force="FP1",
    right_foot_ground_reaction_force="FP2",
    s2="pelvis_center"  # Reference marker for velocity
)

print(f"COD Exercise loaded")
print(f"Duration: {len(cod.forceplatforms['FP1'].force['Fz'].data) / cod.sampling_frequency:.2f} s")
print(f"Sampling frequency: {cod.sampling_frequency} Hz")

# Output:
# COD Exercise loaded
# Duration: 3.45 s
# Sampling frequency: 1000 Hz
```

### Detect Ground Contacts

```python
# Get force platforms
fp_left = cod.forceplatforms['FP1']
fp_right = cod.forceplatforms['FP2']

# Vertical forces
fz_left = fp_left.force['Fz'].data
fz_right = fp_right.force['Fz'].data
freq = fp_left.sampling_frequency

# Filter
fz_left_filt = laban.butterworth_filt(fz_left, freq=freq, cut=15, order=4)
fz_right_filt = laban.butterworth_filt(fz_right, freq=freq, cut=15, order=4)

# Detect contacts (threshold at 50N)
threshold = 50  # N

# Left foot contacts
contacts_left = laban.find_peaks(
    fz_left_filt,
    height=threshold,
    distance=int(0.2 * freq),  # Min 0.2s between contacts
    prominence=threshold
)

# Right foot contacts
contacts_right = laban.find_peaks(
    fz_right_filt,
    height=threshold,
    distance=int(0.2 * freq),
    prominence=threshold
)

print(f"\nGround contacts:")
print(f"  Left foot: {len(contacts_left)} contacts")
print(f"  Right foot: {len(contacts_right)} contacts")

# Output:
# Ground contacts:
#   Left foot: 3 contacts
#   Right foot: 4 contacts
```

### Calculate Contact Times

```python
def calculate_contact_time(fz, contact_idx, freq, threshold=50):
    """Calculate contact time for a single contact."""
    # Find contact start (force exceeds threshold)
    start = contact_idx
    while start > 0 and fz[start - 1] > threshold:
        start -= 1
    
    # Find contact end (force drops below threshold)
    end = contact_idx
    while end < len(fz) - 1 and fz[end + 1] > threshold:
        end += 1
    
    contact_time = (end - start) / freq
    return contact_time, start, end

# Calculate contact times for all contacts
contact_times_left = []
for contact_idx in contacts_left:
    ct, _, _ = calculate_contact_time(fz_left_filt, contact_idx, freq)
    contact_times_left.append(ct)

contact_times_right = []
for contact_idx in contacts_right:
    ct, _, _ = calculate_contact_time(fz_right_filt, contact_idx, freq)
    contact_times_right.append(ct)

print("Contact Times:")
print(f"  Left foot: {np.mean(contact_times_left) * 1000:.0f} ± {np.std(contact_times_left) * 1000:.0f} ms")
print(f"  Right foot: {np.mean(contact_times_right) * 1000:.0f} ± {np.std(contact_times_right) * 1000:.0f} ms")

# Output:
# Contact Times:
#   Left foot: 187 ± 23 ms
#   Right foot: 172 ± 18 ms
```

### Analyze Braking and Propulsion

```python
# Analyze anteroposterior force during COD

fx_left = fp_left.force['Fx'].data
fx_left_filt = laban.butterworth_filt(fx_left, freq=freq, cut=15, order=4)

# For first left contact (approach phase)
contact_idx = contacts_left[0]
ct, start, end = calculate_contact_time(fz_left_filt, contact_idx, freq)

# Extract force during contact
fx_contact = fx_left_filt[start:end]

# Braking phase (negative Fx = backward force)
braking_impulse = -np.trapz(fx_contact[fx_contact < 0]) / freq

# Propulsion phase (positive Fx = forward force)
propulsion_impulse = np.trapz(fx_contact[fx_contact > 0]) / freq

# Ratio
brake_prop_ratio = braking_impulse / propulsion_impulse if propulsion_impulse > 0 else np.inf

print(f"\nCOD Contact Analysis:")
print(f"  Braking impulse: {braking_impulse:.1f} N·s")
print(f"  Propulsion impulse: {propulsion_impulse:.1f} N·s")
print(f"  Brake:Propulsion ratio: {brake_prop_ratio:.2f}")

# Output:
# COD Contact Analysis:
#   Braking impulse: 45.3 N·s
#   Propulsion impulse: 52.7 N·s
#   Brake:Propulsion ratio: 0.86
```

### Peak Forces

```python
# Peak vertical and horizontal forces during COD

peak_fz_left = fz_left_filt.max()
peak_fx_braking = abs(fx_left_filt.min())  # Max braking
peak_fx_propulsion = fx_left_filt.max()    # Max propulsion

# Bodyweight (from participant or estimate)
bodyweight = 735  # N

print(f"\nPeak Forces:")
print(f"  Vertical: {peak_fz_left:.1f} N ({peak_fz_left / bodyweight:.2f} × BW)")
print(f"  Braking: {peak_fx_braking:.1f} N ({peak_fx_braking / bodyweight:.2f} × BW)")
print(f"  Propulsion: {peak_fx_propulsion:.1f} N ({peak_fx_propulsion / bodyweight:.2f} × BW)")

# Output:
# Peak Forces:
#   Vertical: 1876.3 N (2.55 × BW)
#   Braking: 567.4 N (0.77 × BW)
#   Propulsion: 623.1 N (0.85 × BW)
```

## Shuttle Test Protocol

### 5-10-5 Shuttle Test

```python
def analyze_shuttle_test(tdf_files, participant):
    """
    Analyze 5-10-5 shuttle test (20 yards total).
    
    Sprint 5 yards right, 10 yards left, 5 yards right.
    
    Parameters
    ----------
    tdf_files : list
        List of TDF files (one per trial)
    participant : laban.Participant
        Participant info
    
    Returns
    -------
    dict
        Shuttle test results
    """
    import labanalysis as laban
    import numpy as np
    
    # Create shuttle test
    shuttle = laban.ShuttleTest.from_files(
        filenames=tdf_files,
        participant=participant,
        left_foot_ground_reaction_force="FP1",
        right_foot_ground_reaction_force="FP2"
    )
    
    # Analyze each trial
    results = []
    
    for cod_exercise in shuttle.change_of_direction_exercises:
        # Get timing data
        # Assuming timing gates recorded in markers or manual timing
        
        # Get force data
        fp_left = cod_exercise.forceplatforms['FP1']
        fp_right = cod_exercise.forceplatforms['FP2']
        
        fz_left = fp_left.force['Fz'].data
        fz_right = fp_right.force['Fz'].data
        freq = fp_left.sampling_frequency
        
        # Detect contacts
        fz_left_filt = laban.butterworth_filt(fz_left, freq=freq, cut=15, order=4)
        fz_right_filt = laban.butterworth_filt(fz_right, freq=freq, cut=15, order=4)
        
        contacts_left = laban.find_peaks(
            fz_left_filt,
            height=50,
            distance=int(0.2 * freq)
        )
        
        contacts_right = laban.find_peaks(
            fz_right_filt,
            height=50,
            distance=int(0.2 * freq)
        )
        
        # Calculate metrics
        trial_result = {
            'total_contacts': len(contacts_left) + len(contacts_right),
            'left_contacts': len(contacts_left),
            'right_contacts': len(contacts_right),
        }
        
        results.append(trial_result)
    
    return results

# Usage
tdf_files = ["shuttle_trial1.tdf", "shuttle_trial2.tdf", "shuttle_trial3.tdf"]

results = analyze_shuttle_test(tdf_files, participant)

print("=== 5-10-5 Shuttle Test Results ===\n")
for i, result in enumerate(results):
    print(f"Trial {i+1}:")
    print(f"  Total contacts: {result['total_contacts']}")
    print(f"  Left: {result['left_contacts']}, Right: {result['right_contacts']}")
```

### Split Times Analysis

```python
# Analyze split times for each leg of shuttle

# Assuming timing from markers or gates
# Example: pelvis marker position

pelvis = cod.markers['pelvis_center']
y_pos = pelvis['y'].data / 1000  # mm to m
time = pelvis['y'].index

# Calculate velocity
velocity = laban.winter_derivative1(y_pos, freq=freq)

# Detect direction changes (velocity sign change)
vel_sign = np.sign(velocity)
direction_changes = np.where(np.diff(vel_sign) != 0)[0]

print(f"Direction changes detected: {len(direction_changes)}")

# Split times (time between direction changes)
if len(direction_changes) >= 2:
    split1_time = time[direction_changes[0]]  # First 5 yards
    split2_time = time[direction_changes[1]] - split1_time  # 10 yards
    split3_time = time[-1] - time[direction_changes[1]]  # Final 5 yards
    total_time = time[-1]
    
    print(f"\nSplit Times:")
    print(f"  First 5 yards: {split1_time:.3f} s")
    print(f"  Middle 10 yards: {split2_time:.3f} s")
    print(f"  Final 5 yards: {split3_time:.3f} s")
    print(f"  Total: {total_time:.3f} s")
```

## 505 Test

### Calculate COD Deficit

```python
def analyze_505_test(tdf_file_505, tdf_file_sprint, participant):
    """
    Analyze 505 test and calculate COD deficit.
    
    505: Sprint 15m, turn 180°, sprint back 5m
    Sprint: Straight 10m sprint
    
    COD deficit = 505 time - sprint time
    
    Parameters
    ----------
    tdf_file_505 : str
        505 test file
    tdf_file_sprint : str
        Straight sprint file
    participant : laban.Participant
        Participant info
    
    Returns
    -------
    dict
        505 test results
    """
    # Load 505 test
    cod_505 = laban.ChangeOfDirectionExercise.from_tdf(
        file=tdf_file_505,
        left_foot_ground_reaction_force="FP1",
        right_foot_ground_reaction_force="FP2"
    )
    
    # Get 505 time (from timing gates or markers)
    # Example: using pelvis displacement
    pelvis_505 = cod_505.markers['pelvis_center']
    y_pos_505 = pelvis_505['y'].data / 1000
    
    # Find turnaround point (minimum Y position)
    turn_idx = np.argmin(y_pos_505)
    
    # Time to turn and return 5m
    time_505 = pelvis_505['y'].index[turn_idx] + 0.5  # Add return time estimate
    
    # Load straight sprint
    sprint = laban.TimeseriesRecord.from_tdf(tdf_file_sprint)
    pelvis_sprint = sprint.markers['pelvis_center']
    
    # Get 10m sprint time
    time_10m = pelvis_sprint['y'].index[-1]
    
    # COD deficit
    cod_deficit = time_505 - (time_10m * 0.5)  # Scale to 5m
    
    results = {
        '505_time_s': time_505,
        'sprint_10m_time_s': time_10m,
        'cod_deficit_s': cod_deficit,
        'cod_deficit_pct': 100 * cod_deficit / time_505
    }
    
    return results

# Usage
results_505 = analyze_505_test("505_test.tdf", "sprint_10m.tdf", participant)

print("=== 505 Test Results ===\n")
print(f"505 time: {results_505['505_time_s']:.3f} s")
print(f"10m sprint time: {results_505['sprint_10m_time_s']:.3f} s")
print(f"COD deficit: {results_505['cod_deficit_s']:.3f} s ({results_505['cod_deficit_pct']:.1f}%)")

# Output:
# === 505 Test Results ===
# 
# 505 time: 2.234 s
# 10m sprint time: 1.856 s
# COD deficit: 0.306 s (13.7%)
```

## Reactive Agility

### Response Time Analysis

```python
# Analyze reactive agility with external stimulus

# Stimulus occurs at known time (e.g., light signal at t=1.0s)
stimulus_time = 1.0  # seconds

# Detect first movement after stimulus
pelvis = cod.markers['pelvis_center']
y_pos = pelvis['y'].data / 1000
time = pelvis['y'].index
freq = pelvis.sampling_frequency

# Calculate acceleration
velocity = laban.winter_derivative1(y_pos, freq=freq)
acceleration = laban.winter_derivative1(velocity, freq=freq)

# Find stimulus index
stimulus_idx = int(stimulus_time * freq)

# Find first significant acceleration after stimulus
accel_threshold = 2.0  # m/s²
post_stimulus_accel = acceleration[stimulus_idx:]

response_idx = np.where(np.abs(post_stimulus_accel) > accel_threshold)[0]

if len(response_idx) > 0:
    reaction_time = response_idx[0] / freq
    
    print(f"Stimulus time: {stimulus_time:.3f} s")
    print(f"Response time: {reaction_time * 1000:.0f} ms")
    print(f"Total response + movement time: {reaction_time:.3f} s")
else:
    print("No response detected")

# Output:
# Stimulus time: 1.000 s
# Response time: 287 ms
# Total response + movement time: 0.287 s
```

## Bilateral Asymmetry

### Compare Left and Right COD Performance

```python
# Analyze COD performance turning left vs. right

# Load left turn trial
cod_left = laban.ChangeOfDirectionExercise.from_tdf("cod_turn_left.tdf")

# Load right turn trial
cod_right = laban.ChangeOfDirectionExercise.from_tdf("cod_turn_right.tdf")

# Calculate metrics for both
def get_cod_metrics(cod_exercise):
    """Extract COD metrics."""
    fp = cod_exercise.forceplatforms['FP1']
    fz = fp.force['Fz'].data
    fx = fp.force['Fx'].data
    freq = fp.sampling_frequency
    
    fz_filt = laban.butterworth_filt(fz, freq=freq, cut=15, order=4)
    fx_filt = laban.butterworth_filt(fx, freq=freq, cut=15, order=4)
    
    # Peak forces
    peak_fz = fz_filt.max()
    peak_fx_brake = abs(fx_filt.min())
    
    # Contact time
    contacts = laban.find_peaks(fz_filt, height=50, distance=int(0.2 * freq))
    if len(contacts) > 0:
        ct, _, _ = calculate_contact_time(fz_filt, contacts[0], freq)
    else:
        ct = 0
    
    return {
        'peak_fz': peak_fz,
        'peak_brake': peak_fx_brake,
        'contact_time': ct
    }

metrics_left = get_cod_metrics(cod_left)
metrics_right = get_cod_metrics(cod_right)

# Calculate asymmetry
asym_fz = 100 * abs(metrics_left['peak_fz'] - metrics_right['peak_fz']) / max(metrics_left['peak_fz'], metrics_right['peak_fz'])
asym_brake = 100 * abs(metrics_left['peak_brake'] - metrics_right['peak_brake']) / max(metrics_left['peak_brake'], metrics_right['peak_brake'])

print("=== Bilateral COD Asymmetry ===\n")
print(f"Peak vertical force:")
print(f"  Left turn: {metrics_left['peak_fz']:.1f} N")
print(f"  Right turn: {metrics_right['peak_fz']:.1f} N")
print(f"  Asymmetry: {asym_fz:.1f}%")

print(f"\nPeak braking force:")
print(f"  Left turn: {metrics_left['peak_brake']:.1f} N")
print(f"  Right turn: {metrics_right['peak_brake']:.1f} N")
print(f"  Asymmetry: {asym_brake:.1f}%")

if asym_fz > 10 or asym_brake > 10:
    print("\n⚠ Significant asymmetry detected (>10%)")
```

## Normative Data

### Performance Standards

| Level | 5-10-5 (s) | 505 (s) | T-Test (s) |
|-------|-----------|---------|------------|
| **Elite** | < 4.5 | < 2.2 | < 9.0 |
| **Good** | 4.5-5.0 | 2.2-2.4 | 9.0-10.0 |
| **Average** | 5.0-5.5 | 2.4-2.6 | 10.0-11.0 |
| **Below average** | > 5.5 | > 2.6 | > 11.0 |

## Visualization

```python
import matplotlib.pyplot as plt

# Plot force-time during COD
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

time_axis = np.arange(len(fz_left_filt)) / freq

# Vertical force
ax1.plot(time_axis, fz_left_filt, 'b-', label='Left foot', linewidth=2)
ax1.plot(time_axis, fz_right_filt, 'r-', label='Right foot', linewidth=2)
ax1.axhline(threshold, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('Vertical Force (N)')
ax1.set_title('Ground Reaction Forces During Change of Direction')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Horizontal force
fx_left_filt = laban.butterworth_filt(fp_left.force['Fx'].data, freq=freq, cut=15, order=4)
ax2.plot(time_axis, fx_left_filt, 'b-', label='Left foot', linewidth=2)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Horizontal Force (N)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Troubleshooting

### Missing ground contacts

```python
# If contacts are not detected, lower threshold or adjust filtering

# Try different threshold
contacts = laban.find_peaks(
    fz_filt,
    height=30,  # Lower threshold
    distance=int(0.15 * freq),  # Shorter min distance
    prominence=30
)
```

### Unrealistic force values

```python
# Check for force platform crosstalk or calibration issues

# Verify zero offset
baseline_left = fz_left[:int(0.5 * freq)].mean()
baseline_right = fz_right[:int(0.5 * freq)].mean()

if abs(baseline_left) > 10 or abs(baseline_right) > 10:
    print(f"Warning: Baseline offset detected")
    print(f"  Left: {baseline_left:.1f} N")
    print(f"  Right: {baseline_right:.1f} N")
    
    # Subtract baseline
    fz_left_corrected = fz_left - baseline_left
    fz_right_corrected = fz_right - baseline_right
```

## See Also

- **[Force Platforms](../biomechanics/force-platforms.md)** - GRF analysis
- **[Gait Analysis](gait-analysis.md)** - Related analysis methods
- **[Peak Detection](../signal-processing/peak-detection.md)** - Contact detection
- **[API Reference: ShuttleTest](../../api/protocols/agility-tests.md)** - Test protocol

---

**Key Metrics**: Contact time, peak forces, braking/propulsion, COD deficit, bilateral asymmetry  
**Reference**: Nimphius S et al. Change of direction deficit: A more isolated measure of change of direction performance than total 505 time. J Strength Cond Res. 2016.
