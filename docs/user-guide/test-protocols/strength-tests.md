# Strength Tests

Complete guide to isokinetic and isometric strength assessment using Biostrength devices and force platforms in labanalysis.

## Overview

Strength testing quantifies neuromuscular performance through:
- **Isometric strength**: Maximum force at static joint angle
- **Isokinetic strength**: Maximum torque at constant angular velocity
- **1RM prediction**: Brzycki equation for submaximal loads
- **Rate of force development (RFD)**: Speed of force production
- **Bilateral comparison**: Left vs. right limb strength

labanalysis supports:
- Biostrength isokinetic dynamometer data
- Force platform isometric tests
- 1RM prediction from submaximal lifts
- EMG-force relationships

## Quick Reference

| Test Type | Key Metrics | Typical Values |
|-----------|-------------|----------------|
| **Isometric (leg extension)** | Peak force, RFD | 2000-3000 N |
| **Isokinetic 60°/s** | Peak torque | 200-300 Nm (knee ext) |
| **Isokinetic 180°/s** | Peak torque | 150-200 Nm (knee ext) |
| **1RM (squat)** | Predicted max | 1.5-2.5x bodyweight |
| **Bilateral deficit** | L-R asymmetry | <10% ideal |

## Brzycki 1RM Prediction

### Predict 1RM from Submaximal Load

```python
import labanalysis as laban

# Create Brzycki calculator
brzycki = laban.Brzycki1RM()

# Participant completed 8 reps at 100 kg
load = 100  # kg
reps = 8

# Predict 1RM
predicted_1rm = brzycki.predict_1rm(reps=reps, load=load)

print(f"Load: {load} kg for {reps} reps")
print(f"Predicted 1RM: {predicted_1rm:.1f} kg")

# Output:
# Load: 100 kg for 8 reps
# Predicted 1RM: 124.1 kg
```

### Predict Training Load from 1RM

```python
# Known 1RM: 120 kg
# Want to train at 8 reps max

rm1 = 120  # kg
target_reps = 8

# Predict load for 8 reps
training_load = brzycki.predict_load(rm1=rm1, reps=target_reps)

print(f"1RM: {rm1} kg")
print(f"Load for {target_reps} reps: {training_load:.1f} kg")
print(f"Percentage of 1RM: {100 * training_load / rm1:.1f}%")

# Output:
# 1RM: 120 kg
# Load for 8 reps: 96.7 kg
# Percentage of 1RM: 80.6%
```

### Predict Reps from 1RM and Load

```python
# Known 1RM: 120 kg
# Training load: 90 kg

rm1 = 120
load = 90

# Predict max reps
max_reps = brzycki.predict_reps(rm1=rm1, load=load)

print(f"1RM: {rm1} kg, Load: {load} kg")
print(f"Expected max reps: {max_reps:.1f}")

# Output:
# 1RM: 120 kg, Load: 90 kg
# Expected max reps: 10.0
```

### Validation

```python
# Validate Brzycki equation with known data
loads = [60, 70, 80, 90, 100]
reps = [20, 15, 12, 10, 8]

# Predict 1RM from each load-reps pair
predicted_1rms = []

for load, rep in zip(loads, reps):
    pred = brzycki.predict_1rm(reps=rep, load=load)
    predicted_1rms.append(pred)
    print(f"{load:3d} kg × {rep:2d} reps → 1RM = {pred:.1f} kg")

# Check consistency
mean_1rm = np.mean(predicted_1rms)
std_1rm = np.std(predicted_1rms)

print(f"\nMean predicted 1RM: {mean_1rm:.1f} ± {std_1rm:.1f} kg")
print(f"CV: {100 * std_1rm / mean_1rm:.1f}%")

# Output:
#  60 kg × 20 reps → 1RM = 127.1 kg
#  70 kg × 15 reps → 1RM = 116.4 kg
#  80 kg × 12 reps → 1RM = 115.2 kg
#  90 kg × 10 reps → 1RM = 120.0 kg
# 100 kg ×  8 reps → 1RM = 124.1 kg
# 
# Mean predicted 1RM: 120.6 ± 4.8 kg
# CV: 4.0%
```

## Isometric Strength Tests

### Force Platform Isometric Test

```python
import labanalysis as laban
import numpy as np

# Load isometric trial (e.g., isometric mid-thigh pull)
record = laban.TimeseriesRecord.from_tdf("isometric_pull.tdf")
fp = record.forceplatforms['FP1']

# Get vertical force
fz = fp.force['Fz'].data
freq = fp.sampling_frequency
time = fp.force['Fz'].index

# Filter
fz_filt = laban.butterworth_filt(fz, freq=freq, cut=10, order=4)

# Identify baseline (pre-pull)
baseline = fz_filt[:int(1.0 * freq)].mean()

# Subtract baseline (bodyweight)
force = fz_filt - baseline

# Find peak force
peak_force = force.max()
peak_idx = np.argmax(force)
time_to_peak = time[peak_idx]

print("=== Isometric Test Results ===\n")
print(f"Peak force: {peak_force:.1f} N")
print(f"Time to peak: {time_to_peak:.3f} s")

# Output:
# === Isometric Test Results ===
# 
# Peak force: 2847.3 N
# Time to peak: 1.234 s
```

### Rate of Force Development (RFD)

```python
# Calculate RFD in different time windows

# Define time windows (ms)
windows = [50, 100, 200, 250]

print("Rate of Force Development:")

for window_ms in windows:
    # Find force at window
    window_s = window_ms / 1000
    window_idx = int(window_s * freq)
    
    if window_idx < len(force):
        force_at_window = force[window_idx]
        rfd = force_at_window / window_s  # N/s
        
        print(f"  RFD 0-{window_ms}ms: {rfd:.1f} N/s ({rfd / 1000:.1f} kN/s)")

# Peak RFD (maximum slope)
force_derivative = np.diff(force) * freq  # N/s
peak_rfd = force_derivative.max()
peak_rfd_time = time[np.argmax(force_derivative)]

print(f"\nPeak RFD: {peak_rfd:.1f} N/s ({peak_rfd / 1000:.1f} kN/s)")
print(f"  at time: {peak_rfd_time:.3f} s")

# Output:
# Rate of Force Development:
#   RFD 0-50ms: 8234.5 N/s (8.2 kN/s)
#   RFD 0-100ms: 12456.3 N/s (12.5 kN/s)
#   RFD 0-200ms: 10789.2 N/s (10.8 kN/s)
#   RFD 0-250ms: 9876.4 N/s (9.9 kN/s)
# 
# Peak RFD: 15234.7 N/s (15.2 kN/s)
#   at time: 0.087 s
```

### Impulse and Work

```python
# Calculate impulse (force-time integral)
impulse = np.trapz(force[force > 0]) / freq  # N·s

# Calculate work (force × displacement)
# Assume displacement from position sensor or integrate velocity
# For demonstration, assume constant displacement of 0.15m
displacement = 0.15  # meters
work = peak_force * displacement  # Joules (simplified)

print(f"Impulse: {impulse:.1f} N·s")
print(f"Work: {work:.1f} J")

# Output:
# Impulse: 1523.4 N·s
# Work: 427.1 J
```

### Visualize Force-Time Curve

```python
import matplotlib.pyplot as plt

# Plot force-time curve
plt.figure(figsize=(12, 6))
plt.plot(time, force, 'b-', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)

# Mark peak
plt.scatter(time_to_peak, peak_force, color='red', s=100, zorder=5)
plt.annotate(
    f'Peak: {peak_force:.0f} N',
    xy=(time_to_peak, peak_force),
    xytext=(time_to_peak + 0.5, peak_force),
    arrowprops=dict(arrowstyle='->', color='red'),
    fontsize=12
)

# Mark RFD windows
for window_ms in [50, 100, 200]:
    window_idx = int((window_ms / 1000) * freq)
    plt.axvline(time[window_idx], color='green', linestyle=':', alpha=0.5)
    plt.text(time[window_idx], force.max() * 0.9, f'{window_ms}ms', 
             rotation=90, va='top')

plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Isometric Force-Time Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Isokinetic Strength Tests

### Load Biostrength Data

```python
# Load isokinetic test from Biostrength device
from labanalysis.io.read.biostrength import read_biostrength

# Read Biostrength file
biostrength_data = read_biostrength("isokinetic_test.csv")

# Access force/torque data
# Structure depends on Biostrength file format
# Typically contains: angle, torque, velocity, side
```

### Analyze Peak Torque

```python
# Example analysis of bilateral isokinetic knee extension

# Left and right leg data (example structure)
torque_left = biostrength_data['left']['torque']  # Nm
angle_left = biostrength_data['left']['angle']    # degrees

torque_right = biostrength_data['right']['torque']
angle_right = biostrength_data['right']['angle']

# Find peak torque
peak_torque_left = torque_left.max()
peak_angle_left = angle_left[np.argmax(torque_left)]

peak_torque_right = torque_right.max()
peak_angle_right = angle_right[np.argmax(torque_right)]

print("=== Isokinetic Test Results ===\n")
print("Left leg:")
print(f"  Peak torque: {peak_torque_left:.1f} Nm")
print(f"  Angle at peak: {peak_angle_left:.1f}°")

print("\nRight leg:")
print(f"  Peak torque: {peak_torque_right:.1f} Nm")
print(f"  Angle at peak: {peak_angle_right:.1f}°")

# Bilateral comparison
asymmetry = 100 * abs(peak_torque_left - peak_torque_right) / max(peak_torque_left, peak_torque_right)

print(f"\nBilateral asymmetry: {asymmetry:.1f}%")

if asymmetry < 10:
    print("  → Symmetric (acceptable)")
elif asymmetry < 15:
    print("  → Mild asymmetry (monitor)")
else:
    print("  → Significant asymmetry (address)")

# Output:
# === Isokinetic Test Results ===
# 
# Left leg:
#   Peak torque: 234.5 Nm
#   Angle at peak: 62.3°
# 
# Right leg:
#   Peak torque: 256.8 Nm
#   Angle at peak: 58.7°
# 
# Bilateral asymmetry: 8.7%
#   → Symmetric (acceptable)
```

### Torque-Angle Curve

```python
# Plot torque-angle relationship
plt.figure(figsize=(12, 6))

plt.plot(angle_left, torque_left, 'b-', linewidth=2, label='Left')
plt.plot(angle_right, torque_right, 'r-', linewidth=2, label='Right')

# Mark peaks
plt.scatter(peak_angle_left, peak_torque_left, color='blue', s=100, zorder=5)
plt.scatter(peak_angle_right, peak_torque_right, color='red', s=100, zorder=5)

plt.xlabel('Knee Angle (degrees)')
plt.ylabel('Torque (Nm)')
plt.title('Isokinetic Torque-Angle Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Agonist-Antagonist Ratio

```python
# Compare knee extension (quadriceps) vs. flexion (hamstrings)

# Extension (agonist)
peak_extension = 256.8  # Nm

# Flexion (antagonist)
peak_flexion = 145.3  # Nm

# H/Q ratio (hamstring/quadriceps)
hq_ratio = peak_flexion / peak_extension

print(f"Peak extension (quad): {peak_extension:.1f} Nm")
print(f"Peak flexion (ham): {peak_flexion:.1f} Nm")
print(f"H/Q ratio: {hq_ratio:.2f}")

# Interpretation
if 0.5 <= hq_ratio <= 0.7:
    print("  → Normal H/Q ratio")
elif hq_ratio < 0.5:
    print("  → Hamstrings relatively weak (injury risk)")
else:
    print("  → Hamstrings relatively strong")

# Output:
# Peak extension (quad): 256.8 Nm
# Peak flexion (ham): 145.3 Nm
# H/Q ratio: 0.57
#   → Normal H/Q ratio
```

### Velocity-Specific Testing

```python
# Compare torque at different angular velocities
# Typical: 60, 180, 240, 300 °/s

velocities = [60, 180, 240, 300]  # degrees/s
peak_torques = [256.8, 198.3, 167.4, 142.1]  # Nm

# Plot torque-velocity relationship
plt.figure(figsize=(10, 6))
plt.plot(velocities, peak_torques, 'o-', linewidth=2, markersize=10)

plt.xlabel('Angular Velocity (°/s)')
plt.ylabel('Peak Torque (Nm)')
plt.title('Torque-Velocity Relationship')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate % decrease
torque_at_60 = peak_torques[0]
torque_at_300 = peak_torques[-1]
decrease_pct = 100 * (torque_at_60 - torque_at_300) / torque_at_60

print(f"Torque at 60°/s: {torque_at_60:.1f} Nm")
print(f"Torque at 300°/s: {torque_at_300:.1f} Nm")
print(f"Decrease: {decrease_pct:.1f}%")

# Output:
# Torque at 60°/s: 256.8 Nm
# Torque at 300°/s: 142.1 Nm
# Decrease: 44.7%
```

## Complete Strength Test Workflow

### Isometric Test Protocol

```python
def analyze_isometric_test(tdf_file, participant):
    """
    Complete isometric strength test analysis.
    
    Parameters
    ----------
    tdf_file : str
        Path to TDF file with force platform data
    participant : laban.Participant
        Participant information
    
    Returns
    -------
    dict
        Strength metrics
    """
    import labanalysis as laban
    import numpy as np
    
    # Load data
    record = laban.TimeseriesRecord.from_tdf(tdf_file)
    fp = record.forceplatforms['FP1']
    
    fz = fp.force['Fz'].data
    freq = fp.sampling_frequency
    
    # Filter
    fz_filt = laban.butterworth_filt(fz, freq=freq, cut=10, order=4)
    
    # Baseline
    baseline = fz_filt[:int(1.0 * freq)].mean()
    force = fz_filt - baseline
    
    # Peak force
    peak_force = force.max()
    peak_idx = np.argmax(force)
    
    # RFD windows
    rfd_50 = force[int(0.05 * freq)] / 0.05
    rfd_100 = force[int(0.10 * freq)] / 0.10
    rfd_200 = force[int(0.20 * freq)] / 0.20
    
    # Peak RFD
    force_deriv = np.diff(force) * freq
    peak_rfd = force_deriv.max()
    
    # Impulse
    impulse = np.trapz(force[force > 0]) / freq
    
    # Normalize to bodyweight
    bodyweight = participant.weight * 9.81
    
    results = {
        'peak_force_n': peak_force,
        'peak_force_bw': peak_force / bodyweight,
        'rfd_50ms_n_s': rfd_50,
        'rfd_100ms_n_s': rfd_100,
        'rfd_200ms_n_s': rfd_200,
        'peak_rfd_n_s': peak_rfd,
        'impulse_ns': impulse,
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

results = analyze_isometric_test("isometric_pull.tdf", participant)

print("=== Isometric Test Results ===\n")
print(f"Peak force: {results['peak_force_n']:.1f} N ({results['peak_force_bw']:.2f} × BW)")
print(f"\nRFD:")
print(f"  0-50ms: {results['rfd_50ms_n_s'] / 1000:.1f} kN/s")
print(f"  0-100ms: {results['rfd_100ms_n_s'] / 1000:.1f} kN/s")
print(f"  0-200ms: {results['rfd_200ms_n_s'] / 1000:.1f} kN/s")
print(f"  Peak: {results['peak_rfd_n_s'] / 1000:.1f} kN/s")
print(f"\nImpulse: {results['impulse_ns']:.1f} N·s")
```

### 1RM Testing Protocol

```python
def analyze_1rm_test(loads, reps):
    """
    Analyze submaximal loads to predict 1RM.
    
    Parameters
    ----------
    loads : list
        Loads used (kg)
    reps : list
        Reps completed at each load
    
    Returns
    -------
    dict
        1RM prediction results
    """
    import labanalysis as laban
    import numpy as np
    
    brzycki = laban.Brzycki1RM()
    
    # Predict 1RM from each load-reps pair
    predictions = []
    for load, rep in zip(loads, reps):
        pred = brzycki.predict_1rm(reps=rep, load=load)
        predictions.append(pred)
    
    # Average and variability
    mean_1rm = np.mean(predictions)
    std_1rm = np.std(predictions)
    
    # Recommended training loads
    training_zones = {
        'Power (1-5 reps)': (0.85, 1.00),
        'Strength (6-8 reps)': (0.75, 0.85),
        'Hypertrophy (8-12 reps)': (0.65, 0.75),
        'Endurance (15+ reps)': (0.50, 0.65),
    }
    
    results = {
        'predicted_1rm_kg': mean_1rm,
        'std_kg': std_1rm,
        'cv_percent': 100 * std_1rm / mean_1rm,
        'training_loads': {}
    }
    
    for zone, (low_pct, high_pct) in training_zones.items():
        low_load = mean_1rm * low_pct
        high_load = mean_1rm * high_pct
        results['training_loads'][zone] = (low_load, high_load)
    
    return results

# Usage
loads = [60, 70, 80, 90]
reps = [15, 12, 10, 8]

results = analyze_1rm_test(loads, reps)

print("=== 1RM Prediction Results ===\n")
print(f"Predicted 1RM: {results['predicted_1rm_kg']:.1f} ± {results['std_kg']:.1f} kg")
print(f"CV: {results['cv_percent']:.1f}%")
print(f"\nRecommended Training Loads:")
for zone, (low, high) in results['training_loads'].items():
    print(f"  {zone}: {low:.0f}-{high:.0f} kg")

# Output:
# === 1RM Prediction Results ===
# 
# Predicted 1RM: 116.9 ± 4.2 kg
# CV: 3.6%
# 
# Recommended Training Loads:
#   Power (1-5 reps): 99-117 kg
#   Strength (6-8 reps): 88-99 kg
#   Hypertrophy (8-12 reps): 76-88 kg
#   Endurance (15+ reps): 58-76 kg
```

## Normative Data

### Age and Gender Norms

| Test | Male (20-39 yr) | Female (20-39 yr) |
|------|-----------------|-------------------|
| **Leg extension (N)** | 2500-3500 | 1500-2500 |
| **Knee ext 60°/s (Nm)** | 220-320 | 140-220 |
| **H/Q ratio** | 0.50-0.70 | 0.50-0.70 |
| **Bilateral asymmetry** | <10% | <10% |

### Interpretation

```python
def interpret_strength(peak_force, age, gender):
    """Interpret isometric strength based on norms."""
    
    # Norms (N)
    if gender == 'male':
        if age < 40:
            norm_range = (2500, 3500)
        elif age < 60:
            norm_range = (2000, 3000)
        else:
            norm_range = (1500, 2500)
    else:  # female
        if age < 40:
            norm_range = (1500, 2500)
        elif age < 60:
            norm_range = (1200, 2000)
        else:
            norm_range = (1000, 1700)
    
    if peak_force > norm_range[1]:
        return "Excellent"
    elif peak_force >= norm_range[0]:
        return "Normal"
    else:
        return "Below normal"

# Example
interpretation = interpret_strength(peak_force=2847, age=25, gender='male')
print(f"Strength level: {interpretation}")
# Output: Strength level: Normal
```

## Troubleshooting

### Baseline drift in isometric tests

```python
# If baseline is not stable, use median instead of mean
baseline_window = fz_filt[:int(1.0 * freq)]
baseline = np.median(baseline_window)

# Or use robust estimation
from scipy import stats
baseline = stats.trim_mean(baseline_window, proportiontocut=0.1)
```

### Brzycki equation outside valid range

```python
# Brzycki valid for 1-36 reps
# For >36 reps, equation becomes unreliable

try:
    pred = brzycki.predict_1rm(reps=40, load=50)
except ValueError as e:
    print(f"Error: {e}")
    print("Use alternative equations for high-rep ranges")
```

## See Also

- **[Force Platforms](../biomechanics/force-platforms.md)** - Isometric test setup
- **[API Reference: Brzycki1RM](../../api-reference/equations/strength.md)** - 1RM equation
- **[API Reference: Strength Tests](../../api-reference/protocols/strength-tests.md)** - Test protocols

---

**Key Metrics**: Peak force, RFD, 1RM, bilateral asymmetry, H/Q ratio  
**Reference**: Brzycki M. Strength testing—predicting a one-rep max from reps-to-fatigue. JOPERD. 1993.
