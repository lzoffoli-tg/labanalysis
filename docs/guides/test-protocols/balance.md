# Balance Tests

Complete guide to static and dynamic balance assessment using force platforms in labanalysis.

## Overview

Balance testing evaluates postural control through:
- **Static balance**: Quiet standing on stable/unstable surfaces
- **Dynamic balance**: Responses to perturbations
- **Postural sway**: COP displacement and velocity
- **Stability limits**: Maximum lean angles

labanalysis supports:
- COP-based sway analysis
- Sway area and velocity metrics
- Frequency analysis of postural control
- Eyes open vs. eyes closed comparisons
- Dual-task assessment

## Quick Reference

| Metric | Typical Values (Young Adult) | Interpretation |
|--------|------------------------------|----------------|
| **Sway area (95%)** | 100-300 mm² | Smaller = better balance |
| **Sway velocity** | 10-20 mm/s | Lower = better control |
| **COP range (ML)** | 10-20 mm | Smaller = more stable |
| **COP range (AP)** | 20-40 mm | AP typically > ML |
| **Romberg quotient** | 1.0-2.0 | >2.0 suggests vestibular issues |

## Basic Setup

### Load Balance Test Data

```python
import labanalysis as laban
import numpy as np
import matplotlib.pyplot as plt

# Load force platform data
record = laban.TimeseriesRecord.from_tdf("balance_test.tdf")
fp = record.forceplatforms['FP1']

# Get COP data
cop = fp.cop
cop_x = cop['COPx'].data  # Mediolateral (mm)
cop_y = cop['COPy'].data  # Anteroposterior (mm)
freq = cop.sampling_frequency

# Get vertical force to identify standing phase
fz = fp.force['Fz'].data
bodyweight = np.median(fz[fz > 0.8 * fz.max()])

print(f"Test duration: {len(cop_x) / freq:.1f} s")
print(f"Sampling frequency: {freq} Hz")
print(f"Bodyweight: {bodyweight:.1f} N ({bodyweight / 9.81:.1f} kg)")

# Output:
# Test duration: 30.0 s
# Sampling frequency: 100 Hz
# Bodyweight: 735.0 N (74.9 kg)
```

### Segment Test Phases

```python
# Identify quiet standing phase (stable force)
threshold = 0.9 * bodyweight

# Find continuous standing region
standing = fz > threshold
standing_indices = np.where(standing)[0]

# Get largest continuous segment
diff = np.diff(standing_indices)
breaks = np.where(diff > 1)[0]

if len(breaks) > 0:
    # Find longest segment
    segment_lengths = np.diff(np.concatenate([[0], breaks + 1, [len(standing_indices)]]))
    longest_segment_idx = np.argmax(segment_lengths)
    
    if longest_segment_idx == 0:
        start = standing_indices[0]
        end = standing_indices[breaks[0]]
    else:
        start = standing_indices[breaks[longest_segment_idx - 1] + 1]
        if longest_segment_idx < len(breaks):
            end = standing_indices[breaks[longest_segment_idx]]
        else:
            end = standing_indices[-1]
else:
    start = standing_indices[0]
    end = standing_indices[-1]

# Extract standing phase
cop_x_stand = cop_x[start:end]
cop_y_stand = cop_y[start:end]
duration = (end - start) / freq

print(f"\nStanding phase: {duration:.1f} s ({(end-start)} samples)")

# Output:
# Standing phase: 28.5 s (2850 samples)
```

## COP-Based Sway Metrics

### Sway Area

95% confidence ellipse area.

```python
def calculate_sway_area(cop_x, cop_y, percentile=95):
    """
    Calculate sway area using confidence ellipse.
    
    Parameters
    ----------
    cop_x, cop_y : array
        COP coordinates (mm)
    percentile : float
        Confidence level (default 95%)
    
    Returns
    -------
    float
        Sway area in mm²
    """
    # Standard deviations
    std_x = np.std(cop_x)
    std_y = np.std(cop_y)
    
    # Chi-square value for confidence level
    # 95% → 2.447, 90% → 2.146, 99% → 3.035
    chi2_val = {90: 2.146, 95: 2.447, 99: 3.035}[percentile]
    
    # Ellipse area
    area = np.pi * chi2_val * std_x * std_y
    
    return area

# Calculate sway area
sway_area_95 = calculate_sway_area(cop_x_stand, cop_y_stand, percentile=95)

print(f"Sway area (95%): {sway_area_95:.2f} mm²")

# Output:
# Sway area (95%): 245.32 mm²
```

### Sway Path Length and Velocity

```python
# Calculate sway path (total distance traveled by COP)
dx = np.diff(cop_x_stand)
dy = np.diff(cop_y_stand)
distances = np.sqrt(dx**2 + dy**2)
sway_path = distances.sum()

# Sway velocity
sway_velocity = sway_path / duration

print(f"Sway path length: {sway_path:.1f} mm")
print(f"Sway velocity: {sway_velocity:.2f} mm/s")

# Output:
# Sway path length: 437.8 mm
# Sway velocity: 15.36 mm/s
```

### COP Range and RMS

```python
# Range (min to max displacement)
range_x = cop_x_stand.max() - cop_x_stand.min()
range_y = cop_y_stand.max() - cop_y_stand.min()

# RMS (root mean square)
rms_x = np.sqrt(np.mean(cop_x_stand**2))
rms_y = np.sqrt(np.mean(cop_y_stand**2))

print("COP Range:")
print(f"  Mediolateral: {range_x:.2f} mm")
print(f"  Anteroposterior: {range_y:.2f} mm")

print("\nCOP RMS:")
print(f"  Mediolateral: {rms_x:.2f} mm")
print(f"  Anteroposterior: {rms_y:.2f} mm")

# Output:
# COP Range:
#   Mediolateral: 24.56 mm
#   Anteroposterior: 38.72 mm
# 
# COP RMS:
#   Mediolateral: 5.23 mm
#   Anteroposterior: 8.45 mm
```

### Visualize COP Trajectory

```python
# Plot COP path
plt.figure(figsize=(10, 10))

# Plot trajectory
plt.plot(cop_x_stand, cop_y_stand, 'b-', alpha=0.3, linewidth=0.5)
plt.scatter(cop_x_stand[0], cop_y_stand[0], color='green', s=100, 
            marker='o', label='Start', zorder=5)
plt.scatter(cop_x_stand[-1], cop_y_stand[-1], color='red', s=100, 
            marker='x', label='End', zorder=5)

# Plot mean position
mean_x = cop_x_stand.mean()
mean_y = cop_y_stand.mean()
plt.scatter(mean_x, mean_y, color='black', s=100, 
            marker='+', label='Mean', zorder=5)

# Plot 95% confidence ellipse
from matplotlib.patches import Ellipse

std_x = np.std(cop_x_stand)
std_y = np.std(cop_y_stand)
ellipse = Ellipse(
    (mean_x, mean_y),
    width=2 * 2.447 * std_x,
    height=2 * 2.447 * std_y,
    edgecolor='red',
    facecolor='none',
    linewidth=2,
    label='95% Ellipse'
)
plt.gca().add_patch(ellipse)

plt.xlabel('ML Displacement (mm)')
plt.ylabel('AP Displacement (mm)')
plt.title('Center of Pressure Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

## Frequency Analysis

### Postural Control Frequency Bands

```python
# Calculate PSD of COP
freqs_x, psd_x = laban.psd(cop_x_stand, freq=freq, nperseg=1024)
freqs_y, psd_y = laban.psd(cop_y_stand, freq=freq, nperseg=1024)

# Define frequency bands
# Low: 0-0.5 Hz (visual/vestibular)
# Medium: 0.5-2.0 Hz (cerebellar)
# High: 2.0-10 Hz (proprioceptive/motor)

def power_in_band(freqs, psd, f_low, f_high):
    """Calculate power in frequency band."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    return np.trapz(psd[mask], freqs[mask])

# Calculate band powers
bands = {
    'Low (0-0.5 Hz)': (0, 0.5),
    'Medium (0.5-2 Hz)': (0.5, 2.0),
    'High (2-10 Hz)': (2.0, 10.0)
}

print("=== Frequency Analysis - Mediolateral ===")
total_power_x = np.trapz(psd_x, freqs_x)

for band_name, (f_low, f_high) in bands.items():
    power = power_in_band(freqs_x, psd_x, f_low, f_high)
    pct = 100 * power / total_power_x
    print(f"{band_name}: {pct:.1f}% of total power")

print("\n=== Frequency Analysis - Anteroposterior ===")
total_power_y = np.trapz(psd_y, freqs_y)

for band_name, (f_low, f_high) in bands.items():
    power = power_in_band(freqs_y, psd_y, f_low, f_high)
    pct = 100 * power / total_power_y
    print(f"{band_name}: {pct:.1f}% of total power")

# Output:
# === Frequency Analysis - Mediolateral ===
# Low (0-0.5 Hz): 78.3% of total power
# Medium (0.5-2 Hz): 18.2% of total power
# High (2-10 Hz): 3.5% of total power
# 
# === Frequency Analysis - Anteroposterior ===
# Low (0-0.5 Hz): 82.1% of total power
# Medium (0.5-2 Hz): 15.4% of total power
# High (2-10 Hz): 2.5% of total power

# Plot PSD
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.semilogy(freqs_x, psd_x)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (mm²/Hz)')
plt.title('COP PSD - Mediolateral')
plt.xlim(0, 5)
plt.grid(True, alpha=0.3)

# Mark frequency bands
plt.axvline(0.5, color='red', linestyle='--', alpha=0.5)
plt.axvline(2.0, color='red', linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.semilogy(freqs_y, psd_y)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (mm²/Hz)')
plt.title('COP PSD - Anteroposterior')
plt.xlim(0, 5)
plt.grid(True, alpha=0.3)

plt.axvline(0.5, color='red', linestyle='--', alpha=0.5)
plt.axvline(2.0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

### Mean Frequency

```python
# Calculate mean frequency (centroid of PSD)
def mean_frequency(freqs, psd):
    """Calculate mean frequency of signal."""
    return np.sum(freqs * psd) / np.sum(psd)

mean_freq_x = mean_frequency(freqs_x, psd_x)
mean_freq_y = mean_frequency(freqs_y, psd_y)

print(f"Mean frequency (ML): {mean_freq_x:.3f} Hz")
print(f"Mean frequency (AP): {mean_freq_y:.3f} Hz")

# Output:
# Mean frequency (ML): 0.342 Hz
# Mean frequency (AP): 0.318 Hz

# Higher mean frequency indicates more rapid corrections
# Lower mean frequency suggests more stable control
```

## Condition Comparisons

### Eyes Open vs. Eyes Closed (Romberg Test)

```python
# Load both conditions
record_eo = laban.TimeseriesRecord.from_tdf("balance_eyes_open.tdf")
record_ec = laban.TimeseriesRecord.from_tdf("balance_eyes_closed.tdf")

fp_eo = record_eo.forceplatforms['FP1']
fp_ec = record_ec.forceplatforms['FP1']

# Extract COP
cop_eo = fp_eo.cop
cop_ec = fp_ec.cop

cop_x_eo = cop_eo['COPx'].data
cop_y_eo = cop_eo['COPy'].data

cop_x_ec = cop_ec['COPx'].data
cop_y_ec = cop_ec['COPy'].data

# Calculate metrics for both conditions
sway_area_eo = calculate_sway_area(cop_x_eo, cop_y_eo)
sway_area_ec = calculate_sway_area(cop_x_ec, cop_y_ec)

# Romberg quotient (EC / EO)
romberg_quotient = sway_area_ec / sway_area_eo

print("=== Eyes Open vs. Eyes Closed ===\n")
print(f"Sway area (EO): {sway_area_eo:.2f} mm²")
print(f"Sway area (EC): {sway_area_ec:.2f} mm²")
print(f"Romberg quotient: {romberg_quotient:.2f}")

if romberg_quotient > 2.0:
    print("  → High dependence on vision (possible vestibular issue)")
elif romberg_quotient > 1.5:
    print("  → Moderate dependence on vision (normal)")
else:
    print("  → Low dependence on vision (good proprioception)")

# Output:
# === Eyes Open vs. Eyes Closed ===
# 
# Sway area (EO): 187.34 mm²
# Sway area (EC): 312.58 mm²
# Romberg quotient: 1.67
#   → Moderate dependence on vision (normal)
```

### Stable vs. Unstable Surface

```python
# Compare firm vs. foam surface
sway_firm = calculate_sway_area(cop_x_firm, cop_y_firm)
sway_foam = calculate_sway_area(cop_x_foam, cop_y_foam)

foam_effect = (sway_foam - sway_firm) / sway_firm * 100

print(f"Sway area (firm): {sway_firm:.2f} mm²")
print(f"Sway area (foam): {sway_foam:.2f} mm²")
print(f"Foam effect: +{foam_effect:.1f}%")

# Typical: 50-150% increase on foam
```

### Single-Leg vs. Double-Leg Stance

```python
# Compare standing conditions
sway_double = calculate_sway_area(cop_x_double, cop_y_double)
sway_single = calculate_sway_area(cop_x_single, cop_y_single)

difficulty_ratio = sway_single / sway_double

print(f"Sway area (double-leg): {sway_double:.2f} mm²")
print(f"Sway area (single-leg): {sway_single:.2f} mm²")
print(f"Single-leg difficulty: {difficulty_ratio:.1f}x harder")

# Typical: 3-5x more sway on single leg
```

## Dynamic Balance Tests

### Limits of Stability

```python
# Test maximum voluntary lean in different directions

# Load trial where subject leans forward maximally
record_lean = laban.TimeseriesRecord.from_tdf("lean_forward.tdf")
fp_lean = record_lean.forceplatforms['FP1']

cop_lean = fp_lean.cop
cop_x_lean = cop_lean['COPx'].data
cop_y_lean = cop_lean['COPy'].data

# Find maximum excursion
max_forward = cop_y_lean.max()
max_backward = cop_y_lean.min()
max_right = cop_x_lean.max()
max_left = cop_x_lean.min()

print("=== Limits of Stability ===\n")
print(f"Maximum forward lean: {max_forward:.1f} mm")
print(f"Maximum backward lean: {abs(max_backward):.1f} mm")
print(f"Maximum right lean: {max_right:.1f} mm")
print(f"Maximum left lean: {abs(max_left):.1f} mm")

# Calculate stability margin (% of base of support)
# Typical foot length ~250mm, width ~100mm
foot_length = 250  # mm
foot_width = 100   # mm

forward_pct = 100 * max_forward / (foot_length / 2)
backward_pct = 100 * abs(max_backward) / (foot_length / 2)

print(f"\nAs % of base of support:")
print(f"  Forward: {forward_pct:.1f}%")
print(f"  Backward: {backward_pct:.1f}%")

# Output:
# === Limits of Stability ===
# 
# Maximum forward lean: 87.3 mm
# Maximum backward lean: 62.1 mm
# Maximum right lean: 34.2 mm
# Maximum left lean: 31.8 mm
# 
# As % of base of support:
#   Forward: 69.8%
#   Backward: 49.7%
```

### Postural Responses to Perturbations

```python
# Analyze response to external perturbation

# Load perturbation trial (e.g., platform translation)
record_pert = laban.TimeseriesRecord.from_tdf("perturbation.tdf")
fp_pert = record_pert.forceplatforms['FP1']

cop_pert = fp_pert.cop
cop_y_pert = cop_pert['COPy'].data
freq = cop_pert.sampling_frequency
time = cop_pert['COPy'].index

# Perturbation occurs at t=2.0s (known from protocol)
pert_time = 2.0
pert_idx = int(pert_time * freq)

# Extract response window (2s after perturbation)
response_window = slice(pert_idx, pert_idx + int(2.0 * freq))
cop_response = cop_y_pert[response_window]
time_response = time[response_window] - time[pert_idx]

# Calculate response metrics
# 1. Latency (time to first peak)
baseline = cop_y_pert[pert_idx - int(0.5 * freq):pert_idx].mean()
response_magnitude = np.abs(cop_response - baseline)
first_peak_idx = np.argmax(response_magnitude[:int(0.5 * freq)])
latency = time_response[first_peak_idx]

# 2. Peak displacement
peak_displacement = response_magnitude.max()

# 3. Time to stabilization (return to baseline ± 2 SD)
baseline_std = np.std(cop_y_pert[pert_idx - int(0.5 * freq):pert_idx])
threshold = 2 * baseline_std

stable_mask = response_magnitude < threshold
if np.any(stable_mask):
    # Find first sustained period (>0.5s) within threshold
    stable_runs = []
    current_run = 0
    for i, is_stable in enumerate(stable_mask):
        if is_stable:
            current_run += 1
        else:
            if current_run > 0:
                stable_runs.append((i - current_run, current_run))
            current_run = 0
    
    # Find first run > 0.5s
    for start_idx, length in stable_runs:
        if length > int(0.5 * freq):
            stabilization_time = time_response[start_idx]
            break
    else:
        stabilization_time = None
else:
    stabilization_time = None

print("=== Perturbation Response ===\n")
print(f"Response latency: {latency * 1000:.0f} ms")
print(f"Peak displacement: {peak_displacement:.1f} mm")
if stabilization_time:
    print(f"Time to stabilization: {stabilization_time:.2f} s")
else:
    print("Did not stabilize within 2s")

# Output:
# === Perturbation Response ===
# 
# Response latency: 127 ms
# Peak displacement: 45.3 mm
# Time to stabilization: 0.87 s

# Plot response
plt.figure(figsize=(12, 6))
plt.plot(time_response, cop_response - baseline, 'b-', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
plt.axhline(threshold, color='red', linestyle=':', alpha=0.5, label='Stability threshold')
plt.axhline(-threshold, color='red', linestyle=':', alpha=0.5)

if stabilization_time:
    plt.axvline(stabilization_time, color='green', linestyle='--', 
                label=f'Stabilization ({stabilization_time:.2f}s)')

plt.xlabel('Time after perturbation (s)')
plt.ylabel('COP displacement (mm)')
plt.title('Postural Response to Perturbation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Dual-Task Assessment

### Balance During Cognitive Task

```python
# Compare single-task vs. dual-task balance

# Single task: quiet standing only
sway_single_task = calculate_sway_area(cop_x_single_task, cop_y_single_task)

# Dual task: standing + cognitive task (e.g., counting backwards)
sway_dual_task = calculate_sway_area(cop_x_dual_task, cop_y_dual_task)

# Dual-task cost
dt_cost = (sway_dual_task - sway_single_task) / sway_single_task * 100

print("=== Dual-Task Assessment ===\n")
print(f"Sway area (single-task): {sway_single_task:.2f} mm²")
print(f"Sway area (dual-task): {sway_dual_task:.2f} mm²")
print(f"Dual-task cost: +{dt_cost:.1f}%")

if dt_cost > 30:
    print("  → High dual-task cost (attention-demanding)")
elif dt_cost > 15:
    print("  → Moderate dual-task cost (normal)")
else:
    print("  → Low dual-task cost (automatic control)")

# Output:
# === Dual-Task Assessment ===
# 
# Sway area (single-task): 198.45 mm²
# Sway area (dual-task): 257.32 mm²
# Dual-task cost: +29.7%
#   → Moderate dual-task cost (normal)
```

## Complete Balance Test Protocol

```python
def analyze_balance_test(tdf_file, participant, condition='eyes_open'):
    """
    Complete balance test analysis.
    
    Parameters
    ----------
    tdf_file : str
        Path to TDF file
    participant : laban.Participant
        Participant info
    condition : str
        Test condition ('eyes_open', 'eyes_closed', 'foam', etc.)
    
    Returns
    -------
    dict
        Balance metrics
    """
    import labanalysis as laban
    import numpy as np
    
    # Load data
    record = laban.TimeseriesRecord.from_tdf(tdf_file)
    fp = record.forceplatforms['FP1']
    
    # Get COP
    cop = fp.cop
    cop_x = cop['COPx'].data
    cop_y = cop['COPy'].data
    freq = cop.sampling_frequency
    
    # Get standing phase
    fz = fp.force['Fz'].data
    bodyweight = np.median(fz[fz > 0.8 * fz.max()])
    
    standing = fz > 0.9 * bodyweight
    standing_indices = np.where(standing)[0]
    
    # Extract longest continuous segment
    start = standing_indices[0]
    end = standing_indices[-1]
    
    cop_x_stand = cop_x[start:end]
    cop_y_stand = cop_y[start:end]
    duration = (end - start) / freq
    
    # Calculate metrics
    # 1. Sway area
    std_x = np.std(cop_x_stand)
    std_y = np.std(cop_y_stand)
    sway_area = np.pi * 2.447 * std_x * std_y
    
    # 2. Sway path and velocity
    dx = np.diff(cop_x_stand)
    dy = np.diff(cop_y_stand)
    sway_path = np.sqrt(dx**2 + dy**2).sum()
    sway_velocity = sway_path / duration
    
    # 3. COP range
    range_x = cop_x_stand.max() - cop_x_stand.min()
    range_y = cop_y_stand.max() - cop_y_stand.min()
    
    # 4. RMS
    rms_x = np.sqrt(np.mean(cop_x_stand**2))
    rms_y = np.sqrt(np.mean(cop_y_stand**2))
    
    # 5. Frequency analysis
    freqs_x, psd_x = laban.psd(cop_x_stand, freq=freq, nperseg=1024)
    mean_freq_x = np.sum(freqs_x * psd_x) / np.sum(psd_x)
    
    # Compile results
    results = {
        'condition': condition,
        'duration': duration,
        'sway_area_mm2': sway_area,
        'sway_velocity_mm_s': sway_velocity,
        'cop_range_ml_mm': range_x,
        'cop_range_ap_mm': range_y,
        'cop_rms_ml_mm': rms_x,
        'cop_rms_ap_mm': rms_y,
        'mean_freq_ml_hz': mean_freq_x,
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

results = analyze_balance_test("balance_eo.tdf", participant, condition='eyes_open')

print("=== Balance Test Results ===\n")
print(f"Condition: {results['condition']}")
print(f"Duration: {results['duration']:.1f} s\n")
print("Sway Metrics:")
print(f"  Area (95%): {results['sway_area_mm2']:.2f} mm²")
print(f"  Velocity: {results['sway_velocity_mm_s']:.2f} mm/s")
print(f"\nCOP Range:")
print(f"  ML: {results['cop_range_ml_mm']:.2f} mm")
print(f"  AP: {results['cop_range_ap_mm']:.2f} mm")
print(f"\nFrequency:")
print(f"  Mean freq (ML): {results['mean_freq_ml_hz']:.3f} Hz")
```

## Normative Data and Interpretation

### Age-Related Norms

| Age Group | Sway Area (mm²) | Sway Velocity (mm/s) |
|-----------|-----------------|----------------------|
| 20-39 years | 100-300 | 10-20 |
| 40-59 years | 150-400 | 15-25 |
| 60-79 years | 200-600 | 20-35 |
| 80+ years | 300-1000 | 30-50 |

### Interpretation Guidelines

```python
def interpret_balance_results(results, age):
    """Interpret balance test results based on age norms."""
    
    sway_area = results['sway_area_mm2']
    sway_vel = results['sway_velocity_mm_s']
    
    # Age-based thresholds
    if age < 40:
        area_thresh = (100, 300)
        vel_thresh = (10, 20)
    elif age < 60:
        area_thresh = (150, 400)
        vel_thresh = (15, 25)
    elif age < 80:
        area_thresh = (200, 600)
        vel_thresh = (20, 35)
    else:
        area_thresh = (300, 1000)
        vel_thresh = (30, 50)
    
    print(f"=== Interpretation (Age {age}) ===\n")
    
    # Sway area
    if sway_area < area_thresh[0]:
        print(f"Sway area: Excellent ({sway_area:.0f} mm²)")
    elif sway_area <= area_thresh[1]:
        print(f"Sway area: Normal ({sway_area:.0f} mm²)")
    else:
        print(f"Sway area: Below normal ({sway_area:.0f} mm²) - consider fall risk assessment")
    
    # Sway velocity
    if sway_vel < vel_thresh[0]:
        print(f"Sway velocity: Excellent ({sway_vel:.1f} mm/s)")
    elif sway_vel <= vel_thresh[1]:
        print(f"Sway velocity: Normal ({sway_vel:.1f} mm/s)")
    else:
        print(f"Sway velocity: Below normal ({sway_vel:.1f} mm/s) - increased correction effort")

interpret_balance_results(results, age=25)
```

## Troubleshooting

### COP calculation issues

```python
# Check if COP is valid
if np.any(np.isnan(cop_x)) or np.any(np.isnan(cop_y)):
    print("Warning: COP contains NaN values")
    print("  - Check if vertical force is sufficient")
    print("  - COP undefined when Fz ≈ 0")
    
    # Filter out low-force samples
    valid_force = fz > 0.1 * bodyweight
    cop_x_clean = cop_x[valid_force]
    cop_y_clean = cop_y[valid_force]
```

### Unrealistic sway values

```python
# Check for outliers
cop_x_median = np.median(cop_x_stand)
cop_x_mad = np.median(np.abs(cop_x_stand - cop_x_median))

threshold = 5 * 1.4826 * cop_x_mad  # 5 MAD threshold

outliers = np.abs(cop_x_stand - cop_x_median) > threshold

if outliers.sum() > 0:
    print(f"Warning: {outliers.sum()} outlier samples detected")
    print("  - Remove outliers or check force platform calibration")
    
    # Remove outliers
    cop_x_clean = cop_x_stand[~outliers]
    cop_y_clean = cop_y_stand[~outliers]
```

## See Also

- **[Force Platforms](../biomechanics/force-platforms.md)** - COP calculation details
- **[Frequency Analysis](../signal-processing/frequency-analysis.md)** - PSD analysis
- **[Peak Detection](../signal-processing/peak-detection.md)** - Event detection
- **[API Reference: ForcePlatform](../../api/records/records.md#forceplatform)** - ForcePlatform class

---

**Key Metrics**: Sway area, sway velocity, COP range, Romberg quotient  
**Reference**: Prieto TE et al. Measures of postural steadiness: differences between healthy young and elderly adults. IEEE Trans Biomed Eng. 1996.
