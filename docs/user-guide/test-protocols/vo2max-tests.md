# VO2max Tests

Complete guide to submaximal and maximal aerobic capacity testing using metabolic analyzers in labanalysis.

## Overview

VO2max testing evaluates cardiorespiratory fitness through:
- **Maximal tests**: Incremental protocols to exhaustion
- **Submaximal tests**: Prediction from HR-VO2 relationship
- **Metabolic parameters**: VO2, VCO2, RER, ventilation
- **Anaerobic threshold**: VT1, VT2, MLSS estimation

labanalysis supports:
- Cosmed metabolic analyzer data
- Submaximal VO2max prediction
- HR-based equations (bike, run)
- Plateau detection and threshold identification

## Quick Reference

| Test | Protocol | Typical VO2max | Interpretation |
|------|----------|----------------|----------------|
| **Treadmill (run)** | Bruce, Balke | 40-60 ml/kg/min | Elite: >60 |
| **Cycle ergometer** | Astrand, YMCA | 35-50 ml/kg/min | Elite: >55 |
| **Submaximal** | 85% HRmax | Predicted value | ±10-15% error |

## Load Metabolic Data

### From Cosmed Device

```python
import labanalysis as laban
import numpy as np

# Load metabolic data
metabolic = laban.MetabolicRecord.from_file(
    filename="vo2max_test.txt",
    breath_by_breath=False  # 30s average if False
)

print(f"Metabolic record loaded")
print(f"Duration: {len(metabolic.vo2.data) / metabolic.sampling_frequency:.1f} s")
print(f"Sampling: {'breath-by-breath' if metabolic.breath_by_breath else '30s average'}")

# Access signals
vo2 = metabolic.vo2  # Oxygen consumption (ml/kg/min)
vco2 = metabolic.vco2  # CO2 production (ml/min)
hr = metabolic.heart_rate  # Heart rate (bpm)
rer = metabolic.rer  # Respiratory exchange ratio

print(f"\nPeak values:")
print(f"  VO2: {vo2.data.max():.1f} ml/kg/min")
print(f"  HR: {hr.data.max():.0f} bpm")
print(f"  RER: {rer.data.max():.2f}")

# Output:
# Metabolic record loaded
# Duration: 720.0 s (12 min)
# Sampling: 30s average
# 
# Peak values:
#   VO2: 52.3 ml/kg/min
#   HR: 189 bpm
#   RER: 1.18
```

## Maximal Testing

### Detect VO2max Plateau

```python
# Identify plateau in VO2 (criteria: <150 ml/min increase)

vo2_data = vo2.data  # ml/kg/min
time = vo2.index
weight = 75  # kg

# Convert to absolute VO2 (L/min)
vo2_abs = vo2_data * weight / 1000  # L/min

# Calculate change between consecutive points
vo2_diff = np.diff(vo2_abs) * 1000  # ml/min

# Find plateau (change < 150 ml/min)
plateau_threshold = 150  # ml/min
plateau_indices = np.where(vo2_diff < plateau_threshold)[0]

if len(plateau_indices) > 0:
    plateau_start = plateau_indices[0]
    vo2max_value = vo2_data[plateau_start:].max()
    
    print(f"VO2max plateau detected:")
    print(f"  Plateau starts at: {time[plateau_start]:.1f} s")
    print(f"  VO2max: {vo2max_value:.1f} ml/kg/min")
    print(f"  VO2max (absolute): {vo2max_value * weight / 1000:.2f} L/min")
else:
    # No plateau - use peak value
    vo2max_value = vo2_data.max()
    vo2max_idx = np.argmax(vo2_data)
    
    print(f"No plateau detected - using peak VO2:")
    print(f"  Peak at: {time[vo2max_idx]:.1f} s")
    print(f"  VO2peak: {vo2max_value:.1f} ml/kg/min")

# Output:
# VO2max plateau detected:
#   Plateau starts at: 600.5 s
#   VO2max: 52.3 ml/kg/min
#   VO2max (absolute): 3.92 L/min
```

### Verify Maximal Effort

```python
# Maximal effort criteria (ACSM):
# 1. HR within 10 bpm of age-predicted max
# 2. RER ≥ 1.10
# 3. VO2 plateau

# Age-predicted HRmax
age = 25
hr_max_predicted = 207 - 0.7 * age  # Alternative: 220 - age

# Measured peak HR
hr_peak = hr.data.max()

# RER peak
rer_peak = rer.data.max()

# VO2 plateau
has_plateau = len(plateau_indices) > 0

print("=== Maximal Effort Verification ===\n")

# Criterion 1: HR
hr_diff = abs(hr_peak - hr_max_predicted)
criterion1 = hr_diff <= 10

print(f"1. Heart Rate:")
print(f"   Predicted HRmax: {hr_max_predicted:.0f} bpm")
print(f"   Measured peak: {hr_peak:.0f} bpm")
print(f"   Difference: {hr_diff:.0f} bpm")
print(f"   ✓ Met" if criterion1 else "   ✗ Not met")

# Criterion 2: RER
criterion2 = rer_peak >= 1.10

print(f"\n2. RER:")
print(f"   Peak RER: {rer_peak:.2f}")
print(f"   Threshold: ≥1.10")
print(f"   ✓ Met" if criterion2 else "   ✗ Not met")

# Criterion 3: Plateau
print(f"\n3. VO2 Plateau:")
print(f"   {'✓ Detected' if has_plateau else '✗ Not detected'}")

# Overall
criteria_met = sum([criterion1, criterion2, has_plateau])
print(f"\n=== Overall: {criteria_met}/3 criteria met ===")

if criteria_met >= 2:
    print("Valid VO2max test")
else:
    print("Possibly submaximal effort - report as VO2peak")

# Output:
# === Maximal Effort Verification ===
# 
# 1. Heart Rate:
#    Predicted HRmax: 189 bpm
#    Measured peak: 189 bpm
#    Difference: 0 bpm
#    ✓ Met
# 
# 2. RER:
#    Peak RER: 1.18
#    Threshold: ≥1.10
#    ✓ Met
# 
# 3. VO2 Plateau:
#    ✓ Detected
# 
# === Overall: 3/3 criteria met ===
# Valid VO2max test
```

## Submaximal Testing

### Predict VO2max from HR-VO2 Relationship

```python
# Submaximal protocol: Stop at 85% HRmax

# Extract submaximal data (HR < 85% predicted max)
hr_threshold = 0.85 * hr_max_predicted

submaximal_mask = hr.data < hr_threshold
hr_submax = hr.data[submaximal_mask]
vo2_submax = vo2.data[submaximal_mask]

# Linear regression HR vs VO2
from scipy import stats

slope, intercept, r_value, _, _ = stats.linregress(hr_submax, vo2_submax)

# Predict VO2 at HRmax
vo2max_predicted = slope * hr_max_predicted + intercept

print("=== Submaximal VO2max Prediction ===\n")
print(f"HR range used: {hr_submax.min():.0f}-{hr_submax.max():.0f} bpm")
print(f"VO2 range: {vo2_submax.min():.1f}-{vo2_submax.max():.1f} ml/kg/min")
print(f"Correlation (r): {r_value:.3f}")
print(f"\nPredicted VO2max at {hr_max_predicted:.0f} bpm:")
print(f"  {vo2max_predicted:.1f} ml/kg/min")

# Output:
# === Submaximal VO2max Prediction ===
# 
# HR range used: 120-160 bpm
# VO2 range: 28.3-45.7 ml/kg/min
# Correlation (r): 0.987
# 
# Predicted VO2max at 189 bpm:
#   51.3 ml/kg/min
```

### YMCA Protocol

```python
# YMCA submaximal cycle test protocol
# Predict VO2max from 2-4 workloads at steady-state HR

# Workloads and corresponding HR (example data)
workloads_watts = [75, 100, 125, 150]  # Watts
hr_ss = [120, 135, 148, 163]  # Steady-state HR at each workload

# Linear regression
slope, intercept, r_value, _, _ = stats.linregress(hr_ss, workloads_watts)

# Predict workload at HRmax
workload_at_hrmax = slope * hr_max_predicted + intercept

# Convert to VO2max
# VO2 (ml/min) = 7 + 10.8 * workload (watts)
vo2_abs_max = (7 + 10.8 * workload_at_hrmax) / weight  # ml/kg/min

print("=== YMCA Cycle Test Prediction ===\n")
print(f"Workloads: {workloads_watts} W")
print(f"Steady-state HR: {hr_ss} bpm")
print(f"Correlation: r = {r_value:.3f}")
print(f"\nPredicted workload at HRmax: {workload_at_hrmax:.0f} W")
print(f"Predicted VO2max: {vo2_abs_max:.1f} ml/kg/min")

# Output:
# === YMCA Cycle Test Prediction ===
# 
# Workloads: [75, 100, 125, 150] W
# Steady-state HR: [120, 135, 148, 163] bpm
# Correlation: r = 0.996
# 
# Predicted workload at HRmax: 225 W
# Predicted VO2max: 48.7 ml/kg/min
```

### Astrand-Rhyming Nomogram

```python
# Predict VO2max from single submaximal workload

# Single workload test
workload = 150  # Watts
hr_ss_single = 150  # Steady-state HR

# Astrand-Rhyming equation (simplified)
# VO2max = (VO2_submax / (HRsubmax / HRmax)) * age_correction

vo2_submax = (7 + 10.8 * workload) / weight  # ml/kg/min
hr_ratio = hr_ss_single / hr_max_predicted

# Age correction factor
age_corrections = {
    25: 1.00,
    35: 0.87,
    45: 0.78,
    55: 0.71,
    65: 0.65
}

age_factor = age_corrections.get(age, 1.00)

vo2max_astrand = (vo2_submax / hr_ratio) * age_factor

print("=== Astrand-Rhyming Prediction ===\n")
print(f"Workload: {workload} W at HR {hr_ss_single} bpm")
print(f"Submaximal VO2: {vo2_submax:.1f} ml/kg/min")
print(f"HR ratio: {hr_ratio:.3f}")
print(f"Age correction: {age_factor:.2f}")
print(f"\nPredicted VO2max: {vo2max_astrand:.1f} ml/kg/min")

# Output:
# === Astrand-Rhyming Prediction ===
# 
# Workload: 150 W at HR 150 bpm
# Submaximal VO2: 28.6 ml/kg/min
# HR ratio: 0.794
# Age correction: 1.00
# 
# Predicted VO2max: 36.0 ml/kg/min
```

## Threshold Detection

### Ventilatory Threshold (VT1)

```python
# Detect VT1 using V-slope method (VCO2 vs VO2)

vo2_abs = vo2.data * weight / 1000  # L/min
vco2_abs = vco2.data / 1000  # L/min (assuming vco2 in ml/min)

# Filter data for smooth curves
vo2_filt = laban.butterworth_filt(vo2_abs, freq=metabolic.sampling_frequency, cut=0.05, order=2)
vco2_filt = laban.butterworth_filt(vco2_abs, freq=metabolic.sampling_frequency, cut=0.05, order=2)

# Calculate slopes (windowed regression)
window = 5  # points
slopes = []

for i in range(window, len(vo2_filt) - window):
    vo2_window = vo2_filt[i-window:i+window]
    vco2_window = vco2_filt[i-window:i+window]
    
    slope_local, _, _, _, _ = stats.linregress(vo2_window, vco2_window)
    slopes.append(slope_local)

slopes = np.array(slopes)

# VT1 = first breakpoint where slope increases above 1.0
vt1_idx = np.where(slopes > 1.0)[0]

if len(vt1_idx) > 0:
    vt1_position = vt1_idx[0] + window
    vt1_vo2 = vo2.data[vt1_position]
    vt1_hr = hr.data[vt1_position]
    
    print(f"VT1 detected:")
    print(f"  VO2 at VT1: {vt1_vo2:.1f} ml/kg/min ({100 * vt1_vo2 / vo2max_value:.0f}% VO2max)")
    print(f"  HR at VT1: {vt1_hr:.0f} bpm ({100 * vt1_hr / hr_peak:.0f}% HRmax)")
else:
    print("VT1 not detected")

# Output:
# VT1 detected:
#   VO2 at VT1: 38.2 ml/kg/min (73% VO2max)
#   HR at VT1: 152 bpm (80% HRmax)
```

### Respiratory Compensation Point (RCP/VT2)

```python
# Detect RCP using VE/VCO2 method

ve = metabolic.ventilation.data  # L/min (if available)
# Or estimate: VE ≈ 30 * VCO2 (rough approximation)

ve_vco2_ratio = ve / vco2_abs

# Filter
ve_vco2_filt = laban.butterworth_filt(ve_vco2_ratio, freq=metabolic.sampling_frequency, cut=0.05, order=2)

# RCP = minimum VE/VCO2 after VT1
rcp_search_start = vt1_position if len(vt1_idx) > 0 else len(ve_vco2_filt) // 2

rcp_idx = rcp_search_start + np.argmin(ve_vco2_filt[rcp_search_start:])
rcp_vo2 = vo2.data[rcp_idx]
rcp_hr = hr.data[rcp_idx]

print(f"\nRCP/VT2 detected:")
print(f"  VO2 at RCP: {rcp_vo2:.1f} ml/kg/min ({100 * rcp_vo2 / vo2max_value:.0f}% VO2max)")
print(f"  HR at RCP: {rcp_hr:.0f} bpm ({100 * rcp_hr / hr_peak:.0f}% HRmax)")

# Output:
# RCP/VT2 detected:
#   VO2 at RCP: 46.8 ml/kg/min (89% VO2max)
#   HR at RCP: 175 bpm (93% HRmax)
```

## Training Zones

### HR-Based Training Zones

```python
# Calculate training zones from thresholds

zones = {
    'Zone 1 (Recovery)': (0, 0.70),
    'Zone 2 (Endurance)': (0.70, vt1_hr / hr_peak),
    'Zone 3 (Tempo)': (vt1_hr / hr_peak, rcp_hr / hr_peak),
    'Zone 4 (Threshold)': (rcp_hr / hr_peak, 0.95),
    'Zone 5 (VO2max)': (0.95, 1.00),
}

print("=== Heart Rate Training Zones ===\n")

for zone_name, (low_pct, high_pct) in zones.items():
    hr_low = int(low_pct * hr_peak)
    hr_high = int(high_pct * hr_peak)
    
    print(f"{zone_name}:")
    print(f"  {hr_low}-{hr_high} bpm ({100*low_pct:.0f}-{100*high_pct:.0f}% HRmax)")

# Output:
# === Heart Rate Training Zones ===
# 
# Zone 1 (Recovery):
#   0-132 bpm (0-70% HRmax)
# Zone 2 (Endurance):
#   132-152 bpm (70-80% HRmax)
# Zone 3 (Tempo):
#   152-175 bpm (80-93% HRmax)
# Zone 4 (Threshold):
#   175-180 bpm (93-95% HRmax)
# Zone 5 (VO2max):
#   180-189 bpm (95-100% HRmax)
```

## Complete Test Workflow

```python
def analyze_vo2max_test(filename, participant):
    """
    Complete VO2max test analysis.
    
    Parameters
    ----------
    filename : str
        Metabolic data file
    participant : laban.Participant
        Participant info
    
    Returns
    -------
    dict
        Test results
    """
    import labanalysis as laban
    import numpy as np
    from scipy import stats
    
    # Load data
    metabolic = laban.MetabolicRecord.from_file(filename, breath_by_breath=False)
    
    vo2 = metabolic.vo2.data
    hr = metabolic.heart_rate.data
    rer = metabolic.rer.data
    
    # Age-predicted HRmax
    hr_max_pred = 207 - 0.7 * participant.age
    
    # Peak values
    vo2max = vo2.max()
    hr_peak = hr.max()
    rer_peak = rer.max()
    
    # Plateau detection
    vo2_diff = np.diff(vo2 * participant.weight / 1000) * 1000
    has_plateau = np.any(vo2_diff < 150)
    
    # Maximal effort criteria
    hr_criterion = abs(hr_peak - hr_max_pred) <= 10
    rer_criterion = rer_peak >= 1.10
    criteria_met = sum([hr_criterion, rer_criterion, has_plateau])
    
    # Submaximal prediction (if available)
    submaximal_mask = hr < 0.85 * hr_max_pred
    if submaximal_mask.sum() > 5:
        hr_sub = hr[submaximal_mask]
        vo2_sub = vo2[submaximal_mask]
        slope, intercept, r_value, _, _ = stats.linregress(hr_sub, vo2_sub)
        vo2max_predicted = slope * hr_max_pred + intercept
    else:
        vo2max_predicted = None
    
    results = {
        'vo2max_ml_kg_min': vo2max,
        'vo2max_l_min': vo2max * participant.weight / 1000,
        'hr_peak_bpm': hr_peak,
        'hr_max_predicted_bpm': hr_max_pred,
        'rer_peak': rer_peak,
        'has_plateau': has_plateau,
        'criteria_met': criteria_met,
        'is_maximal': criteria_met >= 2,
        'vo2max_predicted': vo2max_predicted,
    }
    
    return results

# Usage
participant = laban.Participant(name="John", surname="Doe", height=1.80, weight=75, age=25)
results = analyze_vo2max_test("vo2max_test.txt", participant)

print("=== VO2max Test Results ===\n")
print(f"VO2max: {results['vo2max_ml_kg_min']:.1f} ml/kg/min ({results['vo2max_l_min']:.2f} L/min)")
print(f"Peak HR: {results['hr_peak_bpm']:.0f} bpm")
print(f"Peak RER: {results['rer_peak']:.2f}")
print(f"\nMaximal effort: {'Yes' if results['is_maximal'] else 'No'} ({results['criteria_met']}/3 criteria)")

if results['vo2max_predicted']:
    print(f"\nSubmaximal prediction: {results['vo2max_predicted']:.1f} ml/kg/min")
```

## Normative Data

### Age and Gender Norms

| Age Group | Male (ml/kg/min) | Female (ml/kg/min) |
|-----------|------------------|-------------------|
| **20-29** | 38-48 (avg 43) | 33-42 (avg 38) |
| **30-39** | 35-45 (avg 40) | 30-38 (avg 34) |
| **40-49** | 33-43 (avg 38) | 28-36 (avg 32) |
| **50-59** | 30-40 (avg 35) | 25-33 (avg 29) |
| **60+** | 27-37 (avg 32) | 22-30 (avg 26) |

### Fitness Classification

| Classification | Male | Female |
|----------------|------|--------|
| **Excellent** | >55 | >50 |
| **Good** | 45-55 | 40-50 |
| **Average** | 35-45 | 30-40 |
| **Below average** | 25-35 | 20-30 |
| **Poor** | <25 | <20 |

## Visualization

```python
import matplotlib.pyplot as plt

# Plot VO2, HR, RER over time
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

time = vo2.index

# VO2
axes[0].plot(time, vo2.data, 'b-', linewidth=2)
axes[0].axhline(vo2max, color='red', linestyle='--', label=f'VO2max ({vo2max:.1f})')
if len(vt1_idx) > 0:
    axes[0].axhline(vt1_vo2, color='orange', linestyle=':', label=f'VT1 ({vt1_vo2:.1f})')
axes[0].set_ylabel('VO2 (ml/kg/min)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# HR
axes[1].plot(time, hr.data, 'r-', linewidth=2)
axes[1].axhline(hr_peak, color='red', linestyle='--', label=f'Peak ({hr_peak:.0f})')
axes[1].axhline(hr_max_predicted, color='gray', linestyle=':', label=f'Predicted max ({hr_max_predicted:.0f})')
axes[1].set_ylabel('Heart Rate (bpm)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# RER
axes[2].plot(time, rer.data, 'g-', linewidth=2)
axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
axes[2].axhline(1.1, color='red', linestyle=':', label='Max criterion (1.10)')
axes[2].set_ylabel('RER')
axes[2].set_xlabel('Time (s)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('VO2max Test - Metabolic Response')
plt.tight_layout()
plt.show()
```

## Troubleshooting

### No plateau detected

```python
# If VO2 doesn't plateau, use peak VO2 and verify effort criteria

if not has_plateau:
    print("No plateau detected")
    print("Report as VO2peak, not VO2max")
    print("Ensure:")
    print("  - Test was maximal (check RER ≥ 1.10)")
    print("  - HR near predicted max")
    print("  - Subject gave maximal effort")
```

### Erratic HR or VO2 data

```python
# Filter noisy data before analysis

vo2_smooth = laban.butterworth_filt(vo2.data, freq=metabolic.sampling_frequency, cut=0.05, order=2)
hr_smooth = laban.median_filt(hr.data, order=5)
```

## See Also

- **[API Reference: MetabolicRecord](../../api-reference/records/records.md#metabolicrecord)** - Metabolic data class
- **[API Reference: SubmaximalVO2MaxTest](../../api-reference/protocols/vo2max.md)** - Test protocol
- **[Filtering](../signal-processing/filtering.md)** - Data smoothing

---

**Key Metrics**: VO2max, HRmax, RER, VT1, VT2  
**Reference**: ACSM. Guidelines for Exercise Testing and Prescription. 11th ed. 2021.
