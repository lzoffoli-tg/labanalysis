# labanalysis.records.posture

Posture analysis classes for upright standing and prone (plank) positions.

**Source**: `src/labanalysis/records/posture.py`

## Overview

The `posture` module provides classes for analyzing static postural stability:

- **UprightPosture**: Standing balance and postural sway analysis
- **PronePosture**: Plank position core stability and endurance

Both classes extend WholeBody and utilize force platform data for center of pressure (COP) analysis and postural stability metrics.

## Classes

### UprightPosture

Upright standing posture for balance assessment.

```python
class UprightPosture(WholeBody):
    """
    Represents an upright standing posture for balance and stability analysis.
    
    Used for balance tests and postural stability assessments with center of
    pressure (COP) tracking and sway metrics.
    
    Parameters
    ----------
    left_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for left foot contact
    right_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for right foot contact
    left_acromion : Point3D, optional
        Left shoulder marker for trunk sway analysis
    right_acromion : Point3D, optional
        Right shoulder marker for trunk sway analysis
    **signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals (markers, EMG, etc.)
    
    Attributes
    ----------
    side : str
        'bilateral', 'left', or 'right' depending on available force data
    output_metrics : pd.DataFrame
        COP sway metrics (path length, velocity, area, range)
    
    Notes
    -----
    At least one foot force platform must be provided.
    
    Common test conditions:
    - Eyes open vs eyes closed
    - Firm surface vs foam
    - Bilateral vs unilateral stance
    - Narrow vs wide base of support
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Load balance test data
    >>> data = laban.read_tdf(
    ...     "balance_eyes_open.tdf",
    ...     forceplatform_keys=[".*"]
    ... )
    >>> 
    >>> # Create upright posture object
    >>> posture = laban.UprightPosture(**data)
    >>> 
    >>> # Get COP metrics
    >>> metrics = posture.output_metrics
    >>> print(f"COP path length: {metrics['cop_path_length_mm'].values[0]:.1f} mm")
    >>> print(f"COP velocity: {metrics['cop_mean_velocity_mm_s'].values[0]:.2f} mm/s")
    >>> print(f"COP area: {metrics['cop_area_mm2'].values[0]:.1f} mm²")
    """
```

**Example - Eyes Open vs Eyes Closed:**

```python
import labanalysis as laban

# Load both conditions
data_eo = laban.read_tdf("balance_eyes_open.tdf", forceplatform_keys=[".*"])
data_ec = laban.read_tdf("balance_eyes_closed.tdf", forceplatform_keys=[".*"])

# Analyze
posture_eo = laban.UprightPosture(**data_eo)
posture_ec = laban.UprightPosture(**data_ec)

# Compare sway
metrics_eo = posture_eo.output_metrics
metrics_ec = posture_ec.output_metrics

print("Eyes Open:")
print(f"  Path length: {metrics_eo['cop_path_length_mm'].values[0]:.1f} mm")
print(f"  Velocity: {metrics_eo['cop_mean_velocity_mm_s'].values[0]:.2f} mm/s")

print("\nEyes Closed:")
print(f"  Path length: {metrics_ec['cop_path_length_mm'].values[0]:.1f} mm")
print(f"  Velocity: {metrics_ec['cop_mean_velocity_mm_s'].values[0]:.2f} mm/s")

# Romberg quotient (EC / EO)
romberg = metrics_ec['cop_mean_velocity_mm_s'].values[0] / metrics_eo['cop_mean_velocity_mm_s'].values[0]
print(f"\nRomberg quotient: {romberg:.2f}")
```

**Example - COP Visualization:**

```python
import labanalysis as laban
import matplotlib.pyplot as plt

# Load data
data = laban.read_tdf("balance.tdf", forceplatform_keys=[".*"])
posture = laban.UprightPosture(**data)

# Get COP trajectory
fp = posture.resultant_force
cop_x = fp.origin['X'].to_numpy()
cop_y = fp.origin['Y'].to_numpy()

# Plot COP path
plt.figure(figsize=(8, 8))
plt.plot(cop_x, cop_y, 'b-', linewidth=0.5, alpha=0.7)
plt.plot(cop_x[0], cop_y[0], 'go', markersize=10, label='Start')
plt.plot(cop_x[-1], cop_y[-1], 'ro', markersize=10, label='End')
plt.xlabel('ML Position (mm)')
plt.ylabel('AP Position (mm)')
plt.title('Center of Pressure Trajectory')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
```

---

### PronePosture

Prone (plank) position for core stability assessment.

```python
class PronePosture(WholeBody):
    """
    Represents a prone (plank) posture for core stability analysis.
    
    Used for core stability and endurance tests with force distribution
    analysis across hands and feet.
    
    Parameters
    ----------
    left_foot_ground_reaction_force : ForcePlatform
        Force platform data for left foot (required)
    right_foot_ground_reaction_force : ForcePlatform
        Force platform data for right foot (required)
    left_hand_ground_reaction_force : ForcePlatform
        Force platform data for left hand (required)
    right_hand_ground_reaction_force : ForcePlatform
        Force platform data for right hand (required)
    left_acromion : Point3D, optional
        Left shoulder marker for trunk position tracking
    right_acromion : Point3D, optional
        Right shoulder marker for trunk position tracking
    **signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals (markers, EMG for core muscles, etc.)
    
    Attributes
    ----------
    output_metrics : pd.DataFrame
        Force distribution and stability metrics
    
    Notes
    -----
    All four force platforms (both feet and both hands) are required.
    
    Common test protocols:
    - Static plank hold (endurance)
    - Dynamic plank (alternating limb lifts)
    - Side plank variations
    
    Typical force distribution in optimal plank:
    - Hands: ~60-65% of body weight
    - Feet: ~35-40% of body weight
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Load plank test data
    >>> data = laban.read_tdf(
    ...     "plank_test.tdf",
    ...     forceplatform_keys=[".*"]
    ... )
    >>> 
    >>> # Create prone posture object
    >>> plank = laban.PronePosture(**data)
    >>> 
    >>> # Get force distribution
    >>> lh_force = plank.left_hand_ground_reaction_force.force['Z'].to_numpy().mean()
    >>> rh_force = plank.right_hand_ground_reaction_force.force['Z'].to_numpy().mean()
    >>> lf_force = plank.left_foot_ground_reaction_force.force['Z'].to_numpy().mean()
    >>> rf_force = plank.right_foot_ground_reaction_force.force['Z'].to_numpy().mean()
    >>> 
    >>> total_force = lh_force + rh_force + lf_force + rf_force
    >>> hands_pct = (lh_force + rh_force) / total_force * 100
    >>> feet_pct = (lf_force + rf_force) / total_force * 100
    >>> 
    >>> print(f"Hands: {hands_pct:.1f}% | Feet: {feet_pct:.1f}%")
    """
```

**Example - Plank Endurance Test:**

```python
import labanalysis as laban
import numpy as np
import matplotlib.pyplot as plt

# Load plank data (e.g., 60-second hold)
data = laban.read_tdf("plank_60s.tdf", forceplatform_keys=[".*"])
plank = laban.PronePosture(**data)

# Extract forces over time
time = plank.left_hand_ground_reaction_force.index

lh = plank.left_hand_ground_reaction_force.force['Z'].to_numpy()
rh = plank.right_hand_ground_reaction_force.force['Z'].to_numpy()
lf = plank.left_foot_ground_reaction_force.force['Z'].to_numpy()
rf = plank.right_foot_ground_reaction_force.force['Z'].to_numpy()

hands_force = lh + rh
feet_force = lf + rf
total_force = hands_force + feet_force

# Calculate force distribution over time
hands_pct = (hands_force / total_force) * 100

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Force over time
ax1.plot(time, lh, label='Left Hand')
ax1.plot(time, rh, label='Right Hand')
ax1.plot(time, lf, label='Left Foot')
ax1.plot(time, rf, label='Right Foot')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Force (N)')
ax1.set_title('Plank Force Distribution')
ax1.legend()
ax1.grid(True)

# Hands % over time
ax2.plot(time, hands_pct, 'b-')
ax2.axhline(60, color='g', linestyle='--', label='Target (60%)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Hands Force (%)')
ax2.set_title('Hands Load Percentage')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

**Example - Symmetry Analysis:**

```python
import labanalysis as laban

# Load data
data = laban.read_tdf("plank.tdf", forceplatform_keys=[".*"])
plank = laban.PronePosture(**data)

# Calculate symmetry indices
lh = plank.left_hand_ground_reaction_force.force['Z'].to_numpy().mean()
rh = plank.right_hand_ground_reaction_force.force['Z'].to_numpy().mean()
lf = plank.left_foot_ground_reaction_force.force['Z'].to_numpy().mean()
rf = plank.right_foot_ground_reaction_force.force['Z'].to_numpy().mean()

# Hand symmetry
hand_asymmetry = abs(lh - rh) / ((lh + rh) / 2) * 100

# Foot symmetry
foot_asymmetry = abs(lf - rf) / ((lf + rf) / 2) * 100

# Diagonal symmetry (cross-pattern stability)
left_diagonal = lh + rf
right_diagonal = rh + lf
diagonal_asymmetry = abs(left_diagonal - right_diagonal) / ((left_diagonal + right_diagonal) / 2) * 100

print(f"Hand asymmetry: {hand_asymmetry:.1f}%")
print(f"Foot asymmetry: {foot_asymmetry:.1f}%")
print(f"Diagonal asymmetry: {diagonal_asymmetry:.1f}%")
```

---

## Common Workflows

### 1. Balance Test Battery

```python
import labanalysis as laban
import pandas as pd

# Define test conditions
conditions = [
    ('firm_eyes_open', 'Firm surface, eyes open'),
    ('firm_eyes_closed', 'Firm surface, eyes closed'),
    ('foam_eyes_open', 'Foam surface, eyes open'),
    ('foam_eyes_closed', 'Foam surface, eyes closed'),
]

# Analyze all conditions
results = []
for filename, description in conditions:
    data = laban.read_tdf(f"{filename}.tdf", forceplatform_keys=[".*"])
    posture = laban.UprightPosture(**data)
    metrics = posture.output_metrics
    metrics['condition'] = description
    results.append(metrics)

# Combine
df = pd.concat(results, ignore_index=True)
print(df[['condition', 'cop_path_length_mm', 'cop_mean_velocity_mm_s', 'cop_area_mm2']])
```

### 2. Unilateral Stance Comparison

```python
import labanalysis as laban

# Load left and right leg stance
data_left = laban.read_tdf("single_leg_left.tdf", forceplatform_keys=["left.*"])
data_right = laban.read_tdf("single_leg_right.tdf", forceplatform_keys=["right.*"])

posture_left = laban.UprightPosture(**data_left)
posture_right = laban.UprightPosture(**data_right)

# Compare
metrics_l = posture_left.output_metrics
metrics_r = posture_right.output_metrics

print(f"Left leg sway: {metrics_l['cop_mean_velocity_mm_s'].values[0]:.2f} mm/s")
print(f"Right leg sway: {metrics_r['cop_mean_velocity_mm_s'].values[0]:.2f} mm/s")
```

---

## Troubleshooting

### Issue: "No force platform data"

**Cause**: Missing required force platforms

**Solution for UprightPosture**: Provide at least one foot platform
```python
# Check available force platforms
print(list(data.keys()))
posture = laban.UprightPosture(
    left_foot_ground_reaction_force=data.get('left_foot_ground_reaction_force'),
    right_foot_ground_reaction_force=data.get('right_foot_ground_reaction_force')
)
```

**Solution for PronePosture**: All four platforms required
```python
# Verify all platforms present
required = ['left_hand_ground_reaction_force', 'right_hand_ground_reaction_force',
            'left_foot_ground_reaction_force', 'right_foot_ground_reaction_force']
for key in required:
    if key not in data:
        print(f"Missing: {key}")
```

---

## See Also

- [WholeBody](bodies.md) - Full body biomechanical model
- [ForcePlatform](records.md#forceplatform) - Force platform data structure
- [Balance Tests](../protocols/balance-tests.md) - Balance test protocols
- [Balance Tutorial](../../guides/test-protocols/balance.md) - Complete workflow

---

**Analyze postural stability with COP tracking and force distribution metrics.**
