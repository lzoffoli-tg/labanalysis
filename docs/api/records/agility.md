# labanalysis.records.agility

Change-of-direction analysis for agility assessment.

**Source**: `src/labanalysis/records/agility.py`

## Overview

The `agility` module provides the `ChangeOfDirectionExercise` class for analyzing directional changes during agility tests. Tracks contact phases, loading/propulsion phases, and velocity changes using force platform and marker data.

## Classes

### ChangeOfDirectionExercise

Single change-of-direction step analysis.

```python
class ChangeOfDirectionExercise(WholeBody):
    """
    Represents a single step during a change of direction movement.
    
    Analyzes the contact phase during directional changes, detecting:
    - Loading phase: Deceleration (braking)
    - Propulsion phase: Acceleration (push-off in new direction)
    - Inversion time: Transition between loading and propulsion
    
    Parameters
    ----------
    left_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for left foot
    right_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for right foot
    left_hand_ground_reaction_force : ForcePlatform, optional
        Force platform data for left hand (rarely used)
    right_hand_ground_reaction_force : ForcePlatform, optional
        Force platform data for right hand (rarely used)
    s2 : Point3D, optional
        S2 sacral marker (critical for detecting inversion time)
    left_acromion : Point3D, optional
        Left shoulder marker
    right_acromion : Point3D, optional
        Right shoulder marker
    **signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals (markers, EMG, joint angles, etc.)
    
    Attributes
    ----------
    side : str
        'bilateral', 'left', or 'right' depending on available force data
    contact_phase : WholeBody
        Data during entire contact phase
    contact_time : float
        Duration of contact phase (seconds)
    loading_phase : WholeBody
        Data during loading (braking) phase
    propulsion_phase : WholeBody
        Data during propulsion phase
    inversion_time : float
        Time instant marking transition from loading to propulsion (seconds)
    velocity : Signal3D
        Velocity of S2 marker (m/s)
    
    Notes
    -----
    The S2 (sacral) marker is critical for detecting the inversion time,
    which is defined as the time when the anteroposterior velocity of S2
    reaches its maximum (peak deceleration before re-acceleration).
    
    At least one foot force platform must be provided.
    
    Typical change-of-direction patterns:
    - 90° cuts (lateral directional change)
    - 180° turns (reversal of direction)
    - V-cuts (diagonal cuts)
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Load COD step data
    >>> data = laban.read_tdf(
    ...     "cod_90deg.tdf",
    ...     marker_keys=["s2", ".*ankle.*", ".*knee.*"],
    ...     forceplatform_keys=[".*"]
    ... )
    >>> 
    >>> # Create COD object
    >>> cod = laban.ChangeOfDirectionExercise(**data)
    >>> 
    >>> # Get phases
    >>> print(f"Contact time: {cod.contact_time:.3f} s")
    >>> print(f"Inversion time: {cod.inversion_time:.3f} s")
    >>> 
    >>> # Analyze loading vs propulsion
    >>> loading = cod.loading_phase
    >>> propulsion = cod.propulsion_phase
    >>> 
    >>> loading_time = loading.index[-1] - loading.index[0]
    >>> propulsion_time = propulsion.index[-1] - propulsion.index[0]
    >>> 
    >>> print(f"Loading: {loading_time:.3f} s ({loading_time/cod.contact_time*100:.1f}%)")
    >>> print(f"Propulsion: {propulsion_time:.3f} s ({propulsion_time/cod.contact_time*100:.1f}%)")
    """
```

**Key Properties:**

- `contact_phase` - Entire ground contact period
- `loading_phase` - Braking/deceleration phase (contact start to inversion)
- `propulsion_phase` - Push-off/acceleration phase (inversion to contact end)
- `inversion_time` - Transition point between loading and propulsion
- `velocity` - S2 marker velocity (3D)
- `contact_time` - Total contact duration

**Example - COD Force Analysis:**

```python
import labanalysis as laban
import matplotlib.pyplot as plt
import numpy as np

# Load COD step
data = laban.read_tdf("cod_step.tdf", marker_keys=["s2"], forceplatform_keys=[".*"])
cod = laban.ChangeOfDirectionExercise(**data)

# Get force data
force = cod.resultant_force.force['Z'].to_numpy()
time = cod.resultant_force.index

# Get velocity
velocity = cod.velocity
vel_ap = velocity['Y'].to_numpy()  # Anteroposterior velocity

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Force
ax1.plot(time, force, 'b-', linewidth=2)
ax1.axvline(cod.inversion_time, color='r', linestyle='--', label='Inversion time')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Vertical Force (N)')
ax1.set_title('Ground Reaction Force')
ax1.legend()
ax1.grid(True)

# Velocity
ax2.plot(velocity.index, vel_ap, 'g-', linewidth=2)
ax2.axvline(cod.inversion_time, color='r', linestyle='--', label='Inversion time')
ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('AP Velocity (m/s)')
ax2.set_title('S2 Anteroposterior Velocity')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

**Example - Loading vs Propulsion Comparison:**

```python
import labanalysis as laban
import numpy as np

# Load COD step
data = laban.read_tdf("cod_step.tdf", marker_keys=["s2"], forceplatform_keys=[".*"])
cod = laban.ChangeOfDirectionExercise(**data)

# Extract phases
loading = cod.loading_phase
propulsion = cod.propulsion_phase

# Loading phase metrics
loading_force = loading.resultant_force.force['Z'].to_numpy()
loading_time = loading.index[-1] - loading.index[0]
loading_peak_force = loading_force.max()
loading_impulse = np.trapz(loading_force, loading.index)

# Propulsion phase metrics
propulsion_force = propulsion.resultant_force.force['Z'].to_numpy()
propulsion_time = propulsion.index[-1] - propulsion.index[0]
propulsion_peak_force = propulsion_force.max()
propulsion_impulse = np.trapz(propulsion_force, propulsion.index)

# Print comparison
print("Loading Phase (Braking):")
print(f"  Duration: {loading_time*1000:.0f} ms")
print(f"  Peak force: {loading_peak_force:.0f} N")
print(f"  Impulse: {loading_impulse:.1f} N·s")

print("\nPropulsion Phase (Push-off):")
print(f"  Duration: {propulsion_time*1000:.0f} ms")
print(f"  Peak force: {propulsion_peak_force:.0f} N")
print(f"  Impulse: {propulsion_impulse:.1f} N·s")

# Ratios
print(f"\nLoading/Propulsion time ratio: {loading_time/propulsion_time:.2f}")
print(f"Loading/Propulsion impulse ratio: {loading_impulse/propulsion_impulse:.2f}")
```

**Example - Velocity Change Analysis:**

```python
import labanalysis as laban
import numpy as np

# Load COD step
data = laban.read_tdf("cod_step.tdf", marker_keys=["s2"], forceplatform_keys=[".*"])
cod = laban.ChangeOfDirectionExercise(**data)

# Get velocity
velocity = cod.velocity
vel_ap = velocity['Y'].to_numpy()  # Anteroposterior
vel_ml = velocity['X'].to_numpy()  # Mediolateral

# Entry velocity (first samples)
entry_vel_ap = np.mean(vel_ap[:10])
entry_vel_ml = np.mean(vel_ml[:10])
entry_speed = np.sqrt(entry_vel_ap**2 + entry_vel_ml**2)

# Exit velocity (last samples)
exit_vel_ap = np.mean(vel_ap[-10:])
exit_vel_ml = np.mean(vel_ml[-10:])
exit_speed = np.sqrt(exit_vel_ap**2 + exit_vel_ml**2)

# Velocity change
delta_vel_ap = exit_vel_ap - entry_vel_ap
delta_vel_ml = exit_vel_ml - entry_vel_ml

print(f"Entry velocity: {entry_speed:.2f} m/s (AP: {entry_vel_ap:.2f}, ML: {entry_vel_ml:.2f})")
print(f"Exit velocity: {exit_speed:.2f} m/s (AP: {exit_vel_ap:.2f}, ML: {exit_vel_ml:.2f})")
print(f"Velocity change: ΔAP = {delta_vel_ap:.2f} m/s, ΔML = {delta_vel_ml:.2f} m/s")
print(f"Speed retention: {exit_speed/entry_speed*100:.1f}%")
```

---

## Common Workflows

### 1. Multiple COD Steps Analysis

```python
import labanalysis as laban
import pandas as pd

# Analyze multiple COD steps from shuttle run
steps_data = []
for i in range(1, 6):  # 5 directional changes
    data = laban.read_tdf(f"shuttle_step_{i}.tdf", marker_keys=["s2"], forceplatform_keys=[".*"])
    cod = laban.ChangeOfDirectionExercise(**data)
    
    steps_data.append({
        'step': i,
        'contact_time_ms': cod.contact_time * 1000,
        'inversion_time_s': cod.inversion_time,
        'loading_pct': (cod.inversion_time - cod.index[0]) / cod.contact_time * 100
    })

df = pd.DataFrame(steps_data)
print(df)
print(f"\nMean contact time: {df['contact_time_ms'].mean():.0f} ± {df['contact_time_ms'].std():.0f} ms")
```

### 2. Left vs Right Comparison

```python
import labanalysis as laban

# Load left and right COD steps
data_left = laban.read_tdf("cod_left.tdf", marker_keys=["s2"], forceplatform_keys=["left.*"])
data_right = laban.read_tdf("cod_right.tdf", marker_keys=["s2"], forceplatform_keys=["right.*"])

cod_left = laban.ChangeOfDirectionExercise(**data_left)
cod_right = laban.ChangeOfDirectionExercise(**data_right)

# Compare contact times
print(f"Left leg: {cod_left.contact_time*1000:.0f} ms")
print(f"Right leg: {cod_right.contact_time*1000:.0f} ms")

# Asymmetry
asymmetry = abs(cod_left.contact_time - cod_right.contact_time) / ((cod_left.contact_time + cod_right.contact_time) / 2) * 100
print(f"Asymmetry: {asymmetry:.1f}%")
```

---

## Troubleshooting

### Issue: "Cannot detect inversion time"

**Cause**: S2 marker not provided or velocity not calculated correctly

**Solution**: Ensure S2 marker is included
```python
# Verify S2 marker present
if 's2' not in data:
    print("S2 marker missing - inversion time cannot be calculated")
    
# Load with S2
data = laban.read_tdf("trial.tdf", marker_keys=["s2", ".*"], forceplatform_keys=[".*"])
```

### Issue: "Loading phase empty"

**Cause**: Inversion time equals contact start (no deceleration phase detected)

**Solution**: Check velocity profile - may be pure acceleration step (e.g., sprint start)
```python
# Check velocity profile
velocity = cod.velocity
vel_ap = velocity['Y'].to_numpy()

import matplotlib.pyplot as plt
plt.plot(velocity.index, vel_ap)
plt.xlabel('Time (s)')
plt.ylabel('AP Velocity (m/s)')
plt.title('S2 Velocity Profile')
plt.grid(True)
plt.show()
```

---

## See Also

- [WholeBody](bodies.md) - Full body biomechanical model
- [ForcePlatform](records.md#forceplatform) - Force platform data structure
- [Agility Tests](../protocols/agility-tests.md) - Agility test protocols
- [Locomotion](locomotion.md) - Gait analysis classes

---

**Analyze change-of-direction movements with phase detection and velocity tracking.**
