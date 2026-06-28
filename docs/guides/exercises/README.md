# Exercise Analysis Guide

This guide covers the `exercises` module for biomechanical exercise analysis.

## Overview

The exercises module provides specialized classes for analyzing different types of exercises:
- **Jumps**: Single jumps, drop jumps, repeated jump sequences
- **Balance**: Upright and prone posture stability
- **Agility**: Change of direction movements

All exercise classes extend `WholeBody`, inheriting full biomechanical analysis capabilities (joint angles, segment lengths, etc.) while adding exercise-specific metrics.

---

## Jump Analysis

### Single Jump

Analyze vertical jumps including squat jumps (SJ) and counter-movement jumps (CMJ).

**Load from file:**
```python
import labanalysis as laban

jump = laban.SingleJump.from_tdf(
    file="cmj_trial.tdf",
    bodymass_kg=75.0,
    left_foot_ground_reaction_force="left_fp",
    right_foot_ground_reaction_force="right_fp"
)
```

**Key metrics:**
```python
# Performance metrics
print(f"Jump height: {jump.jump_height:.1f} cm")
print(f"Peak power: {jump.peak_power:.0f} W")
print(f"Peak force: {jump.peak_vertical_force:.0f} N")
print(f"Takeoff velocity: {jump.takeoff_velocity:.2f} m/s")

# Timing metrics
print(f"Contact time: {jump.contact_time:.3f} s")
print(f"Flight time: {jump.flight_time:.3f} s")
print(f"Eccentric phase: {jump.eccentric_phase_duration:.3f} s")
print(f"Concentric phase: {jump.concentric_phase_duration:.3f} s")

# Performance index
print(f"RSI: {jump.reactive_strength_index:.2f}")
```

**Phase detection:**
```python
# Ground contact phase
contact_phase = jump.contact_phase
print(f"Contact starts: {contact_phase.start:.3f} s")
print(f"Contact ends: {contact_phase.end:.3f} s")

# Flight phase
flight_phase = jump.flight_phase
print(f"Flight duration: {flight_phase.duration:.3f} s")

# Eccentric phase (downward movement)
eccentric = jump.eccentric_phase
print(f"Eccentric from {eccentric.start:.3f} to {eccentric.end:.3f} s")

# Concentric phase (upward propulsion)
concentric = jump.concentric_phase
print(f"Concentric from {concentric.start:.3f} to {concentric.end:.3f} s")
```

**Full biomechanical analysis:**
```python
# Joint angles during jump (inherited from WholeBody)
knee_angle = jump.left_knee_flexionextension
hip_angle = jump.left_hip_flexionextension
ankle_angle = jump.left_ankle_flexionextension

# Find maximum flexion
max_knee_flexion = knee_angle.min()
print(f"Maximum knee flexion: {max_knee_flexion:.1f}°")

# Export all angles
all_angles = jump.joint_angles.to_dataframe()
all_angles.to_csv('jump_kinematics.csv')
```

---

### Drop Jump

Plyometric drop jumps from elevated surfaces.

**Load with box height:**
```python
dj = laban.DropJump.from_tdf(
    file="dj_40cm.tdf",
    bodymass_kg=75.0,
    box_height_cm=40.0,
    left_foot_ground_reaction_force="left_fp",
    right_foot_ground_reaction_force="right_fp"
)

print(f"Box height: {dj.box_height_cm} cm")
print(f"Contact time: {dj.contact_time*1000:.0f} ms")
print(f"Flight time: {dj.flight_time*1000:.0f} ms")
print(f"RSI: {dj.reactive_strength_index:.2f}")
```

**Landing vs takeoff phases:**
```python
# DropJump has modified phase detection
landing = dj.landing_phase  # First ground contact
flight = dj.flight_phase    # Time in air after takeoff

print(f"Landing contact: {landing.duration:.3f} s")
print(f"Subsequent flight: {flight.duration:.3f} s")
```

---

### Repeated Jumps

Continuous jump sequences for fatigue analysis.

**Load and analyze:**
```python
rj = laban.RepeatedJumps.from_tdf(
    file="repeated_jumps_10x.tdf",
    bodymass_kg=75.0,
    left_foot_ground_reaction_force="left_fp"
)

# Access individual jumps
print(f"Total jumps detected: {len(rj.jumps)}")

for i, jump in enumerate(rj.jumps, 1):
    print(f"Jump {i}: "
          f"Height={jump.jump_height:.1f} cm, "
          f"Contact={jump.contact_time*1000:.0f} ms, "
          f"Power={jump.peak_power:.0f} W")
```

**Fatigue analysis:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Extract metrics for each jump
heights = [j.jump_height for j in rj.jumps]
contact_times = [j.contact_time for j in rj.jumps]
powers = [j.peak_power for j in rj.jumps]

# Calculate fatigue index
fatigue_index = (max(heights) - min(heights)) / max(heights) * 100
print(f"Fatigue index: {fatigue_index:.1f}%")

# Regression across jumps
jump_numbers = np.arange(1, len(heights) + 1)
slope, intercept = np.polyfit(jump_numbers, heights, 1)
print(f"Height decrease: {-slope:.2f} cm per jump")

# Plot fatigue curve
plt.figure(figsize=(10, 6))
plt.plot(jump_numbers, heights, 'o-', label='Height')
plt.plot(jump_numbers, np.polyval([slope, intercept], jump_numbers), '--', label='Trend')
plt.xlabel('Jump Number')
plt.ylabel('Jump Height (cm)')
plt.title('Jump Height Fatigue')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Balance Analysis

### Upright Posture

Standing balance assessment with center of pressure (COP) analysis.

**Load and analyze:**
```python
balance = laban.UprightPosture.from_tdf(
    file="balance_eyes_open.tdf",
    bodymass_kg=75.0,
    left_foot_ground_reaction_force="left_fp",
    right_foot_ground_reaction_force="right_fp"
)

# COP metrics
print(f"Sway area: {balance.sway_area:.0f} mm²")
print(f"Mean velocity: {balance.sway_velocity:.1f} mm/s")
print(f"ML sway: {balance.mediolateral_sway:.1f} mm")
print(f"AP sway: {balance.anteroposterior_sway:.1f} mm")
```

**COP trajectory:**
```python
# Center of pressure path
cop = balance.center_of_pressure

# Plot COP trajectory
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.plot(cop.data[:, 0], cop.data[:, 1])
plt.xlabel('Mediolateral (mm)')
plt.ylabel('Anteroposterior (mm)')
plt.title('Center of Pressure Trajectory')
plt.axis('equal')
plt.grid(True)
plt.show()
```

**Eyes open vs closed comparison:**
```python
# Load both conditions
eyes_open = laban.UprightPosture.from_tdf("balance_eo.tdf", bodymass_kg=75)
eyes_closed = laban.UprightPosture.from_tdf("balance_ec.tdf", bodymass_kg=75)

# Compare sway
print("Eyes Open:")
print(f"  Sway area: {eyes_open.sway_area:.0f} mm²")
print(f"  Velocity: {eyes_open.sway_velocity:.1f} mm/s")

print("\nEyes Closed:")
print(f"  Sway area: {eyes_closed.sway_area:.0f} mm²")
print(f"  Velocity: {eyes_closed.sway_velocity:.1f} mm/s")

# Calculate Romberg quotient
romberg = eyes_closed.sway_area / eyes_open.sway_area
print(f"\nRomberg quotient: {romberg:.2f}")
```

---

### Prone Posture

Plank position core stability assessment.

**Load with 4 force platforms:**
```python
plank = laban.PronePosture.from_tdf(
    file="plank_30s.tdf",
    bodymass_kg=75.0,
    left_hand_ground_reaction_force="left_hand_fp",
    right_hand_ground_reaction_force="right_hand_fp",
    left_foot_ground_reaction_force="left_foot_fp",
    right_foot_ground_reaction_force="right_foot_fp"
)

# Force distribution
total_force = plank.ground_reaction_force.module.mean()
print(f"Mean total force: {total_force:.0f} N")
print(f"Expected (bodyweight): {plank.bodymass_kg * 9.81:.0f} N")
```

**Hand vs foot loading:**
```python
# Extract individual platform forces
left_hand_force = plank.left_hand_ground_reaction_force.force.module.mean()
right_hand_force = plank.right_hand_ground_reaction_force.force.module.mean()
left_foot_force = plank.left_foot_ground_reaction_force.force.module.mean()
right_foot_force = plank.right_foot_ground_reaction_force.force.module.mean()

hand_force = left_hand_force + right_hand_force
foot_force = left_foot_force + right_foot_force

hand_percentage = hand_force / (hand_force + foot_force) * 100
print(f"Hand support: {hand_percentage:.1f}%")
print(f"Foot support: {100-hand_percentage:.1f}%")
```

---

## Agility Analysis

### Change of Direction Exercise

Shuttle run and agility movement analysis.

**Load and analyze:**
```python
cod = laban.ChangeOfDirectionExercise.from_tdf(
    file="shuttle_left.tdf",
    bodymass_kg=75.0,
    left_foot_ground_reaction_force="left_fp",
    right_foot_ground_reaction_force="right_fp"
)

# Movement direction
print(f"Direction: {cod.side}")

# Timing metrics
print(f"Contact time: {cod.contact_time:.3f} s")
print(f"Loading time: {cod.loading_time*1000:.0f} ms ({cod.loading_time/cod.contact_time*100:.0f}%)")
print(f"Propulsion time: {cod.propulsion_time*1000:.0f} ms ({cod.propulsion_time/cod.contact_time*100:.0f}%)")

# Velocity
print(f"Max velocity: {cod.maximum_velocity:.2f} m/s")
```

**Loading vs propulsion phases:**
```python
# Loading phase (deceleration)
loading_time = cod.loading_time
loading_percentage = (loading_time / cod.contact_time) * 100

# Propulsion phase (acceleration in new direction)
propulsion_time = cod.propulsion_time
propulsion_percentage = (propulsion_time / cod.contact_time) * 100

print(f"Loading: {loading_time*1000:.0f} ms ({loading_percentage:.0f}%)")
print(f"Propulsion: {propulsion_time*1000:.0f} ms ({propulsion_percentage:.0f}%)")

# Efficiency index
efficiency = propulsion_time / loading_time
print(f"Efficiency (prop/load): {efficiency:.2f}")
```

---

## Common Patterns

### Processing Pipeline

Apply signal processing before analysis:

```python
from labanalysis.pipelines import get_default_processing_pipeline

# Load raw data
jump = laban.SingleJump.from_tdf("jump_raw.tdf", bodymass_kg=75)

# Apply processing
pipeline = get_default_processing_pipeline()
pipeline(jump, inplace=True)

# Now analyze processed data
print(f"Jump height: {jump.jump_height:.1f} cm")
```

### Exporting Data

Export exercise data to pandas DataFrames:

```python
# Method 1: Export timeseries directly
df = jump.ground_reaction_force.to_dataframe()
df.to_csv('grf.csv')

# Method 2: Export specific signals
knee_df = jump.left_knee_flexionextension.to_dataframe()
ankle_df = jump.left_ankle_flexionextension.to_dataframe()

# Method 3: Export aggregated data
all_angles = jump.joint_angles.to_dataframe()
all_lengths = jump.segment_lengths.to_dataframe()

combined = all_angles.join(all_lengths)
combined.to_csv('full_biomechanics.csv')
```

### Copying Exercises

Create independent copies:

```python
jump_copy = jump.copy()

# Modify copy without affecting original
pipeline(jump_copy, inplace=True)
assert jump.jump_height != jump_copy.jump_height  # Different values
```

---

## See Also

- [Exercises API](../../api/exercises.md) - Full API reference
- [WholeBody Guide](../biomechanics/wholebody.md) - Inherited biomechanical capabilities
- [Pipelines Guide](../pipelines/processing-pipelines.md) - Signal processing
- [Test Protocols](../test-protocols/overview.md) - Complete test protocol workflows
