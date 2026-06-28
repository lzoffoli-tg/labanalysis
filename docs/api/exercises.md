# Exercises Module

The `exercises` module provides classes for analyzing biomechanical exercise data, including jumps, balance exercises, and agility movements.

## Overview

Exercise classes extend the `WholeBody` biomechanical model with exercise-specific analysis capabilities. They detect movement phases, calculate performance metrics, and provide specialized visualization.

## Classes

### SingleJump

Single vertical jump analysis with phase detection and performance metrics.

**Module:** `labanalysis.exercises.single_jump`

**Inherits from:** `WholeBody`

**Description:**  
Analyzes single vertical jumps including squat jumps (SJ) and counter-movement jumps (CMJ). Automatically detects eccentric, concentric, flight, and contact phases from ground reaction force data.

**Parameters:**
- `bodymass_kg` (float, required): Subject body mass in kilograms
- `left_foot_ground_reaction_force` (ForcePlatform, optional): Left foot force platform
- `right_foot_ground_reaction_force` (ForcePlatform, optional): Right foot force platform
- All WholeBody marker parameters (40+ anatomical markers)

**Key Properties:**
- `flight_time` (float): Flight phase duration in seconds
- `contact_time` (float): Ground contact duration in seconds
- `jump_height` (float): Calculated jump height in centimeters
- `takeoff_velocity` (float): Vertical velocity at takeoff in m/s
- `peak_vertical_force` (float): Maximum vertical ground reaction force in Newtons
- `peak_power` (float): Maximum power output in Watts
- `reactive_strength_index` (float): RSI = jump_height / contact_time
- `eccentric_phase_duration` (float): Eccentric phase duration in seconds
- `concentric_phase_duration` (float): Concentric phase duration in seconds

**Methods:**
- `from_tdf(file, ...)` (classmethod): Load from BTS TDF file
- `copy()`: Create deep copy

**Example:**
```python
import labanalysis as laban

# Load single jump from file
jump = laban.SingleJump.from_tdf(
    file="cmj_trial.tdf",
    bodymass_kg=75.0,
    left_foot_ground_reaction_force="left_fp",
    right_foot_ground_reaction_force="right_fp"
)

# Access performance metrics
print(f"Jump height: {jump.jump_height:.1f} cm")
print(f"Peak power: {jump.peak_power:.0f} W")
print(f"RSI: {jump.reactive_strength_index:.2f}")

# Phase durations
print(f"Contact time: {jump.contact_time:.3f} s")
print(f"Flight time: {jump.flight_time:.3f} s")
```

---

### DropJump

Drop jump (depth jump) analysis with box height consideration.

**Module:** `labanalysis.exercises.drop_jump`

**Inherits from:** `SingleJump`

**Description:**  
Extends SingleJump for plyometric drop jumps from elevated surfaces. Includes box height parameter and modified phase detection for landing-takeoff sequences.

**Additional Parameters:**
- `box_height_cm` (float, optional): Height of the box in centimeters

**Additional Properties:**
- `box_height_cm` (float): Box height
- `landing_phase`: Ground contact batch before flight phase
- Overrides `flight_phase` to handle landing-takeoff sequence

**Methods:**
- `from_tdf(file, box_height_cm, ...)` (classmethod): Load with box height
- `set_box_height_cm(height)`: Set box height
- `copy()`: Create deep copy preserving box height

**Example:**
```python
# Drop jump from 40cm box
dj = laban.DropJump.from_tdf(
    file="dj_40cm.tdf",
    bodymass_kg=75.0,
    box_height_cm=40.0,
    left_foot_ground_reaction_force="left_fp",
    right_foot_ground_reaction_force="right_fp"
)

print(f"Box height: {dj.box_height_cm} cm")
print(f"Contact time: {dj.contact_time*1000:.0f} ms")
print(f"RSI: {dj.reactive_strength_index:.2f}")
```

---

### RepeatedJumps

Analysis of continuous repeated jump sequences.

**Module:** `labanalysis.exercises.repeated_jumps`

**Inherits from:** `WholeBody`

**Description:**  
Detects and analyzes multiple consecutive jumps in a single trial. Useful for fatigue assessment and endurance testing.

**Key Properties:**
- `jumps` (list of SingleJump): Individual jumps detected in sequence
- `excluded_jumps` (list): Jumps excluded from analysis
- Individual jump metrics for each detected jump

**Methods:**
- `from_tdf(file, ...)` (classmethod): Load and auto-detect jumps
- `copy()`: Create deep copy

**Example:**
```python
# Load repeated jump sequence
rj = laban.RepeatedJumps.from_tdf(
    file="repeated_jumps_10x.tdf",
    bodymass_kg=75.0,
    left_foot_ground_reaction_force="left_fp"
)

# Analyze individual jumps
for i, jump in enumerate(rj.jumps, 1):
    print(f"Jump {i}: {jump.jump_height:.1f} cm, Contact: {jump.contact_time*1000:.0f} ms")

# Calculate fatigue index
heights = [j.jump_height for j in rj.jumps]
fatigue_index = (max(heights) - min(heights)) / max(heights) * 100
print(f"Fatigue index: {fatigue_index:.1f}%")
```

---

### UprightPosture

Upright balance and postural stability analysis.

**Module:** `labanalysis.exercises.upright_posture`

**Inherits from:** `WholeBody`

**Description:**  
Analyzes upright standing balance through center of pressure (COP) movement, sway metrics, and force distribution.

**Key Properties:**
- `center_of_pressure`: COP trajectory (Point3D)
- `sway_area`: 95% confidence ellipse area (mm²)
- `sway_velocity`: Mean COP velocity (mm/s)
- `mediolateral_sway`: ML sway amplitude (mm)
- `anteroposterior_sway`: AP sway amplitude (mm)

**Methods:**
- `from_tdf(file, ...)` (classmethod): Load from file
- `copy()`: Create deep copy

**Example:**
```python
# Upright balance test
balance = laban.UprightPosture.from_tdf(
    file="balance_eyes_open.tdf",
    bodymass_kg=75.0,
    left_foot_ground_reaction_force="left_fp",
    right_foot_ground_reaction_force="right_fp"
)

print(f"Sway area: {balance.sway_area:.0f} mm²")
print(f"Mean velocity: {balance.sway_velocity:.1f} mm/s")
```

---

### PronePosture

Prone (plank) posture stability analysis.

**Module:** `labanalysis.exercises.prone_posture`

**Inherits from:** `WholeBody`

**Description:**  
Analyzes core stability during plank exercises with hands and feet on force platforms.

**Key Properties:**
- Similar to UprightPosture but adapted for prone position
- Hand and foot force distribution
- Core stability metrics

**Methods:**
- `from_tdf(file, ...)` (classmethod): Load from file
- `copy()`: Create deep copy

**Example:**
```python
# Plank stability test
plank = laban.PronePosture.from_tdf(
    file="plank_30s.tdf",
    bodymass_kg=75.0,
    left_hand_ground_reaction_force="left_hand_fp",
    right_hand_ground_reaction_force="right_hand_fp",
    left_foot_ground_reaction_force="left_foot_fp",
    right_foot_ground_reaction_force="right_foot_fp"
)

# Analyze hand vs foot load distribution
total_force = plank.ground_reaction_force.module.mean()
print(f"Mean total force: {total_force:.0f} N")
```

---

### ChangeOfDirectionExercise

Change of direction and agility exercise analysis.

**Module:** `labanalysis.exercises.change_of_direction`

**Inherits from:** `WholeBody`

**Description:**  
Analyzes shuttle runs and change of direction movements. Detects contact phases, loading/propulsion times, and maximum velocity.

**Key Properties:**
- `contact_time`: Ground contact duration
- `loading_time`: Deceleration phase duration
- `propulsion_time`: Acceleration phase duration
- `maximum_velocity`: Peak velocity during exercise
- `side`: Movement direction ('left', 'right', 'bilateral')

**Methods:**
- `from_tdf(file, ...)` (classmethod): Load from file
- `copy()`: Create deep copy

**Example:**
```python
# Change of direction analysis
cod = laban.ChangeOfDirectionExercise.from_tdf(
    file="shuttle_left.tdf",
    bodymass_kg=75.0,
    left_foot_ground_reaction_force="left_fp",
    right_foot_ground_reaction_force="right_fp"
)

print(f"Side: {cod.side}")
print(f"Contact time: {cod.contact_time:.3f} s")
print(f"Loading time: {cod.loading_time*1000:.0f} ms ({cod.loading_time/cod.contact_time*100:.0f}%)")
print(f"Propulsion time: {cod.propulsion_time*1000:.0f} ms ({cod.propulsion_time/cod.contact_time*100:.0f}%)")
print(f"Max velocity: {cod.maximum_velocity:.2f} m/s")
```

---

## Common Patterns

### Loading from Files

All exercise classes support loading from BTS Bioengineering TDF files:

```python
exercise = ExerciseClass.from_tdf(
    file="path/to/file.tdf",
    bodymass_kg=75.0,
    # Force platform names (match TDF channel names)
    left_foot_ground_reaction_force="left_fp",
    right_foot_ground_reaction_force="right_fp",
    # Marker names (optional, uses defaults if not specified)
    left_heel="LHEE",
    right_heel="RHEE",
    # ... other markers
)
```

### Creating Copies

All exercises support deep copying:

```python
exercise_copy = exercise.copy()
# exercise_copy is independent of exercise
```

### Accessing Biomechanical Properties

Exercises inherit all WholeBody properties:

```python
# Joint angles
left_knee_angle = exercise.left_knee_flexionextension
left_hip_angle = exercise.left_hip_flexionextension

# Segment lengths
thigh_length = exercise.left_thigh_length

# Export to DataFrame
all_angles = exercise.joint_angles.to_dataframe()
all_lengths = exercise.segment_lengths.to_dataframe()
```

---

## See Also

- [WholeBody API](records/bodies.md) - Full body biomechanical model
- [Protocols API](protocols/protocols.md) - Test protocol classes
- [Exercises Guide](../guides/exercises/README.md) - Exercise analysis guide
