# Biomechanics

Guide to biomechanical analysis tools in labanalysis including the WholeBody model, force platforms, joint angles, and coordinate systems.

## Overview

labanalysis provides comprehensive tools for biomechanical analysis:

- **WholeBody Model** - 86 properties (42 markers, 36 joint angles, 8 computed properties)
- **Force Platforms** - Ground reaction force analysis
- **Joint Angles** - 36 joint angles across all major joints
- **Coordinate Systems** - Reference frame transformations
- **EMG Signals** - Electromyography analysis

## Quick Reference

| Topic | Description | Guide |
|-------|-------------|-------|
| **[WholeBody Model](whole-body-model.md)** | Full body with 86 properties | Complete model guide |
| **[Force Platforms](force-platforms.md)** | GRF and COP analysis | Force analysis guide |
| **[Joint Angles](joint-angles.md)** | 36 joint angles | Angle calculation guide |
| **[Coordinate Systems](coordinate-systems.md)** | Reference frames | Transformation guide |
| **[EMG Signals](emg-signals.md)** | Muscle activation | EMG processing guide |

## Quick Start

### Load WholeBody Model

```python
import labanalysis as laban

# Load motion capture data with anatomical markers
body = laban.WholeBody.from_tdf(
    "mocap_data.tdf",
    # Foot markers (metatarsals for accurate foot plane)
    left_first_metatarsal_head="LFM1",
    left_fifth_metatarsal_head="LFM5",
    left_heel="LHEE",
    right_first_metatarsal_head="RFM1",
    right_fifth_metatarsal_head="RFM5",
    right_heel="RHEE",
    # Spine markers
    c7="C7",
    t5="T5",
    sc="SC",
    # Head markers (4 cranial markers for head center)
    head_anterior="HANT",
    head_posterior="HPOST",
    head_left="HLEFT",
    head_right="HRIGHT",
    # Pelvis
    left_asis="LASIS",
    right_asis="RASIS",
    left_psis="LPSIS",
    right_psis="RPSIS"
    # ... additional markers
)

print(f"WholeBody model created with {len(body._markers)} markers")
```

### Access Joint Angles

```python
# Access specific joint angles
knee_flexion = body.left_knee_flexionextension
hip_abduction = body.left_hip_abduction
ankle_dorsiflexion = body.left_ankle_dorsiflexion

print(f"Left knee flexion range: {knee_flexion.data.min():.1f}° to {knee_flexion.data.max():.1f}°")

# Get all 36 joint angles
all_angles = body._angular_measures
print(f"Available angles: {len(all_angles)}")
```

### Analyze Force Platform

```python
# Extract force platform
fp = body.left_foot_ground_reaction_force

# Access force components
force_vertical = fp.force['Fz']
force_ap = fp.force['Fx']  # Anterior-posterior
force_ml = fp.force['Fy']  # Medial-lateral

# Calculate center of pressure
cop_x = -fp.torque['My'].data / fp.force['Fz'].data
cop_y = fp.torque['Mx'].data / fp.force['Fz'].data

print(f"Vertical force range: {force_vertical.data.min():.1f} to {force_vertical.data.max():.1f} N")
```

## WholeBody Model (86 Properties)

The WholeBody class provides a complete biomechanical model:

### 42 Anatomical Markers

**Foot Markers (8):**
- First/fifth metatarsal heads (bilateral)
- Heel markers (bilateral)
- Toe markers (bilateral)

**Lower Limb (12):**
- Ankle medial/lateral malleoli (bilateral)
- Knee medial/lateral epicondyles (bilateral)
- Greater trochanters (bilateral)

**Pelvis (4):**
- ASIS (bilateral)
- PSIS (bilateral)

**Spine (3):**
- C7, T5, Sternoclavicular joint

**Upper Limb (8):**
- Shoulder anterior/posterior (bilateral)
- Acromions (bilateral)
- Elbow medial/lateral (bilateral)
- Wrist medial/lateral (bilateral)

**Head (4):**
- Anterior, posterior, left, right cranial markers

**Hands (2):**
- Hand markers (bilateral)

[→ Complete WholeBody guide](whole-body-model.md)

### 36 Joint Angles

**Lower Limb (18):**
- Hip: flexion/extension, abduction/adduction, internal/external rotation (bilateral)
- Knee: flexion/extension, internal/external rotation, varus/valgus (bilateral)
- Ankle: dorsi/plantar flexion, inversion/eversion, internal/external rotation (bilateral)

**Spine & Pelvis (9):**
- Pelvis: anterior/posterior tilt, obliquity, rotation
- Trunk: flexion/extension, lateral flexion, rotation
- Neck: flexion/extension, lateral tilt, rotation

**Upper Limb (9):**
- Shoulder: flexion/extension, abduction/adduction, internal/external rotation (bilateral)
- Elbow: flexion/extension (bilateral)
- Forearm: pronation/supination (bilateral)

[→ Complete joint angles guide](joint-angles.md)

### 8 Computed Properties

- Head center (centroid of 4 cranial markers)
- Neck base (midpoint SC-C7)
- Foot planes (from 4 foot markers per foot)
- Joint centers (computed from marker positions)
- Segment reference frames
- Center of mass estimates

[→ Complete WholeBody guide](whole-body-model.md)

## Common Workflows

### Gait Analysis

```python
# Load walking/running data
from labanalysis.records.locomotion import WalkingExercise

walking = WalkingExercise.from_tdf(
    "walking.tdf",
    left_heel="LHEE",
    right_heel="RHEE",
    left_toe="LTOE",
    right_toe="RTOE",
    left_foot_ground_reaction_force='FP1'
)

# Extract gait parameters
stride_length = walking.stride_length
cadence = walking.cadence
walking_speed = walking.speed

print(f"Stride length: {stride_length.mean():.2f} m")
print(f"Cadence: {cadence:.1f} steps/min")
print(f"Speed: {walking_speed:.2f} m/s")
```

### Jump Analysis

```python
from labanalysis.records.jumping import SingleJump

# Load jump data
jump = SingleJump.from_tdf(
    "cmj.tdf",
    left_foot_ground_reaction_force='FP1'
)

# Analyze jump mechanics
print(f"Flight time: {jump.flight_time:.3f} s")
print(f"Jump height: {jump.jump_height:.2f} m")
```

### Posture Analysis

```python
from labanalysis.records.posture import UprightPosture

# Load standing posture data
posture = UprightPosture.from_tdf(
    "balance.tdf",
    c7="C7",
    left_asis="LASIS",
    right_asis="RASIS",
    left_foot_ground_reaction_force='FP1'
)

# Analyze posture alignment
trunk_lean = posture.trunk_lean
```

## Force Platform Analysis

### Ground Reaction Forces

```python
# Load force platform data
fp = laban.ForcePlatform(
    origin=np.array([0., 0., 0.]),
    force={'Fx': fx, 'Fy': fy, 'Fz': fz},
    torque={'Mx': mx, 'My': my, 'Mz': mz}
)

# Filter forces
fz_filtered = laban.butterworth_filt(
    signal=fp.force['Fz'].data,
    freq=1000,
    cut=10,
    order=4
)

# Detect contact phases
threshold = 20  # N
contacts = laban.crossings(fz_filtered - threshold, direction='both')
```

[→ Complete force platform guide](force-platforms.md)

### Center of Pressure

```python
# Calculate COP trajectory
cop_x = -fp.torque['My'].data / fp.force['Fz'].data
cop_y = fp.torque['Mx'].data / fp.force['Fz'].data

# COP displacement statistics
cop_displacement = np.sqrt(cop_x**2 + cop_y**2)
print(f"COP excursion: {cop_displacement.max()*100:.1f} cm")
```

## Coordinate Systems

### Transform Between Reference Frames

```python
# Get marker in global frame
marker_global = body.left_heel.data  # Shape: (n, 3)

# Transform to pelvis reference frame
pelvis_frame = body.pelvis_reference_frame
marker_pelvis = laban.to_reference_frame(
    marker_global,
    origin=pelvis_frame.origin,
    axes=pelvis_frame.axes
)
```

[→ Complete coordinate systems guide](coordinate-systems.md)

## EMG Signal Processing

```python
# Load EMG signal
emg = laban.EMGSignal.from_tdf("data.tdf", column="Biceps")

# Standard EMG processing pipeline
# 1. High-pass filter (remove baseline)
hp = laban.butterworth_filt(emg.data, freq=2000, cut=20, order=4, filt_type='high')

# 2. Full-wave rectification
rectified = np.abs(hp)

# 3. Low-pass filter (linear envelope)
envelope = laban.butterworth_filt(rectified, freq=2000, cut=6, order=4, filt_type='low')

# 4. Normalize to MVC
emg_normalized = (envelope / emg_mvc_max) * 100  # % MVC
```

[→ Complete EMG guide](emg-signals.md)

## Topic Guides

Detailed guides for each topic:

1. **[WholeBody Model](whole-body-model.md)** - Complete model with 86 properties
2. **[Force Platforms](force-platforms.md)** - GRF analysis and COP calculations
3. **[Joint Angles](joint-angles.md)** - All 36 joint angles explained
4. **[Coordinate Systems](coordinate-systems.md)** - Reference frames and transformations
5. **[EMG Signals](emg-signals.md)** - Electromyography processing

## See Also

- **[API Reference: Records Module](../../api-reference/records/README.md)** - WholeBody, ForcePlatform API
- **[Test Protocols](../test-protocols/README.md)** - Standardized biomechanical tests
- **[Signal Processing](../signal-processing/README.md)** - Filter and analyze biomechanical signals
- **[Examples](../../examples/biomechanics/)** - Biomechanics code examples

---

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
