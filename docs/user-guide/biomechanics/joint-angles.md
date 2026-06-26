# Joint Angles

Guide to accessing and analyzing the 37 joint angles automatically calculated by the WholeBody class.

## Overview

The `WholeBody` class automatically calculates 37 joint angles from 3D marker positions. All angles follow biomechanical conventions with consistent sign interpretations across the body.

**Angle Categories:**
- **Lower Limb** (8): Ankle (4) + Knee (4) + Hip (6) = 14 angles
- **Pelvis & Trunk** (7): Pelvis (3) + Trunk (4) = 7 angles
- **Upper Limb** (10): Shoulder girdle (4) + Shoulder (6) + Elbow (2) = 12 angles
- **Head & Spine** (5): Neck (3) + Spine curvature (2) = 5 angles

All angles are returned as `Signal1D` objects in degrees.

## Quick Reference

```python
import labanalysis as laban

# Create WholeBody from markers
body = laban.WholeBody.from_tdf(
    "gait.tdf",
    left_asis="LASI", right_asis="RASI",
    left_psis="LPSI", right_psis="RPSI",
    left_knee_lateral="LKNEL", left_knee_medial="LKNEM",
    left_ankle_lateral="LANKL", left_ankle_medial="LANKM",
    # ... other markers
)

# Access joint angles (Signal1D objects)
knee_angle = body.left_knee_flexionextension
hip_angle = body.left_hip_flexionextension
ankle_angle = body.left_ankle_flexionextension

# Get angle data and statistics
print(f"Knee range: {knee_angle.data.min():.1f}° to {knee_angle.data.max():.1f}°")
print(f"Mean hip angle: {hip_angle.data.mean():.1f}°")

# All angles use consistent sign convention
# Positive = flexion, abduction, internal rotation (typically)
```

## Lower Limb Angles

### Ankle Angles (4)

| Property | Description | Positive | Negative | Normal ROM |
|----------|-------------|----------|----------|------------|
| `left_ankle_flexionextension` | Sagittal plane | Dorsiflexion | Plantarflexion | -20° to +30° |
| `right_ankle_flexionextension` | Sagittal plane | Dorsiflexion | Plantarflexion | -20° to +30° |
| `left_ankle_inversioneversion` | Frontal plane | Eversion | Inversion | -30° to +20° |
| `right_ankle_inversioneversion` | Frontal plane | Eversion | Inversion | -30° to +20° |

**Example: Ankle Angle During Gait**

```python
# Get ankle dorsiflexion angle
ankle_flex = body.left_ankle_flexionextension

# Detect heel strike (peak dorsiflexion)
from labanalysis import find_peaks
heel_strikes, _ = find_peaks(ankle_flex.data, distance=100)

# Analyze ankle ROM during gait cycle
for i in range(len(heel_strikes) - 1):
    start = heel_strikes[i]
    end = heel_strikes[i + 1]
    
    cycle_data = ankle_flex.data[start:end]
    max_dorsi = cycle_data.max()
    max_plantar = cycle_data.min()
    rom = max_dorsi - max_plantar
    
    print(f"Stride {i+1}: ROM = {rom:.1f}° (dorsi: {max_dorsi:.1f}°, plantar: {max_plantar:.1f}°)")
```

### Knee Angles (4)

| Property | Description | Positive | Negative | Normal ROM |
|----------|-------------|----------|----------|------------|
| `left_knee_flexionextension` | Sagittal plane | Flexion | Extension | 0° to 140° |
| `right_knee_flexionextension` | Sagittal plane | Flexion | Extension | 0° to 140° |
| `left_knee_varusvalgus` | Frontal plane alignment | Varus (bow-leg) | Valgus (knock-knee) | -5° to +5° |
| `right_knee_varusvalgus` | Frontal plane alignment | Varus (bow-leg) | Valgus (knock-knee) | -5° to +5° |

**Note**: Sign convention - **Positive = Varus** (bow-legged), **Negative = Valgus** (knock-knee) for both knees. Angles are normalized to [-180°, +180°] to ensure values near 0° for aligned knees (prevents wrapping to ~360°).

**Example: Knee Varus/Valgus Assessment**

```python
# Get knee alignment
varus_valgus = body.left_knee_varusvalgus

# Calculate mean alignment (during stance phase)
# Assuming standing or mid-stance analysis
mean_alignment = varus_valgus.data.mean()

if mean_alignment > 5:
    print(f"Varus alignment: {mean_alignment:.1f}° (bow-legged)")
elif mean_alignment < -5:
    print(f"Valgus alignment: {abs(mean_alignment):.1f}° (knock-knee)")
else:
    print(f"Neutral alignment: {mean_alignment:.1f}° (perfect)")
```

### Hip Angles (6)

| Property | Description | Positive | Negative | Normal ROM |
|----------|-------------|----------|----------|------------|
| `left_hip_flexionextension` | Sagittal plane | Flexion | Extension | -20° to 120° |
| `right_hip_flexionextension` | Sagittal plane | Flexion | Extension | -20° to 120° |
| `left_hip_abductionadduction` | Frontal plane | Abduction | Adduction | -20° to 45° |
| `right_hip_abductionadduction` | Frontal plane | Abduction | Adduction | -20° to 45° |
| `left_hip_internalexternalrotation` | Transverse plane | Internal | External | -45° to 45° |
| `right_hip_internalexternalrotation` | Transverse plane | Internal | External | -45° to 45° |

**Note**: Hip angles return approximately 0° when the thigh is vertical (neutral standing position). Flexion/extension angles are measured relative to the vertical axis.

**Example: Hip Kinematics During Squat**

```python
# Get hip flexion during squat
hip_flex = body.left_hip_flexionextension

# Find squat depth (maximum flexion)
max_flexion = hip_flex.data.max()
squat_depth_time = hip_flex.index[np.argmax(hip_flex.data)]

print(f"Maximum hip flexion: {max_flexion:.1f}° at t={squat_depth_time:.2f}s")

# Classify squat depth
if max_flexion > 100:
    depth = "Deep squat (>100°)"
elif max_flexion > 90:
    depth = "Parallel squat (90-100°)"
else:
    depth = "Partial squat (<90°)"
    
print(f"Squat classification: {depth}")
```

## Pelvis & Trunk Angles

### Pelvis Angles (7)

Pelvis angles available in both global (earth-fixed) and local (body-segment) reference frames.

| Property | Description | Positive | Negative | Neutral |
|----------|-------------|----------|----------|---------|
| `pelvis_anteroposterior_tilt_global` | Sagittal tilt | Posterior tilt | Anterior tilt | 0° |
| `pelvis_lateraltilt_global` | Frontal tilt (global) | Left hip higher | Right hip higher | 0° |
| `pelvis_lateraltilt_local` | Frontal tilt (trunk-relative) | Left hip higher | Right hip higher | 0° |
| `pelvis_rotation_global` | Transverse rotation (global) | Left hip forward | Right hip forward | 0° |
| `pelvis_rotation_local` | Transverse rotation (neck-relative) | Left hip forward | Right hip forward | 0° |

**Note**: 
- Global measurements are relative to gravity (earth-fixed coordinate system)
- Local measurements are relative to trunk orientation (body-segment coordinate system)
- All pelvis angles are 0° in neutral position (level, aligned pelvis)

**Example: Pelvic Tilt Assessment**

```python
# Get anterior/posterior pelvic tilt
pelvic_tilt = body.pelvis_anteroposterior_tilt_global

# Analyze static posture
mean_tilt = pelvic_tilt.data.mean()

if mean_tilt < -10:
    print(f"Anterior pelvic tilt: {abs(mean_tilt):.1f}° (lordotic posture)")
elif mean_tilt > 10:
    print(f"Posterior pelvic tilt: {mean_tilt:.1f}° (flat back)")
else:
    print(f"Neutral pelvic tilt: {mean_tilt:.1f}°")

# Get lateral pelvic tilt (global = absolute space orientation)
lateral_tilt_global = body.pelvis_lateraltilt_global
mean_lateral_global = lateral_tilt_global.data.mean()

# Get lateral pelvic tilt (local = relative to trunk)
lateral_tilt_local = body.pelvis_lateraltilt_local
mean_lateral_local = lateral_tilt_local.data.mean()

if abs(mean_lateral_global) < 2:
    print(f"Level pelvis (global): {mean_lateral_global:.1f}°")
elif mean_lateral_global > 0:
    print(f"Left hip higher (global): {mean_lateral_global:.1f}°")
else:
    print(f"Right hip higher (global): {abs(mean_lateral_global):.1f}°")

# Compare global vs local to assess trunk compensation
difference = abs(mean_lateral_global - mean_lateral_local)
if difference > 5:
    print(f"Significant trunk compensation detected: {difference:.1f}° difference")
```

### Trunk Angles (4)

| Property | Description | Positive | Negative |
|----------|-------------|----------|----------|
| `trunk_flexionextension_global` | Sagittal flexion (global) | Flexion | Extension |
| `trunk_lateralflexion` | Frontal flexion (pelvis frame) | Right flexion | Left flexion |
| `trunk_rotation_global` | Transverse rotation (global) | Right rotation | Left rotation |
| `trunk_rotation_local` | Transverse rotation (relative to pelvis) | Right rotation | Left rotation |

**Example: Trunk-Pelvis Dissociation During Gait**

```python
# Get trunk and pelvis rotation (both global)
trunk_rot_global = body.trunk_rotation_global
pelvis_rot_global = body.pelvis_rotation_global

# Calculate dissociation (trunk relative to pelvis)
dissociation = trunk_rot_global.data - pelvis_rot_global.data

# Or use the pre-calculated local rotation
trunk_local = body.trunk_rotation_local  # Same as dissociation

# Analyze coordination
max_dissociation = np.max(np.abs(trunk_local.data))
print(f"Maximum trunk-pelvis dissociation: {max_dissociation:.1f}°")

# Normal gait: 5-10° dissociation
if max_dissociation < 5:
    print("Reduced trunk-pelvis dissociation (rigid rotation)")
elif max_dissociation > 15:
    print("Excessive trunk-pelvis dissociation")
```

## Upper Limb Angles

### Shoulder Girdle Angles (4)

| Property | Description | Positive | Negative |
|----------|-------------|----------|----------|
| `shoulder_lateraltilt_global` | Shoulder elevation (global) | Left shoulder higher | Right shoulder higher |
| `shoulder_lateraltilt_local` | Shoulder elevation (trunk-relative) | Left shoulder higher | Right shoulder higher |
| `left_scapular_protractionretraction` | Scapular position (transverse) | Protraction | Retraction |
| `right_scapular_protractionretraction` | Scapular position (transverse) | Protraction | Retraction |

**Note**: 
- Global measurements show shoulder orientation relative to gravity
- Local measurements show shoulder orientation relative to trunk axis
- Useful for assessing postural compensation patterns

**Example: Scapular Protraction Analysis**

```python
# Get scapular position
scap_prot_left = body.left_scapular_protractionretraction
scap_prot_right = body.right_scapular_protractionretraction

# Assess resting posture
mean_left = scap_prot_left.data.mean()
mean_right = scap_prot_right.data.mean()

print(f"Left scapula: {mean_left:.1f}° ({'protracted' if mean_left > 0 else 'retracted'})")
print(f"Right scapula: {mean_right:.1f}° ({'protracted' if mean_right > 0 else 'retracted'})")

# Bilateral asymmetry
asymmetry = abs(mean_left - mean_right)
if asymmetry > 10:
    print(f"Warning: Bilateral asymmetry = {asymmetry:.1f}°")
```

### Shoulder Joint Angles (6)

| Property | Description | Positive | Negative | Normal ROM |
|----------|-------------|----------|----------|------------|
| `left_shoulder_flexionextension` | Sagittal plane | Flexion | Extension | -60° to 180° |
| `right_shoulder_flexionextension` | Sagittal plane | Flexion | Extension | -60° to 180° |
| `left_shoulder_abductionadduction` | Frontal plane | Abduction | Adduction | 0° to 180° |
| `right_shoulder_abductionadduction` | Frontal plane | Abduction | Adduction | 0° to 180° |
| `left_shoulder_internalexternalrotation` | Transverse plane | Internal | External | -90° to 90° |
| `right_shoulder_internalexternalrotation` | Transverse plane | Internal | External | -90° to 90° |

**Note**: Shoulder angles return approximately 0° when the arm hangs vertically at the side (neutral standing position). Flexion/extension and abduction/adduction are measured relative to the vertical axis.

**Example: Shoulder ROM Assessment**

```python
# Get shoulder abduction
shoulder_abd = body.left_shoulder_abductionadduction

# Find maximum abduction
max_abd = shoulder_abd.data.max()

print(f"Maximum shoulder abduction: {max_abd:.1f}°")

# Classify ROM
if max_abd > 170:
    print("Full ROM (overhead reach)")
elif max_abd > 90:
    print("Functional ROM (shoulder level)")
else:
    print("Limited ROM (<90°)")
```

### Elbow Angles (2)

| Property | Description | Positive | Negative | Normal ROM |
|----------|-------------|----------|----------|------------|
| `left_elbow_flexionextension` | Sagittal plane | Flexion | Extension | 0° to 150° |
| `right_elbow_flexionextension` | Sagittal plane | Flexion | Extension | 0° to 150° |

## Head & Spine Angles

### Neck Angles (3)

| Property | Description | Positive | Negative |
|----------|-------------|----------|----------|
| `neck_lateralflexion` | Frontal lateral flexion | Right flexion | Left flexion |
| `neck_flexionextension` | Sagittal flexion/extension | Flexion (forward) | Extension (backward) |

**Example: Forward Head Posture Assessment**

```python
# Get neck flexion (global)
neck_global = body.neck_flexionextension_global

# Assess forward head posture
mean_angle = neck_global.data.mean()

if mean_angle > 15:
    print(f"Forward head posture: {mean_angle:.1f}° (protracted)")
elif mean_angle < -10:
    print(f"Retracted head posture: {mean_angle:.1f}°")
else:
    print(f"Neutral head position: {mean_angle:.1f}°")
```

### Spine Curvature Angles (2)

| Property | Description | Normal Range | Interpretation |
|----------|-------------|--------------|----------------|
| `lumbar_lordosis` | Lumbar curvature angle at L2 | 140-160° | Smaller = more curved (hyperlordosis) |
| `dorsal_kyphosis` | Thoracic curvature angle at T5 | 140-160° | Smaller = more curved (hyperkyphosis) |

**Note**: These are **internal angles** at vertebrae (T5-L2-PSIS_mid for lordosis, C7-T5-L2 for kyphosis). Smaller angles indicate greater spinal curvature.

**Calculation Details**:
- **Lumbar lordosis**: Angle at L2 vertex, formed by T5 (superior) → L2 → PSIS midpoint (inferior)
- **Thoracic kyphosis**: Angle at T5 vertex, formed by C7 (superior) → T5 → L2 (inferior)

**Example: Spinal Curvature Assessment**

```python
# Get spine curvature angles
lordosis = body.lumbar_lordosis
kyphosis = body.dorsal_kyphosis

# Analyze static posture
mean_lordosis = lordosis.data.mean()
mean_kyphosis = kyphosis.data.mean()

print(f"Lumbar lordosis: {mean_lordosis:.1f}°")
if mean_lordosis < 140:
    print("  → Hyperlordosis (excessive lumbar curve, sway back)")
elif mean_lordosis > 160:
    print("  → Hypolordosis (flat lumbar spine)")
else:
    print("  → Normal lumbar curvature")

print(f"Thoracic kyphosis: {mean_kyphosis:.1f}°")
if mean_kyphosis < 140:
    print("  → Hyperkyphosis (rounded upper back, hunchback)")
elif mean_kyphosis > 160:
    print("  → Hypokyphosis (flat upper back)")
else:
    print("  → Normal thoracic curvature")
```

## Bilateral Comparison

```python
# Compare left and right sides
left_knee = body.left_knee_flexionextension
right_knee = body.right_knee_flexionextension

# Calculate asymmetry during a movement
left_max = left_knee.data.max()
right_max = right_knee.data.max()

asymmetry = abs(left_max - right_max) / ((left_max + right_max) / 2) * 100

print(f"Left knee max flexion: {left_max:.1f}°")
print(f"Right knee max flexion: {right_max:.1f}°")
print(f"Bilateral asymmetry: {asymmetry:.1f}%")

# Asymmetry > 10% may indicate imbalance
if asymmetry > 10:
    print("Warning: Significant bilateral asymmetry detected")
```

## Export Angles to DataFrame

```python
import pandas as pd

# Export specific angles
ankle_df = body.left_ankle_flexionextension.to_dataframe()
knee_df = body.left_knee_flexionextension.to_dataframe()
hip_df = body.left_hip_flexionextension.to_dataframe()

# Combine angles into one DataFrame
angles_df = pd.concat([ankle_df, knee_df, hip_df], axis=1)

# Export to Excel
angles_df.to_excel("joint_angles.xlsx")

# Export to CSV
angles_df.to_csv("joint_angles.csv")
```

## Filtering Angles

```python
import labanalysis as laban

# Filter a single angle
knee_angle = body.left_knee_flexionextension
freq = 100  # Hz

knee_filtered = laban.butterworth_filt(
    knee_angle.data,
    freq=freq,
    cut=6,  # 6 Hz cutoff
    order=4
)

# Update angle data
knee_angle.data = knee_filtered

# Or filter entire WholeBody at once
body_filtered = body.apply(
    laban.butterworth_filt,
    axis=0,
    inplace=False,
    freq=100,
    cut=6,
    order=4
)

# Now all angles are filtered
knee_filtered = body_filtered.left_knee_flexionextension
```

## Troubleshooting

### Issue: "ValueError: marker not found"

Some angles require specific markers:

```python
# Ankle angles require: ankle_lateral, ankle_medial, heel, toe, metatarsal markers
# Knee angles require: knee_lateral, knee_medial, hip, ankle
# Hip angles require: pelvis markers (asis, psis), knee

# Check which markers are available
print(body.keys())

# Attempt to access angle will raise clear error if markers missing
try:
    ankle = body.left_ankle_flexionextension
except ValueError as e:
    print(f"Missing markers: {e}")
```

### Issue: Angle Values Seem Incorrect

Check axis conventions:

```python
# Verify axes match your data
print(f"Vertical axis: {body.vertical_axis}")  # Should be 'Y'
print(f"Anteroposterior axis: {body.anteroposterior_axis}")  # Should be 'Z'

# If incorrect, markers may have wrong axis labels
# Recreate WholeBody with correct axis specification
```

### Issue: Noisy Angle Data

Filter before analysis:

```python
# Apply low-pass filter
knee_angle_filtered = laban.butterworth_filt(
    body.left_knee_flexionextension.data,
    freq=100,
    cut=6,
    order=4
)
```

## See Also

- [WholeBody Model](whole-body-model.md) - Complete WholeBody guide with all 88 properties
- [Coordinate Systems](coordinate-systems.md) - Understanding reference frames
- [API Reference: WholeBody](../../api-reference/records/bodies.md) - Complete angle API
- [Test Protocols: Gait Analysis](../test-protocols/gait-analysis.md) - Using angles in gait analysis

---

**42 Joint Angles**: Comprehensive kinematic analysis from 3D marker data using biomechanical conventions.
