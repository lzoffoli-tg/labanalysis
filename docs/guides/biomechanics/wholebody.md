# WholeBody Model

Complete guide to the WholeBody biomechanical model with 104+ properties including 37 angular measures, anatomical markers, joint centers, and reference frames.

## Overview

The `WholeBody` class is a comprehensive full-body biomechanical model that automatically calculates joint angles and segment properties from 3D marker positions. It's the most feature-rich class in labanalysis for kinematic analysis.

**Key Features:**
- 42 joint angular measures (Signal1D) - all ISB-compliant
- Anatomical marker positions (Point3D)
- Joint centers and reference frames
- Segment properties (mass, length, etc.)
- Automatic joint angle calculation using biomechanical conventions
- Support for incomplete marker sets with fallback calculations

## Quick Start

```python
import labanalysis as laban

# Load WholeBody from motion capture data
body = laban.WholeBody.from_tdf(
    "mocap.tdf",
    # Required cranial markers
    left_psis="LPSI",
    right_psis="RPSI",
    left_asis="LASI",
    right_asis="RASI",
    
    # Left leg markers
    left_knee_medial="LKNEM",
    left_knee_lateral="LKNEL",
    left_ankle_medial="LANKM",
    left_ankle_lateral="LANKL",
    left_heel="LHEE",
    left_toe="LTOE",
    
    # Right leg markers
    right_knee_medial="RKNEM",
    right_knee_lateral="RKNEL",
    right_ankle_medial="RANKM",
    right_ankle_lateral="RANKL",
    right_heel="RHEE",
    right_toe="RTOE",
    
    # Additional markers as needed...
)

# Access joint angles
knee_flexion = body.left_knee_flexionextension
print(f"Knee flexion range: {knee_flexion.data.min():.1f}° to {knee_flexion.data.max():.1f}°")

# Access marker positions
c7_position = body.c7_vertebra
print(f"C7 trajectory: {c7_position.data.shape}")
```

## Key Properties

The WholeBody class provides 104+ properties organized into categories:

### 42 Anatomical Markers (Point3D)

All marker properties return `Point3D` objects with 3D coordinates (x, y, z) over time.

#### Cranial Markers (8)

| Property | Description | Label |
|----------|-------------|-------|
| `left_psis` | Left posterior superior iliac spine | LPSI |
| `right_psis` | Right posterior superior iliac spine | RPSI |
| `left_asis` | Left anterior superior iliac spine | LASI |
| `right_asis` | Right anterior superior iliac spine | RASI |
| `sacrum` | Sacrum (computed midpoint) | SAC |
| `left_acromion` | Left acromion | LACR |
| `right_acromion` | Right acromion | RACR |
| `c7_vertebra` | C7 vertebra | C7 |

#### Left Leg Markers (10)

| Property | Description | Label |
|----------|-------------|-------|
| `left_thigh` | Left thigh lateral | LTH |
| `left_knee_medial` | Left knee medial epicondyle | LKNEM |
| `left_knee_lateral` | Left knee lateral epicondyle | LKNEL |
| `left_knee` | Left knee (computed midpoint) | LKNE |
| `left_shank` | Left shank lateral | LSH |
| `left_ankle_medial` | Left ankle medial malleolus | LANKM |
| `left_ankle_lateral` | Left ankle lateral malleolus | LANKL |
| `left_ankle` | Left ankle (computed midpoint) | LANK |
| `left_heel` | Left heel | LHEE |
| `left_toe` | Left toe | LTOE |

#### Right Leg Markers (10)

Mirror of left leg markers (RTHIGH, RKNEM, RKNEL, etc.)

#### Left Arm Markers (7)

| Property | Description | Label |
|----------|-------------|-------|
| `left_elbow_medial` | Left elbow medial epicondyle | LEBM |
| `left_elbow_lateral` | Left elbow lateral epicondyle | LEBL |
| `left_elbow` | Left elbow (computed midpoint) | LELB |
| `left_wrist_medial` | Left wrist medial | LWRM |
| `left_wrist_lateral` | Left wrist lateral | LWRL |
| `left_wrist` | Left wrist (computed midpoint) | LWRI |
| `left_hand` | Left hand | LHAN |

#### Right Arm Markers (7)

Mirror of left arm markers (REBM, REBL, RELB, etc.)

### 36 Joint Angles (Signal1D)

All angle properties return `Signal1D` objects with angles in degrees over time.

#### Hip Angles (6)

| Property | Description | Range |
|----------|-------------|-------|
| `left_hip_flexionextension` | Left hip flexion(+) / extension(-) | -20° to 120° |
| `left_hip_abduction` | Left hip abduction(+) / adduction(-) | -20° to 45° |
| `left_hip_rotation` | Left hip internal(+) / external(-) rotation | -45° to 45° |
| `right_hip_flexionextension` | Right hip flexion/extension | -20° to 120° |
| `right_hip_abduction` | Right hip abduction/adduction | -20° to 45° |
| `right_hip_rotation` | Right hip rotation | -45° to 45° |

**Convention**: Angles calculated using pelvis as reference frame, following ISB recommendations.

#### Knee Angles (6)

| Property | Description | Range |
|----------|-------------|-------|
| `left_knee_flexionextension` | Left knee flexion(+) / extension(-) | 0° to 140° |
| `left_knee_abduction` | Left knee valgus(+) / varus(-) | -15° to 15° |
| `left_knee_rotation` | Left knee internal(+) / external(-) rotation | -30° to 30° |
| `right_knee_flexionextension` | Right knee flexion/extension | 0° to 140° |
| `right_knee_abduction` | Right knee valgus/varus | -15° to 15° |
| `right_knee_rotation` | Right knee rotation | -30° to 30° |

**Convention**: Angles calculated using thigh and shank segments.

#### Ankle Angles (6)

| Property | Description | Range |
|----------|-------------|-------|
| `left_ankle_flexionextension` | Left ankle dorsiflexion(+) / plantarflexion(-) | -30° to 45° |
| `left_ankle_abduction` | Left ankle eversion(+) / inversion(-) | -30° to 30° |
| `left_ankle_rotation` | Left ankle rotation | -20° to 20° |
| `right_ankle_flexionextension` | Right ankle dorsi/plantarflexion | -30° to 45° |
| `right_ankle_abduction` | Right ankle eversion/inversion | -30° to 30° |
| `right_ankle_rotation` | Right ankle rotation | -20° to 20° |

**Convention**: Angles calculated using shank and foot segments.

#### Shoulder Angles (6)

| Property | Description | Range |
|----------|-------------|-------|
| `left_shoulder_flexionextension` | Left shoulder flexion(+) / extension(-) | -60° to 180° |
| `left_shoulder_abduction` | Left shoulder abduction(+) / adduction(-) | 0° to 180° |
| `left_shoulder_rotation` | Left shoulder internal(+) / external(-) rotation | -90° to 90° |
| `right_shoulder_flexionextension` | Right shoulder flexion/extension | -60° to 180° |
| `right_shoulder_abduction` | Right shoulder abduction/adduction | 0° to 180° |
| `right_shoulder_rotation` | Right shoulder rotation | -90° to 90° |

**Convention**: Angles calculated using thorax and upper arm segments.

#### Elbow Angles (6)

| Property | Description | Range |
|----------|-------------|-------|
| `left_elbow_flexionextension` | Left elbow flexion(+) / extension(-) | 0° to 150° |
| `left_elbow_abduction` | Left elbow carrying angle | -20° to 20° |
| `left_elbow_rotation` | Left elbow pronation(+) / supination(-) | -90° to 90° |
| `right_elbow_flexionextension` | Right elbow flexion/extension | 0° to 150° |
| `right_elbow_abduction` | Right elbow carrying angle | -20° to 20° |
| `right_elbow_rotation` | Right elbow pronation/supination | -90° to 90° |

**Convention**: Angles calculated using upper arm and forearm segments.

#### Wrist Angles (6)

| Property | Description | Range |
|----------|-------------|-------|
| `left_wrist_flexionextension` | Left wrist flexion(+) / extension(-) | -70° to 80° |
| `left_wrist_abduction` | Left wrist radial(+) / ulnar(-) deviation | -40° to 20° |
| `left_wrist_rotation` | Left wrist rotation | -45° to 45° |
| `right_wrist_flexionextension` | Right wrist flexion/extension | -70° to 80° |
| `right_wrist_abduction` | Right wrist radial/ulnar deviation | -40° to 20° |
| `right_wrist_rotation` | Right wrist rotation | -45° to 45° |

**Convention**: Angles calculated using forearm and hand segments.

### 8 Computed Properties

| Property | Type | Description |
|----------|------|-------------|
| `mass` | float | Body mass in kg (from participant) |
| `height` | float | Body height in m (from participant) |
| `pelvis` | Segment | Pelvis segment |
| `thorax` | Segment | Thorax segment |
| `left_foot` | Segment | Left foot segment |
| `right_foot` | Segment | Right foot segment |
| `left_thigh_segment` | Segment | Left thigh segment |
| `right_thigh_segment` | Segment | Right thigh segment |

## Creating a WholeBody Instance

### Method 1: From TDF File

```python
body = laban.WholeBody.from_tdf(
    "mocap.tdf",
    # Provide marker names as they appear in the TDF file
    left_psis="LPSI",
    right_psis="RPSI",
    left_asis="LASI",
    right_asis="RASI",
    # ... all other markers
)
```

### Method 2: From Existing Record

```python
# Load record first
record = laban.TimeseriesRecord.from_tdf("mocap.tdf")

# Create WholeBody from record
body = laban.WholeBody(
    left_psis=record['MKRS']['LPSI'],
    right_psis=record['MKRS']['RPSI'],
    # ... other markers
)
```

### Method 3: Partial Marker Set

WholeBody supports incomplete marker sets with automatic fallbacks:

```python
# Minimum viable: just pelvis and leg markers
body = laban.WholeBody.from_tdf(
    "mocap.tdf",
    # Pelvis (required)
    left_psis="LPSI",
    right_psis="RPSI",
    left_asis="LASI",
    right_asis="RASI",
    
    # Left leg (required for left leg angles)
    left_knee_medial="LKNEM",
    left_knee_lateral="LKNEL",
    left_ankle_medial="LANKM",
    left_ankle_lateral="LANKL",
    left_heel="LHEE",
    left_toe="LTOE",
    
    # Right leg omitted - right leg angles will be None
)

# Check what's available
if body.left_knee_flexionextension is not None:
    print("Left knee angle available")
else:
    print("Left knee angle not calculated (missing markers)")
```

## Common Workflows

### Extract Gait Cycle Joint Angles

```python
import labanalysis as laban
import numpy as np

# Load full body kinematics
body = laban.WholeBody.from_tdf("gait.tdf", ...)

# Get joint angles
hip_flex = body.left_hip_flexionextension
knee_flex = body.left_knee_flexionextension
ankle_flex = body.left_ankle_flexionextension

# Find heel strikes (example with force platform)
record = laban.TimeseriesRecord.from_tdf("gait.tdf")
fp = record['FP1']
fz = fp.force['Fz'].data

# Detect heel strikes
from scipy.signal import find_peaks
heel_strikes, _ = find_peaks(fz, height=50, distance=500)

# Extract one gait cycle
cycle_start = heel_strikes[0]
cycle_end = heel_strikes[1]

hip_cycle = hip_flex.data[cycle_start:cycle_end]
knee_cycle = knee_flex.data[cycle_start:cycle_end]
ankle_cycle = ankle_flex.data[cycle_start:cycle_end]

# Normalize to 0-100% gait cycle
gait_percent = np.linspace(0, 100, len(hip_cycle))

# Plot
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=gait_percent, y=hip_cycle, name='Hip'))
fig.add_trace(go.Scatter(x=gait_percent, y=knee_cycle, name='Knee'))
fig.add_trace(go.Scatter(x=gait_percent, y=ankle_cycle, name='Ankle'))
fig.update_layout(
    title='Left Leg Sagittal Plane Angles',
    xaxis_title='Gait Cycle (%)',
    yaxis_title='Angle (°)'
)
fig.show()
```

### Compare Bilateral Symmetry

```python
# Load body
body = laban.WholeBody.from_tdf("squat.tdf", ...)

# Get bilateral knee angles
left_knee = body.left_knee_flexionextension.data
right_knee = body.right_knee_flexionextension.data

# Calculate symmetry metrics
max_diff = np.abs(left_knee - right_knee).max()
mean_diff = np.abs(left_knee - right_knee).mean()

# Symmetry angle
symmetry_angle = np.arctan2(2 * mean_diff, left_knee.mean() + right_knee.mean()) * 180 / np.pi

print(f"Max bilateral difference: {max_diff:.1f}°")
print(f"Mean bilateral difference: {mean_diff:.1f}°")
print(f"Symmetry angle: {symmetry_angle:.1f}°")

# Visual comparison
fig = go.Figure()
fig.add_trace(go.Scatter(y=left_knee, name='Left Knee'))
fig.add_trace(go.Scatter(y=right_knee, name='Right Knee'))
fig.add_trace(go.Scatter(y=left_knee - right_knee, name='Difference'))
fig.update_layout(title='Bilateral Knee Symmetry', yaxis_title='Angle (°)')
fig.show()
```

### Export Joint Angles to DataFrame

```python
import pandas as pd

# Load body
body = laban.WholeBody.from_tdf("motion.tdf", ...)

# Collect all available joint angles
angles_data = {
    'time': body.left_knee_flexionextension.index,
}

# Add all angles if available
if body.left_hip_flexionextension is not None:
    angles_data['left_hip_flex'] = body.left_hip_flexionextension.data
if body.left_knee_flexionextension is not None:
    angles_data['left_knee_flex'] = body.left_knee_flexionextension.data
if body.left_ankle_flexionextension is not None:
    angles_data['left_ankle_flex'] = body.left_ankle_flexionextension.data

# Create DataFrame
df = pd.DataFrame(angles_data)

# Export
df.to_excel("joint_angles.xlsx", index=False)
print(f"Exported {len(df)} samples to joint_angles.xlsx")
```

## Fallback Calculations

WholeBody implements intelligent fallbacks when markers are missing:

### Joint Midpoints

When medial/lateral epicondyle markers are missing, joints are estimated from segment endpoints:

```python
# With both knee markers
body = laban.WholeBody(
    left_knee_medial=marker1,
    left_knee_lateral=marker2,
    # ...
)
# → left_knee = midpoint(medial, lateral)

# With only one knee marker (fallback)
body = laban.WholeBody(
    left_knee_lateral=marker2,  # medial missing
    # ...
)
# → left_knee = lateral marker position
```

### Shoulder Calculation

Shoulders can be calculated from acromion markers OR estimated from C7 and pelvis:

```python
# Preferred: acromion markers
body = laban.WholeBody(
    left_acromion=acr_marker,
    # ...
)

# Fallback: estimate from C7 and pelvis width
body = laban.WholeBody(
    c7_vertebra=c7_marker,
    left_asis=lasi,
    right_asis=rasi,
    # ...
)
# → shoulder estimated from trunk geometry
```

## Biomechanical Conventions

### Coordinate Systems

All angles follow ISB (International Society of Biomechanics) recommendations:

- **X-axis**: Forward (anterior direction)
- **Y-axis**: Upward (superior direction)  
- **Z-axis**: Right (lateral direction)

### Angle Sign Conventions

| Joint | Positive | Negative |
|-------|----------|----------|
| Hip | Flexion | Extension |
| Knee | Flexion | Extension |
| Ankle | Dorsiflexion | Plantarflexion |
| Shoulder | Flexion | Extension |
| Elbow | Flexion | Extension |
| All | Abduction | Adduction |
| All | Internal rotation | External rotation |

### Segment Definitions

Segments are defined by anatomical landmarks following Winter (2009):

- **Pelvis**: ASIS and PSIS markers
- **Thigh**: Hip joint center to knee joint center
- **Shank**: Knee joint center to ankle joint center
- **Foot**: Heel to toe
- **Thorax**: C7 to mid-pelvis
- **Upper arm**: Shoulder to elbow
- **Forearm**: Elbow to wrist
- **Hand**: Wrist to hand marker

## Performance Considerations

WholeBody calculations are performed lazily:

```python
body = laban.WholeBody.from_tdf("large_file.tdf", ...)
# ↑ Fast - only loads marker data

angle = body.left_knee_flexionextension
# ↑ Slower - calculates angle on first access

angle2 = body.left_knee_flexionextension
# ↑ Fast - returns cached result
```

For batch processing, pre-calculate needed angles:

```python
# Pre-calculate all angles (force evaluation)
angles = [
    body.left_hip_flexionextension,
    body.left_knee_flexionextension,
    body.left_ankle_flexionextension,
    # ... etc
]

# Now all angles are cached for fast access
```

## Validation and Quality Checks

### Check Marker Completeness

```python
# Check which markers are available
available_markers = []
missing_markers = []

for marker_name in ['left_knee_medial', 'left_knee_lateral', 'left_ankle_medial']:
    marker = getattr(body, marker_name, None)
    if marker is not None and not np.isnan(marker.data).all():
        available_markers.append(marker_name)
    else:
        missing_markers.append(marker_name)

print(f"Available: {available_markers}")
print(f"Missing: {missing_markers}")
```

### Detect Invalid Angles

```python
# Check for physiologically impossible values
knee_angle = body.left_knee_flexionextension.data

# Knee flexion should be 0-140°
invalid = (knee_angle < -10) | (knee_angle > 150)

if invalid.any():
    print(f"Warning: {invalid.sum()} samples with invalid knee angles")
    print(f"Range: {knee_angle[invalid].min():.1f}° to {knee_angle[invalid].max():.1f}°")
```

### Check Marker Trajectory Quality

```python
# Calculate marker velocities to detect outliers
c7 = body.c7_vertebra
velocity = np.linalg.norm(np.diff(c7.data, axis=0), axis=1) * c7.sampling_frequency

# Flag high velocities (likely artifacts)
threshold = 5.0  # m/s
outliers = velocity > threshold

if outliers.any():
    print(f"Warning: {outliers.sum()} samples with high C7 velocity (>{threshold} m/s)")
```

## See Also

- **[Force Platforms Guide](force-platforms.md)** - Combine kinematics with kinetics
- **[Joint Angles Reference](joint-angles.md)** - Detailed angle definitions
- **[API Reference: WholeBody](../../api/records/bodies.md)** - Complete API documentation
- **[Tutorial: Full Body Kinematics](../../tutorials/03-full-body-kinematics.md)** - Complete workflow

---

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
