# labanalysis.records.bodies

Full-body biomechanical models with automated joint angle calculation.

## Classes

### WholeBody

Complete full-body biomechanical model with 104+ properties including 38 angular measures, anatomical markers, joint centers, and reference frames.

**Source**: `src/labanalysis/records/bodies.py`

```python
class WholeBody:
    """
    Full-body biomechanical model with automatic joint angle calculation.
    
    Provides access to 42 anatomical marker positions (Point3D), 38 joint 
    angles (Signal1D), and 8 computed properties. Joint angles are calculated
    using biomechanical conventions following ISB recommendations.
    
    Parameters
    ----------
    left_psis : Point3D
        Left posterior superior iliac spine marker
    right_psis : Point3D
        Right posterior superior iliac spine marker
    left_asis : Point3D
        Left anterior superior iliac spine marker
    right_asis : Point3D
        Right anterior superior iliac spine marker
    left_knee_medial : Point3D, optional
        Left knee medial epicondyle marker
    left_knee_lateral : Point3D, optional
        Left knee lateral epicondyle marker
    left_ankle_medial : Point3D, optional
        Left ankle medial malleolus marker
    left_ankle_lateral : Point3D, optional
        Left ankle lateral malleolus marker
    left_heel : Point3D, optional
        Left heel marker
    left_toe : Point3D, optional
        Left toe marker
    ... (similar for right leg, arms, trunk)
    
    Notes
    -----
    - Pelvis markers (PSIS, ASIS) are required
    - Other markers are optional; angles are computed when sufficient markers available
    - Joint midpoints are computed from medial/lateral markers when both available
    - Fallback calculations used when only one marker present
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Load from TDF file
    >>> body = laban.WholeBody.from_tdf(
    ...     "mocap.tdf",
    ...     left_psis="LPSI",
    ...     right_psis="RPSI",
    ...     left_asis="LASI",
    ...     right_asis="RASI",
    ...     left_knee_medial="LKNEM",
    ...     left_knee_lateral="LKNEL",
    ...     left_ankle_medial="LANKM",
    ...     left_ankle_lateral="LANKL",
    ...     left_heel="LHEE",
    ...     left_toe="LTOE"
    ... )
    >>> 
    >>> # Access joint angle
    >>> knee_angle = body.left_knee_flexionextension
    >>> print(f"Knee flexion range: {knee_angle.data.min():.1f}° to {knee_angle.data.max():.1f}°")
    Knee flexion range: 2.3° to 87.5°
    >>> 
    >>> # Access marker position
    >>> c7 = body.c7_vertebra
    >>> print(f"C7 position shape: {c7.data.shape}")
    C7 position shape: (10000, 3)
    """
```

#### Class Methods

##### from_tdf()

Load WholeBody from BTS TDF file.

```python
@classmethod
def from_tdf(
    cls,
    file_path: str,
    left_psis: str,
    right_psis: str,
    left_asis: str,
    right_asis: str,
    **marker_labels
) -> WholeBody
```

**Parameters:**
- `file_path` (str): Path to TDF file
- `left_psis` (str): Label for left PSIS marker in TDF
- `right_psis` (str): Label for right PSIS marker in TDF
- `left_asis` (str): Label for left ASIS marker in TDF
- `right_asis` (str): Label for right ASIS marker in TDF
- `**marker_labels`: Optional marker labels (e.g., `left_knee_medial="LKNEM"`)

**Returns:**
- `WholeBody`: Instance with loaded marker data

**Example:**
```python
body = laban.WholeBody.from_tdf(
    "gait.tdf",
    left_psis="LPSI",
    right_psis="RPSI",
    left_asis="LASI",
    right_asis="RASI",
    left_knee_medial="LKNEM",
    left_knee_lateral="LKNEL"
)
```

#### Properties: 42 Anatomical Markers

All marker properties return `Point3D` objects with shape `(n_samples, 3)` representing (x, y, z) coordinates.

##### Cranial Markers (8)

| Property | Type | Description |
|----------|------|-------------|
| `left_psis` | Point3D | Left posterior superior iliac spine |
| `right_psis` | Point3D | Right posterior superior iliac spine |
| `left_asis` | Point3D | Left anterior superior iliac spine |
| `right_asis` | Point3D | Right anterior superior iliac spine |
| `sacrum` | Point3D | Sacrum (computed midpoint of PSIS markers) |
| `left_acromion` | Point3D | Left acromion |
| `right_acromion` | Point3D | Right acromion |
| `c7_vertebra` | Point3D | C7 vertebra |

##### Left Leg Markers (10)

| Property | Type | Description |
|----------|------|-------------|
| `left_thigh` | Point3D | Left thigh lateral marker |
| `left_knee_medial` | Point3D | Left knee medial epicondyle |
| `left_knee_lateral` | Point3D | Left knee lateral epicondyle |
| `left_knee` | Point3D | Left knee joint center (computed midpoint) |
| `left_shank` | Point3D | Left shank lateral marker |
| `left_ankle_medial` | Point3D | Left ankle medial malleolus |
| `left_ankle_lateral` | Point3D | Left ankle lateral malleolus |
| `left_ankle` | Point3D | Left ankle joint center (computed midpoint) |
| `left_heel` | Point3D | Left heel marker |
| `left_toe` | Point3D | Left toe marker |

##### Right Leg Markers (10)

Mirror of left leg markers with `right_` prefix.

##### Left Arm Markers (7)

| Property | Type | Description |
|----------|------|-------------|
| `left_elbow_medial` | Point3D | Left elbow medial epicondyle |
| `left_elbow_lateral` | Point3D | Left elbow lateral epicondyle |
| `left_elbow` | Point3D | Left elbow joint center (computed midpoint) |
| `left_wrist_medial` | Point3D | Left wrist medial marker |
| `left_wrist_lateral` | Point3D | Left wrist lateral marker |
| `left_wrist` | Point3D | Left wrist joint center (computed midpoint) |
| `left_hand` | Point3D | Left hand marker |

##### Right Arm Markers (7)

Mirror of left arm markers with `right_` prefix.

#### Properties: 38 Joint Angles

All angle properties return `Signal1D` objects with angles in degrees. Calculated lazily on first access.

**Important - Semantic Axis Convention:**
All angle calculations use semantic axis names (lateral_axis, vertical_axis, anteroposterior_axis) and their corresponding column indices in rotation matrices:
- Column 0 = lateral_axis (mediolateral direction)
- Column 1 = vertical_axis (superior-inferior direction)
- Column 2 = anteroposterior_axis (anterior-posterior direction)

This approach is **coordinate-system independent** - the code works regardless of whether your lab uses X=vertical, Y=vertical, or any other global axis convention. Never reference global axis letters (X/Y/Z) in calculations; always use semantic column indices.

##### Ankle Angles (4)

| Property | Type | Description | Positive | Negative | Neutral |
|----------|------|-------------|----------|----------|---------|
| `left_ankle_flexionextension` | Signal1D | Left ankle sagittal plane | Dorsiflexion (toe up) | Plantarflexion (toe down) | 0° (foot ⊥ shin at 90°) |
| `left_ankle_inversioneversion` | Signal1D | Left ankle frontal plane | Eversion | Inversion | 0° |
| `right_ankle_flexionextension` | Signal1D | Right ankle sagittal plane | Dorsiflexion (toe up) | Plantarflexion (toe down) | 0° (foot ⊥ shin at 90°) |
| `right_ankle_inversioneversion` | Signal1D | Right ankle frontal plane | Eversion | Inversion | 0° |

**Reference frames:**
- Origin: Ankle center (midpoint of lateral and medial ankle markers)
- lateral_axis: LEFT/RIGHT (ankle_lateral → ankle_medial)
- vertical_axis: UP (ankle → knee)
- anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

**Calculation:** Foot plane projected onto sagittal plane (anteroposterior_axis and vertical_axis). Uses semantic column indices: Column 0 = lateral_axis, Column 1 = vertical_axis, Column 2 = anteroposterior_axis.

##### Knee Angles (4)

| Property | Type | Description | Positive | Negative | Neutral |
|----------|------|-------------|----------|----------|---------|
| `left_knee_flexionextension` | Signal1D | Left knee sagittal plane | Flexion | Extension | 0° |
| `left_knee_varusvalgus` | Signal1D | Left knee frontal plane alignment | Varus (bow-leg, ginocchio varo) | Valgus (knock-knee, ginocchio valgo) | 0° (perfect alignment) |
| `right_knee_flexionextension` | Signal1D | Right knee sagittal plane | Flexion | Extension | 0° |
| `right_knee_varusvalgus` | Signal1D | Right knee frontal plane alignment | Varus (bow-leg, ginocchio varo) | Valgus (knock-knee, ginocchio valgo) | 0° (perfect alignment) |

**Reference frames:**
- Origin: Knee center (midpoint of lateral and medial knee markers)
- lateral_axis: LEFT/RIGHT (knee_lateral → knee_medial)
- vertical_axis: UP (knee → hip)
- anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

**Calculation:** Ankle position transformed to knee reference frame and projected onto frontal plane (lateral_axis and vertical_axis). Uses semantic rotation matrix indices:
- rotation_matrix[:, :, 0] = lateral_axis component (Column 0)
- rotation_matrix[:, :, 1] = vertical_axis component (Column 1)

Positive lateral deviation (ankle lateral to knee) indicates varus. Angles automatically normalized to [-180°, +180°] range to prevent wrapping issues.

##### Hip Angles (6)

| Property | Type | Description | Positive | Negative | Neutral |
|----------|------|-------------|----------|----------|---------|
| `left_hip_flexionextension` | Signal1D | Left hip sagittal plane | Flexion | Extension | 0° |
| `left_hip_abductionadduction` | Signal1D | Left hip frontal plane | Abduction | Adduction | 0° |
| `left_hip_internalexternalrotation` | Signal1D | Left hip transverse plane | Internal rotation | External rotation | 0° |
| `right_hip_flexionextension` | Signal1D | Right hip sagittal plane | Flexion | Extension | 0° |
| `right_hip_abductionadduction` | Signal1D | Right hip frontal plane | Abduction | Adduction | 0° |
| `right_hip_internalexternalrotation` | Signal1D | Right hip transverse plane | Internal rotation | External rotation | 0° |

**Reference frames:** Uses pelvis reference frame translated to hip joint center (from De Leva 1996 regression).

**Example:**
```python
hip_flex = body.left_hip_flexionextension
print(f"Hip flexion: mean={hip_flex.data.mean():.1f}°, max={hip_flex.data.max():.1f}°")
```

##### Pelvis Angles (3)

| Property | Type | Description | Positive | Negative | Neutral |
|----------|------|-------------|----------|----------|---------|
| `pelvis_anteroposteriortilt_global` | Signal1D | Pelvis sagittal tilt | Posterior tilt | Anterior tilt | 0° |
| `pelvis_lateraltilt_global` | Signal1D | Pelvis frontal tilt | Right tilt (right hip drop) | Left tilt (left hip drop) | 0° (level) |
| `pelvis_rotation_global` | Signal1D | Pelvis transverse rotation | Right rotation | Left rotation | 0° (aligned) |

**Reference frames:**
- Origin: Pelvis center (computed from ASIS and PSIS markers)
- lateral_axis: LEFT (right midpoint → left midpoint)
- vertical_axis: UP (pelvis_center → neck_base)
- anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

**Note:** All pelvis angles return 0° when the pelvis is in neutral position (level and aligned with global axes). Angles measured relative to global coordinate system.

##### Trunk Angles (4)

| Property | Type | Description | Positive | Negative | Neutral |
|----------|------|-------------|----------|----------|---------|
| `trunk_flexionextension_global` | Signal1D | Trunk sagittal flexion | Flexion (forward) | Extension (backward) | 0° |
| `trunk_lateralflexion_global` | Signal1D | Trunk frontal flexion | Left tilt | Right tilt | 0° |
| `trunk_rotation_global` | Signal1D | Trunk transverse rotation (global) | Right rotation | Left rotation | 0° |
| `trunk_rotation_local` | Signal1D | Trunk transverse rotation (relative to pelvis) | Right rotation | Left rotation | 0° |

**Reference frames:** Trunk angles use shoulder and neck markers to define trunk orientation relative to global axes or pelvis frame.

##### Shoulder Girdle Angles (4)

| Property | Type | Description | Positive | Negative | Neutral |
|----------|------|-------------|----------|----------|---------|
| `shoulder_lateraltilt_global` | Signal1D | Shoulder frontal tilt (global) | Right tilt | Left tilt | 0° (level) |
| `shoulder_lateraltilt_local` | Signal1D | Shoulder frontal tilt (relative to trunk) | Right tilt | Left tilt | 0° (level) |
| `left_scapular_protractionretraction` | Signal1D | Left scapular transverse position | Protraction (forward) | Retraction (backward) | 0° |
| `right_scapular_protractionretraction` | Signal1D | Right scapular transverse position | Protraction (forward) | Retraction (backward) | 0° |

**Reference frames:** Uses acromion and C7 markers to define shoulder orientation.

##### Shoulder Joint Angles (6)

| Property | Type | Description | Positive | Negative | Neutral |
|----------|------|-------------|----------|----------|---------|
| `left_shoulder_flexionextension` | Signal1D | Left shoulder sagittal plane | Flexion (forward) | Extension (backward) | 0° |
| `left_shoulder_abductionadduction` | Signal1D | Left shoulder frontal plane | Abduction (away) | Adduction (toward body) | 0° |
| `left_shoulder_internalexternalrotation` | Signal1D | Left shoulder transverse plane | Internal rotation | External rotation | 0° |
| `right_shoulder_flexionextension` | Signal1D | Right shoulder sagittal plane | Flexion (forward) | Extension (backward) | 0° |
| `right_shoulder_abductionadduction` | Signal1D | Right shoulder frontal plane | Abduction (away) | Adduction (toward body) | 0° |
| `right_shoulder_internalexternalrotation` | Signal1D | Right shoulder transverse plane | Internal rotation | External rotation | 0° |

**Reference frames:**
- Origin: Shoulder joint center (from anterior/posterior markers or De Leva regression)
- lateral_axis: LEFT/RIGHT (neck_base → shoulder, points outward)
- vertical_axis: UP (pelvis_center → neck_base)
- anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

##### Elbow Angles (2)

| Property | Type | Description | Positive | Negative | Neutral |
|----------|------|-------------|----------|----------|---------|
| `left_elbow_flexionextension` | Signal1D | Left elbow sagittal plane | Flexion | Extension | 0° |
| `right_elbow_flexionextension` | Signal1D | Right elbow sagittal plane | Flexion | Extension | 0° |

**Reference frames:**
- Origin: Elbow center (midpoint of lateral and medial elbow markers)
- lateral_axis: LEFT/RIGHT (elbow_lateral → elbow_medial)
- vertical_axis: UP (elbow → shoulder)
- anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

##### Neck and Head Angles (3)

| Property | Type | Description | Positive | Negative | Neutral |
|----------|------|-------------|----------|----------|---------|
| `neck_lateral_tilt` | Signal1D | Neck frontal tilt | Right tilt | Left tilt | 0° (centered) |
| `neck_flexionextension_local` | Signal1D | Neck flexion (relative to trunk) | Flexion (forward) | Extension (backward) | 0° |
| `neck_flexionextension_global` | Signal1D | Neck flexion (global) | Forward | Backward | 0° |

**Reference frames:**
- Origin: Neck base (C7 marker or computed from shoulders)
- vertical_axis: UP (pelvis_center → neck_base)
- anteroposterior_axis: FORWARD (head_anterior → head_posterior or derived)
- lateral_axis: LEFT (cross product, Gram-Schmidt)

##### Spine Curvature Angles (2)

| Property | Type | Description | Normal Range | Calculation |
|----------|------|-------------|--------------|-------------|
| `lumbar_lordosis` | Signal1D | Lumbar spine curvature angle at L2 | 140-160° | T5 → L2 → PSIS_mid |
| `dorsal_kyphosis` | Signal1D | Thoracic spine curvature angle at T5 | 140-160° | C7 → T5 → L2 |

**Interpretation:**
- These are internal angles at vertebral vertices
- Smaller angles (<140°) indicate greater curvature (hyperlordosis/hyperkyphosis)
- Larger angles (>160°) indicate flatter spine (hypolordosis/hypokyphosis)
- Normal range: 140-160° represents healthy spinal curvature

**Note:** These angles are measured using 3-point geometry (angle at middle vertebra) rather than reference frame transformations.

#### Properties: 10 Computed Properties

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
| **`segment_lengths`** | **Timeseries** | **All segment lengths and widths combined (NEW)** |
| **`joint_angles`** | **Timeseries** | **All joint angles combined (NEW)** |

##### segment_lengths

**NEW in version 207**: Aggregate property that combines all segment dimensions into a single Timeseries.

Returns a Timeseries containing:
- **Foot dimensions**: heights, lengths, widths (left/right)
- **Joint widths**: ankle, knee, elbow (left/right)  
- **Limb lengths**: leg, thigh, arm, forearm, lower/upper limb totals (left/right)
- **Body dimensions**: shoulder_width, hip_width, trunk_length, pelvis_height

**Example:**
```python
lengths = body.segment_lengths
print(f"Available: {lengths.columns}")
# ['left_foot_height', 'right_foot_length', 'left_thigh_length', 
#  'shoulder_width', 'hip_width', 'trunk_length', ...]

# Access specific dimension
left_thigh = lengths['left_thigh_length']

# Export to pandas
df = lengths.to_dataframe()
```

##### joint_angles

**NEW in version 207**: Aggregate property that combines all joint angles into a single Timeseries.

Returns a Timeseries containing all 28+ computed angles:
- **Lower limb**: ankle flexion/extension & inversion/eversion, knee flexion/extension & varus/valgus, hip flexion/extension & abduction/adduction & rotation (left/right)
- **Pelvis & trunk**: pelvis anteroposterior tilt, trunk rotation
- **Upper limb**: shoulder flexion/extension & abduction/adduction & rotation & elevation/depression & lateral tilt, scapular protraction/retraction, elbow flexion/extension (left/right)
- **Neck & spine**: neck flexion/extension & lateral tilt, lumbar lordosis, dorsal kyphosis

**Example:**
```python
angles = body.joint_angles
print(f"Available: {len(angles.columns)} angles")

# Access specific angle
knee_flex = angles['left_knee_flexionextension']

# Export all to pandas
df = angles.to_dataframe()
df.to_csv('joint_angles.csv')
```

**Note**: Only includes angles computable from available markers. Properties gracefully skip missing markers.

#### Methods

##### to_dataframe()

Convert all angles to pandas DataFrame.

```python
def to_dataframe(self, angles_only: bool = True) -> pd.DataFrame
```

**Parameters:**
- `angles_only` (bool): If True, include only joint angles. If False, include markers too.

**Returns:**
- `pd.DataFrame`: DataFrame with time index and angle/marker columns

**Example:**
```python
df = body.to_dataframe(angles_only=True)
df.to_excel("joint_angles.xlsx", index=False)
```

## Usage Examples

### Complete Gait Analysis

```python
import labanalysis as laban
import numpy as np

# Load full body model
body = laban.WholeBody.from_tdf(
    "gait.tdf",
    left_psis="LPSI", right_psis="RPSI",
    left_asis="LASI", right_asis="RASI",
    left_knee_medial="LKNEM", left_knee_lateral="LKNEL",
    left_ankle_medial="LANKM", left_ankle_lateral="LANKL",
    left_heel="LHEE", left_toe="LTOE",
    right_knee_medial="RKNEM", right_knee_lateral="RKNEL",
    right_ankle_medial="RANKM", right_ankle_lateral="RANKL",
    right_heel="RHEE", right_toe="RTOE"
)

# Get sagittal plane angles
hip = body.left_hip_flexionextension
knee = body.left_knee_flexionextension
ankle = body.left_ankle_flexionextension

# Calculate range of motion
rom = {
    'hip': hip.data.max() - hip.data.min(),
    'knee': knee.data.max() - knee.data.min(),
    'ankle': ankle.data.max() - ankle.data.min()
}

print("Range of Motion:")
for joint, value in rom.items():
    print(f"  {joint.capitalize()}: {value:.1f}°")
```

### Bilateral Asymmetry Analysis

```python
# Compare left and right legs
left_knee = body.left_knee_flexionextension.data
right_knee = body.right_knee_flexionextension.data

# Calculate symmetry index
symmetry_index = (left_knee - right_knee) / ((left_knee + right_knee) / 2) * 100

print(f"Mean bilateral asymmetry: {np.abs(symmetry_index).mean():.1f}%")
print(f"Max bilateral asymmetry: {np.abs(symmetry_index).max():.1f}%")
```

### Export to OpenSim Format

```python
# Export marker trajectories for OpenSim
markers_dict = {
    'LPSI': body.left_psis,
    'RPSI': body.right_psis,
    'LASI': body.left_asis,
    'RASI': body.right_asis,
    'LKNE': body.left_knee,
    'RKNE': body.right_knee,
    'LANK': body.left_ankle,
    'RANK': body.right_ankle
}

laban.write_opensim(markers_dict, "output.mot")
```

## See Also

- **[User Guide: WholeBody Model](../../user-guide/biomechanics/whole-body-model.md)** - Complete usage guide
- **[Tutorial: Full Body Kinematics](../../tutorials/03-full-body-kinematics.md)** - Step-by-step tutorial
- **[Point3D](timeseries.md#point3d)** - 3D marker data structure
- **[Signal1D](timeseries.md#signal1d)** - 1D signal data structure

---

**Reference**: Winter DA (2009). Biomechanics and Motor Control of Human Movement. 4th ed.
