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

##### Ankle Angles (4)

| Property | Type | Description | Positive | Negative |
|----------|------|-------------|----------|----------|
| `left_ankle_flexionextension` | Signal1D | Left ankle sagittal plane | Dorsiflexion | Plantarflexion |
| `left_ankle_inversioneversion` | Signal1D | Left ankle frontal plane | Eversion | Inversion |
| `right_ankle_flexionextension` | Signal1D | Right ankle sagittal plane | Dorsiflexion | Plantarflexion |
| `right_ankle_inversioneversion` | Signal1D | Right ankle frontal plane | Eversion | Inversion |

##### Knee Angles (4)

| Property | Type | Description | Positive | Negative |
|----------|------|-------------|----------|----------|
| `left_knee_flexionextension` | Signal1D | Left knee sagittal plane | Flexion | Extension |
| `left_knee_varusvalgus` | Signal1D | Left knee frontal plane alignment | Varus (bow-leg) | Valgus (knock-knee) |
| `right_knee_flexionextension` | Signal1D | Right knee sagittal plane | Flexion | Extension |
| `right_knee_varusvalgus` | Signal1D | Right knee frontal plane alignment | Varus (bow-leg) | Valgus (knock-knee) |

**Note**: Sign convention - Positive = Varus, Negative = Valgus, 0° = Perfect alignment. Angles are automatically normalized to [-180°, +180°] range to prevent wrapping issues.

##### Hip Angles (6)

| Property | Type | Description | Positive | Negative |
|----------|------|-------------|----------|----------|
| `left_hip_flexionextension` | Signal1D | Left hip sagittal plane | Flexion | Extension |
| `left_hip_abductionadduction` | Signal1D | Left hip frontal plane | Abduction | Adduction |
| `left_hip_internalexternalrotation` | Signal1D | Left hip transverse plane | Internal | External |
| `right_hip_flexionextension` | Signal1D | Right hip sagittal plane | Flexion | Extension |
| `right_hip_abductionadduction` | Signal1D | Right hip frontal plane | Abduction | Adduction |
| `right_hip_internalexternalrotation` | Signal1D | Right hip transverse plane | Internal | External |

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

**Note**: All pelvis angles return 0° when the pelvis is in neutral position (level and aligned with global axes)

##### Trunk Angles (4)

| Property | Type | Description | Positive | Negative |
|----------|------|-------------|----------|----------|
| `trunk_flexionextension_global` | Signal1D | Trunk sagittal flexion | Flexion | Extension |
| `trunk_lateralflexion_global` | Signal1D | Trunk frontal flexion | Left tilt | Right tilt |
| `trunk_rotation_global` | Signal1D | Trunk transverse rotation (global) | Right rotation | Left rotation |
| `trunk_rotation_local` | Signal1D | Trunk transverse rotation (local to pelvis) | Right rotation | Left rotation |

##### Shoulder Girdle Angles (4)

| Property | Type | Description | Positive | Negative |
|----------|------|-------------|----------|----------|
| `shoulder_lateraltilt_global` | Signal1D | Shoulder frontal tilt (global) | Right tilt | Left tilt |
| `shoulder_lateraltilt_local` | Signal1D | Shoulder frontal tilt (relative to trunk) | Right tilt | Left tilt |
| `left_scapular_protractionretraction` | Signal1D | Left scapular transverse position | Protraction | Retraction |
| `right_scapular_protractionretraction` | Signal1D | Right scapular transverse position | Protraction | Retraction |

##### Shoulder Joint Angles (6)

| Property | Type | Description | Positive | Negative |
|----------|------|-------------|----------|----------|
| `left_shoulder_flexionextension` | Signal1D | Left shoulder sagittal plane | Flexion | Extension |
| `left_shoulder_abductionadduction` | Signal1D | Left shoulder frontal plane | Abduction | Adduction |
| `left_shoulder_internalexternalrotation` | Signal1D | Left shoulder transverse plane | Internal | External |
| `right_shoulder_flexionextension` | Signal1D | Right shoulder sagittal plane | Flexion | Extension |
| `right_shoulder_abductionadduction` | Signal1D | Right shoulder frontal plane | Abduction | Adduction |
| `right_shoulder_internalexternalrotation` | Signal1D | Right shoulder transverse plane | Internal | External |

##### Elbow Angles (2)

| Property | Type | Description | Positive | Negative |
|----------|------|-------------|----------|----------|
| `left_elbow_flexionextension` | Signal1D | Left elbow sagittal plane | Flexion | Extension |
| `right_elbow_flexionextension` | Signal1D | Right elbow sagittal plane | Flexion | Extension |

##### Neck and Head Angles (3)

| Property | Type | Description | Positive | Negative |
|----------|------|-------------|----------|----------|
| `neck_lateral_tilt` | Signal1D | Neck frontal tilt | Right tilt | Left tilt |
| `neck_flexionextension_local` | Signal1D | Neck flexion (relative to trunk) | Flexion | Extension |
| `neck_flexionextension_global` | Signal1D | Neck flexion (global) | Forward | Backward |

##### Spine Curvature Angles (2)

| Property | Type | Description | Normal Range | Calculation |
|----------|------|-------------|--------------|-------------|
| `lumbar_lordosis` | Signal1D | Lumbar spine curvature angle at L2 | 140-160° | T5 → L2 → PSIS_mid |
| `dorsal_kyphosis` | Signal1D | Thoracic spine curvature angle at T5 | 140-160° | C7 → T5 → L2 |

**Note**: These are internal angles at vertebral vertices. Smaller angles (<140°) indicate greater curvature (hyperlordosis/hyperkyphosis), larger angles (>160°) indicate flatter spine (hypolordosis/hypokyphosis)

#### Properties: 8 Computed Properties

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
