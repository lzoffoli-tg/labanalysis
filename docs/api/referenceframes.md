# Reference Frames Module

The `referenceframes` module provides 3D coordinate system transformations for biomechanical analysis.

## Overview

ReferenceFrame represents an oriented 3D coordinate system defined by an origin point and three orthogonal unit vectors. It enables coordinate transformations between different anatomical and laboratory reference frames.

---

## ReferenceFrame

3D coordinate system with origin and orientation.

**Module:** `labanalysis.referenceframes`

**Description:**  
A ReferenceFrame consists of an origin (Point3D) and three orthonormal vectors (Signal3D) defining the X, Y, and Z axes. It supports transformation of points and vectors between reference frames.

**Parameters:**
- `origin` (Point3D, required): Origin point of the reference frame
- `x` (Signal3D, required): X-axis unit vector (normalized)
- `y` (Signal3D, required): Y-axis unit vector (normalized, orthogonal to X)
- `z` (Signal3D, required): Z-axis unit vector (normalized, orthogonal to X and Y)

**Properties:**
- `origin` (Point3D): Reference frame origin
- `x` (Signal3D): X-axis unit vector
- `y` (Signal3D): Y-axis unit vector  
- `z` (Signal3D): Z-axis unit vector
- `rotation_matrix` (numpy.ndarray): 3x3 rotation matrix at each time point
- `is_orthonormal` (bool): True if axes form orthonormal basis

**Methods:**
- `transform_point(point, target_frame)`: Transform point to another reference frame
- `transform_vector(vector, target_frame)`: Transform vector to another reference frame
- `copy()`: Create deep copy

**Example:**
```python
import labanalysis as laban
import numpy as np

# Define time vector
time = np.linspace(0, 1, 100)

# Create lab reference frame (global)
origin_lab = laban.Point3D(
    data=np.zeros((100, 3)),  # Origin at [0, 0, 0]
    time=time,
    unit='mm'
)

x_lab = laban.Signal3D(
    data=np.tile([1, 0, 0], (100, 1)),  # X = [1, 0, 0]
    time=time,
    unit=''
)

y_lab = laban.Signal3D(
    data=np.tile([0, 1, 0], (100, 1)),  # Y = [0, 1, 0]
    time=time,
    unit=''
)

z_lab = laban.Signal3D(
    data=np.tile([0, 0, 1], (100, 1)),  # Z = [0, 0, 1]
    time=time,
    unit=''
)

lab_frame = laban.ReferenceFrame(origin=origin_lab, x=x_lab, y=y_lab, z=z_lab)

# Verify orthonormality
assert lab_frame.is_orthonormal
```

---

## Creating Anatomical Reference Frames

### Pelvis Reference Frame

```python
# Pelvis frame from ASIS and PSIS markers
def create_pelvis_frame(left_asis, right_asis, left_psis, right_psis):
    """
    Create pelvis reference frame.
    
    X-axis: Points anteriorly (perpendicular to frontal plane)
    Y-axis: Points to the right
    Z-axis: Points superiorly
    """
    import numpy as np
    
    # Origin at mid-ASIS
    origin = (left_asis + right_asis) / 2
    
    # Y-axis: from left to right ASIS
    y_axis = (right_asis - left_asis).normalize()
    
    # Temporary anterior vector (mid-ASIS to mid-PSIS)
    mid_psis = (left_psis + right_psis) / 2
    anterior_temp = origin - mid_psis
    
    # Z-axis: perpendicular to plane formed by ASIS and PSIS markers
    z_axis = y_axis.cross(anterior_temp).normalize()
    
    # X-axis: perpendicular to Y and Z (points anteriorly)
    x_axis = y_axis.cross(z_axis).normalize()
    
    return laban.ReferenceFrame(origin=origin, x=x_axis, y=y_axis, z=z_axis)

# Use with WholeBody
body = laban.WholeBody.from_tdf("trial.tdf", bodymass_kg=75)
pelvis_frame = create_pelvis_frame(
    body.left_asis,
    body.right_asis,
    body.left_psis,
    body.right_psis
)
```

### Foot Reference Frame

```python
def create_foot_frame(heel, toe, first_met, fifth_met):
    """
    Create foot reference frame.
    
    X-axis: Points anteriorly (heel to toe direction)
    Y-axis: Points medially (5th to 1st metatarsal)
    Z-axis: Points superiorly (perpendicular to foot plane)
    """
    # Origin at heel
    origin = heel
    
    # X-axis: from heel to midpoint of metatarsals
    mid_met = (first_met + fifth_met) / 2
    x_axis = (mid_met - heel).normalize()
    
    # Y-axis: from 5th to 1st metatarsal (medial direction)
    y_axis_temp = (first_met - fifth_met).normalize()
    
    # Z-axis: perpendicular to foot plane
    z_axis = x_axis.cross(y_axis_temp).normalize()
    
    # Recalculate Y to ensure orthogonality
    y_axis = z_axis.cross(x_axis).normalize()
    
    return laban.ReferenceFrame(origin=origin, x=x_axis, y=y_axis, z=z_axis)
```

---

## Coordinate Transformations

### Transform Points Between Frames

```python
# Create two reference frames
frame_a = create_pelvis_frame(...)
frame_b = create_foot_frame(...)

# Point defined in frame A
point_in_a = laban.Point3D(data=..., time=..., unit='mm')

# Transform point from frame A to frame B
point_in_b = frame_a.transform_point(point_in_a, target_frame=frame_b)

# Now point_in_b expresses the same physical location in frame B coordinates
```

### Transform Vectors Between Frames

```python
# Velocity vector in lab frame
velocity_lab = laban.Signal3D(data=..., time=..., unit='m/s')

# Transform velocity to pelvis frame
velocity_pelvis = lab_frame.transform_vector(velocity_lab, target_frame=pelvis_frame)

# Angular velocity is now expressed relative to pelvis orientation
```

---

## Common Use Cases

### Joint Angle Calculations

Reference frames enable calculation of relative orientations:

```python
# Thigh and shank reference frames
thigh_frame = create_segment_frame(hip_center, knee_center, ...)
shank_frame = create_segment_frame(knee_center, ankle_center, ...)

# Knee flexion angle = rotation of shank relative to thigh
# (Implemented internally in WholeBody.left_knee_flexionextension)
```

### Center of Mass in Pelvis Frame

```python
# Global COM position
com_global = body.center_of_mass

# Transform to pelvis frame for pelvis-relative COM
pelvis_frame = create_pelvis_frame(...)
com_in_pelvis = lab_frame.transform_point(com_global, target_frame=pelvis_frame)

# Now com_in_pelvis shows COM position relative to pelvis orientation
```

### Force Vector Decomposition

```python
# Ground reaction force in lab frame
grf_lab = force_platform.force

# Transform to foot frame for anterior-posterior vs medial-lateral components
foot_frame = create_foot_frame(...)
grf_foot = lab_frame.transform_vector(grf_lab, target_frame=foot_frame)

# grf_foot components now represent:
# X: anterior-posterior force
# Y: medial-lateral force  
# Z: vertical force (perpendicular to foot)
```

---

## Technical Details

### Rotation Matrix

The rotation matrix transforms vectors from the reference frame to the global (lab) frame:

```python
R = [x.to_numpy(), y.to_numpy(), z.to_numpy()]  # 3x3 matrix

# At each time point t:
# point_lab = origin + R @ point_local
# point_local = R.T @ (point_lab - origin)
```

### Orthonormality Verification

Reference frames should maintain orthonormality:

```python
# Check orthonormality
frame = laban.ReferenceFrame(origin, x, y, z)

if not frame.is_orthonormal:
    print("Warning: Reference frame is not orthonormal")
    print("This may indicate:")
    print("- Marker tracking errors")
    print("- Improper frame construction")
    print("- Need for axis re-orthogonalization")
```

### Gram-Schmidt Orthogonalization

If constructing frames from measured markers, apply Gram-Schmidt:

```python
def orthogonalize_frame(x_approx, y_approx, z_approx):
    """
    Orthogonalize approximate axes using Gram-Schmidt process.
    
    Keeps X-axis as primary, adjusts Y and Z to be orthogonal.
    """
    # Normalize X
    x = x_approx.normalize()
    
    # Y perpendicular to X
    y_temp = y_approx - x * (x.dot(y_approx))
    y = y_temp.normalize()
    
    # Z perpendicular to both
    z_temp = z_approx - x * (x.dot(z_approx)) - y * (y.dot(z_approx))
    z = z_temp.normalize()
    
    return x, y, z
```

---

## ISB Recommendations

The International Society of Biomechanics (ISB) recommends standardized segment coordinate systems. Reference frames in labanalysis can implement ISB conventions:

**ISB Pelvis Frame:**
- Origin: Midpoint between left and right ASIS
- X: Anteriorly in sagittal plane
- Y: To the right in frontal plane
- Z: Superiorly

**ISB Foot Frame:**
- Origin: Midpoint between medial and lateral malleoli
- X: Anteriorly (parallel to 2nd metatarsal projection)
- Y: Medially
- Z: Superiorly (perpendicular to foot sole)

**ISB Thigh Frame:**
- Origin: Hip joint center
- X: Anteriorly
- Y: To the right
- Z: Superiorly (along femur long axis)

---

## See Also

- [WholeBody API](records/bodies.md) - Biomechanical model using reference frames
- [Signal3D API](records/timeseries.md) - 3D signal containers
- [Point3D API](records/timeseries.md) - 3D point trajectories
- [Reference Frames Guide](../guides/biomechanics/reference-frames.md) - Detailed usage guide

## References

1. Wu G, et al. (2002). ISB recommendation on definitions of joint coordinate system of various joints for the reporting of human joint motion—part I: ankle, hip, and spine. *Journal of Biomechanics*, 35(4), 543-548.

2. Wu G, et al. (2005). ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of human joint motion—Part II: shoulder, elbow, wrist and hand. *Journal of Biomechanics*, 38(5), 981-992.
