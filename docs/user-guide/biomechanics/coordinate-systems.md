# Coordinate Systems

Guide to working with coordinate systems and reference frame transformations in biomechanical analysis.

## Overview

Biomechanical data is typically collected in a **global (laboratory) reference frame** but often needs to be expressed in **local (anatomical) reference frames** for meaningful interpretation. labanalysis provides tools for:

- Defining custom coordinate systems using Gram-Schmidt orthonormalization
- Transforming 3D data between reference frames
- Working with anatomical reference frames (pelvis, trunk, limbs)

**Key Concepts:**
- **Global frame**: Laboratory/world coordinates (fixed in space)
- **Local frame**: Anatomical coordinates (moves with body segments)
- **Transformation**: Converting data from one frame to another
- **Orthonormalization**: Creating perpendicular unit vectors from non-orthogonal inputs

## Quick Reference

```python
import labanalysis as laban
import numpy as np

# Create pelvis coordinate system from markers
l_asis = body.left_asis   # Point3D
r_asis = body.right_asis
l_psis = body.left_psis
r_psis = body.right_psis

# Define pelvis axes
i = (r_asis - l_asis).to_numpy()  # Mediolateral (left to right)
j = ((l_asis + r_asis) / 2 - (l_psis + r_psis) / 2).to_numpy()  # Anteroposterior

# Create orthonormal reference frame
R_pelvis = laban.gram_schmidt(i, j, k=None)  # Returns (N, 3, 3)

# Transform marker to pelvis frame
marker_pelvis = marker.change_reference_frame(
    new_x=R_pelvis[:, :, 0],
    new_y=R_pelvis[:, :, 1],
    new_z=R_pelvis[:, :, 2],
    new_origin=pelvis_center
)
```

## Global vs Local Reference Frames

### Global (Laboratory) Frame

The global frame is fixed in space, typically defined by the motion capture system:

- **X-axis**: Often mediolateral (left → right)
- **Y-axis**: Often vertical (ground → ceiling)
- **Z-axis**: Often anteroposterior (back → front)

```python
# Markers in global frame
marker_global = body.left_knee  # Point3D in lab coordinates

# Global coordinates
x_global = marker_global['X']  # Mediolateral
y_global = marker_global['Y']  # Vertical
z_global = marker_global['Z']  # Anteroposterior
```

### Local (Anatomical) Frame

Local frames move with body segments, making angles and movements more interpretable:

**Example: Pelvis frame**
- **X-axis**: Mediolateral (right hip → left hip)
- **Y-axis**: Vertical (inferior → superior)
- **Z-axis**: Anteroposterior (posterior → anterior)

**Why use local frames?**
- Joint angles become independent of global orientation
- Bilateral comparisons are easier
- Movement patterns are clearer (e.g., knee flexion vs global vertical)

## Gram-Schmidt Orthonormalization

The `gram_schmidt()` function creates an orthonormal coordinate system (3 perpendicular unit vectors) from 2-3 non-orthogonal input vectors.

### Function Signature

```python
R = laban.gram_schmidt(
    i,  # First axis direction (N, 3)
    j,  # Second axis direction (N, 3)
    k=None  # Optional third axis (N, 3)
)
# Returns: rotation matrix R (N, 3, 3)
```

**Process:**
1. **i-axis**: Normalized from input `i`
2. **k-axis**: Perpendicular to `i` and `j` (cross product)
3. **j-axis**: Perpendicular to both `i` and `k` (cross product)

Result: Right-handed orthonormal frame with i, j, k as columns.

### Example: Create Pelvis Frame

```python
import labanalysis as laban

# Get pelvis markers
l_asis = body.left_asis
r_asis = body.right_asis
l_psis = body.left_psis
r_psis = body.right_psis

# Calculate pelvis center
pelvis_center = (l_asis + r_asis + l_psis + r_psis) / 4

# Define desired axes (not yet orthonormal)
i = (r_asis - l_asis).to_numpy()  # Mediolateral
j = ((l_asis + r_asis) / 2 - (l_psis + r_psis) / 2).to_numpy()  # AP

# Create orthonormal frame
R_pelvis = laban.gram_schmidt(i, j, k=None)

# R_pelvis shape: (N_samples, 3, 3)
# R_pelvis[:, :, 0] = i-axis (mediolateral)
# R_pelvis[:, :, 1] = j-axis (anteroposterior, orthonormalized)
# R_pelvis[:, :, 2] = k-axis (vertical, from i×j)
```

### Example: Create Thigh Frame

```python
# Define thigh axes
hip = body.left_hip
knee = body.left_knee

# Longitudinal axis (hip to knee)
k = (knee - hip).to_numpy()  # Distal direction

# Mediolateral axis (perpendicular to sagittal plane)
# Use pelvis orientation as reference
pelvis_ml = (body.right_hip - body.left_hip).to_numpy()
i = pelvis_ml  # Approximate mediolateral

# Create orthonormal thigh frame
R_thigh = laban.gram_schmidt(i, k=k, j=None)

# R_thigh[:, :, 2] = longitudinal (hip → knee)
# R_thigh[:, :, 0] = mediolateral (orthonormalized)
# R_thigh[:, :, 1] = anteroposterior (from k×i)
```

## Reference Frame Transformations

### Transform Point3D to New Frame

Use `change_reference_frame()` method to express 3D data in a different coordinate system:

```python
# Transform marker from global to pelvis frame
marker_global = body.left_knee  # Point3D in global frame

# Get pelvis frame (R_pelvis from Gram-Schmidt)
# Get pelvis origin
pelvis_center = (body.left_asis + body.right_asis + 
                 body.left_psis + body.right_psis) / 4

# Transform
marker_pelvis = marker_global.change_reference_frame(
    new_x=R_pelvis[:, :, 0],  # New x-axis direction
    new_y=R_pelvis[:, :, 1],  # New y-axis direction
    new_z=R_pelvis[:, :, 2],  # New z-axis direction
    new_origin=pelvis_center  # New origin point
)

# Now marker_pelvis is in pelvis coordinates
# marker_pelvis['X'] = mediolateral position relative to pelvis
# marker_pelvis['Y'] = anteroposterior position relative to pelvis
# marker_pelvis['Z'] = vertical position relative to pelvis
```

### Transform Signal3D

Same syntax works for `Signal3D` (forces, velocities, etc.):

```python
# Transform force from global to pelvis frame
force_global = fp.force  # Signal3D in global frame

force_pelvis = force_global.change_reference_frame(
    new_x=R_pelvis[:, :, 0],
    new_y=R_pelvis[:, :, 1],
    new_z=R_pelvis[:, :, 2],
    new_origin=pelvis_center  # Origin shift (optional for forces)
)

# Now force components are in pelvis axes
# force_pelvis['X'] = mediolateral force (pelvis frame)
# force_pelvis['Y'] = anteroposterior force (pelvis frame)
# force_pelvis['Z'] = vertical force (pelvis frame)
```

## Common Anatomical Frames

### Pelvis Frame (ISB Convention)

```python
def create_pelvis_frame(body):
    """Create pelvis reference frame following ISB recommendations."""
    # Project markers to pelvis plane (for cleaner axes)
    l_asis, r_asis, l_psis, r_psis = body._get_projected_pelvis_points()
    
    # Center
    center = (l_asis + r_asis + l_psis + r_psis) / 4
    
    # i-axis: right ASIS → left ASIS (mediolateral)
    i = ((l_asis + l_psis) / 2 - center).to_numpy()
    
    # k-axis: mid-ASIS → mid-PSIS (approximate vertical)
    k = ((l_asis + r_asis) / 2 - center).to_numpy()
    
    # Orthonormalize
    R = laban.gram_schmidt(i, k=k)
    
    return center, R

pelvis_origin, R_pelvis = create_pelvis_frame(body)
```

### Thigh Frame

```python
def create_thigh_frame(body, side='left'):
    """Create thigh reference frame."""
    if side == 'left':
        hip = body.left_hip
        knee = body.left_knee
    else:
        hip = body.right_hip
        knee = body.right_knee
    
    # Longitudinal axis (proximal → distal)
    k = (knee - hip).to_numpy()
    
    # Mediolateral axis (from pelvis)
    i = (body.left_hip - body.right_hip).to_numpy()
    
    # Orthonormalize
    R = laban.gram_schmidt(i, k=k)
    
    return hip, R

hip_origin, R_thigh = create_thigh_frame(body, side='left')
```

### Trunk Frame

```python
def create_trunk_frame(body):
    """Create trunk reference frame."""
    # Get spine landmarks
    c7 = body.c7
    l_asis, r_asis, l_psis, r_psis = body._get_projected_pelvis_points()
    base = (l_psis + r_psis) / 2
    
    # Vertical axis (pelvis → C7)
    k = (c7 - base).to_numpy()
    
    # Mediolateral axis (from pelvis)
    i = (body.left_shoulder - body.right_shoulder).to_numpy()
    
    # Orthonormalize
    R = laban.gram_schmidt(i, k=k)
    
    # Origin at mid-trunk
    origin = (c7 + base) / 2
    
    return origin, R

trunk_origin, R_trunk = create_trunk_frame(body)
```

## Practical Applications

### Express Knee Position in Thigh Frame

```python
# Create thigh frame
hip, R_thigh = create_thigh_frame(body, side='left')

# Get knee marker
knee_global = body.left_knee

# Transform to thigh frame
knee_thigh = knee_global.change_reference_frame(
    new_x=R_thigh[:, :, 0],
    new_y=R_thigh[:, :, 1],
    new_z=R_thigh[:, :, 2],
    new_origin=hip
)

# knee_thigh['Z'] is now distance along femur (thigh length)
thigh_length = knee_thigh['Z'].data.mean()
print(f"Thigh length: {thigh_length:.1f} mm")

# Lateral deviation (knee varus/valgus indicator)
lateral_dev = knee_thigh['X'].data
print(f"Knee lateral deviation: {lateral_dev.mean():.1f} ± {lateral_dev.std():.1f} mm")
```

### Calculate Joint Angle in Local Frame

```python
# Get knee flexion in thigh coordinate system
ankle_global = body.left_ankle
ankle_thigh = ankle_global.change_reference_frame(
    new_x=R_thigh[:, :, 0],
    new_y=R_thigh[:, :, 1],
    new_z=R_thigh[:, :, 2],
    new_origin=hip
)

# Ankle position relative to thigh axes
# ankle_thigh['Y'] = anteroposterior (flexion indicator)
# ankle_thigh['Z'] = longitudinal (extension indicator)

# Calculate flexion angle manually
import numpy as np
knee_flex_rad = np.arctan2(ankle_thigh['Y'].data, ankle_thigh['Z'].data)
knee_flex_deg = np.degrees(knee_flex_rad)

# Compare to automatic calculation
knee_flex_auto = body.left_knee_flexionextension
print(f"Manual flexion: {knee_flex_deg.mean():.1f}°")
print(f"Automatic flexion: {knee_flex_auto.data.mean():.1f}°")
```

### Bilateral Symmetry Analysis

```python
# Create reference frames for both limbs
hip_L, R_thigh_L = create_thigh_frame(body, side='left')
hip_R, R_thigh_R = create_thigh_frame(body, side='right')

# Transform knees to respective thigh frames
knee_L_local = body.left_knee.change_reference_frame(
    new_x=R_thigh_L[:, :, 0], new_y=R_thigh_L[:, :, 1], 
    new_z=R_thigh_L[:, :, 2], new_origin=hip_L
)

knee_R_local = body.right_knee.change_reference_frame(
    new_x=R_thigh_R[:, :, 0], new_y=R_thigh_R[:, :, 1],
    new_z=R_thigh_R[:, :, 2], new_origin=hip_R
)

# Compare thigh lengths
length_L = knee_L_local['Z'].data.mean()
length_R = knee_R_local['Z'].data.mean()

asymmetry = abs(length_L - length_R) / ((length_L + length_R) / 2) * 100
print(f"Leg length asymmetry: {asymmetry:.1f}%")
```

## Advanced Topics

### Time-Varying Rotations

Reference frames change over time (e.g., pelvis rotates during gait). The rotation matrix `R` has shape `(N, 3, 3)` where `N` is the number of time samples:

```python
# R_pelvis[0, :, :] = rotation matrix at first time point
# R_pelvis[100, :, :] = rotation matrix at 101st time point

# Axes vary over time
x_axis_t0 = R_pelvis[0, :, 0]  # X-axis at t=0
x_axis_t100 = R_pelvis[100, :, 0]  # X-axis at t=100 (may be different)
```

### Extracting Euler Angles from Rotation Matrix

```python
import numpy as np

# Extract Euler angles (ZYX convention) from rotation matrix
# R: (N, 3, 3)

def rotation_matrix_to_euler_zyx(R):
    """Convert rotation matrices to ZYX Euler angles."""
    # R from gram_schmidt has shape (N, 3, 3)
    
    # Extract angles
    pitch = np.arcsin(-R[:, 2, 0])  # Rotation about Y
    
    # Handle gimbal lock
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])  # Rotation about X
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])   # Rotation about Z
    
    # Convert to degrees
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

# Example: Pelvis orientation angles
roll, pitch, yaw = rotation_matrix_to_euler_zyx(R_pelvis)
# roll = pelvis lateral tilt
# pitch = pelvis anterior/posterior tilt
# yaw = pelvis rotation
```

## Troubleshooting

### Issue: Axes Not Perpendicular

If input vectors to `gram_schmidt()` are nearly parallel, results may be unstable:

```python
# Check angle between input vectors
i_norm = i / np.linalg.norm(i, axis=1, keepdims=True)
j_norm = j / np.linalg.norm(j, axis=1, keepdims=True)

dot_product = np.sum(i_norm * j_norm, axis=1)
angle = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))

print(f"Angle between i and j: {angle.mean():.1f}° ± {angle.std():.1f}°")

# If angle is close to 0° or 180°, vectors are nearly parallel
# Solution: Choose different input vectors
```

### Issue: Left-Handed vs Right-Handed Frames

`gram_schmidt()` always produces a right-handed frame. If you need left-handed, flip one axis:

```python
# Right-handed (default)
R = laban.gram_schmidt(i, j)

# Convert to left-handed by flipping k-axis
R[:, :, 2] = -R[:, :, 2]  # Flip vertical axis
```

### Issue: Reference Frame Flipping Over Time

If axes suddenly flip sign, markers may have crossed or reference markers are swapped:

```python
# Check for sign flips
x_axis = R[:, :, 0]
sign_flips = np.where(np.diff(np.sign(x_axis[:, 0])))[0]

if len(sign_flips) > 0:
    print(f"Warning: Axis flips detected at frames {sign_flips}")
    # Investigate marker data at these frames
```

## See Also

- [Signal Processing: Transformations](../signal-processing/transformations.md) - Detailed `change_reference_frame()` guide
- [WholeBody Model](whole-body-model.md) - Pre-defined anatomical frames
- [Joint Angles](joint-angles.md) - Angles calculated in local frames
- [API Reference: signalprocessing.gram_schmidt()](../../api-reference/signalprocessing.md#gram_schmidt) - Complete API

---

**Coordinate Systems**: Essential for expressing biomechanical data in anatomically meaningful reference frames.
