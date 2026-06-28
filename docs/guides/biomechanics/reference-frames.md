# Reference Frames Guide

Guide to using anatomical coordinate systems and transformations.

## Overview

Reference frames define oriented 3D coordinate systems for biomechanical analysis. They enable:
- Expressing positions and vectors relative to anatomical segments
- Calculating joint angles as rotations between segments
- Decomposing forces and velocities into anatomically meaningful directions

---

## Quick Start

### Creating a Reference Frame

A reference frame consists of an origin (Point3D) and three orthonormal axes (Signal3D):

```python
import labanalysis as laban
import numpy as np

# Load biomechanical data
body = laban.WholeBody.from_tdf("trial.tdf", bodymass_kg=75)

# Create pelvis reference frame
def create_pelvis_frame(body):
    # Origin at midpoint between ASIS markers
    origin = (body.left_asis + body.right_asis) / 2
    
    # Y-axis: points to the right (left to right ASIS)
    y_axis = (body.right_asis - body.left_asis).normalize()
    
    # Anterior direction (ASIS to PSIS midpoint)
    mid_psis = (body.left_psis + body.right_psis) / 2
    anterior = origin - mid_psis
    
    # Z-axis: perpendicular to pelvis plane (upward)
    z_axis = y_axis.cross(anterior).normalize()
    
    # X-axis: perpendicular to Y and Z (forward)
    x_axis = y_axis.cross(z_axis).normalize()
    
    return laban.ReferenceFrame(origin=origin, x=x_axis, y=y_axis, z=z_axis)

pelvis_frame = create_pelvis_frame(body)
print(f"Pelvis frame created with {len(pelvis_frame.origin.time)} time points")
```

---

## ISB-Compliant Segment Frames

The International Society of Biomechanics (ISB) recommends standardized coordinate systems.

### Pelvis Frame (ISB)

```python
def create_isb_pelvis_frame(left_asis, right_asis, left_psis, right_psis):
    """
    ISB pelvis reference frame.
    
    Origin: Midpoint between left and right ASIS
    X: Anteriorly (perpendicular to frontal plane)
    Y: To the right (in frontal plane)
    Z: Superiorly (perpendicular to transverse plane)
    """
    # Origin
    origin = (left_asis + right_asis) / 2
    
    # Y-axis (mediolateral)
    y_axis = (right_asis - left_asis).normalize()
    
    # Temporary anterior vector
    mid_psis = (left_psis + right_psis) / 2
    anterior_temp = origin - mid_psis
    
    # Z-axis (superior direction, perpendicular to pelvis plane)
    z_axis = y_axis.cross(anterior_temp).normalize()
    
    # X-axis (anterior direction)
    x_axis = y_axis.cross(z_axis).normalize()
    
    return laban.ReferenceFrame(origin=origin, x=x_axis, y=y_axis, z=z_axis)
```

### Foot Frame (ISB)

```python
def create_isb_foot_frame(heel, first_met, fifth_met):
    """
    ISB foot reference frame.
    
    Origin: Heel marker
    X: Anteriorly (parallel to long axis of foot)
    Y: Medially (from 5th to 1st metatarsal)
    Z: Superiorly (perpendicular to foot sole)
    """
    # Origin at heel
    origin = heel
    
    # X-axis: from heel toward midpoint of metatarsals
    mid_met = (first_met + fifth_met) / 2
    x_axis = (mid_met - heel).normalize()
    
    # Y-axis temp: from 5th to 1st metatarsal
    y_temp = (first_met - fifth_met).normalize()
    
    # Z-axis: perpendicular to foot plane
    z_axis = x_axis.cross(y_temp).normalize()
    
    # Y-axis: orthogonal to X and Z
    y_axis = z_axis.cross(x_axis).normalize()
    
    return laban.ReferenceFrame(origin=origin, x=x_axis, y=y_axis, z=z_axis)
```

### Thigh Frame (ISB)

```python
def create_isb_thigh_frame(hip_center, knee_center, knee_medial, knee_lateral):
    """
    ISB thigh reference frame.
    
    Origin: Hip joint center
    X: Anteriorly
    Y: To the right
    Z: Superiorly (along femur long axis)
    """
    # Origin at hip
    origin = hip_center
    
    # Z-axis: from knee to hip (superior direction along femur)
    z_axis = (hip_center - knee_center).normalize()
    
    # Y-axis temp: from medial to lateral knee markers
    y_temp = (knee_lateral - knee_medial).normalize()
    
    # X-axis: perpendicular to Y and Z (anterior)
    x_axis = y_temp.cross(z_axis).normalize()
    
    # Y-axis: orthogonal to X and Z
    y_axis = z_axis.cross(x_axis).normalize()
    
    return laban.ReferenceFrame(origin=origin, x=x_axis, y=y_axis, z=z_axis)
```

---

## Coordinate Transformations

### Transform Points Between Frames

Express a point in different coordinate systems:

```python
# Create laboratory (global) and pelvis frames
lab_frame = create_laboratory_frame()  # Global reference
pelvis_frame = create_isb_pelvis_frame(...)

# Center of mass in global coordinates
com_global = body.center_of_mass

# Transform COM to pelvis-relative coordinates
com_in_pelvis = lab_frame.transform_point(com_global, target_frame=pelvis_frame)

# Now com_in_pelvis shows COM position relative to pelvis:
# X: anterior-posterior distance from pelvis origin
# Y: mediolateral distance
# Z: superior-inferior distance
```

### Transform Vectors Between Frames

Express velocities or forces in segment-relative terms:

```python
# Ground reaction force in lab frame
grf_lab = force_platform.force  # Signal3D in global coordinates

# Transform to foot frame
foot_frame = create_isb_foot_frame(...)
grf_foot = lab_frame.transform_vector(grf_lab, target_frame=foot_frame)

# Now grf_foot components represent:
# X: anterior-posterior force (propulsion/braking)
# Y: medial-lateral force
# Z: vertical force (perpendicular to foot)
```

---

## Use Cases

### Joint Angle Calculation

Reference frames enable relative orientation calculations:

```python
# Thigh and shank frames
thigh_frame = create_isb_thigh_frame(hip_center, knee_center, ...)
shank_frame = create_isb_shank_frame(knee_center, ankle_center, ...)

# Knee flexion angle = rotation of shank relative to thigh
# (This is what WholeBody.left_knee_flexionextension calculates internally)

# Access the result
knee_angle = body.left_knee_flexionextension  # Signal1D
print(f"Knee flexion: {knee_angle.data.min():.1f}° to {knee_angle.data.max():.1f}°")
```

### Center of Mass Relative to Pelvis

Track COM movement in pelvis-relative coordinates:

```python
# Global COM
com_global = body.center_of_mass

# Pelvis frame
pelvis = create_isb_pelvis_frame(body.left_asis, body.right_asis, ...)

# Transform to pelvis coordinates
lab = create_laboratory_frame()
com_pelvis = lab.transform_point(com_global, target_frame=pelvis)

# Plot anterior-posterior COM displacement
import matplotlib.pyplot as plt
plt.plot(com_pelvis.time, com_pelvis.data[:, 0])  # X component (AP)
plt.xlabel('Time (s)')
plt.ylabel('COM anterior displacement (mm)')
plt.title('Center of Mass in Pelvis Frame')
plt.show()
```

### Force Decomposition

Analyze forces in anatomically meaningful directions:

```python
# Global ground reaction force
grf_global = jump.ground_reaction_force.force

# Create foot frame
foot = create_isb_foot_frame(jump.left_heel, jump.left_first_metatarsal_head, ...)

# Transform force to foot coordinates
lab = create_laboratory_frame()
grf_foot = lab.transform_vector(grf_global, target_frame=foot)

# Extract components
anterior_force = grf_foot.data[:, 0]  # X: AP force
mediolateral_force = grf_foot.data[:, 1]  # Y: ML force  
vertical_force = grf_foot.data[:, 2]  # Z: Normal to foot

# Analyze propulsion phase
propulsion_index = np.where(anterior_force > 0)[0]
max_propulsion = anterior_force[propulsion_index].max()
print(f"Max anterior propulsion: {max_propulsion:.0f} N")
```

---

## Verification and Quality Control

### Check Orthonormality

Reference frames should maintain orthonormal axes:

```python
frame = create_isb_pelvis_frame(...)

if not frame.is_orthonormal:
    print("Warning: Frame is not orthonormal!")
    print("Possible causes:")
    print("- Marker tracking errors")
    print("- Improper construction")
    print("- Need Gram-Schmidt orthogonalization")
```

### Gram-Schmidt Orthogonalization

Re-orthogonalize approximate axes:

```python
def orthogonalize_axes(x_approx, y_approx, z_approx):
    """
    Orthogonalize three approximate axes.
    
    Keeps X-axis as primary, adjusts Y and Z.
    """
    # Normalize X
    x = x_approx.normalize()
    
    # Make Y perpendicular to X
    y_temp = y_approx - x * (x.dot(y_approx))
    y = y_temp.normalize()
    
    # Make Z perpendicular to both X and Y
    z_temp = z_approx - x * (x.dot(z_approx)) - y * (y.dot(z_approx))
    z = z_temp.normalize()
    
    return x, y, z

# Use when constructing frames from noisy marker data
x_ortho, y_ortho, z_ortho = orthogonalize_axes(x_measured, y_measured, z_measured)
frame = laban.ReferenceFrame(origin, x_ortho, y_ortho, z_ortho)
```

---

## Advanced Topics

### Time-Varying Reference Frames

Reference frames change over time as segments move:

```python
# Pelvis frame orientation changes during movement
pelvis = create_isb_pelvis_frame(body.left_asis, body.right_asis, ...)

# Rotation matrix at each time point
R = pelvis.rotation_matrix  # (n_frames, 3, 3) array

# Pelvis tilt angle over time
pelvis_tilt = body.pelvis_anteroposterior_tilt  # Signal1D
plt.plot(pelvis_tilt.time, pelvis_tilt.data)
plt.ylabel('Pelvis Anterior Tilt (degrees)')
plt.xlabel('Time (s)')
```

### Multiple Coordinate Transformations

Chain transformations through multiple frames:

```python
# Point in foot frame → shank frame → thigh frame → pelvis frame → lab frame

# This is how joint angles propagate through kinematic chain
# Each joint contributes rotation between adjacent segments
```

---

## Practical Example: Gait Analysis

Complete workflow analyzing gait kinematics:

```python
import labanalysis as laban
import matplotlib.pyplot as plt

# Load gait trial
body = laban.WholeBody.from_tdf("gait.tdf", bodymass_kg=75)

# Create segment frames
pelvis = create_isb_pelvis_frame(...)
thigh = create_isb_thigh_frame(...)
shank = create_isb_shank_frame(...)

# Extract joint angles (computed from relative orientations)
hip_angle = body.left_hip_flexionextension
knee_angle = body.left_knee_flexionextension
ankle_angle = body.left_ankle_flexionextension

# Plot sagittal plane angles
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(hip_angle.time, hip_angle.data)
axes[0].set_ylabel('Hip Flexion (°)')
axes[0].grid(True)

axes[1].plot(knee_angle.time, knee_angle.data)
axes[1].set_ylabel('Knee Flexion (°)')
axes[1].grid(True)

axes[2].plot(ankle_angle.time, ankle_angle.data)
axes[2].set_ylabel('Ankle Dorsiflexion (°)')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True)

plt.suptitle('Gait Kinematics - Left Leg')
plt.tight_layout()
plt.show()
```

---

## See Also

- [Reference Frames API](../../api/referenceframes.md) - Full API reference
- [WholeBody Guide](wholebody.md) - Using computed joint angles
- [Joint Angles](joint-angles.md) - Joint angle calculations

## References

1. Wu G, et al. (2002). ISB recommendation on definitions of joint coordinate system of various joints. *Journal of Biomechanics*, 35(4), 543-548.

2. Wu G, et al. (2005). ISB recommendation on definitions of joint coordinate systems—Part II: shoulder, elbow, wrist and hand. *Journal of Biomechanics*, 38(5), 981-992.
