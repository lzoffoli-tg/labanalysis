# Tutorial: Custom Reference Frames

Complete guide to creating and using anatomical reference frames with semantic axis naming.

**Duration**: 25 minutes  
**Level**: Advanced  
**Prerequisites**: Understanding of 3D geometry, biomechanics basics, completed Tutorial 03

## What You'll Learn

- Create anatomical reference frames using `ReferenceFrame` class
- Use semantic axis naming (`lateral_axis`, `vertical_axis`, `anteroposterior_axis`)
- Transform 3D vectors into local coordinate systems
- Work with non-default coordinate configurations
- Calculate joint angles manually from reference frames
- Understand the difference between right-handed and left-handed frames

## Introduction to Reference Frames

In biomechanical analysis, data is typically collected in a **global (laboratory) reference frame** but must be expressed in **local (anatomical) reference frames** for meaningful interpretation.

### Why Semantic Axis Naming?

Traditional approaches use generic names like `axis_1`, `axis_2`, `axis_3` or coordinate-specific names like `x_axis`, `y_axis`, `z_axis`. However, these approaches have problems:

- **Generic names** don't convey meaning: Is `axis_1` lateral or vertical?
- **Coordinate-specific names** assume a fixed coordinate system: What if the user's setup uses X=vertical instead of Z=vertical?

The `ReferenceFrame` class solves this with **semantic axis naming**:
- `lateral_axis` - Always means mediolateral direction
- `vertical_axis` - Always means superior-inferior direction  
- `anteroposterior_axis` - Always means forward-backward direction

This makes your code **coordinate-system independent**.

## Step 1: Creating a Simple Reference Frame

```python
import labanalysis as laban
import numpy as np

# Define axes (these can point in any global direction)
origin = np.array([[0.5, 1.0, 0.2]])  # Shape (1, 3)
lateral_axis = np.array([[1.0, 0.0, 0.0]])      # Points in +X direction
vertical_axis = np.array([[0.0, 1.0, 0.0]])     # Points in +Y direction

# Create reference frame
ref_frame = laban.ReferenceFrame(
    origin=origin,
    lateral_axis=lateral_axis,
    vertical_axis=vertical_axis
)

# The anteroposterior_axis is computed automatically as lateral × vertical
print(f"Lateral axis: {ref_frame.lateral_axis[0]}")
print(f"Vertical axis: {ref_frame.vertical_axis[0]}")
print(f"Anteroposterior axis: {ref_frame.rotation_matrix[0, :, 2]}")
```

**Output:**
```
Lateral axis: [1. 0. 0.]
Vertical axis: [0. 1. 0.]
Anteroposterior axis: [0. 0. 1.]
```

### Key Points

1. Input axes don't need to be orthogonal - Gram-Schmidt orthonormalization is applied automatically
2. Input axes don't need to be unit length - they're normalized automatically
3. The third axis (`anteroposterior_axis`) is computed from the cross product if not provided

### Using Point3D Objects Directly (New in v206)

You can pass Point3D, Signal3D, or any Timeseries object directly to ReferenceFrame without calling `.to_numpy()`:

```python
import labanalysis as laban

# Load WholeBody data
body = laban.WholeBody.from_tdf_file("walking_trial.tdf")

# Get markers as Point3D objects
hip = body.left_hip
knee = body.left_knee
lateral_vec = body.right_hip - body.left_hip

# Create reference frame - no .to_numpy() needed!
thigh_rf = laban.ReferenceFrame(
    origin=hip,                # Point3D accepted directly
    lateral_axis=lateral_vec,  # Point3D from subtraction
    vertical_axis=knee - hip   # Point3D expression
)

print(f"Origin shape: {thigh_rf.origin.shape}")
print(f"Number of samples: {thigh_rf._n_samples}")
```

**Before (v205 and earlier):**
```python
thigh_rf = laban.ReferenceFrame(
    origin=hip.to_numpy(),              # Manual conversion
    lateral_axis=lateral_vec.to_numpy(),
    vertical_axis=(knee - hip).to_numpy()
)
```

ReferenceFrame now automatically extracts the numpy data from Timeseries objects, making code cleaner and more readable.

## Step 2: Understanding Semantic Parameters

The semantic parameter names tell you exactly what each axis represents anatomically:

```python
# Example: Pelvis reference frame
pelvis_lateral = body.right_hip - body.left_hip      # LEFT to RIGHT (Point3D)
pelvis_vertical = body.midpoint_asis - body.midpoint_psis  # POSTERIOR to ANTERIOR (Point3D)
pelvis_origin = body.pelvis_center  # Point3D

pelvis_rf = laban.ReferenceFrame(
    origin=pelvis_origin,
    lateral_axis=pelvis_lateral,      # Semantic: mediolateral
    vertical_axis=pelvis_vertical     # Semantic: anteroposterior (will be orthogonalized)
)

# You KNOW what each axis means, regardless of global coordinate system
print("Lateral axis represents: mediolateral direction (left → right)")
print("Vertical axis represents: superior-inferior direction (down → up)")
print("Anteroposterior axis represents: posterior → anterior direction (back → front)")
```

## Step 3: Transforming Vectors with einsum

Once you have a reference frame, you can transform vectors from global to local coordinates:

```python
# Get knee position in global coordinates
knee_global = body.left_knee.to_numpy()  # Shape (N, 3)

# Get hip reference frame
hip_rf = body.left_hip_referenceframe
rmat = hip_rf.rotation_matrix  # Shape (N, 3, 3)
origin = hip_rf.origin  # Shape (N, 3)

# Transform knee to hip local coordinates
knee_vec = knee_global - origin  # Vector from hip to knee
knee_local = np.einsum("nij,nj->ni", rmat, knee_vec)

# Now knee_local has components in hip reference frame:
lateral_component = knee_local[:, 0]  # Mediolateral position
vertical_component = knee_local[:, 1]  # Superior-inferior position
anteroposterior_component = knee_local[:, 2]  # Anterior-posterior position

print(f"Knee lateral offset: {lateral_component.mean():.3f} m")
print(f"Knee vertical distance (thigh length): {vertical_component.mean():.3f} m")
print(f"Knee AP offset: {anteroposterior_component.mean():.3f} m")
```

### The einsum Formula

```python
knee_local = np.einsum("nij,nj->ni", rmat, knee_vec)
```

Breaking this down:
- `"nij,nj->ni"`: Einstein summation notation
- `n`: Time samples (frame index)
- `i,j`: Spatial dimensions (3D)
- Operation: For each frame `n`, multiply rotation matrix `[i,j]` by vector `[j]` to get result `[i]`

This is equivalent to:
```python
knee_local = np.zeros((N, 3))
for n in range(N):
    knee_local[n, :] = rmat[n, :, :] @ knee_vec[n, :]
```

But `einsum` is vectorized and much faster!

## Step 4: Practical Example - Pelvis Reference Frame

```python
import labanalysis as laban

# Load WholeBody data
body = laban.WholeBody.from_tdf_file("walking_trial.tdf")

# Access built-in pelvis reference frame
pelvis_rf = body.pelvis_referenceframe

# Inspect the semantic axes
print("Pelvis reference frame axes:")
print(f"  Lateral axis (first frame): {pelvis_rf.lateral_axis[0]}")
print(f"  Vertical axis (first frame): {pelvis_rf.vertical_axis[0]}")

# The rotation matrix has axes as columns:
# Column 0 = lateral_axis
# Column 1 = vertical_axis
# Column 2 = anteroposterior_axis

print(f"\nRotation matrix shape: {pelvis_rf.rotation_matrix.shape}")
print(f"Column 0 (lateral): {pelvis_rf.rotation_matrix[0, :, 0]}")
print(f"Column 1 (vertical): {pelvis_rf.rotation_matrix[0, :, 1]}")
print(f"Column 2 (anteroposterior): {pelvis_rf.rotation_matrix[0, :, 2]}")

# Transform left hip position to pelvis frame
left_hip_global = body.left_hip.to_numpy()
left_hip_pelvis = np.einsum(
    "nij,nj->ni",
    pelvis_rf.rotation_matrix,
    left_hip_global - pelvis_rf.origin
)

print(f"\nLeft hip in pelvis frame (mediolateral component): {left_hip_pelvis[:, 0].mean():.3f} m")
```

## Step 5: Working with Non-Default Coordinate Systems

The beauty of semantic naming is that your code works regardless of the user's coordinate configuration:

```python
# Example: User configures vertical_axis="X" instead of default "Z"
# (This would be set via Point3D properties, not shown here)

# Your code using semantic names still works:
ref_frame = laban.ReferenceFrame(
    origin=origin,
    lateral_axis=some_lateral_vector,  # Doesn't matter which global axis this aligns with
    vertical_axis=some_vertical_vector
)

# After transformation, you can still access components semantically:
transformed = np.einsum("nij,nj->ni", ref_frame.rotation_matrix, vector)

# Index 0 is ALWAYS lateral, index 1 is ALWAYS vertical, index 2 is ALWAYS anteroposterior
# This is guaranteed by the ReferenceFrame construction, independent of global coordinates
lateral = transformed[:, 0]
vertical = transformed[:, 1]
anteroposterior = transformed[:, 2]
```

### The Key Insight

After transformation with a `ReferenceFrame`:
- **Index 0** always represents the `lateral_axis` component
- **Index 1** always represents the `vertical_axis` component
- **Index 2** always represents the `anteroposterior_axis` component

This mapping is **fixed by construction**, not by the global coordinate system.

## Step 6: Manual Angle Calculation

You can calculate joint angles manually using reference frames:

```python
# Calculate knee flexion angle manually
knee_rf = body.left_knee_referenceframe
ankle_vec = (body.left_ankle - body.left_knee).to_numpy()

# Transform ankle to knee reference frame
ankle_local = np.einsum("nij,nj->ni", knee_rf.rotation_matrix, ankle_vec)

# Extract components
anteroposterior = ankle_local[:, 2]  # Forward-backward
vertical = ankle_local[:, 1]  # Up-down

# Knee flexion angle in sagittal plane
flexion_rad = np.arctan2(-anteroposterior, -vertical)
flexion_deg = np.degrees(flexion_rad)

# Compare with automatic calculation
auto_flexion = body.left_knee_flexionextension.to_numpy()

print(f"Manual calculation: {flexion_deg.mean():.1f}°")
print(f"Automatic calculation: {auto_flexion.mean():.1f}°")
print(f"Difference: {abs(flexion_deg.mean() - auto_flexion.mean()):.2f}°")
```

## Step 7: Left vs Right Side Frames

Important: Left and right side reference frames have different handedness:

```python
# Left side: right-handed frame (det(R) = +1)
left_knee_rf = body.left_knee_referenceframe
left_det = np.linalg.det(left_knee_rf.rotation_matrix[0])
print(f"Left knee frame determinant: {left_det:.1f}")  # +1.0

# Right side: left-handed frame (det(R) = -1)
right_knee_rf = body.right_knee_referenceframe
right_det = np.linalg.det(right_knee_rf.rotation_matrix[0])
print(f"Right knee frame determinant: {right_det:.1f}")  # -1.0

# Why? To keep the anteroposterior axis pointing FORWARD on both sides
# Left side: lateral points LEFT, vertical points UP, anteroposterior points FORWARD
# Right side: lateral points RIGHT, vertical points UP, anteroposterior points FORWARD
# This requires flipping the frame handedness
```

The sign of the determinant tells you the frame's handedness:
- **det(R) = +1**: Right-handed coordinate system
- **det(R) = -1**: Left-handed coordinate system

## Troubleshooting and Best Practices

### Problem: Axes are nearly parallel

```python
# Check angle between input axes before creating frame
axis1_norm = axis1 / np.linalg.norm(axis1, axis=1, keepdims=True)
axis2_norm = axis2 / np.linalg.norm(axis2, axis=1, keepdims=True)
dot_product = np.sum(axis1_norm * axis2_norm, axis=1)
angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))

if np.any(angle_deg < 10) or np.any(angle_deg > 170):
    print("WARNING: Input axes are nearly parallel!")
    print(f"Angle between axes: {angle_deg.mean():.1f}°")
```

**Solution**: Choose different anatomical landmarks that create more perpendicular axes.

### Problem: Reference frame flips over time

```python
# Check for sign flips in axes
lateral_axis = ref_frame.lateral_axis
sign_flips = np.where(np.diff(np.sign(lateral_axis[:, 0])))[0]

if len(sign_flips) > 0:
    print(f"WARNING: Axis flips detected at frames {sign_flips}")
```

**Solution**: Check for marker swaps or occlusions in the source data.

### Best Practice: Always use semantic names

```python
# GOOD: Semantic naming
ref_frame = laban.ReferenceFrame(
    origin=origin,
    lateral_axis=mediolateral_vector,
    vertical_axis=superoinferior_vector
)

# BAD: Coordinate-specific naming (assumes X/Y/Z mapping)
# Don't do this - breaks with non-default coordinate systems
ref_frame = laban.ReferenceFrame(
    origin=origin,
    lateral_axis=x_vector,  # Assumes X = lateral
    vertical_axis=y_vector  # Assumes Y = vertical
)
```

## Summary

**Key Takeaways:**

1. ✅ Use `lateral_axis`, `vertical_axis`, `anteroposterior_axis` parameters for semantic clarity
2. ✅ Input axes are automatically orthonormalized via Gram-Schmidt
3. ✅ After transformation, indices [0], [1], [2] always map to lateral, vertical, anteroposterior
4. ✅ This mapping is coordinate-system independent
5. ✅ Left/right sides have different handedness (right-handed vs left-handed frames)
6. ✅ Always check for nearly-parallel input axes

**Next Steps:**

- Practice creating custom frames for different body segments
- Explore bilateral symmetry analysis using reference frames
- Calculate joint angles manually and compare with automatic calculations

## See Also

- [Coordinate Systems Guide](../user-guide/biomechanics/coordinate-systems.md) - Detailed coordinate system documentation
- [WholeBody Model](../user-guide/biomechanics/whole-body-model.md) - Pre-defined anatomical frames
- [Tutorial 03: Full Body Kinematics](03-full-body-kinematics.md) - Using built-in reference frames
- [Example: Reference Frame Transformations](../examples/biomechanics/reference-frames.py) - Complete working example
