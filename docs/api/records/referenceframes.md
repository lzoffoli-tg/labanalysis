# labanalysis.records.referenceframes

Anatomical reference frame transformations with semantic axis naming.

**Source**: `src/labanalysis/records/referenceframes.py`

## Overview

The `referenceframes` module provides the `ReferenceFrame` class for creating and working with anatomical coordinate systems. Key features:

- **Semantic axis naming**: `lateral_axis`, `vertical_axis`, `anteroposterior_axis` (coordinate-system independent)
- **Automatic orthonormalization**: Gram-Schmidt process creates perpendicular unit vectors
- **Transform any 3D data**: numpy arrays, DataFrames, Point3D, Signal3D, ForcePlatform
- **Bidirectional transforms**: `apply()` and `apply_inverse()`
- **Point3D/Signal3D support** (v206+): Pass Timeseries objects directly without `.to_numpy()`

## Classes

### ReferenceFrame

Creates an orthonormal reference frame from anatomical axes.

```python
class ReferenceFrame:
    """
    Anatomical reference frame for coordinate transformations.
    
    Creates an orthonormal (perpendicular unit vector) coordinate system
    from 2-3 input axis directions using Gram-Schmidt orthonormalization.
    
    Parameters
    ----------
    origin : np.ndarray or list or tuple or Timeseries
        Origin point of the reference frame. Shape (N, 3) or (3,).
        
        If Timeseries (Point3D, Signal3D): ._data is extracted automatically.
        Must have exactly 3 columns.
    
    lateral_axis : np.ndarray or list or tuple or Timeseries
        First axis direction (mediolateral). Shape (N, 3) or (3,).
        
        If Timeseries (Point3D, Signal3D): ._data is extracted automatically.
        Must have exactly 3 columns.
    
    vertical_axis : np.ndarray or list or tuple or Timeseries
        Second axis direction (superior-inferior). Shape (N, 3) or (3,).
        
        If Timeseries (Point3D, Signal3D): ._data is extracted automatically.
        Must have exactly 3 columns.
    
    anteroposterior_axis : np.ndarray or list or tuple or Timeseries or None, optional
        Third axis direction (anterior-posterior). Shape (N, 3) or (3,).
        If None (default), computed as lateral_axis × vertical_axis.
        
        If Timeseries (Point3D, Signal3D): ._data is extracted automatically.
        Must have exactly 3 columns.
    
    Attributes
    ----------
    origin : np.ndarray
        Origin point, shape (N, 3)
    lateral_axis : np.ndarray
        Orthonormalized lateral axis, shape (N, 3)
    vertical_axis : np.ndarray
        Orthonormalized vertical axis, shape (N, 3)
    anteroposterior_axis : np.ndarray or None
        Input anteroposterior axis (if provided), shape (N, 3)
    rotation_matrix : np.ndarray
        Rotation matrix (N, 3, 3). Columns represent semantic axes:
        Column 0 = lateral_axis, Column 1 = vertical_axis, Column 2 = anteroposterior_axis.
    
    Examples
    --------
    Example 1: Create from numpy arrays
    
    >>> import labanalysis as laban
    >>> import numpy as np
    >>> 
    >>> origin = np.array([[0.5, 1.0, 0.2]])
    >>> lateral = np.array([[1.0, 0.0, 0.0]])
    >>> vertical = np.array([[0.0, 1.0, 0.0]])
    >>> 
    >>> ref_frame = laban.ReferenceFrame(origin, lateral, vertical)
    >>> print(ref_frame.rotation_matrix.shape)
    (1, 3, 3)
    
    Example 2: Non-orthogonal inputs (auto-orthonormalized)
    
    >>> lateral = np.array([[1.0, 0.1, 0.0]])
    >>> vertical = np.array([[0.0, 1.0, 0.1]])
    >>> 
    >>> ref_frame = laban.ReferenceFrame([0, 0, 0], lateral, vertical)
    >>> R = ref_frame.rotation_matrix[0]
    >>> identity = R.T @ R
    >>> np.allclose(identity, np.eye(3))
    True
    
    Example 3: Using Point3D objects directly (new in v206)
    
    >>> hip = body.left_hip          # Point3D with shape (1000, 3)
    >>> knee = body.left_knee        # Point3D
    >>> lateral_vec = body.right_hip - body.left_hip  # Point3D arithmetic
    >>> 
    >>> ref_frame = laban.ReferenceFrame(
    ...     origin=hip,                # Point3D accepted directly
    ...     lateral_axis=lateral_vec,  # Point3D from subtraction
    ...     vertical_axis=knee - hip   # Point3D expression
    ... )
    >>> # No .to_numpy() needed! Arrays extracted automatically.
    
    Notes
    -----
    - Input axes do NOT need to be orthogonal or unit length
    - Gram-Schmidt orthonormalization is applied automatically
    - The third axis is computed as lateral × vertical if not provided
    - After construction, rotation_matrix columns are orthonormal
    """
```

#### Properties

##### origin

Reference frame origin point.

```python
@property
def origin(self) -> np.ndarray
```

**Returns:**
- `np.ndarray`: Origin coordinates, shape (N, 3)

**Example:**
```python
rf = laban.ReferenceFrame(origin=[1, 2, 3], lateral_axis=[1, 0, 0], vertical_axis=[0, 1, 0])
print(rf.origin)  # [[1. 2. 3.]]
```

##### lateral_axis

Orthonormalized lateral (mediolateral) axis.

```python
@property
def lateral_axis(self) -> np.ndarray
```

**Returns:**
- `np.ndarray`: Unit vector in lateral direction, shape (N, 3)

**Semantic meaning:** Mediolateral direction (left → right for left side, right → left for right side)

##### vertical_axis

Orthonormalized vertical (superior-inferior) axis.

```python
@property
def vertical_axis(self) -> np.ndarray
```

**Returns:**
- `np.ndarray`: Unit vector in vertical direction, shape (N, 3)

**Semantic meaning:** Superior-inferior direction (down → up)

##### rotation_matrix

Rotation matrix for transformations.

```python
@property
def rotation_matrix(self) -> np.ndarray
```

**Returns:**
- `np.ndarray`: Rotation matrix, shape (N, 3, 3)

**Column structure (semantic axes):**
- Column 0 = lateral_axis (mediolateral direction)
- Column 1 = vertical_axis (superior-inferior direction)  
- Column 2 = anteroposterior_axis (anterior-posterior direction)

**Example:**
```python
rf = laban.ReferenceFrame([0, 0, 0], [1, 0, 0], [0, 1, 0])
R = rf.rotation_matrix[0]

# Extract semantic axes from rotation matrix
lateral_axis = R[:, 0]           # Column 0
vertical_axis = R[:, 1]          # Column 1
anteroposterior_axis = R[:, 2]   # Column 2
```

#### Methods

##### apply()

Transform data from global to local reference frame.

```python
def apply(
    self,
    obj: np.ndarray | pd.DataFrame | Timeseries | ForcePlatform,
    inplace: bool = False
) -> np.ndarray | pd.DataFrame | Timeseries | ForcePlatform
```

**Parameters:**
- `obj` (np.ndarray | DataFrame | Timeseries | ForcePlatform): Data to transform
  - `np.ndarray`: Shape (N, 3) or (3,)
  - `pd.DataFrame`: Must have exactly 3 numeric columns
  - `Timeseries`: Must have 3 columns (Point3D, Signal3D)
  - `ForcePlatform`: Transforms origin, force, and torque
- `inplace` (bool): Modify object in place (only for DataFrame/Timeseries/ForcePlatform)

**Returns:**
- Same type as input, transformed to local coordinates

**Raises:**
- `ValueError`: If input has wrong number of columns or mismatched sample count
- `TypeError`: If input type is not supported

**Examples:**

Transform numpy array:
```python
# Global coordinates
knee_global = body.left_knee.to_numpy()  # (N, 3)

# Transform to hip reference frame
hip_rf = body.left_hip_referenceframe
knee_local = hip_rf.apply(knee_global)

# Now in hip local coordinates
lateral_offset = knee_local[:, 0]
vertical_distance = knee_local[:, 1]
ap_offset = knee_local[:, 2]
```

Transform Point3D:
```python
# Transform marker
marker_global = body.left_ankle  # Point3D
marker_local = knee_rf.apply(marker_global)  # Returns Point3D

# Units and metadata preserved
print(marker_local.unit)  # 'm'
print(marker_local.shape)  # Same as input
```

Transform ForcePlatform:
```python
# Transform force platform to pelvis frame
fp_global = record['FP1']
fp_pelvis = pelvis_rf.apply(fp_global)

# Force components now in pelvis coordinates
fx_pelvis = fp_pelvis.force['Fx']  # Mediolateral
fy_pelvis = fp_pelvis.force['Fy']  # Superior-inferior  
fz_pelvis = fp_pelvis.force['Fz']  # Anteroposterior
```

In-place transformation:
```python
# Modify Timeseries in place
# Note: Column labels can be anything; semantic meaning comes from indices after transformation
signal = laban.Signal3D(data, index, unit='m', columns=['GlobalX', 'GlobalY', 'GlobalZ'])
rf.apply(signal, inplace=True)  # signal modified, returns None
# After transformation: [:, 0]=lateral, [:, 1]=vertical, [:, 2]=anteroposterior
```

##### apply_inverse()

Transform data from local back to global reference frame.

```python
def apply_inverse(
    self,
    obj: np.ndarray | pd.DataFrame | Timeseries | ForcePlatform,
    inplace: bool = False
) -> np.ndarray | pd.DataFrame | Timeseries | ForcePlatform
```

**Parameters:**
- Same as `apply()`

**Returns:**
- Same type as input, transformed to global coordinates

**Example:**
```python
# Forward transform
knee_local = hip_rf.apply(knee_global)

# Inverse transform (should recover original)
knee_recovered = hip_rf.apply_inverse(knee_local)

# Verify roundtrip
np.allclose(knee_recovered, knee_global)  # True
```

##### \_\_call\_\_()

Callable interface (shorthand for `apply()`).

```python
def __call__(
    self,
    obj: np.ndarray | pd.DataFrame | Timeseries | ForcePlatform,
    inplace: bool = False
) -> np.ndarray | pd.DataFrame | Timeseries | ForcePlatform
```

**Example:**
```python
rf = laban.ReferenceFrame([0, 0, 0], [1, 0, 0], [0, 1, 0])

# These are equivalent
result1 = rf.apply(data)
result2 = rf(data)
```

## Usage Examples

### Creating Reference Frames

From anatomical markers:
```python
import labanalysis as laban

# Load body
body = laban.WholeBody.from_tdf_file("walking.tdf")

# Create pelvis reference frame from markers
lateral = body.right_asis - body.left_asis  # Point3D
vertical = (body.left_asis + body.right_asis) / 2 - \
           (body.left_psis + body.right_psis) / 2  # Point3D

pelvis_rf = laban.ReferenceFrame(
    origin=body.pelvis_center,  # Point3D
    lateral_axis=lateral,        # Point3D
    vertical_axis=vertical       # Point3D
)
```

### Transforming Vectors

Using einsum for custom transformations:
```python
# Get reference frame components
R = rf.rotation_matrix  # (N, 3, 3)
origin = rf.origin       # (N, 3)

# Manual transformation
vec_global = knee - hip  # Global coordinates
vec_centered = vec_global - origin  # Center at origin
vec_local = np.einsum("nij,nj->ni", R, vec_centered)  # Rotate

# Equivalent to:
vec_local = rf.apply(vec_global)
```

### Manual Angle Calculation

Calculate knee flexion manually:
```python
# Get knee reference frame
knee_rf = body.left_knee_referenceframe

# Transform ankle to knee frame
ankle_vec = body.left_ankle - body.left_knee
ankle_local = knee_rf.apply(ankle_vec)

# Extract components
anteroposterior = ankle_local[:, 2]  # Forward-backward
vertical = ankle_local[:, 1]         # Up-down

# Knee flexion in sagittal plane
flexion_rad = np.arctan2(-anteroposterior, -vertical)
flexion_deg = np.degrees(flexion_rad)

# Compare with automatic
auto_flexion = body.left_knee_flexionextension.to_numpy()
print(f"Manual: {flexion_deg.mean():.1f}°")
print(f"Auto: {auto_flexion.mean():.1f}°")
```

### Bilateral Symmetry Analysis

Compare left and right reference frames:
```python
left_knee_rf = body.left_knee_referenceframe
right_knee_rf = body.right_knee_referenceframe

# Check determinants
left_det = np.linalg.det(left_knee_rf.rotation_matrix[0])
right_det = np.linalg.det(right_knee_rf.rotation_matrix[0])

print(f"Left knee: det(R) = {left_det:+.1f}")   # +1.0 (right-handed)
print(f"Right knee: det(R) = {right_det:+.1f}")  # -1.0 (left-handed)

# This ensures anteroposterior_axis points FORWARD on both sides
```

### Multi-Sample Reference Frames

Time-varying reference frames:
```python
# Create time-varying pelvis frame (N=1000 samples)
n_samples = 1000

lateral = body.right_asis - body.left_asis  # Point3D (1000, 3)
vertical = body.midpoint_asis - body.midpoint_psis  # Point3D (1000, 3)

pelvis_rf = laban.ReferenceFrame(
    origin=body.pelvis_center,  # (1000, 3)
    lateral_axis=lateral,        # (1000, 3)
    vertical_axis=vertical       # (1000, 3)
)

# Transform marker trajectory
hip_pelvis = pelvis_rf.apply(body.left_hip)  # (1000, 3) in pelvis frame

# Each time sample has its own reference frame orientation
```

### Working with Non-Default Coordinates

Semantic naming works regardless of global coordinate system:
```python
# Works with ANY global coordinate convention
# (e.g., X=vertical, Y=anteroposterior, Z=lateral - or any other mapping)
# Code remains unchanged because we use semantic axis names

ref_frame = laban.ReferenceFrame(
    origin=origin,
    lateral_axis=some_lateral_vector,      # Anatomical direction, not X/Y/Z
    vertical_axis=some_vertical_vector     # Anatomical direction, not X/Y/Z
)

# After transformation, column indices are ALWAYS semantic
transformed = ref_frame.apply(data)

lateral_component = transformed[:, 0]           # ALWAYS mediolateral
vertical_component = transformed[:, 1]          # ALWAYS superior-inferior
anteroposterior_component = transformed[:, 2]   # ALWAYS anterior-posterior

# Never use axis letters (X/Y/Z) - they vary between labs
# Always use semantic column indices (0=lateral, 1=vertical, 2=anteroposterior)
```

## Advanced Topics

### Gram-Schmidt Orthonormalization

The orthonormalization process:

1. **Lateral axis**: Normalized from input `lateral_axis`
   ```
   e1 = lateral_axis / ||lateral_axis||
   ```

2. **Anteroposterior axis**: Perpendicular to lateral and vertical
   ```
   e3_temp = lateral_axis × vertical_axis
   e3 = e3_temp / ||e3_temp||
   ```

3. **Vertical axis**: Perpendicular to lateral and anteroposterior
   ```
   e2 = e3 × e1
   ```

Result: `[e1, e2, e3]` form an orthonormal basis.

### Rotation Matrix Structure

The rotation matrix stores semantic axes as columns, independent of the global coordinate system:

```python
rotation_matrix = [
    [lateral_x,  vertical_x,  anteroposterior_x],  # Global coordinate components
    [lateral_y,  vertical_y,  anteroposterior_y],  # (rows represent global axes,
    [lateral_z,  vertical_z,  anteroposterior_z]   #  but naming depends on lab setup)
]
```

**Semantic column meanings (coordinate-system independent):**
- Column 0 = lateral_axis (mediolateral direction)
- Column 1 = vertical_axis (superior-inferior direction)
- Column 2 = anteroposterior_axis (anterior-posterior direction)

**Important:** The rows represent global coordinate components, but their anatomical meaning varies between labs. Always reference columns by semantic axis names, not by letter (X/Y/Z).

### Left vs Right Handedness

**Left side frames** (right-handed, det(R) = +1):
- Lateral points LEFT
- Vertical points UP
- Anteroposterior points FORWARD
- Right-hand rule: lateral × vertical = anteroposterior

**Right side frames** (left-handed, det(R) = -1):
- Lateral points RIGHT (negated)
- Vertical points UP (same)
- Anteroposterior points FORWARD (same)
- Left-hand rule: lateral × vertical = -anteroposterior

This ensures anteroposterior axes point FORWARD on both sides.

### Broadcasting

Single-frame reference applied to multi-sample data:
```python
# Single reference frame (1 sample)
rf = laban.ReferenceFrame([0, 0, 0], [1, 0, 0], [0, 1, 0])

# Multi-sample data (1000 samples)
data = np.random.rand(1000, 3)

# Broadcasting: same frame applied to all samples
result = rf.apply(data)  # (1000, 3)
```

Multi-frame reference requires matching sample count:
```python
# Multi-frame reference (1000 samples)
rf = laban.ReferenceFrame(origin_1000, lateral_1000, vertical_1000)

# Must match
data_1000 = np.random.rand(1000, 3)  # ✓ Works
data_500 = np.random.rand(500, 3)    # ✗ ValueError
```

## Troubleshooting

### Nearly Parallel Input Axes

```python
# Check angle between axes
axis1_norm = lateral / np.linalg.norm(lateral, axis=1, keepdims=True)
axis2_norm = vertical / np.linalg.norm(vertical, axis=1, keepdims=True)
dot_product = np.sum(axis1_norm * axis2_norm, axis=1)
angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))

if np.any(angle_deg < 10) or np.any(angle_deg > 170):
    print(f"WARNING: Axes nearly parallel! Angle: {angle_deg.mean():.1f}°")
```

**Solution:** Choose different anatomical landmarks that create more perpendicular axes.

### Reference Frame Flips

```python
# Detect sign flips in axis over time
lateral_axis = rf.lateral_axis
sign_flips = np.where(np.diff(np.sign(lateral_axis[:, 0])))[0]

if len(sign_flips) > 0:
    print(f"WARNING: Axis flips at frames {sign_flips}")
```

**Solution:** Check for marker swaps or occlusions in source data.

### Shape Mismatches

```python
# All inputs must have same number of rows
try:
    rf = laban.ReferenceFrame(
        origin=np.random.rand(5, 3),
        lateral_axis=np.random.rand(10, 3),  # Different!
        vertical_axis=np.random.rand(5, 3)
    )
except ValueError as e:
    print(e)  # "All inputs must have same number of rows"
```

**Solution:** Ensure all inputs have matching first dimension.

## See Also

- **[Tutorial: Custom Reference Frames](../../tutorials/09-custom-reference-frames.md)** - Complete guide
- **[User Guide: Coordinate Systems](../../guides/biomechanics/coordinates.md)** - Coordinate system concepts
- **[WholeBody](bodies.md)** - Pre-defined anatomical reference frames
- **[Signal Processing](../signalprocessing.md)** - `gram_schmidt()` function

---

**Module**: `src/labanalysis/records/referenceframes.py`
