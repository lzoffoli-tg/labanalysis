# Coordinate Transformations

Guide to changing reference frames and creating custom coordinate systems using Gram-Schmidt orthonormalization in labanalysis.

## Overview

Coordinate transformations are essential when:
- Converting between laboratory and anatomical reference frames
- Aligning data from different sensors/devices
- Creating segment-based local coordinate systems
- Expressing vectors in different bases

labanalysis provides:
- `change_reference_frame()` - Transform Signal3D/Point3D to new coordinate system
- `gram_schmidt()` - Create orthonormal basis from non-orthogonal vectors

## Quick Reference

| Task | Method | Input | Output |
|------|--------|-------|--------|
| Transform marker to new frame | `marker.change_reference_frame()` | Basis vectors + origin | Rotated & translated marker |
| Create orthonormal basis | `gram_schmidt(i, j, k)` | 3 direction vectors | Rotation matrix (N, 3, 3) |
| Express force in new axes | `force.change_reference_frame()` | Anatomical axes | Force in anatomical frame |

## Gram-Schmidt Orthonormalization

### Basic Concept

Given non-orthogonal vectors, create an orthonormal basis (perpendicular unit vectors).

**Process:**
1. Normalize first vector → **e₁**
2. Remove e₁ component from second vector, normalize → **e₂**
3. Compute third vector as **e₃ = e₁ × e₂** (or orthogonalize third input)

**Result:** Rotation matrix with orthonormal basis `[e₁, e₂, e₃]` as columns.

### Usage

```python
import labanalysis as laban
import numpy as np

# Define three direction vectors (not necessarily orthogonal)
i = np.array([[1.0, 0.1, 0.0]])  # Approximately X
j = np.array([[0.0, 1.0, 0.1]])  # Approximately Y
k = None  # Will be computed as i × j

# Create orthonormal basis
R = laban.gram_schmidt(i, j, k)

print(f"Rotation matrix shape: {R.shape}")
# Output: Rotation matrix shape: (1, 3, 3)

print(f"e1: {R[0, :, 0]}")  # First column
print(f"e2: {R[0, :, 1]}")  # Second column
print(f"e3: {R[0, :, 2]}")  # Third column

# Output:
# e1: [0.995 0.100 0.000]
# e2: [-0.100 0.995 0.100]
# e3: [0.010 -0.099 0.995]

# Verify orthonormality
e1 = R[0, :, 0]
e2 = R[0, :, 1]
e3 = R[0, :, 2]

print(f"e1 · e2 = {np.dot(e1, e2):.6f}")  # Should be ~0
print(f"||e1|| = {np.linalg.norm(e1):.6f}")  # Should be 1
# Output:
# e1 · e2 = 0.000000
# ||e1|| = 1.000000
```

### Batch Processing

Gram-Schmidt handles multiple frames simultaneously:

```python
# Create coordinate systems for each time sample
# Example: pelvis coordinate system from LASI, RASI, LPSI, RPSI markers

record = laban.TimeseriesRecord.from_tdf("walking.tdf")

# Load pelvic markers
LASI = record.markers['LASI'].data  # (N, 3)
RASI = record.markers['RASI'].data
LPSI = record.markers['LPSI'].data
RPSI = record.markers['RPSI'].data

# Define pelvis coordinate system
# Midpoint calculations
mid_ASIS = (LASI + RASI) / 2  # Anterior midpoint
mid_PSIS = (LPSI + RPSI) / 2  # Posterior midpoint

# Basis vectors
i = RASI - LASI  # Mediolateral (right to left)
j = mid_ASIS - mid_PSIS  # Anteroposterior
k = None  # Will be computed as i × j (vertical)

# Create rotation matrices for all time samples
R_pelvis = laban.gram_schmidt(i, j, k)

print(f"Pelvis frames: {R_pelvis.shape}")
# Output: Pelvis frames: (1247, 3, 3)  (one frame per time sample)

# Now R_pelvis[t, :, :] is the pelvis coordinate system at time t
```

## Change Reference Frame

### Basic Transformation

Transform a `Signal3D` or `Point3D` to a new coordinate system.

```python
import labanalysis as laban
import numpy as np

# Load marker in laboratory frame
marker = laban.Point3D.from_tdf("trial.tdf", marker_name="C7")

# Define new reference frame (e.g., aligned with motion direction)
# Laboratory: X=right, Y=forward, Z=up
# New frame: X=forward, Y=up, Z=right

new_x = [0, 1, 0]  # Forward (was Y)
new_y = [0, 0, 1]  # Up (was Z)
new_z = [1, 0, 0]  # Right (was X)
new_origin = [0, 0, 0]  # Same origin

# Transform marker
marker_transformed = marker.change_reference_frame(
    new_x=new_x,
    new_y=new_y,
    new_z=new_z,
    new_origin=new_origin,
    inplace=False
)

# Check transformation
print("Original coordinates (lab frame):")
print(f"  X: {marker['x'].data[0]:.1f} mm")
print(f"  Y: {marker['y'].data[0]:.1f} mm")
print(f"  Z: {marker['z'].data[0]:.1f} mm")

print("\nTransformed coordinates (motion frame):")
print(f"  X (forward): {marker_transformed['x'].data[0]:.1f} mm")
print(f"  Y (up): {marker_transformed['y'].data[0]:.1f} mm")
print(f"  Z (right): {marker_transformed['z'].data[0]:.1f} mm")
```

### Time-Varying Reference Frames

Use time-varying basis vectors for anatomical coordinate systems:

```python
# Create pelvis coordinate system (from previous example)
R_pelvis = laban.gram_schmidt(i, j, k)

# Extract basis vectors
pelvis_x = R_pelvis[:, :, 0]  # Mediolateral
pelvis_y = R_pelvis[:, :, 1]  # Anteroposterior  
pelvis_z = R_pelvis[:, :, 2]  # Vertical

# Origin = pelvis center
pelvis_origin = (mid_ASIS + mid_PSIS) / 2

# Transform thigh marker to pelvis frame
hip_marker = record.markers['LASIS']

# Transform
hip_in_pelvis = hip_marker.change_reference_frame(
    new_x=pelvis_x,
    new_y=pelvis_y,
    new_z=pelvis_z,
    new_origin=pelvis_origin,
    inplace=False
)

# Now hip_in_pelvis expresses hip position in pelvis-centered coordinates
print(f"Hip relative to pelvis (X=ML, Y=AP, Z=vertical):")
print(f"  X: {hip_in_pelvis['x'].data.mean():.1f} ± {hip_in_pelvis['x'].data.std():.1f} mm")
print(f"  Y: {hip_in_pelvis['y'].data.mean():.1f} ± {hip_in_pelvis['y'].data.std():.1f} mm")
print(f"  Z: {hip_in_pelvis['z'].data.mean():.1f} ± {hip_in_pelvis['z'].data.std():.1f} mm")
# Output shows hip position relative to pelvis center in anatomical axes
```

## Complete Workflow Examples

### Example 1: Express GRF in Gait Direction

Transform force platform forces to align with walking direction.

```python
import labanalysis as laban
import numpy as np

# Load force platform and markers
record = laban.TimeseriesRecord.from_tdf("walking.tdf")
fp = record.forceplatforms['FP1']

# Load heel and toe markers to determine gait direction
heel = record.markers['heel_R']
toe = record.markers['toe_R']

# Calculate gait direction (heel to toe)
gait_vector = toe.data - heel.data  # (N, 3)

# Average direction over contact phase
contact_start = 500
contact_end = 800
gait_direction = gait_vector[contact_start:contact_end, :].mean(axis=0)
gait_direction = gait_direction / np.linalg.norm(gait_direction)

print(f"Gait direction (unit vector): {gait_direction}")
# Output: Gait direction (unit vector): [0.02 0.98 0.01]  (mostly forward)

# Define new coordinate system
# X = gait direction (forward)
# Y = vertical (unchanged)
# Z = mediolateral (right-hand rule)

new_x = gait_direction
new_y = np.array([0, 0, 1])  # Vertical (Z in lab frame)
new_z = None  # Will be computed as X × Y

# Broadcast to all time samples
n_samples = len(fp.force['Fx'].data)
new_x_full = np.tile(new_x, (n_samples, 1))
new_y_full = np.tile(new_y, (n_samples, 1))

# Transform force
force_gait_frame = fp.force.change_reference_frame(
    new_x=new_x_full,
    new_y=new_y_full,
    new_z=None,
    new_origin=[0, 0, 0],
    inplace=False
)

# Now force components are:
# Fx = anteroposterior (braking/propulsion)
# Fy = vertical
# Fz = mediolateral

fx_gait = force_gait_frame['Fx'].data
fy_gait = force_gait_frame['Fy'].data
fz_gait = force_gait_frame['Fz'].data

print(f"\nForce in gait frame:")
print(f"  Fx (A-P): {fx_gait[contact_start:contact_end].mean():.1f} N (braking)")
print(f"  Fy (vertical): {fy_gait[contact_start:contact_end].mean():.1f} N")
print(f"  Fz (M-L): {fz_gait[contact_start:contact_end].mean():.1f} N")
```

### Example 2: Segment-Based Coordinate System

Create thigh coordinate system and express knee marker in thigh frame.

```python
# Load hip, knee, ankle markers
hip = record.markers['hip_R']
knee = record.markers['knee_R']
ankle = record.markers['ankle_R']

# Define thigh coordinate system
# Y-axis: hip to knee (longitudinal)
# X-axis: perpendicular to sagittal plane
# Z-axis: X × Y

# Longitudinal axis
thigh_long = knee.data - hip.data  # (N, 3)

# Approximate mediolateral (assuming bilateral symmetry)
# Use hip_L to define frontal plane
hip_L = record.markers['hip_L']
hip_R = record.markers['hip_R']
frontal_normal = hip_R.data - hip_L.data

# Thigh X = frontal normal (lateral)
# Thigh Y = longitudinal (distal)
# Thigh Z = X × Y (anterior)

R_thigh = laban.gram_schmidt(
    i=frontal_normal,
    j=thigh_long,
    k=None
)

# Extract basis vectors
thigh_x = R_thigh[:, :, 0]
thigh_y = R_thigh[:, :, 1]
thigh_z = R_thigh[:, :, 2]

# Origin = hip joint center
thigh_origin = hip.data

# Express knee in thigh frame
knee_in_thigh = knee.change_reference_frame(
    new_x=thigh_x,
    new_y=thigh_y,
    new_z=thigh_z,
    new_origin=thigh_origin,
    inplace=False
)

# Knee position in thigh frame
# X should be near-zero (knee is in sagittal plane)
# Y is thigh length (positive distal)
# Z varies with knee flexion

x_knee = knee_in_thigh['x'].data
y_knee = knee_in_thigh['y'].data
z_knee = knee_in_thigh['z'].data

print(f"Knee in thigh frame:")
print(f"  X (mediolateral): {x_knee.mean():.1f} ± {x_knee.std():.1f} mm")
print(f"  Y (distal): {y_knee.mean():.1f} ± {y_knee.std():.1f} mm (thigh length)")
print(f"  Z (anterior): {z_knee.mean():.1f} ± {z_knee.std():.1f} mm")

# Output:
# Knee in thigh frame:
#   X (mediolateral): 2.3 ± 4.1 mm  (near zero, good)
#   Y (distal): -423.7 ± 8.2 mm (thigh length)
#   Z (anterior): 15.6 ± 12.3 mm (varies with flexion)
```

### Example 3: Global to Local Frame for Multiple Segments

Transform all lower limb markers to pelvis frame.

```python
# Create pelvis frame (from earlier)
R_pelvis = laban.gram_schmidt(i, j, k)
pelvis_x = R_pelvis[:, :, 0]
pelvis_y = R_pelvis[:, :, 1]
pelvis_z = R_pelvis[:, :, 2]
pelvis_origin = (mid_ASIS + mid_PSIS) / 2

# Transform all lower limb markers
markers_to_transform = ['hip_R', 'knee_R', 'ankle_R', 'toe_R']
markers_in_pelvis = {}

for marker_name in markers_to_transform:
    marker = record.markers[marker_name]
    
    marker_pelvis = marker.change_reference_frame(
        new_x=pelvis_x,
        new_y=pelvis_y,
        new_z=pelvis_z,
        new_origin=pelvis_origin,
        inplace=False
    )
    
    markers_in_pelvis[marker_name] = marker_pelvis

# Now all markers are expressed in pelvis-centered anatomical frame
# Useful for:
# - Removing trunk motion from leg kinematics
# - Computing joint angles in anatomical axes
# - Comparing gait across trials with different pelvis orientations

# Example: Compute hip abduction angle
hip_pelvis = markers_in_pelvis['hip_R']
x_hip = hip_pelvis['x'].data  # Mediolateral
z_hip = hip_pelvis['z'].data  # Vertical

# Hip abduction angle (in frontal plane)
hip_abd_angle = np.arctan2(x_hip, z_hip) * 180 / np.pi

print(f"Hip abduction angle: {hip_abd_angle.mean():.1f} ± {hip_abd_angle.std():.1f}°")
```

## Advanced: Inverse Transformations

### Transform Back to Original Frame

```python
# Transform marker to pelvis frame
marker_pelvis = marker.change_reference_frame(
    new_x=pelvis_x,
    new_y=pelvis_y,
    new_z=pelvis_z,
    new_origin=pelvis_origin,
    inplace=False
)

# To transform back, use inverse rotation
# R^(-1) = R^T for orthonormal matrices

# Create inverse basis (transpose of original)
R_pelvis_inv = R_pelvis.transpose([0, 2, 1])  # Transpose last two dims

pelvis_x_inv = R_pelvis_inv[:, :, 0]
pelvis_y_inv = R_pelvis_inv[:, :, 1]
pelvis_z_inv = R_pelvis_inv[:, :, 2]

# Transform back
# Note: Origin transformation is different for inverse
# old = R^T * new + origin
# But change_reference_frame does: new = R * (old - origin)
# So we need to manually adjust:

# Step 1: Express in inverse frame
marker_temp = marker_pelvis.change_reference_frame(
    new_x=pelvis_x_inv,
    new_y=pelvis_y_inv,
    new_z=pelvis_z_inv,
    new_origin=[0, 0, 0],  # No translation yet
    inplace=False
)

# Step 2: Add back the origin
marker_reconstructed = marker_temp.copy()
marker_reconstructed.data += pelvis_origin

# Verify
diff = np.abs(marker.data - marker_reconstructed.data).max()
print(f"Max reconstruction error: {diff:.6f} mm")
# Output: Max reconstruction error: 0.000001 mm (numerical precision)
```

## Validation and Quality Control

### Check Orthonormality

```python
# After Gram-Schmidt, verify basis is orthonormal
R = laban.gram_schmidt(i, j, k)

# Extract basis vectors (at first time sample)
e1 = R[0, :, 0]
e2 = R[0, :, 1]
e3 = R[0, :, 2]

# Check orthogonality (dot products should be ~0)
print("Orthogonality checks:")
print(f"  e1 · e2 = {np.abs(np.dot(e1, e2)):.6f}  (should be ~0)")
print(f"  e1 · e3 = {np.abs(np.dot(e1, e3)):.6f}")
print(f"  e2 · e3 = {np.abs(np.dot(e2, e3)):.6f}")

# Check normality (lengths should be 1)
print("\nNormality checks:")
print(f"  ||e1|| = {np.linalg.norm(e1):.6f}  (should be 1)")
print(f"  ||e2|| = {np.linalg.norm(e2):.6f}")
print(f"  ||e3|| = {np.linalg.norm(e3):.6f}")

# Check right-handedness (e1 × e2 should equal e3)
cross = np.cross(e1, e2)
print("\nRight-handedness check:")
print(f"  ||e1 × e2 - e3|| = {np.linalg.norm(cross - e3):.6f}  (should be ~0)")
```

### Visualize Coordinate Systems

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot origin
origin = pelvis_origin[0, :]  # At first time sample
ax.scatter(*origin, color='black', s=100, label='Origin')

# Plot basis vectors
scale = 100  # mm
colors = ['red', 'green', 'blue']
labels = ['X (ML)', 'Y (AP)', 'Z (vertical)']

for i, (color, label) in enumerate(zip(colors, labels)):
    basis = R_pelvis[0, :, i]
    ax.quiver(
        *origin, *basis * scale,
        color=color,
        arrow_length_ratio=0.2,
        linewidth=3,
        label=label
    )

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.legend()
ax.set_title('Pelvis Coordinate System')
plt.show()
```

## Common Patterns

### Pattern 1: Anatomical Segment Frame

```python
def create_segment_frame(proximal, distal, left_marker, right_marker):
    """
    Create anatomical segment coordinate system.
    
    Y-axis: Longitudinal (proximal to distal)
    X-axis: Mediolateral (left to right)
    Z-axis: Anterior (X × Y)
    """
    # Longitudinal
    j = distal - proximal
    
    # Mediolateral
    i = right_marker - left_marker
    
    # Create orthonormal basis
    R = laban.gram_schmidt(i, j, k=None)
    
    return R, proximal  # Return rotation and origin

# Usage
R_thigh, thigh_origin = create_segment_frame(
    proximal=hip.data,
    distal=knee.data,
    left_marker=hip_L.data,
    right_marker=hip_R.data
)
```

### Pattern 2: Plane-Based Frame

```python
def create_plane_frame(p1, p2, p3):
    """
    Create coordinate frame from three points defining a plane.
    
    X-axis: p1 to p2
    Z-axis: Normal to plane (p1-p2) × (p1-p3)
    Y-axis: Z × X
    """
    # Edge vectors
    edge1 = p2 - p1
    edge2 = p3 - p1
    
    # Normal (cross product)
    normal = np.cross(edge1, edge2, axis=1)
    
    # Create basis
    R = laban.gram_schmidt(i=edge1, j=normal, k=None)
    # This gives: X=edge1, Y=normal, Z=X×Y
    # If you want Z=normal, reorder after
    
    return R, p1

# Usage: Foot plane from heel, toe, lateral malleolus
R_foot, foot_origin = create_plane_frame(
    p1=heel.data,
    p2=toe.data,
    p3=lateral_malleolus.data
)
```

## Troubleshooting

### "ValueError: Input vectors are not valid"

```python
# Check input shapes
print(f"i.shape: {i.shape}")  # Should be (N, 3)
print(f"j.shape: {j.shape}")  # Should be (N, 3)

# Check for NaN or zero vectors
if np.any(np.isnan(i)):
    print("Warning: i contains NaN")
    
if np.any(np.linalg.norm(i, axis=1) < 1e-6):
    print("Warning: i contains near-zero vectors")
```

### Transformed data looks wrong

```python
# Common issue: Wrong axis order
# Make sure new_x, new_y, new_z are in the intended directions

# Debug: Transform a known point
test_point = np.array([[100, 0, 0]])  # Point along original X
test_signal = laban.Point3D(
    data=test_point,
    sampling_frequency=100,
    labels=['x', 'y', 'z']
)

test_transformed = test_signal.change_reference_frame(
    new_x=[0, 1, 0],  # Y becomes new X
    new_y=[0, 0, 1],  # Z becomes new Y
    new_z=[1, 0, 0],  # X becomes new Z
    new_origin=[0, 0, 0]
)

print(f"Original: {test_point[0]}")
print(f"Transformed: {test_transformed.data[0]}")
# Expected: [0, 0, 100] (original X is now Z)
```

### Time-varying frames create discontinuities

```python
# Smooth basis vectors before transformation
from scipy.ndimage import uniform_filter1d

# Smooth rotation matrix components
R_smooth = R_pelvis.copy()
for i in range(3):
    for j in range(3):
        R_smooth[:, i, j] = uniform_filter1d(R_pelvis[:, i, j], size=10)

# Re-orthonormalize after smoothing
# Extract smoothed vectors
i_smooth = R_smooth[:, :, 0]
j_smooth = R_smooth[:, :, 1]
k_smooth = R_smooth[:, :, 2]

# Re-apply Gram-Schmidt
R_final = laban.gram_schmidt(i_smooth, j_smooth, k_smooth)
```

## See Also

- **[Signal Processing Overview](README.md)** - All signal processing tools
- **[WholeBody Model](../biomechanics/whole-body-model.md)** - Uses coordinate transformations
- **[API Reference: gram_schmidt()](../../api/signalprocessing.md#gram_schmidt)** - Function documentation
- **[API Reference: Point3D.change_reference_frame()](../../api/records/timeseries.md#change_reference_frame)** - Method documentation

---

**Module**: `src/labanalysis/signalprocessing.py` (gram_schmidt)  
**Module**: `src/labanalysis/records/timeseries.py` (Signal3D.change_reference_frame)  
**Key Concepts**: Gram-Schmidt orthonormalization, rotation matrices, anatomical coordinate systems
