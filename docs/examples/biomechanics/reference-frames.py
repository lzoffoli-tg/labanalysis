"""
Reference Frame Transformations Example
========================================

Demonstrates how to work with anatomical reference frames using semantic axis naming.

Topics covered:
1. Accessing built-in reference frames from WholeBody
2. Inspecting semantic axis properties
3. Transforming vectors into local reference frames
4. Creating custom reference frames
5. Manual angle calculation from transformed coordinates
6. Comparing automatic vs manual calculations

Author: labanalysis team
Date: 2026-06-25
"""

import labanalysis as laban
import numpy as np
import pandas as pd


def main():
    """Main example demonstrating reference frame transformations."""

    # ====================================================================
    # Section 1: Load Data and Create WholeBody
    # ====================================================================
    print("=" * 70)
    print("Section 1: Loading Motion Capture Data")
    print("=" * 70)

    # Load a trial file (replace with your actual file path)
    # body = laban.WholeBody.from_tdf_file("walking_trial.tdf")
    # For this example, we'll create synthetic data
    body = create_synthetic_wholebody()

    print(f"✓ Loaded WholeBody with {body.left_knee.shape[0]} frames")
    print(f"✓ Sample rate: {body.left_knee.sample_rate} Hz")
    print()

    # ====================================================================
    # Section 2: Access Built-in Reference Frames
    # ====================================================================
    print("=" * 70)
    print("Section 2: Accessing Built-in Reference Frames")
    print("=" * 70)

    # WholeBody provides pre-defined anatomical reference frames
    pelvis_rf = body.pelvis_referenceframe
    left_knee_rf = body.left_knee_referenceframe
    right_knee_rf = body.right_knee_referenceframe

    print("Available built-in reference frames:")
    print("  - pelvis_referenceframe")
    print("  - left_hip_referenceframe / right_hip_referenceframe")
    print("  - left_knee_referenceframe / right_knee_referenceframe")
    print("  - left_shoulder_referenceframe / right_shoulder_referenceframe")
    print("  - left_elbow_referenceframe / right_elbow_referenceframe")
    print()

    # ====================================================================
    # Section 3: Inspect Semantic Axis Properties
    # ====================================================================
    print("=" * 70)
    print("Section 3: Inspecting Semantic Axis Properties")
    print("=" * 70)

    # Get semantic axes from pelvis reference frame
    print("Pelvis reference frame (first frame):")
    print(f"  Origin: {pelvis_rf.origin[0]}")
    print(f"  Lateral axis: {pelvis_rf.lateral_axis[0]}")
    print(f"  Vertical axis: {pelvis_rf.vertical_axis[0]}")
    print()

    # The rotation matrix has semantic axes as columns:
    # These columns represent directions in the global coordinate system
    print(f"Rotation matrix shape: {pelvis_rf.rotation_matrix.shape}")
    print("Rotation matrix columns (semantic axes in global coordinates):")
    print(f"  Column 0 = lateral_axis: {pelvis_rf.rotation_matrix[0, :, 0]}")
    print(f"  Column 1 = vertical_axis: {pelvis_rf.rotation_matrix[0, :, 1]}")
    print(f"  Column 2 = anteroposterior_axis: {pelvis_rf.rotation_matrix[0, :, 2]}")
    print()

    # Check handedness (left vs right side frames)
    left_det = np.linalg.det(left_knee_rf.rotation_matrix[0])
    right_det = np.linalg.det(right_knee_rf.rotation_matrix[0])

    print("Reference frame handedness:")
    print(f"  Left knee determinant: {left_det:+.1f} (right-handed)")
    print(f"  Right knee determinant: {right_det:+.1f} (left-handed)")
    print("  → This ensures anteroposterior_axis points FORWARD on both sides")
    print()

    # ====================================================================
    # Section 4: Transform Vectors into Reference Frames
    # ====================================================================
    print("=" * 70)
    print("Section 4: Transforming Vectors with einsum")
    print("=" * 70)

    # Get left hip position in pelvis reference frame
    left_hip_global = body.left_hip.to_numpy()

    # Transform to pelvis frame using einsum
    left_hip_vec = left_hip_global - pelvis_rf.origin
    left_hip_pelvis = np.einsum("nij,nj->ni", pelvis_rf.rotation_matrix, left_hip_vec)

    # Components extracted by index correspond to semantic axes:
    print("Left hip in pelvis reference frame:")
    print(f"  Index [0] = lateral_axis component: {left_hip_pelvis[:, 0].mean():.3f} m")
    print(f"  Index [1] = vertical_axis component: {left_hip_pelvis[:, 1].mean():.3f} m")
    print(f"  Index [2] = anteroposterior_axis component: {left_hip_pelvis[:, 2].mean():.3f} m")
    print()

    # The einsum formula: "nij,nj->ni"
    # n = time samples (frame index)
    # i,j = spatial dimensions (3D)
    # Operation: For each frame n, multiply rotation matrix [i,j] by vector [j]
    print("Understanding einsum formula: 'nij,nj->ni'")
    print("  Equivalent to: result[n, i] = sum_j(rmat[n, i, j] * vec[n, j])")
    print("  This is a vectorized matrix-vector multiplication across time")
    print()

    # ====================================================================
    # Section 5: Create Custom Reference Frame
    # ====================================================================
    print("=" * 70)
    print("Section 5: Creating Custom Reference Frames")
    print("=" * 70)

    # Example: Create a custom thigh reference frame
    # NEW: Point3D objects accepted directly (no .to_numpy() required)
    lateral_axis = body.left_hip - body.right_hip  # Point3D from subtraction
    vertical_axis = body.left_knee - body.left_hip  # Point3D

    # Create reference frame with semantic parameters
    thigh_rf = laban.ReferenceFrame(
        origin=body.left_hip,           # Point3D
        lateral_axis=lateral_axis,       # Point3D (mediolateral)
        vertical_axis=vertical_axis      # Point3D (will be orthonormalized)
        # anteroposterior_axis computed automatically
    )

    print("✨ NEW: Point3D objects accepted directly (no .to_numpy() required)")

    print("Custom thigh reference frame created:")
    print(f"  Origin: Hip joint center")
    print(f"  Lateral axis: From right hip to left hip (mediolateral)")
    print(f"  Vertical axis: From hip to knee (orthonormalized)")
    print(f"  Anteroposterior axis: Computed as lateral × vertical")
    print()

    # Transform knee to thigh frame
    knee_vec = (body.left_knee - body.left_hip).to_numpy()
    knee_local = np.einsum("nij,nj->ni", thigh_rf.rotation_matrix, knee_vec)

    # Components extracted by index correspond to semantic axes:
    print("Knee in thigh reference frame:")
    print(f"  Index [0] = lateral_axis component: {knee_local[:, 0].mean():.3f} m")
    print(f"  Index [1] = vertical_axis component (thigh length): {knee_local[:, 1].mean():.3f} m")
    print(f"  Index [2] = anteroposterior_axis component: {knee_local[:, 2].mean():.3f} m")
    print()

    # ====================================================================
    # Section 6: Manual Angle Calculation
    # ====================================================================
    print("=" * 70)
    print("Section 6: Manual Angle Calculation from Reference Frames")
    print("=" * 70)

    # Calculate knee flexion angle manually
    knee_rf = body.left_knee_referenceframe
    ankle_vec = (body.left_ankle - body.left_knee).to_numpy()

    # Transform ankle to knee reference frame
    ankle_local = np.einsum("nij,nj->ni", knee_rf.rotation_matrix, ankle_vec)

    # Extract components for sagittal plane angle
    anteroposterior = ankle_local[:, 2]  # Forward-backward
    vertical = ankle_local[:, 1]  # Up-down

    # Knee flexion angle in sagittal plane
    flexion_rad = np.arctan2(-anteroposterior, -vertical)
    flexion_deg = np.degrees(flexion_rad)

    # Compare with automatic calculation
    auto_flexion = body.left_knee_flexionextension.to_numpy()

    print("Knee flexion angle comparison:")
    print(f"  Manual calculation: {flexion_deg.mean():.1f}° ± {flexion_deg.std():.1f}°")
    print(f"  Automatic calculation: {auto_flexion.mean():.1f}° ± {auto_flexion.std():.1f}°")
    print(f"  Difference: {abs(flexion_deg.mean() - auto_flexion.mean()):.2f}°")
    print()

    # ====================================================================
    # Section 7: Coordinate System Independence
    # ====================================================================
    print("=" * 70)
    print("Section 7: Coordinate System Independence")
    print("=" * 70)

    print("Key insight: Semantic axis naming makes code coordinate-independent")
    print()
    print("After transformation with ReferenceFrame:")
    print("  Index [0] ALWAYS = lateral_axis component")
    print("  Index [1] ALWAYS = vertical_axis component")
    print("  Index [2] ALWAYS = anteroposterior_axis component")
    print()
    print("This mapping is fixed by ReferenceFrame construction,")
    print("NOT by global X/Y/Z coordinate configuration.")
    print()
    print("The global X/Y/Z values of axes depend on frame orientation;")
    print("semantic meaning (lateral/vertical/anteroposterior) is what matters.")
    print()
    print("Your code works the same whether global coordinates are configured as:")
    print("  - global Y = vertical (default)")
    print("  - global X = vertical (non-standard)")
    print("  - global Z = vertical (alternative)")
    print()

    # ====================================================================
    # Summary
    # ====================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Accessed built-in reference frames from WholeBody")
    print("✓ Inspected semantic axis properties (lateral/vertical/anteroposterior)")
    print("✓ Transformed vectors using einsum formula")
    print("✓ Created custom reference frame with semantic parameters")
    print("✓ Calculated angles manually from transformed coordinates")
    print("✓ Verified coordinate system independence")
    print()
    print("See also:")
    print("  - Tutorial: Custom Reference Frames (docs/tutorials/09-custom-reference-frames.md)")
    print("  - User Guide: Coordinate Systems (docs/user-guide/biomechanics/coordinate-systems.md)")
    print()


def create_synthetic_wholebody():
    """Create synthetic WholeBody for demonstration purposes."""
    from labanalysis.records.timeseries import Point3D

    # Create time vector
    N = 100  # 100 frames
    t = np.linspace(0, 1, N)  # 1 second at 100 Hz
    sample_rate = 100.0

    # Synthetic markers in anatomical position with slight knee flexion
    markers = {}

    # Pelvis markers (in meters)
    markers['left_asis'] = Point3D(
        X=np.full(N, 0.1), Y=np.full(N, 1.0), Z=np.full(N, 0.0),
        sample_rate=sample_rate
    )
    markers['right_asis'] = Point3D(
        X=np.full(N, -0.1), Y=np.full(N, 1.0), Z=np.full(N, 0.0),
        sample_rate=sample_rate
    )
    markers['left_psis'] = Point3D(
        X=np.full(N, 0.1), Y=np.full(N, 1.0), Z=np.full(N, -0.1),
        sample_rate=sample_rate
    )
    markers['right_psis'] = Point3D(
        X=np.full(N, -0.1), Y=np.full(N, 1.0), Z=np.full(N, -0.1),
        sample_rate=sample_rate
    )

    # Hip markers
    markers['left_hip'] = Point3D(
        X=np.full(N, 0.1), Y=np.full(N, 0.9), Z=np.full(N, 0.0),
        sample_rate=sample_rate
    )
    markers['right_hip'] = Point3D(
        X=np.full(N, -0.1), Y=np.full(N, 0.9), Z=np.full(N, 0.0),
        sample_rate=sample_rate
    )

    # Knee markers (with slight flexion varying over time)
    knee_flexion = 10 + 5 * np.sin(2 * np.pi * t)  # 10° ± 5° flexion
    knee_offset_z = 0.4 * np.cos(np.radians(knee_flexion))  # AP component
    knee_offset_y = -0.4 * np.sin(np.radians(knee_flexion))  # Vertical component

    markers['left_knee'] = Point3D(
        X=np.full(N, 0.1),
        Y=0.9 + knee_offset_y,
        Z=knee_offset_z,
        sample_rate=sample_rate
    )
    markers['right_knee'] = Point3D(
        X=np.full(N, -0.1),
        Y=0.9 + knee_offset_y,
        Z=knee_offset_z,
        sample_rate=sample_rate
    )

    # Ankle markers
    markers['left_ankle'] = Point3D(
        X=np.full(N, 0.1),
        Y=np.full(N, 0.1),
        Z=np.full(N, 0.0),
        sample_rate=sample_rate
    )
    markers['right_ankle'] = Point3D(
        X=np.full(N, -0.1),
        Y=np.full(N, 0.1),
        Z=np.full(N, 0.0),
        sample_rate=sample_rate
    )

    # Shoulder markers
    markers['left_shoulder'] = Point3D(
        X=np.full(N, 0.2), Y=np.full(N, 1.4), Z=np.full(N, 0.0),
        sample_rate=sample_rate
    )
    markers['right_shoulder'] = Point3D(
        X=np.full(N, -0.2), Y=np.full(N, 1.4), Z=np.full(N, 0.0),
        sample_rate=sample_rate
    )

    # Elbow markers
    markers['left_elbow'] = Point3D(
        X=np.full(N, 0.2), Y=np.full(N, 1.1), Z=np.full(N, 0.1),
        sample_rate=sample_rate
    )
    markers['right_elbow'] = Point3D(
        X=np.full(N, -0.2), Y=np.full(N, 1.1), Z=np.full(N, 0.1),
        sample_rate=sample_rate
    )

    # Create WholeBody from markers
    body = laban.WholeBody(**markers)

    return body


if __name__ == "__main__":
    main()
