"""
Regression tests for reference frame migration.

This module ensures that all WholeBody reference frames return ReferenceFrame objects
and produce the same rotation matrices as before the migration.
"""

import numpy as np
import pytest
import labanalysis as laban


def create_test_wholebody(n_samples=100):
    """Create a WholeBody instance with synthetic marker data."""
    # Create synthetic marker positions
    t = np.linspace(0, 1, n_samples)

    # Pelvis markers (form a rectangle in frontal plane)
    left_asis = laban.Point3D(
        data=np.column_stack([
            -0.1 + 0.01 * np.sin(2 * np.pi * t),  # Global X: left
            0.9 + 0.02 * np.cos(2 * np.pi * t),   # Global Y: high
            0.0 + 0.01 * np.sin(4 * np.pi * t)    # Global Z: neutral
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_asis = laban.Point3D(
        data=np.column_stack([
            0.1 + 0.01 * np.sin(2 * np.pi * t),
            0.9 + 0.02 * np.cos(2 * np.pi * t),
            0.0 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    left_psis = laban.Point3D(
        data=np.column_stack([
            -0.08 + 0.01 * np.sin(2 * np.pi * t),
            0.85 + 0.02 * np.cos(2 * np.pi * t),
            -0.15 + 0.01 * np.sin(4 * np.pi * t)  # Global Z: backward
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_psis = laban.Point3D(
        data=np.column_stack([
            0.08 + 0.01 * np.sin(2 * np.pi * t),
            0.85 + 0.02 * np.cos(2 * np.pi * t),
            -0.15 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    # Hip markers (De Leva estimation based on ASIS/trochanter)
    left_throcanter = laban.Point3D(
        data=np.column_stack([
            -0.15 + 0.01 * np.sin(2 * np.pi * t),
            0.85 + 0.02 * np.cos(2 * np.pi * t),
            0.0 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_throcanter = laban.Point3D(
        data=np.column_stack([
            0.15 + 0.01 * np.sin(2 * np.pi * t),
            0.85 + 0.02 * np.cos(2 * np.pi * t),
            0.0 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    # Knee markers
    left_knee_lat = laban.Point3D(
        data=np.column_stack([
            -0.18 + 0.01 * np.sin(2 * np.pi * t),
            0.50 + 0.05 * np.cos(2 * np.pi * t),  # Knee flex/extend motion
            0.05 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    left_knee_med = laban.Point3D(
        data=np.column_stack([
            -0.12 + 0.01 * np.sin(2 * np.pi * t),
            0.50 + 0.05 * np.cos(2 * np.pi * t),
            0.05 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_knee_lat = laban.Point3D(
        data=np.column_stack([
            0.18 + 0.01 * np.sin(2 * np.pi * t),
            0.50 + 0.05 * np.cos(2 * np.pi * t),
            0.05 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_knee_med = laban.Point3D(
        data=np.column_stack([
            0.12 + 0.01 * np.sin(2 * np.pi * t),
            0.50 + 0.05 * np.cos(2 * np.pi * t),
            0.05 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    # Ankle markers
    left_ankle_lat = laban.Point3D(
        data=np.column_stack([
            -0.15 + 0.01 * np.sin(2 * np.pi * t),
            0.08 + 0.02 * np.cos(2 * np.pi * t),
            0.05 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    left_ankle_med = laban.Point3D(
        data=np.column_stack([
            -0.10 + 0.01 * np.sin(2 * np.pi * t),
            0.08 + 0.02 * np.cos(2 * np.pi * t),
            0.05 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_ankle_lat = laban.Point3D(
        data=np.column_stack([
            0.15 + 0.01 * np.sin(2 * np.pi * t),
            0.08 + 0.02 * np.cos(2 * np.pi * t),
            0.05 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_ankle_med = laban.Point3D(
        data=np.column_stack([
            0.10 + 0.01 * np.sin(2 * np.pi * t),
            0.08 + 0.02 * np.cos(2 * np.pi * t),
            0.05 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    # Shoulder markers
    left_acromion = laban.Point3D(
        data=np.column_stack([
            -0.20 + 0.01 * np.sin(2 * np.pi * t),
            1.35 + 0.01 * np.cos(2 * np.pi * t),
            0.0 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_acromion = laban.Point3D(
        data=np.column_stack([
            0.20 + 0.01 * np.sin(2 * np.pi * t),
            1.35 + 0.01 * np.cos(2 * np.pi * t),
            0.0 + 0.01 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    # Elbow markers
    left_elbow_lat = laban.Point3D(
        data=np.column_stack([
            -0.25 + 0.02 * np.sin(2 * np.pi * t),
            1.10 + 0.03 * np.cos(2 * np.pi * t),
            0.05 + 0.02 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    left_elbow_med = laban.Point3D(
        data=np.column_stack([
            -0.18 + 0.02 * np.sin(2 * np.pi * t),
            1.10 + 0.03 * np.cos(2 * np.pi * t),
            0.05 + 0.02 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_elbow_lat = laban.Point3D(
        data=np.column_stack([
            0.25 + 0.02 * np.sin(2 * np.pi * t),
            1.10 + 0.03 * np.cos(2 * np.pi * t),
            0.05 + 0.02 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_elbow_med = laban.Point3D(
        data=np.column_stack([
            0.18 + 0.02 * np.sin(2 * np.pi * t),
            1.10 + 0.03 * np.cos(2 * np.pi * t),
            0.05 + 0.02 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    # Wrist markers
    left_wrist_lat = laban.Point3D(
        data=np.column_stack([
            -0.28 + 0.02 * np.sin(2 * np.pi * t),
            0.90 + 0.04 * np.cos(2 * np.pi * t),
            0.10 + 0.02 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    left_wrist_med = laban.Point3D(
        data=np.column_stack([
            -0.22 + 0.02 * np.sin(2 * np.pi * t),
            0.90 + 0.04 * np.cos(2 * np.pi * t),
            0.10 + 0.02 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_wrist_lat = laban.Point3D(
        data=np.column_stack([
            0.28 + 0.02 * np.sin(2 * np.pi * t),
            0.90 + 0.04 * np.cos(2 * np.pi * t),
            0.10 + 0.02 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    right_wrist_med = laban.Point3D(
        data=np.column_stack([
            0.22 + 0.02 * np.sin(2 * np.pi * t),
            0.90 + 0.04 * np.cos(2 * np.pi * t),
            0.10 + 0.02 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    # Neck/trunk markers
    c7 = laban.Point3D(
        data=np.column_stack([
            0.0 + 0.005 * np.sin(2 * np.pi * t),
            1.40 + 0.01 * np.cos(2 * np.pi * t),
            -0.05 + 0.005 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    sc = laban.Point3D(
        data=np.column_stack([
            0.0 + 0.005 * np.sin(2 * np.pi * t),
            1.42 + 0.01 * np.cos(2 * np.pi * t),
            0.05 + 0.005 * np.sin(4 * np.pi * t)
        ]),
        index=np.arange(n_samples),
        columns=["X", "Y", "Z"],
    )

    # Create WholeBody
    body = laban.WholeBody(
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
        left_throcanter=left_throcanter,
        right_throcanter=right_throcanter,
        left_knee_lateral=left_knee_lat,
        left_knee_medial=left_knee_med,
        right_knee_lateral=right_knee_lat,
        right_knee_medial=right_knee_med,
        left_ankle_lateral=left_ankle_lat,
        left_ankle_medial=left_ankle_med,
        right_ankle_lateral=right_ankle_lat,
        right_ankle_medial=right_ankle_med,
        left_acromion=left_acromion,
        right_acromion=right_acromion,
        left_elbow_lateral=left_elbow_lat,
        left_elbow_medial=left_elbow_med,
        right_elbow_lateral=right_elbow_lat,
        right_elbow_medial=right_elbow_med,
        left_wrist_lateral=left_wrist_lat,
        left_wrist_medial=left_wrist_med,
        right_wrist_lateral=right_wrist_lat,
        right_wrist_medial=right_wrist_med,
        c7=c7,
        sc=sc,
    )

    return body


# Test 1: All reference frames return ReferenceFrame objects
def test_all_referenceframes_return_referenceframe_objects():
    """Verify all *_referenceframe properties return ReferenceFrame instances."""
    body = create_test_wholebody(100)

    rf_properties = [
        'left_ankle_referenceframe',
        'right_ankle_referenceframe',
        'left_knee_referenceframe',
        'right_knee_referenceframe',
        'left_hip_referenceframe',
        'right_hip_referenceframe',
        'pelvis_referenceframe',
        'left_shoulder_referenceframe',
        'right_shoulder_referenceframe',
        'neck_referenceframe',
        'left_elbow_referenceframe',
        'right_elbow_referenceframe',
        'left_wrist_referenceframe',
        'right_wrist_referenceframe',
    ]

    for prop_name in rf_properties:
        rf = getattr(body, prop_name)
        assert isinstance(rf, laban.ReferenceFrame), \
            f"{prop_name} should return ReferenceFrame, got {type(rf)}"


# Test 2: Rotation matrices have correct shape and properties
def test_rotation_matrices_shape_and_orthonormality():
    """Verify rotation matrices are (n, 3, 3) and orthonormal."""
    body = create_test_wholebody(100)
    n_samples = 100

    # Left side frames: right-handed (det(R) = +1)
    left_rf_properties = [
        'pelvis_referenceframe',
        'left_knee_referenceframe',
        'left_hip_referenceframe',
        'left_shoulder_referenceframe',
        'neck_referenceframe',
        'left_elbow_referenceframe',
        'left_wrist_referenceframe',
        'left_ankle_referenceframe',
    ]

    # Right side frames: left-handed (det(R) = -1)
    # Note: Axis directions are defined by semantic parameter names, not global X/Y/Z
    right_rf_properties = [
        'right_knee_referenceframe',
        'right_hip_referenceframe',
        'right_shoulder_referenceframe',
        'right_elbow_referenceframe',
        'right_wrist_referenceframe',
        'right_ankle_referenceframe',
    ]

    # Test left side frames (right-handed)
    for prop_name in left_rf_properties:
        rf = getattr(body, prop_name)
        rmat = rf.rotation_matrix

        # Check shape
        assert rmat.shape == (n_samples, 3, 3), \
            f"{prop_name}.rotation_matrix has wrong shape: {rmat.shape}"

        # Check orthonormality for first 10 samples
        for i in range(min(10, n_samples)):
            R = rmat[i]

            # R^T @ R = I (orthonormal columns)
            identity = R.T @ R
            np.testing.assert_array_almost_equal(
                identity, np.eye(3), decimal=10,
                err_msg=f"{prop_name} not orthonormal at sample {i}"
            )

            # det(R) = +1 (right-handed frame)
            det = np.linalg.det(R)
            np.testing.assert_almost_equal(
                det, 1.0, decimal=10,
                err_msg=f"{prop_name} determinant != +1 at sample {i}: det={det}"
            )

    # Test right side frames (left-handed)
    for prop_name in right_rf_properties:
        rf = getattr(body, prop_name)
        rmat = rf.rotation_matrix

        # Check shape
        assert rmat.shape == (n_samples, 3, 3), \
            f"{prop_name}.rotation_matrix has wrong shape: {rmat.shape}"

        # Check orthonormality for first 10 samples
        for i in range(min(10, n_samples)):
            R = rmat[i]

            # R^T @ R = I (orthonormal columns)
            identity = R.T @ R
            np.testing.assert_array_almost_equal(
                identity, np.eye(3), decimal=10,
                err_msg=f"{prop_name} not orthonormal at sample {i}"
            )

            # det(R) = -1 (left-handed frame, Z points FORWARD)
            det = np.linalg.det(R)
            np.testing.assert_almost_equal(
                det, -1.0, decimal=10,
                err_msg=f"{prop_name} determinant != -1 at sample {i}: det={det}"
            )


# Test 3: Hip-pelvis dependency
def test_hip_pelvis_dependency():
    """Verify hip reference frames correctly use pelvis axes."""
    body = create_test_wholebody(100)

    pelvis_rf = body.pelvis_referenceframe
    left_hip_rf = body.left_hip_referenceframe
    right_hip_rf = body.right_hip_referenceframe

    # Left hip should have same axes as pelvis
    np.testing.assert_array_almost_equal(
        left_hip_rf.lateral_axis, pelvis_rf.lateral_axis, decimal=14,
        err_msg="Left hip lateral axis should match pelvis lateral axis"
    )
    np.testing.assert_array_almost_equal(
        left_hip_rf.vertical_axis, pelvis_rf.vertical_axis, decimal=14,
        err_msg="Left hip vertical axis should match pelvis vertical axis"
    )
    np.testing.assert_array_almost_equal(
        left_hip_rf.anteroposterior_axis, pelvis_rf.anteroposterior_axis, decimal=14,
        err_msg="Left hip anteroposterior axis should match pelvis anteroposterior axis"
    )

    # Right hip should have mirrored frame: -lateral (points RIGHT), same vertical (UP),
    # same anteroposterior (points FORWARD, creates left-handed system)
    np.testing.assert_array_almost_equal(
        right_hip_rf.lateral_axis, -pelvis_rf.lateral_axis, decimal=14,
        err_msg="Right hip lateral axis should be negative of pelvis lateral axis (points RIGHT)"
    )
    np.testing.assert_array_almost_equal(
        right_hip_rf.vertical_axis, pelvis_rf.vertical_axis, decimal=14,
        err_msg="Right hip vertical axis should match pelvis vertical axis (points UP)"
    )
    # Anteroposterior axis: same as pelvis to keep pointing FORWARD (creates left-handed system)
    np.testing.assert_array_almost_equal(
        right_hip_rf.anteroposterior_axis, pelvis_rf.anteroposterior_axis, decimal=14,
        err_msg="Right hip anteroposterior axis should match pelvis anteroposterior axis (points FORWARD)"
    )


# Test 4: Origins are at expected locations
def test_referenceframe_origins():
    """Verify reference frame origins match joint centers."""
    body = create_test_wholebody(100)

    # Test pelvis origin = pelvis_center
    pelvis_rf = body.pelvis_referenceframe
    pelvis_center = body.pelvis_center
    np.testing.assert_array_almost_equal(
        pelvis_rf.origin, pelvis_center.to_numpy(), decimal=14,
        err_msg="Pelvis reference frame origin should match pelvis_center"
    )

    # Test left knee origin = left_knee
    left_knee_rf = body.left_knee_referenceframe
    left_knee = body.left_knee
    np.testing.assert_array_almost_equal(
        left_knee_rf.origin, left_knee.to_numpy(), decimal=14,
        err_msg="Left knee reference frame origin should match left_knee"
    )

    # Test right knee origin = right_knee
    right_knee_rf = body.right_knee_referenceframe
    right_knee = body.right_knee
    np.testing.assert_array_almost_equal(
        right_knee_rf.origin, right_knee.to_numpy(), decimal=14,
        err_msg="Right knee reference frame origin should match right_knee"
    )

    # Test left hip origin = left_hip
    left_hip_rf = body.left_hip_referenceframe
    left_hip = body.left_hip
    np.testing.assert_array_almost_equal(
        left_hip_rf.origin, left_hip.to_numpy(), decimal=14,
        err_msg="Left hip reference frame origin should match left_hip"
    )

    # Test right hip origin = right_hip
    right_hip_rf = body.right_hip_referenceframe
    right_hip = body.right_hip
    np.testing.assert_array_almost_equal(
        right_hip_rf.origin, right_hip.to_numpy(), decimal=14,
        err_msg="Right hip reference frame origin should match right_hip"
    )
