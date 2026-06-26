"""
Test to verify angle signs and symmetry after reference frame changes.

This test creates synthetic marker data representing known biomechanical movements
and verifies that:
1. Angle signs are correct (flexion positive, extension negative, etc.)
2. Left and right sides have symmetric angles for symmetric postures
3. The left-handed coordinate system on the right side produces correct angles
"""

import numpy as np
import pytest

import sys
import os
# Adjust path: now we're in test/records/, need to go up two levels to reach src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import labanalysis as laban


@pytest.fixture
def wholebody_neutral():
    """Create WholeBody in neutral anatomical position."""
    # Pelvis markers (rectangle in frontal plane)
    left_asis = laban.Point3D(
        data=np.array([[-0.10, 0.90, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_asis = laban.Point3D(
        data=np.array([[0.10, 0.90, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    left_psis = laban.Point3D(
        data=np.array([[-0.08, 0.85, -0.15]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_psis = laban.Point3D(
        data=np.array([[0.08, 0.85, -0.15]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Hip markers
    left_trochanter = laban.Point3D(
        data=np.array([[-0.15, 0.85, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_trochanter = laban.Point3D(
        data=np.array([[0.15, 0.85, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Knee markers - neutral (straight leg)
    left_knee_lat = laban.Point3D(
        data=np.array([[-0.18, 0.50, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    left_knee_med = laban.Point3D(
        data=np.array([[-0.12, 0.50, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_knee_lat = laban.Point3D(
        data=np.array([[0.18, 0.50, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_knee_med = laban.Point3D(
        data=np.array([[0.12, 0.50, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Ankle markers - neutral
    left_ankle_lat = laban.Point3D(
        data=np.array([[-0.15, 0.08, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    left_ankle_med = laban.Point3D(
        data=np.array([[-0.10, 0.08, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_ankle_lat = laban.Point3D(
        data=np.array([[0.15, 0.08, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_ankle_med = laban.Point3D(
        data=np.array([[0.10, 0.08, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Shoulder markers
    left_acromion = laban.Point3D(
        data=np.array([[-0.20, 1.35, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_acromion = laban.Point3D(
        data=np.array([[0.20, 1.35, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Elbow markers - neutral (straight arm)
    left_elbow_lat = laban.Point3D(
        data=np.array([[-0.25, 1.10, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    left_elbow_med = laban.Point3D(
        data=np.array([[-0.18, 1.10, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_elbow_lat = laban.Point3D(
        data=np.array([[0.25, 1.10, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_elbow_med = laban.Point3D(
        data=np.array([[0.18, 1.10, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Wrist markers - neutral
    left_wrist_lat = laban.Point3D(
        data=np.array([[-0.28, 0.85, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    left_wrist_med = laban.Point3D(
        data=np.array([[-0.22, 0.85, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_wrist_lat = laban.Point3D(
        data=np.array([[0.28, 0.85, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_wrist_med = laban.Point3D(
        data=np.array([[0.22, 0.85, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Neck markers
    c7 = laban.Point3D(
        data=np.array([[0.00, 1.40, -0.05]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    sc = laban.Point3D(
        data=np.array([[0.00, 1.42, 0.05]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    body = laban.WholeBody(
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
        left_trochanter=left_trochanter,
        right_trochanter=right_trochanter,
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


@pytest.fixture
def wholebody_knee_flexed():
    """Create WholeBody with knees flexed (positive angle expected)."""
    # Same as neutral but move ankle forward (Z positive)
    left_asis = laban.Point3D(data=np.array([[-0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_asis = laban.Point3D(data=np.array([[0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_psis = laban.Point3D(data=np.array([[-0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])
    right_psis = laban.Point3D(data=np.array([[0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])
    left_trochanter = laban.Point3D(data=np.array([[-0.15, 0.85, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_trochanter = laban.Point3D(data=np.array([[0.15, 0.85, 0.00]]), index=[0], columns=["X", "Y", "Z"])

    # Knees stay in place
    left_knee_lat = laban.Point3D(data=np.array([[-0.18, 0.50, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_knee_med = laban.Point3D(data=np.array([[-0.12, 0.50, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_knee_lat = laban.Point3D(data=np.array([[0.18, 0.50, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_knee_med = laban.Point3D(data=np.array([[0.12, 0.50, 0.00]]), index=[0], columns=["X", "Y", "Z"])

    # Ankles move forward (positive Z) to create knee flexion
    left_ankle_lat = laban.Point3D(data=np.array([[-0.15, 0.08, 0.20]]), index=[0], columns=["X", "Y", "Z"])
    left_ankle_med = laban.Point3D(data=np.array([[-0.10, 0.08, 0.20]]), index=[0], columns=["X", "Y", "Z"])
    right_ankle_lat = laban.Point3D(data=np.array([[0.15, 0.08, 0.20]]), index=[0], columns=["X", "Y", "Z"])
    right_ankle_med = laban.Point3D(data=np.array([[0.10, 0.08, 0.20]]), index=[0], columns=["X", "Y", "Z"])

    left_acromion = laban.Point3D(data=np.array([[-0.20, 1.35, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_acromion = laban.Point3D(data=np.array([[0.20, 1.35, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_elbow_lat = laban.Point3D(data=np.array([[-0.25, 1.10, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_elbow_med = laban.Point3D(data=np.array([[-0.18, 1.10, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_lat = laban.Point3D(data=np.array([[0.25, 1.10, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_med = laban.Point3D(data=np.array([[0.18, 1.10, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_wrist_lat = laban.Point3D(data=np.array([[-0.28, 0.85, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_wrist_med = laban.Point3D(data=np.array([[-0.22, 0.85, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_wrist_lat = laban.Point3D(data=np.array([[0.28, 0.85, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_wrist_med = laban.Point3D(data=np.array([[0.22, 0.85, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    c7 = laban.Point3D(data=np.array([[0.00, 1.40, -0.05]]), index=[0], columns=["X", "Y", "Z"])
    sc = laban.Point3D(data=np.array([[0.00, 1.42, 0.05]]), index=[0], columns=["X", "Y", "Z"])

    body = laban.WholeBody(
        left_asis=left_asis, right_asis=right_asis, left_psis=left_psis, right_psis=right_psis,
        left_trochanter=left_trochanter, right_trochanter=right_trochanter,
        left_knee_lateral=left_knee_lat, left_knee_medial=left_knee_med,
        right_knee_lateral=right_knee_lat, right_knee_medial=right_knee_med,
        left_ankle_lateral=left_ankle_lat, left_ankle_medial=left_ankle_med,
        right_ankle_lateral=right_ankle_lat, right_ankle_medial=right_ankle_med,
        left_acromion=left_acromion, right_acromion=right_acromion,
        left_elbow_lateral=left_elbow_lat, left_elbow_medial=left_elbow_med,
        right_elbow_lateral=right_elbow_lat, right_elbow_medial=right_elbow_med,
        left_wrist_lateral=left_wrist_lat, left_wrist_medial=left_wrist_med,
        right_wrist_lateral=right_wrist_lat, right_wrist_medial=right_wrist_med,
        c7=c7, sc=sc,
    )
    return body


@pytest.fixture
def wholebody_elbow_flexed():
    """Create WholeBody with elbows flexed (positive angle expected)."""
    left_asis = laban.Point3D(data=np.array([[-0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_asis = laban.Point3D(data=np.array([[0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_psis = laban.Point3D(data=np.array([[-0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])
    right_psis = laban.Point3D(data=np.array([[0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])
    left_trochanter = laban.Point3D(data=np.array([[-0.15, 0.85, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_trochanter = laban.Point3D(data=np.array([[0.15, 0.85, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_knee_lat = laban.Point3D(data=np.array([[-0.18, 0.50, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_knee_med = laban.Point3D(data=np.array([[-0.12, 0.50, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_knee_lat = laban.Point3D(data=np.array([[0.18, 0.50, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_knee_med = laban.Point3D(data=np.array([[0.12, 0.50, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_ankle_lat = laban.Point3D(data=np.array([[-0.15, 0.08, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_ankle_med = laban.Point3D(data=np.array([[-0.10, 0.08, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_ankle_lat = laban.Point3D(data=np.array([[0.15, 0.08, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_ankle_med = laban.Point3D(data=np.array([[0.10, 0.08, 0.00]]), index=[0], columns=["X", "Y", "Z"])

    # Shoulders stay in place
    left_acromion = laban.Point3D(data=np.array([[-0.20, 1.35, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_acromion = laban.Point3D(data=np.array([[0.20, 1.35, 0.00]]), index=[0], columns=["X", "Y", "Z"])

    # Elbows stay in place
    left_elbow_lat = laban.Point3D(data=np.array([[-0.25, 1.10, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_elbow_med = laban.Point3D(data=np.array([[-0.18, 1.10, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_lat = laban.Point3D(data=np.array([[0.25, 1.10, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_med = laban.Point3D(data=np.array([[0.18, 1.10, 0.00]]), index=[0], columns=["X", "Y", "Z"])

    # Wrists move forward (positive Z) to create elbow flexion
    left_wrist_lat = laban.Point3D(data=np.array([[-0.28, 0.85, 0.15]]), index=[0], columns=["X", "Y", "Z"])
    left_wrist_med = laban.Point3D(data=np.array([[-0.22, 0.85, 0.15]]), index=[0], columns=["X", "Y", "Z"])
    right_wrist_lat = laban.Point3D(data=np.array([[0.28, 0.85, 0.15]]), index=[0], columns=["X", "Y", "Z"])
    right_wrist_med = laban.Point3D(data=np.array([[0.22, 0.85, 0.15]]), index=[0], columns=["X", "Y", "Z"])

    c7 = laban.Point3D(data=np.array([[0.00, 1.40, -0.05]]), index=[0], columns=["X", "Y", "Z"])
    sc = laban.Point3D(data=np.array([[0.00, 1.42, 0.05]]), index=[0], columns=["X", "Y", "Z"])

    body = laban.WholeBody(
        left_asis=left_asis, right_asis=right_asis, left_psis=left_psis, right_psis=right_psis,
        left_trochanter=left_trochanter, right_trochanter=right_trochanter,
        left_knee_lateral=left_knee_lat, left_knee_medial=left_knee_med,
        right_knee_lateral=right_knee_lat, right_knee_medial=right_knee_med,
        left_ankle_lateral=left_ankle_lat, left_ankle_medial=left_ankle_med,
        right_ankle_lateral=right_ankle_lat, right_ankle_medial=right_ankle_med,
        left_acromion=left_acromion, right_acromion=right_acromion,
        left_elbow_lateral=left_elbow_lat, left_elbow_medial=left_elbow_med,
        right_elbow_lateral=right_elbow_lat, right_elbow_medial=right_elbow_med,
        left_wrist_lateral=left_wrist_lat, left_wrist_medial=left_wrist_med,
        right_wrist_lateral=right_wrist_lat, right_wrist_medial=right_wrist_med,
        c7=c7, sc=sc,
    )
    return body


def test_reference_frame_handedness(wholebody_neutral):
    """Test that reference frames have correct handedness (left=right-handed, right=left-handed)."""
    # Check reference frames
    left_knee_rf = wholebody_neutral.left_knee_referenceframe
    right_knee_rf = wholebody_neutral.right_knee_referenceframe
    left_elbow_rf = wholebody_neutral.left_elbow_referenceframe
    right_elbow_rf = wholebody_neutral.right_elbow_referenceframe

    left_knee_det = np.linalg.det(left_knee_rf.rotation_matrix[0])
    right_knee_det = np.linalg.det(right_knee_rf.rotation_matrix[0])
    left_elbow_det = np.linalg.det(left_elbow_rf.rotation_matrix[0])
    right_elbow_det = np.linalg.det(right_elbow_rf.rotation_matrix[0])

    # Left side should be right-handed (det = +1)
    assert np.isclose(left_knee_det, 1.0, atol=1e-6), f"Left knee det(R) = {left_knee_det}, expected +1"
    assert np.isclose(left_elbow_det, 1.0, atol=1e-6), f"Left elbow det(R) = {left_elbow_det}, expected +1"

    # Right side should be left-handed (det = -1)
    assert np.isclose(right_knee_det, -1.0, atol=1e-6), f"Right knee det(R) = {right_knee_det}, expected -1"
    assert np.isclose(right_elbow_det, -1.0, atol=1e-6), f"Right elbow det(R) = {right_elbow_det}, expected -1"


def test_anteroposterior_axis_points_forward(wholebody_neutral):
    """Test that anteroposterior axis (Z) points forward on both sides."""
    left_knee_rf = wholebody_neutral.left_knee_referenceframe
    right_knee_rf = wholebody_neutral.right_knee_referenceframe
    left_elbow_rf = wholebody_neutral.left_elbow_referenceframe
    right_elbow_rf = wholebody_neutral.right_elbow_referenceframe

    # Get rotation matrix and extract anteroposterior column (third column, index 2)
    left_knee_z = left_knee_rf.rotation_matrix[0, :, 2]
    right_knee_z = right_knee_rf.rotation_matrix[0, :, 2]
    left_elbow_z = left_elbow_rf.rotation_matrix[0, :, 2]
    right_elbow_z = right_elbow_rf.rotation_matrix[0, :, 2]

    # Z component (index 2) should be negative (pointing forward in our coordinate system where +Z is back)
    # This verifies the axis points in the correct anatomical direction
    assert left_knee_z[2] < 0, f"Left knee Z-axis should point forward (Z<0), got {left_knee_z}"
    assert right_knee_z[2] < 0, f"Right knee Z-axis should point forward (Z<0), got {right_knee_z}"
    assert left_elbow_z[2] < 0, f"Left elbow Z-axis should point forward (Z<0), got {left_elbow_z}"
    assert right_elbow_z[2] < 0, f"Right elbow Z-axis should point forward (Z<0), got {right_elbow_z}"


def test_neutral_position_angles_near_zero(wholebody_neutral):
    """Test that angles in neutral position are close to zero."""
    left_knee_angle = wholebody_neutral.left_knee_flexionextension.to_numpy()[0, 0]
    right_knee_angle = wholebody_neutral.right_knee_flexionextension.to_numpy()[0, 0]
    left_elbow_angle = wholebody_neutral.left_elbow_flexionextension.to_numpy()[0, 0]
    right_elbow_angle = wholebody_neutral.right_elbow_flexionextension.to_numpy()[0, 0]

    # Neutral position should have angles close to 0° (within 15° tolerance for synthetic data)
    assert abs(left_knee_angle) < 15, f"Left knee neutral angle should be ~0°, got {left_knee_angle:.2f}°"
    assert abs(right_knee_angle) < 15, f"Right knee neutral angle should be ~0°, got {right_knee_angle:.2f}°"
    assert abs(left_elbow_angle) < 15, f"Left elbow neutral angle should be ~0°, got {left_elbow_angle:.2f}°"
    assert abs(right_elbow_angle) < 15, f"Right elbow neutral angle should be ~0°, got {right_elbow_angle:.2f}°"


def test_knee_flexion_positive_sign(wholebody_knee_flexed):
    """Test that knee flexion produces positive angles on both sides."""
    left_knee_angle = wholebody_knee_flexed.left_knee_flexionextension.to_numpy()[0, 0]
    right_knee_angle = wholebody_knee_flexed.right_knee_flexionextension.to_numpy()[0, 0]

    # Flexion should be positive
    assert left_knee_angle > 0, f"Left knee flexion should be positive, got {left_knee_angle:.2f}°"
    assert right_knee_angle > 0, f"Right knee flexion should be positive, got {right_knee_angle:.2f}°"

    # Angles should be similar (symmetric movement)
    angle_diff = abs(left_knee_angle - right_knee_angle)
    assert angle_diff < 15, f"Left/right knee angles should be similar, difference: {angle_diff:.2f}°"


def test_elbow_flexion_positive_sign(wholebody_elbow_flexed):
    """Test that elbow flexion produces positive angles on both sides."""
    left_elbow_angle = wholebody_elbow_flexed.left_elbow_flexionextension.to_numpy()[0, 0]
    right_elbow_angle = wholebody_elbow_flexed.right_elbow_flexionextension.to_numpy()[0, 0]

    # Flexion should be positive
    assert left_elbow_angle > 0, f"Left elbow flexion should be positive, got {left_elbow_angle:.2f}°"
    assert right_elbow_angle > 0, f"Right elbow flexion should be positive, got {right_elbow_angle:.2f}°"

    # Angles should be identical for symmetric movement
    angle_diff = abs(left_elbow_angle - right_elbow_angle)
    assert angle_diff < 5, f"Left/right elbow angles should be identical, difference: {angle_diff:.2f}°"
