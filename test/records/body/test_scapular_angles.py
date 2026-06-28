"""
Test scapular protraction/retraction angle calculations.

This test creates synthetic marker data representing different shoulder postures
and verifies that:
1. Neutral posture produces small positive angles (~5-15°) for natural protraction
2. Protraction (forward shoulders) produces positive angles
3. Retraction (backward shoulders) produces negative angles
4. Left and right sides have symmetric angles for symmetric postures
5. Sign convention is consistent: positive = protraction, negative = retraction
"""

import numpy as np
import pytest

from labanalysis.timeseries import Point3D, Signal1D, Timeseries
from labanalysis.records.body import WholeBody



@pytest.fixture
def wholebody_shoulders_neutral():
    """Create WholeBody with shoulders in neutral anatomical position (natural slight protraction)."""
    # Neck markers
    # C7 is posterior (backward), SC is anterior (forward)
    # Global coordinate system used for synthetic markers: +Z is backward, -Z is forward
    # Note: Local reference frame axes may differ - see ReferenceFrame construction
    c7 = Point3D(
        data=np.array([[0.00, 1.40, 0.05]]),  # Posterior: +Z
        index=[0],
        columns=["X", "Y", "Z"],
    )
    sc = Point3D(
        data=np.array([[0.00, 1.42, -0.05]]),  # Anterior: -Z
        index=[0],
        columns=["X", "Y", "Z"],
    )
    # neck_base will be at (0.0, 1.41, 0.0)

    # Shoulders in neutral position (slightly forward = natural protraction)
    # Global coords: Z = -0.03 means slightly forward (negative Z is forward)
    left_acromion = Point3D(
        data=np.array([[-0.20, 1.35, -0.03]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_acromion = Point3D(
        data=np.array([[0.20, 1.35, -0.03]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Elbow markers (needed for shoulder calculation via De Leva regression)
    left_elbow_lat = Point3D(
        data=np.array([[-0.25, 1.10, -0.02]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    left_elbow_med = Point3D(
        data=np.array([[-0.18, 1.10, -0.02]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_elbow_lat = Point3D(
        data=np.array([[0.25, 1.10, -0.02]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_elbow_med = Point3D(
        data=np.array([[0.18, 1.10, -0.02]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Pelvis markers (needed for reference frames)
    left_asis = Point3D(
        data=np.array([[-0.10, 0.90, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_asis = Point3D(
        data=np.array([[0.10, 0.90, 0.00]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    left_psis = Point3D(
        data=np.array([[-0.08, 0.85, -0.15]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_psis = Point3D(
        data=np.array([[0.08, 0.85, -0.15]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    body = WholeBody(
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
        left_acromion=left_acromion,
        right_acromion=right_acromion,
        left_elbow_lateral=left_elbow_lat,
        left_elbow_medial=left_elbow_med,
        right_elbow_lateral=right_elbow_lat,
        right_elbow_medial=right_elbow_med,
        c7=c7,
        sc=sc,
    )
    return body


@pytest.fixture
def wholebody_shoulders_protracted():
    """Create WholeBody with both shoulders protracted (forward) - rounded posture."""
    c7 = Point3D(data=np.array([[0.00, 1.40, 0.05]]), index=[0], columns=["X", "Y", "Z"])  # Posterior: +Z
    sc = Point3D(data=np.array([[0.00, 1.42, -0.05]]), index=[0], columns=["X", "Y", "Z"])  # Anterior: -Z

    # Shoulders moved significantly forward (Z = -0.12)
    left_acromion = Point3D(
        data=np.array([[-0.20, 1.35, -0.12]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_acromion = Point3D(
        data=np.array([[0.20, 1.35, -0.12]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Elbow markers
    left_elbow_lat = Point3D(data=np.array([[-0.25, 1.10, -0.10]]), index=[0], columns=["X", "Y", "Z"])
    left_elbow_med = Point3D(data=np.array([[-0.18, 1.10, -0.10]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_lat = Point3D(data=np.array([[0.25, 1.10, -0.10]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_med = Point3D(data=np.array([[0.18, 1.10, -0.10]]), index=[0], columns=["X", "Y", "Z"])

    left_asis = Point3D(data=np.array([[-0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_asis = Point3D(data=np.array([[0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_psis = Point3D(data=np.array([[-0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])
    right_psis = Point3D(data=np.array([[0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])

    body = WholeBody(
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
        left_acromion=left_acromion,
        right_acromion=right_acromion,
        left_elbow_lateral=left_elbow_lat,
        left_elbow_medial=left_elbow_med,
        right_elbow_lateral=right_elbow_lat,
        right_elbow_medial=right_elbow_med,
        c7=c7,
        sc=sc,
    )
    return body


@pytest.fixture
def wholebody_shoulders_retracted():
    """Create WholeBody with both shoulders retracted (backward) - military posture."""
    c7 = Point3D(data=np.array([[0.00, 1.40, 0.05]]), index=[0], columns=["X", "Y", "Z"])  # Posterior: +Z
    sc = Point3D(data=np.array([[0.00, 1.42, -0.05]]), index=[0], columns=["X", "Y", "Z"])  # Anterior: -Z

    # Shoulders moved backward (Z = +0.08)
    left_acromion = Point3D(
        data=np.array([[-0.20, 1.35, 0.08]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    right_acromion = Point3D(
        data=np.array([[0.20, 1.35, 0.08]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Elbow markers
    left_elbow_lat = Point3D(data=np.array([[-0.25, 1.10, 0.06]]), index=[0], columns=["X", "Y", "Z"])
    left_elbow_med = Point3D(data=np.array([[-0.18, 1.10, 0.06]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_lat = Point3D(data=np.array([[0.25, 1.10, 0.06]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_med = Point3D(data=np.array([[0.18, 1.10, 0.06]]), index=[0], columns=["X", "Y", "Z"])

    left_asis = Point3D(data=np.array([[-0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_asis = Point3D(data=np.array([[0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_psis = Point3D(data=np.array([[-0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])
    right_psis = Point3D(data=np.array([[0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])

    body = WholeBody(
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
        left_acromion=left_acromion,
        right_acromion=right_acromion,
        left_elbow_lateral=left_elbow_lat,
        left_elbow_medial=left_elbow_med,
        right_elbow_lateral=right_elbow_lat,
        right_elbow_medial=right_elbow_med,
        c7=c7,
        sc=sc,
    )
    return body


@pytest.fixture
def wholebody_shoulders_asymmetric():
    """Create WholeBody with left shoulder protracted, right shoulder retracted."""
    c7 = Point3D(data=np.array([[0.00, 1.40, 0.05]]), index=[0], columns=["X", "Y", "Z"])  # Posterior: +Z
    sc = Point3D(data=np.array([[0.00, 1.42, -0.05]]), index=[0], columns=["X", "Y", "Z"])  # Anterior: -Z

    # Left shoulder forward (protracted)
    left_acromion = Point3D(
        data=np.array([[-0.20, 1.35, -0.10]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    # Right shoulder backward (retracted)
    right_acromion = Point3D(
        data=np.array([[0.20, 1.35, 0.08]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    # Elbow markers
    left_elbow_lat = Point3D(data=np.array([[-0.25, 1.10, -0.08]]), index=[0], columns=["X", "Y", "Z"])
    left_elbow_med = Point3D(data=np.array([[-0.18, 1.10, -0.08]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_lat = Point3D(data=np.array([[0.25, 1.10, 0.06]]), index=[0], columns=["X", "Y", "Z"])
    right_elbow_med = Point3D(data=np.array([[0.18, 1.10, 0.06]]), index=[0], columns=["X", "Y", "Z"])

    left_asis = Point3D(data=np.array([[-0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    right_asis = Point3D(data=np.array([[0.10, 0.90, 0.00]]), index=[0], columns=["X", "Y", "Z"])
    left_psis = Point3D(data=np.array([[-0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])
    right_psis = Point3D(data=np.array([[0.08, 0.85, -0.15]]), index=[0], columns=["X", "Y", "Z"])

    body = WholeBody(
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
        left_acromion=left_acromion,
        right_acromion=right_acromion,
        left_elbow_lateral=left_elbow_lat,
        left_elbow_medial=left_elbow_med,
        right_elbow_lateral=right_elbow_lat,
        right_elbow_medial=right_elbow_med,
        c7=c7,
        sc=sc,
    )
    return body


def test_neutral_position_slight_protraction(wholebody_shoulders_neutral):
    """Test that neutral anatomical position produces small positive angles (~5-15°)."""
    left_angle = wholebody_shoulders_neutral.left_scapular_protractionretraction.to_numpy()[0, 0]
    right_angle = wholebody_shoulders_neutral.right_scapular_protractionretraction.to_numpy()[0, 0]

    # Neutral position should have slight protraction (5-15°)
    assert 0 < left_angle < 20, f"Left shoulder neutral angle should be ~5-15°, got {left_angle:.2f}°"
    assert 0 < right_angle < 20, f"Right shoulder neutral angle should be ~5-15°, got {right_angle:.2f}°"

    # Angles should be symmetric for symmetric posture
    angle_diff = abs(left_angle - right_angle)
    assert angle_diff < 2, f"Left/right angles should be similar, difference: {angle_diff:.2f}°"


def test_protraction_positive_sign(wholebody_shoulders_protracted):
    """Test that protraction (forward shoulders) produces positive angles."""
    left_angle = wholebody_shoulders_protracted.left_scapular_protractionretraction.to_numpy()[0, 0]
    right_angle = wholebody_shoulders_protracted.right_scapular_protractionretraction.to_numpy()[0, 0]

    # Protraction should be positive and substantial
    assert left_angle > 20, f"Left shoulder protraction should be > 20°, got {left_angle:.2f}°"
    assert right_angle > 20, f"Right shoulder protraction should be > 20°, got {right_angle:.2f}°"

    # Angles should be symmetric for symmetric movement
    angle_diff = abs(left_angle - right_angle)
    assert angle_diff < 2, f"Left/right protraction angles should be similar, difference: {angle_diff:.2f}°"


def test_retraction_negative_sign(wholebody_shoulders_retracted):
    """Test that retraction (backward shoulders) produces negative angles."""
    left_angle = wholebody_shoulders_retracted.left_scapular_protractionretraction.to_numpy()[0, 0]
    right_angle = wholebody_shoulders_retracted.right_scapular_protractionretraction.to_numpy()[0, 0]

    # Retraction should be negative
    assert left_angle < 0, f"Left shoulder retraction should be negative, got {left_angle:.2f}°"
    assert right_angle < 0, f"Right shoulder retraction should be negative, got {right_angle:.2f}°"

    # Angles should be symmetric for symmetric movement
    angle_diff = abs(left_angle - right_angle)
    assert angle_diff < 2, f"Left/right retraction angles should be similar, difference: {angle_diff:.2f}°"


def test_asymmetric_shoulder_positions(wholebody_shoulders_asymmetric):
    """Test that asymmetric postures produce expected sign patterns."""
    left_angle = wholebody_shoulders_asymmetric.left_scapular_protractionretraction.to_numpy()[0, 0]
    right_angle = wholebody_shoulders_asymmetric.right_scapular_protractionretraction.to_numpy()[0, 0]

    # Left should be protracted (positive)
    assert left_angle > 0, f"Left shoulder protracted should be positive, got {left_angle:.2f}°"

    # Right should be retracted (negative)
    assert right_angle < 0, f"Right shoulder retracted should be negative, got {right_angle:.2f}°"

    # Angles should be different
    assert abs(left_angle - right_angle) > 10, "Asymmetric posture should have different angles"


def test_angles_not_near_180(wholebody_shoulders_neutral):
    """Test that angles are not near ±180° (the original bug)."""
    left_angle = wholebody_shoulders_neutral.left_scapular_protractionretraction.to_numpy()[0, 0]
    right_angle = wholebody_shoulders_neutral.right_scapular_protractionretraction.to_numpy()[0, 0]

    # Angles should not be near ±180° (the bug we're fixing)
    assert abs(left_angle) < 90, f"Left angle should not be near ±180°, got {left_angle:.2f}°"
    assert abs(right_angle) < 90, f"Right angle should not be near ±180°, got {right_angle:.2f}°"
