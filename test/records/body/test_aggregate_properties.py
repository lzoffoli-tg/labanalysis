"""
Test suite for aggregate properties: segment_lengths and joint_angles.

This module tests the two new properties that combine all segment lengths
and all joint angles into single Timeseries objects.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import labanalysis as laban


@pytest.fixture
def wholebody_complete():
    """Create a WholeBody with all markers for complete testing."""
    n_frames = 5
    time_index = list(range(n_frames))

    # Global coordinate system used for synthetic markers: X=lateral, Y=vertical, Z=anteroposterior

    # Pelvis markers
    left_asis = laban.Point3D(
        data=np.array([[-0.10, 0.90, 0.00]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_asis = laban.Point3D(
        data=np.array([[0.10, 0.90, 0.00]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_psis = laban.Point3D(
        data=np.array([[-0.08, 0.85, -0.15]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_psis = laban.Point3D(
        data=np.array([[0.08, 0.85, -0.15]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Hip markers
    left_trochanter = laban.Point3D(
        data=np.array([[-0.15, 0.85, 0.00]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_trochanter = laban.Point3D(
        data=np.array([[0.15, 0.85, 0.00]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Knee markers
    left_knee_lat = laban.Point3D(
        data=np.array([[-0.18, 0.50, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_knee_med = laban.Point3D(
        data=np.array([[-0.12, 0.50, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_knee_lat = laban.Point3D(
        data=np.array([[0.18, 0.50, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_knee_med = laban.Point3D(
        data=np.array([[0.12, 0.50, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Ankle markers
    left_ankle_lat = laban.Point3D(
        data=np.array([[-0.15, 0.08, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_ankle_med = laban.Point3D(
        data=np.array([[-0.10, 0.08, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_ankle_lat = laban.Point3D(
        data=np.array([[0.15, 0.08, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_ankle_med = laban.Point3D(
        data=np.array([[0.10, 0.08, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Foot markers
    left_heel = laban.Point3D(
        data=np.array([[-0.12, 0.02, -0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_heel = laban.Point3D(
        data=np.array([[0.12, 0.02, -0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_toe = laban.Point3D(
        data=np.array([[-0.12, 0.02, 0.15]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_toe = laban.Point3D(
        data=np.array([[0.12, 0.02, 0.15]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_first_met = laban.Point3D(
        data=np.array([[-0.09, 0.02, 0.12]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_fifth_met = laban.Point3D(
        data=np.array([[-0.15, 0.02, 0.12]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_first_met = laban.Point3D(
        data=np.array([[0.09, 0.02, 0.12]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_fifth_met = laban.Point3D(
        data=np.array([[0.15, 0.02, 0.12]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Shoulder markers
    left_acromion = laban.Point3D(
        data=np.array([[-0.20, 1.35, -0.03]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_acromion = laban.Point3D(
        data=np.array([[0.20, 1.35, -0.03]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Elbow markers
    left_elbow_lat = laban.Point3D(
        data=np.array([[-0.25, 1.10, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_elbow_med = laban.Point3D(
        data=np.array([[-0.18, 1.10, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_elbow_lat = laban.Point3D(
        data=np.array([[0.25, 1.10, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_elbow_med = laban.Point3D(
        data=np.array([[0.18, 1.10, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Wrist markers
    left_wrist_lat = laban.Point3D(
        data=np.array([[-0.28, 0.90, 0.10]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_wrist_med = laban.Point3D(
        data=np.array([[-0.22, 0.90, 0.10]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_wrist_lat = laban.Point3D(
        data=np.array([[0.28, 0.90, 0.10]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_wrist_med = laban.Point3D(
        data=np.array([[0.22, 0.90, 0.10]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Neck/trunk markers
    c7 = laban.Point3D(
        data=np.array([[0.00, 1.40, 0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    sc = laban.Point3D(
        data=np.array([[0.00, 1.42, -0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    t5 = laban.Point3D(
        data=np.array([[0.00, 1.10, -0.08]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    l2 = laban.Point3D(
        data=np.array([[0.00, 0.95, -0.05]] * n_frames),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    return laban.WholeBody(
        left_asis=left_asis, right_asis=right_asis,
        left_psis=left_psis, right_psis=right_psis,
        left_trochanter=left_trochanter, right_trochanter=right_trochanter,
        left_knee_lateral=left_knee_lat, left_knee_medial=left_knee_med,
        right_knee_lateral=right_knee_lat, right_knee_medial=right_knee_med,
        left_ankle_lateral=left_ankle_lat, left_ankle_medial=left_ankle_med,
        right_ankle_lateral=right_ankle_lat, right_ankle_medial=right_ankle_med,
        left_heel=left_heel, right_heel=right_heel,
        left_toe=left_toe, right_toe=right_toe,
        left_first_metatarsal_head=left_first_met, left_fifth_metatarsal_head=left_fifth_met,
        right_first_metatarsal_head=right_first_met, right_fifth_metatarsal_head=right_fifth_met,
        left_acromion=left_acromion, right_acromion=right_acromion,
        left_elbow_lateral=left_elbow_lat, left_elbow_medial=left_elbow_med,
        right_elbow_lateral=right_elbow_lat, right_elbow_medial=right_elbow_med,
        left_wrist_lateral=left_wrist_lat, left_wrist_medial=left_wrist_med,
        right_wrist_lateral=right_wrist_lat, right_wrist_medial=right_wrist_med,
        c7=c7, sc=sc, t5=t5, l2=l2,
    )


def test_segment_lengths_returns_timeseries(wholebody_complete):
    """Verify segment_lengths returns a Timeseries object."""
    lengths = wholebody_complete.segment_lengths
    assert isinstance(lengths, laban.Timeseries), \
        "segment_lengths should return a Timeseries object"


def test_segment_lengths_contains_expected_columns(wholebody_complete):
    """Verify segment_lengths contains expected length properties."""
    lengths = wholebody_complete.segment_lengths

    expected_columns = [
        'left_foot_height', 'right_foot_height',
        'left_foot_length', 'right_foot_length',
        'left_foot_width', 'right_foot_width',
        'left_ankle_width', 'right_ankle_width',
        'left_leg_length', 'right_leg_length',
        'left_thigh_length', 'right_thigh_length',
        'left_knee_width', 'right_knee_width',
        'left_lower_limb_length', 'right_lower_limb_length',
        'left_arm_length', 'right_arm_length',
        'left_forearm_length', 'right_forearm_length',
        'left_elbow_width', 'right_elbow_width',
        'left_upper_limb_length', 'right_upper_limb_length',
        'shoulder_width', 'hip_width', 'trunk_length', 'pelvis_height',
    ]

    for col in expected_columns:
        assert col in lengths.columns, f"Missing expected column: {col}"


def test_segment_lengths_values_match_individual_properties(wholebody_complete):
    """Verify segment_lengths values match individual property values."""
    lengths = wholebody_complete.segment_lengths

    # Check a few representative properties
    np.testing.assert_array_almost_equal(
        lengths['left_foot_height'].to_numpy().flatten(),
        wholebody_complete.left_foot_height.to_numpy().flatten(),
        err_msg="left_foot_height values don't match"
    )

    np.testing.assert_array_almost_equal(
        lengths['right_thigh_length'].to_numpy().flatten(),
        wholebody_complete.right_thigh_length.to_numpy().flatten(),
        err_msg="right_thigh_length values don't match"
    )

    np.testing.assert_array_almost_equal(
        lengths['left_elbow_width'].to_numpy().flatten(),
        wholebody_complete.left_elbow_width.to_numpy().flatten(),
        err_msg="left_elbow_width values don't match"
    )

    np.testing.assert_array_almost_equal(
        lengths['shoulder_width'].to_numpy().flatten(),
        wholebody_complete.shoulder_width.to_numpy().flatten(),
        err_msg="shoulder_width values don't match"
    )

    np.testing.assert_array_almost_equal(
        lengths['trunk_length'].to_numpy().flatten(),
        wholebody_complete.trunk_length.to_numpy().flatten(),
        err_msg="trunk_length values don't match"
    )


def test_joint_angles_returns_timeseries(wholebody_complete):
    """Verify joint_angles returns a Timeseries object."""
    angles = wholebody_complete.joint_angles
    assert isinstance(angles, laban.Timeseries), \
        "joint_angles should return a Timeseries object"


def test_joint_angles_contains_expected_columns(wholebody_complete):
    """Verify joint_angles contains expected angle properties."""
    angles = wholebody_complete.joint_angles

    expected_columns = [
        # Ankle angles
        'left_ankle_flexionextension', 'right_ankle_flexionextension',
        'left_ankle_inversioneversion', 'right_ankle_inversioneversion',
        # Knee angles
        'left_knee_flexionextension', 'right_knee_flexionextension',
        'left_knee_varusvalgus', 'right_knee_varusvalgus',
        # Hip angles
        'left_hip_flexionextension', 'right_hip_flexionextension',
        'left_hip_abductionadduction', 'right_hip_abductionadduction',
        'left_hip_internalexternalrotation', 'right_hip_internalexternalrotation',
        # Pelvis angles
        'pelvis_anteroposterior_tilt',
        # Trunk angles
        'trunk_rotation',
        # Elbow angles
        'left_elbow_flexionextension', 'right_elbow_flexionextension',
        # Spine curvature
        'lumbar_lordosis', 'dorsal_kyphosis',
    ]

    for col in expected_columns:
        assert col in angles.columns, f"Missing expected column: {col}"


def test_joint_angles_values_match_individual_properties(wholebody_complete):
    """Verify joint_angles values match individual property values."""
    angles = wholebody_complete.joint_angles

    # Check a few representative properties
    np.testing.assert_array_almost_equal(
        angles['left_knee_flexionextension'].to_numpy().flatten(),
        wholebody_complete.left_knee_flexionextension.to_numpy().flatten(),
        err_msg="left_knee_flexionextension values don't match"
    )

    np.testing.assert_array_almost_equal(
        angles['pelvis_anteroposterior_tilt'].to_numpy().flatten(),
        wholebody_complete.pelvis_anteroposterior_tilt.to_numpy().flatten(),
        err_msg="pelvis_anteroposterior_tilt values don't match"
    )

    np.testing.assert_array_almost_equal(
        angles['lumbar_lordosis'].to_numpy().flatten(),
        wholebody_complete.lumbar_lordosis.to_numpy().flatten(),
        err_msg="lumbar_lordosis values don't match"
    )


def test_segment_lengths_with_partial_markers():
    """Verify segment_lengths works with partial marker sets."""
    # Create body with only lower limb markers
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
    left_knee_lat = laban.Point3D(
        data=np.array([[-0.18, 0.50, 0.05]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    left_knee_med = laban.Point3D(
        data=np.array([[-0.12, 0.50, 0.05]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    body = laban.WholeBody(
        left_asis=left_asis, right_asis=right_asis,
        left_knee_lateral=left_knee_lat, left_knee_medial=left_knee_med,
    )

    lengths = body.segment_lengths

    # Should have at least some columns
    assert len(lengths.columns) > 0, "Should have at least some length measurements"

    # Should have knee width (available)
    assert 'left_knee_width' in lengths.columns


def test_joint_angles_with_partial_markers():
    """Verify joint_angles works with partial marker sets."""
    # Create body with pelvis, spine and neck markers for trunk/spine angles
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
    c7 = laban.Point3D(
        data=np.array([[0.00, 1.40, 0.05]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    t5 = laban.Point3D(
        data=np.array([[0.00, 1.10, -0.08]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )
    l2 = laban.Point3D(
        data=np.array([[0.00, 0.95, -0.05]]),
        index=[0],
        columns=["X", "Y", "Z"],
    )

    body = laban.WholeBody(
        left_asis=left_asis, right_asis=right_asis,
        left_psis=left_psis, right_psis=right_psis,
        c7=c7, t5=t5, l2=l2,
    )

    angles = body.joint_angles

    # Should have at least some columns
    assert len(angles.columns) > 0, "Should have at least some angle measurements"

    # Should have spine curvature angles (available with c7, t5, l2, psis)
    assert 'lumbar_lordosis' in angles.columns or 'dorsal_kyphosis' in angles.columns
