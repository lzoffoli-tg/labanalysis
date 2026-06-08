"""Test WholeBody class with missing medial markers"""

import sys
from os.path import abspath, dirname

import numpy as np
import pytest

# add project root to path
sys.path.append(dirname(dirname(abspath(__file__))))

import src.labanalysis as laban


def create_mock_point3d(n_samples=100, offset=0.0):
    """Create a mock Point3D with random data."""
    data = np.random.randn(n_samples, 3) + offset
    index = np.arange(n_samples) / 100.0  # 100 Hz
    return laban.Point3D(
        data=data,
        index=index,
        columns=["X", "Y", "Z"],
        unit="mm"
    )


def test_ankle_joint_center_with_missing_medial_markers():
    """Test that ankle joint centers are calculated correctly when medial markers are missing."""
    n_samples = 100

    # Create lateral markers only
    left_ankle_lat = create_mock_point3d(n_samples, offset=0.0)
    right_ankle_lat = create_mock_point3d(n_samples, offset=1.0)

    # Create other required markers for a minimal body model
    left_knee_lat = create_mock_point3d(n_samples, offset=2.0)
    right_knee_lat = create_mock_point3d(n_samples, offset=3.0)

    left_throcanter = create_mock_point3d(n_samples, offset=4.0)
    right_throcanter = create_mock_point3d(n_samples, offset=5.0)

    left_asis = create_mock_point3d(n_samples, offset=6.0)
    right_asis = create_mock_point3d(n_samples, offset=7.0)

    left_psis = create_mock_point3d(n_samples, offset=8.0)
    right_psis = create_mock_point3d(n_samples, offset=9.0)

    # Create WholeBody without medial ankle markers
    body = laban.WholeBody(
        left_ankle_lateral=left_ankle_lat,
        right_ankle_lateral=right_ankle_lat,
        # No medial ankle markers
        left_knee_lateral=left_knee_lat,
        right_knee_lateral=right_knee_lat,
        left_throcanter=left_throcanter,
        right_throcanter=right_throcanter,
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
    )

    # Test that ankle joint centers default to lateral markers
    left_ankle = body.left_ankle
    right_ankle = body.right_ankle

    assert left_ankle is not None
    assert right_ankle is not None
    assert left_ankle.shape == (n_samples, 3)
    assert right_ankle.shape == (n_samples, 3)

    # When no medial marker, ankle should equal lateral marker
    np.testing.assert_array_equal(left_ankle.to_numpy(), left_ankle_lat.to_numpy())
    np.testing.assert_array_equal(right_ankle.to_numpy(), right_ankle_lat.to_numpy())


def test_ankle_referenceframe_with_missing_medial_markers():
    """Test that ankle reference frames are calculated without error when medial markers are missing."""
    n_samples = 100

    # Create lateral markers only
    left_ankle_lat = create_mock_point3d(n_samples, offset=0.0)
    right_ankle_lat = create_mock_point3d(n_samples, offset=1.0)

    # Create other required markers
    left_knee_lat = create_mock_point3d(n_samples, offset=2.0)
    right_knee_lat = create_mock_point3d(n_samples, offset=3.0)

    left_throcanter = create_mock_point3d(n_samples, offset=4.0)
    right_throcanter = create_mock_point3d(n_samples, offset=5.0)

    left_asis = create_mock_point3d(n_samples, offset=6.0)
    right_asis = create_mock_point3d(n_samples, offset=7.0)

    left_psis = create_mock_point3d(n_samples, offset=8.0)
    right_psis = create_mock_point3d(n_samples, offset=9.0)

    # Create WholeBody without medial ankle markers
    body = laban.WholeBody(
        left_ankle_lateral=left_ankle_lat,
        right_ankle_lateral=right_ankle_lat,
        # No medial ankle markers
        left_knee_lateral=left_knee_lat,
        right_knee_lateral=right_knee_lat,
        left_throcanter=left_throcanter,
        right_throcanter=right_throcanter,
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
    )

    # Test that reference frames can be calculated without error
    left_ankle_origin, left_rmat = body.left_ankle_referenceframe
    right_ankle_origin, right_rmat = body.right_ankle_referenceframe

    assert left_ankle_origin is not None
    assert right_ankle_origin is not None
    assert left_rmat is not None
    assert right_rmat is not None

    # Check rotation matrix shape
    assert left_rmat.shape == (n_samples, 3, 3)
    assert right_rmat.shape == (n_samples, 3, 3)

    # Check that rotation matrices are valid (orthonormal)
    for i in range(min(5, n_samples)):  # Check first 5 samples
        # Check orthonormality: R @ R.T should be identity
        left_identity = left_rmat[i] @ left_rmat[i].T
        right_identity = right_rmat[i] @ right_rmat[i].T

        np.testing.assert_array_almost_equal(
            left_identity,
            np.eye(3),
            decimal=5,
            err_msg=f"Left ankle rotation matrix not orthonormal at sample {i}"
        )
        np.testing.assert_array_almost_equal(
            right_identity,
            np.eye(3),
            decimal=5,
            err_msg=f"Right ankle rotation matrix not orthonormal at sample {i}"
        )


def test_knee_elbow_wrist_with_missing_medial_markers():
    """Test that knee, elbow, and wrist joint centers work without medial markers."""
    n_samples = 100

    # Create lateral markers only
    left_knee_lat = create_mock_point3d(n_samples, offset=0.0)
    right_knee_lat = create_mock_point3d(n_samples, offset=1.0)

    left_elbow_lat = create_mock_point3d(n_samples, offset=2.0)
    right_elbow_lat = create_mock_point3d(n_samples, offset=3.0)

    left_wrist_lat = create_mock_point3d(n_samples, offset=4.0)
    right_wrist_lat = create_mock_point3d(n_samples, offset=5.0)

    # Create minimal required markers
    left_ankle_lat = create_mock_point3d(n_samples, offset=6.0)
    right_ankle_lat = create_mock_point3d(n_samples, offset=7.0)

    left_shoulder_ant = create_mock_point3d(n_samples, offset=8.0)
    left_shoulder_post = create_mock_point3d(n_samples, offset=9.0)
    right_shoulder_ant = create_mock_point3d(n_samples, offset=10.0)
    right_shoulder_post = create_mock_point3d(n_samples, offset=11.0)

    left_throcanter = create_mock_point3d(n_samples, offset=12.0)
    right_throcanter = create_mock_point3d(n_samples, offset=13.0)

    left_asis = create_mock_point3d(n_samples, offset=14.0)
    right_asis = create_mock_point3d(n_samples, offset=15.0)

    left_psis = create_mock_point3d(n_samples, offset=16.0)
    right_psis = create_mock_point3d(n_samples, offset=17.0)

    # Create WholeBody without medial markers
    body = laban.WholeBody(
        left_knee_lateral=left_knee_lat,
        right_knee_lateral=right_knee_lat,
        left_elbow_lateral=left_elbow_lat,
        right_elbow_lateral=right_elbow_lat,
        left_wrist_lateral=left_wrist_lat,
        right_wrist_lateral=right_wrist_lat,
        left_ankle_lateral=left_ankle_lat,
        right_ankle_lateral=right_ankle_lat,
        left_shoulder_anterior=left_shoulder_ant,
        left_shoulder_posterior=left_shoulder_post,
        right_shoulder_anterior=right_shoulder_ant,
        right_shoulder_posterior=right_shoulder_post,
        left_throcanter=left_throcanter,
        right_throcanter=right_throcanter,
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
    )

    # Test knee joint centers
    left_knee = body.left_knee
    right_knee = body.right_knee
    assert left_knee is not None
    assert right_knee is not None
    np.testing.assert_array_equal(left_knee.to_numpy(), left_knee_lat.to_numpy())
    np.testing.assert_array_equal(right_knee.to_numpy(), right_knee_lat.to_numpy())

    # Test elbow joint centers
    left_elbow = body.left_elbow
    right_elbow = body.right_elbow
    assert left_elbow is not None
    assert right_elbow is not None
    np.testing.assert_array_equal(left_elbow.to_numpy(), left_elbow_lat.to_numpy())
    np.testing.assert_array_equal(right_elbow.to_numpy(), right_elbow_lat.to_numpy())

    # Test wrist joint centers
    left_wrist = body.left_wrist
    right_wrist = body.right_wrist
    assert left_wrist is not None
    assert right_wrist is not None
    np.testing.assert_array_equal(left_wrist.to_numpy(), left_wrist_lat.to_numpy())
    np.testing.assert_array_equal(right_wrist.to_numpy(), right_wrist_lat.to_numpy())


def test_ankle_with_both_markers():
    """Test that ankle joint center is averaged when both markers are present."""
    n_samples = 100

    # Create both lateral and medial markers
    left_ankle_lat = create_mock_point3d(n_samples, offset=0.0)
    left_ankle_med = create_mock_point3d(n_samples, offset=1.0)

    left_knee_lat = create_mock_point3d(n_samples, offset=2.0)

    left_throcanter = create_mock_point3d(n_samples, offset=4.0)
    right_throcanter = create_mock_point3d(n_samples, offset=5.0)

    left_asis = create_mock_point3d(n_samples, offset=6.0)
    right_asis = create_mock_point3d(n_samples, offset=7.0)

    left_psis = create_mock_point3d(n_samples, offset=8.0)
    right_psis = create_mock_point3d(n_samples, offset=9.0)

    # Create WholeBody with both ankle markers
    body = laban.WholeBody(
        left_ankle_lateral=left_ankle_lat,
        left_ankle_medial=left_ankle_med,
        left_knee_lateral=left_knee_lat,
        left_throcanter=left_throcanter,
        right_throcanter=right_throcanter,
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
    )

    # Test that ankle joint center is averaged
    left_ankle = body.left_ankle
    expected = (left_ankle_lat.to_numpy() + left_ankle_med.to_numpy()) / 2

    np.testing.assert_array_almost_equal(left_ankle.to_numpy(), expected)


if __name__ == "__main__":
    test_ankle_joint_center_with_missing_medial_markers()
    test_ankle_referenceframe_with_missing_medial_markers()
    test_knee_elbow_wrist_with_missing_medial_markers()
    test_ankle_with_both_markers()
    print("All tests passed!")
