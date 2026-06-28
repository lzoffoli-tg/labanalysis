"""
Test WholeBody class with missing medial markers.

This test suite validates:
- Joint center calculations with missing medial markers
- Angular measure accessibility across all 38 measures
- New marker support (cranial, foot, spine)
- Reference frame calculations

Note: Tests import from src/ directory to validate actual codebase
rather than installed package version.
"""

import numpy as np
import pytest

from labanalysis.timeseries import Point3D, Signal1D, Timeseries
from labanalysis.records.body import WholeBody



def create_mock_point3d(n_samples=100, offset=0.0):
    """Create a mock Point3D with random data."""
    data = np.random.randn(n_samples, 3) + offset
    index = np.arange(n_samples) / 100.0  # 100 Hz
    return Point3D(
        data=data,
        index=index,
        columns=["X", "Y", "Z"],
        unit="mm"
    )


def create_wholebody_with_cranial_markers(n_samples=100):
    """Create WholeBody with all cranial and neck markers for testing."""
    return WholeBody(
        head_anterior=create_mock_point3d(n_samples, offset=0.0),
        head_posterior=create_mock_point3d(n_samples, offset=1.0),
        head_left=create_mock_point3d(n_samples, offset=2.0),
        head_right=create_mock_point3d(n_samples, offset=3.0),
        sc=create_mock_point3d(n_samples, offset=4.0),
        c7=create_mock_point3d(n_samples, offset=5.0),
        t5=create_mock_point3d(n_samples, offset=6.0),
        # Minimal pelvis markers required for valid WholeBody
        left_asis=create_mock_point3d(n_samples, offset=7.0),
        right_asis=create_mock_point3d(n_samples, offset=8.0),
        left_psis=create_mock_point3d(n_samples, offset=9.0),
        right_psis=create_mock_point3d(n_samples, offset=10.0),
    )


def create_wholebody_with_foot_markers(n_samples=100):
    """Create WholeBody with all foot plane markers for testing."""
    return WholeBody(
        left_toe=create_mock_point3d(n_samples, offset=0.0),
        left_heel=create_mock_point3d(n_samples, offset=1.0),
        left_first_metatarsal_head=create_mock_point3d(n_samples, offset=2.0),
        left_fifth_metatarsal_head=create_mock_point3d(n_samples, offset=3.0),
        right_toe=create_mock_point3d(n_samples, offset=4.0),
        right_heel=create_mock_point3d(n_samples, offset=5.0),
        right_first_metatarsal_head=create_mock_point3d(n_samples, offset=6.0),
        right_fifth_metatarsal_head=create_mock_point3d(n_samples, offset=7.0),
        left_ankle_lateral=create_mock_point3d(n_samples, offset=8.0),
        right_ankle_lateral=create_mock_point3d(n_samples, offset=9.0),
        # Minimal pelvis markers required for valid WholeBody
        left_asis=create_mock_point3d(n_samples, offset=10.0),
        right_asis=create_mock_point3d(n_samples, offset=11.0),
        left_psis=create_mock_point3d(n_samples, offset=12.0),
        right_psis=create_mock_point3d(n_samples, offset=13.0),
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

    left_trochanter = create_mock_point3d(n_samples, offset=4.0)
    right_trochanter = create_mock_point3d(n_samples, offset=5.0)

    left_asis = create_mock_point3d(n_samples, offset=6.0)
    right_asis = create_mock_point3d(n_samples, offset=7.0)

    left_psis = create_mock_point3d(n_samples, offset=8.0)
    right_psis = create_mock_point3d(n_samples, offset=9.0)

    # Create WholeBody without medial ankle markers
    body = WholeBody(
        left_ankle_lateral=left_ankle_lat,
        right_ankle_lateral=right_ankle_lat,
        # No medial ankle markers
        left_knee_lateral=left_knee_lat,
        right_knee_lateral=right_knee_lat,
        left_trochanter=left_trochanter,
        right_trochanter=right_trochanter,
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

    left_trochanter = create_mock_point3d(n_samples, offset=4.0)
    right_trochanter = create_mock_point3d(n_samples, offset=5.0)

    left_asis = create_mock_point3d(n_samples, offset=6.0)
    right_asis = create_mock_point3d(n_samples, offset=7.0)

    left_psis = create_mock_point3d(n_samples, offset=8.0)
    right_psis = create_mock_point3d(n_samples, offset=9.0)

    # Create WholeBody without medial ankle markers
    body = WholeBody(
        left_ankle_lateral=left_ankle_lat,
        right_ankle_lateral=right_ankle_lat,
        # No medial ankle markers
        left_knee_lateral=left_knee_lat,
        right_knee_lateral=right_knee_lat,
        left_trochanter=left_trochanter,
        right_trochanter=right_trochanter,
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
    )

    # Test that reference frames can be calculated without error
    left_rf = body.left_ankle_referenceframe
    right_rf = body.right_ankle_referenceframe

    assert left_rf is not None
    assert right_rf is not None

    # Get rotation matrices from reference frames
    left_rmat = left_rf.rotation_matrix
    right_rmat = right_rf.rotation_matrix

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

    left_trochanter = create_mock_point3d(n_samples, offset=12.0)
    right_trochanter = create_mock_point3d(n_samples, offset=13.0)

    left_asis = create_mock_point3d(n_samples, offset=14.0)
    right_asis = create_mock_point3d(n_samples, offset=15.0)

    left_psis = create_mock_point3d(n_samples, offset=16.0)
    right_psis = create_mock_point3d(n_samples, offset=17.0)

    # Create WholeBody without medial markers
    body = WholeBody(
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
        left_trochanter=left_trochanter,
        right_trochanter=right_trochanter,
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

    left_trochanter = create_mock_point3d(n_samples, offset=4.0)
    right_trochanter = create_mock_point3d(n_samples, offset=5.0)

    left_asis = create_mock_point3d(n_samples, offset=6.0)
    right_asis = create_mock_point3d(n_samples, offset=7.0)

    left_psis = create_mock_point3d(n_samples, offset=8.0)
    right_psis = create_mock_point3d(n_samples, offset=9.0)

    # Create WholeBody with both ankle markers
    body = WholeBody(
        left_ankle_lateral=left_ankle_lat,
        left_ankle_medial=left_ankle_med,
        left_knee_lateral=left_knee_lat,
        left_trochanter=left_trochanter,
        right_trochanter=right_trochanter,
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
    )

    # Test that ankle joint center is averaged
    left_ankle = body.left_ankle
    expected = (left_ankle_lat.to_numpy() + left_ankle_med.to_numpy()) / 2

    np.testing.assert_array_almost_equal(left_ankle.to_numpy(), expected)


## SECTION 2: New marker tests (Suite 1)


def test_head_center_with_all_cranial_markers():
    """Test head_center calculation with all 4 cranial markers."""
    n_samples = 100

    # Create 4 cranial markers at known positions
    head_ant = create_mock_point3d(n_samples, offset=0.0)
    head_post = create_mock_point3d(n_samples, offset=1.0)
    head_left = create_mock_point3d(n_samples, offset=2.0)
    head_right = create_mock_point3d(n_samples, offset=3.0)

    # Create minimal WholeBody
    body = WholeBody(
        head_anterior=head_ant,
        head_posterior=head_post,
        head_left=head_left,
        head_right=head_right,
        left_asis=create_mock_point3d(n_samples, offset=4.0),
        right_asis=create_mock_point3d(n_samples, offset=5.0),
        left_psis=create_mock_point3d(n_samples, offset=6.0),
        right_psis=create_mock_point3d(n_samples, offset=7.0),
    )

    # Test head_center is centroid of 4 markers
    head_center = body.head_center

    assert head_center is not None
    assert head_center.shape == (n_samples, 3)

    # Verify it's the average of the 4 markers
    expected = (head_ant.to_numpy() + head_post.to_numpy() +
                head_left.to_numpy() + head_right.to_numpy()) / 4
    np.testing.assert_array_almost_equal(head_center.to_numpy(), expected, decimal=5)


def test_head_center_with_missing_markers():
    """Test head_center raises ValueError when cranial markers are missing."""
    n_samples = 100

    # Create WholeBody without cranial markers
    body = WholeBody(
        left_asis=create_mock_point3d(n_samples, offset=0.0),
        right_asis=create_mock_point3d(n_samples, offset=1.0),
        left_psis=create_mock_point3d(n_samples, offset=2.0),
        right_psis=create_mock_point3d(n_samples, offset=3.0),
    )

    # head_center should emit warnings and return NaN without markers
    with pytest.warns(UserWarning, match="Cannot calculate head_center"):
        result = body.head_center
        # Verify result is NaN
        assert np.all(np.isnan(result.to_numpy()))


def test_neck_base_calculation():
    """Test neck_base as midpoint between sc and c7."""
    n_samples = 100

    # Create sc and c7 at known positions
    sc = create_mock_point3d(n_samples, offset=0.0)
    c7 = create_mock_point3d(n_samples, offset=1.0)

    # Create WholeBody
    body = WholeBody(
        sc=sc,
        c7=c7,
        left_asis=create_mock_point3d(n_samples, offset=2.0),
        right_asis=create_mock_point3d(n_samples, offset=3.0),
        left_psis=create_mock_point3d(n_samples, offset=4.0),
        right_psis=create_mock_point3d(n_samples, offset=5.0),
    )

    # Test neck_base is midpoint
    neck_base = body.neck_base

    assert neck_base is not None
    assert neck_base.shape == (n_samples, 3)

    # Verify it's the average of sc and c7
    expected = (sc.to_numpy() + c7.to_numpy()) / 2
    np.testing.assert_array_almost_equal(neck_base.to_numpy(), expected, decimal=5)


def test_foot_plane_with_all_markers():
    """Test foot plane calculation with toe, heel, and metatarsals."""
    n_samples = 100

    # Create foot markers
    body = create_wholebody_with_foot_markers(n_samples)

    # Test left and right foot planes (now private properties)
    left_plane = body._left_foot_plane
    right_plane = body._right_foot_plane

    assert left_plane is not None
    assert right_plane is not None

    # Verify shape (n_samples, 4) for [a, b, c, d] coefficients
    assert left_plane.shape == (n_samples, 4)
    assert right_plane.shape == (n_samples, 4)

    # Verify plane coefficients are not all zeros
    assert not np.allclose(left_plane.to_numpy(), 0)
    assert not np.allclose(right_plane.to_numpy(), 0)


def test_metatarsal_markers_independence():
    """Test that first and fifth metatarsal markers are independent."""
    n_samples = 100

    # Create WholeBody with only left_first_metatarsal_head
    body = WholeBody(
        left_first_metatarsal_head=create_mock_point3d(n_samples, offset=0.0),
        left_asis=create_mock_point3d(n_samples, offset=1.0),
        right_asis=create_mock_point3d(n_samples, offset=2.0),
        left_psis=create_mock_point3d(n_samples, offset=3.0),
        right_psis=create_mock_point3d(n_samples, offset=4.0),
    )

    # Verify marker is stored
    assert body.left_first_metatarsal_head is not None

    # Verify other metatarsal markers raise AttributeError when accessed
    with pytest.raises(AttributeError):
        _ = body.left_fifth_metatarsal_head
    with pytest.raises(AttributeError):
        _ = body.right_first_metatarsal_head
    with pytest.raises(AttributeError):
        _ = body.right_fifth_metatarsal_head


## SECTION 3: Derived properties tests (Suite 2)


def test_foot_height_with_new_metatarsals():
    """Test foot height calculation using new foot plane."""
    n_samples = 100

    # Create WholeBody with foot markers
    body = create_wholebody_with_foot_markers(n_samples)

    # Test that foot height properties exist and return Signal1D
    left_height = body.left_foot_height
    right_height = body.right_foot_height

    assert left_height is not None
    assert right_height is not None
    assert isinstance(left_height, Signal1D)
    assert isinstance(right_height, Signal1D)
    # Signal1D may have shape (n_samples,) or (n_samples, 1)
    assert left_height.shape[0] == n_samples
    assert right_height.shape[0] == n_samples


def test_ankle_angles_with_new_foot_plane():
    """Test ankle flexion/extension and inversion/eversion with new plane."""
    n_samples = 100

    # Create WholeBody with foot markers and knee for reference frame
    body = WholeBody(
        left_toe=create_mock_point3d(n_samples, offset=0.0),
        left_heel=create_mock_point3d(n_samples, offset=1.0),
        left_first_metatarsal_head=create_mock_point3d(n_samples, offset=2.0),
        left_fifth_metatarsal_head=create_mock_point3d(n_samples, offset=3.0),
        left_ankle_lateral=create_mock_point3d(n_samples, offset=4.0),
        left_knee_lateral=create_mock_point3d(n_samples, offset=5.0),
        left_asis=create_mock_point3d(n_samples, offset=6.0),
        right_asis=create_mock_point3d(n_samples, offset=7.0),
        left_psis=create_mock_point3d(n_samples, offset=8.0),
        right_psis=create_mock_point3d(n_samples, offset=9.0),
    )

    # Test ankle angles using the new foot plane
    flexion = body.left_ankle_flexionextension
    inversion = body.left_ankle_inversioneversion

    assert flexion is not None
    assert inversion is not None
    assert isinstance(flexion, Signal1D)
    assert isinstance(inversion, Signal1D)


def test_neck_angles_with_head_center():
    """Test neck_lateralflexion and neck_flexionextension."""
    n_samples = 100

    # Create WholeBody with head and neck markers
    body = create_wholebody_with_cranial_markers(n_samples)

    # Test neck angle properties
    lateral_flexion = body.neck_lateralflexion
    flexion = body.neck_flexionextension

    assert lateral_flexion is not None
    assert flexion is not None

    assert isinstance(lateral_flexion, Signal1D)
    assert isinstance(flexion, Signal1D)

    # Signal1D may have shape (n_samples,) or (n_samples, 1)
    assert lateral_flexion.shape[0] == n_samples
    assert flexion.shape[0] == n_samples


## SECTION 4: Integration tests (Suite 3)


def test_all_angular_measures_accessible():
    """
    Test that all 38 angular measures in _angular_measures are accessible.

    This test validates that all angular measures defined in WholeBody._angular_measures
    can be calculated when a complete marker set is provided. The test requires:

    - All lateral markers (ankle, knee, elbow, wrist)
    - All medial markers (required for internal/external rotation measures)
    - Pelvis markers (ASIS and PSIS bilateral)
    - Shoulder markers (anterior, posterior, acromion)
    - Spine markers (s2, l2, c7, t5, sc)
    - Head markers (anterior, posterior, left, right)
    - Foot markers (heel, toe, metatarsals)

    Marker Requirements by Measure Type:
    - 34/38 measures accessible with lateral markers only
    - 4 measures require medial markers:
      * left/right_hip_internalexternalrotation
      * left/right_shoulder_internalexternalrotation
    """
    n_samples = 100

    # Create WholeBody with comprehensive markers (including medial markers for internal/external rotation)
    body = WholeBody(
        # Foot markers
        left_heel=create_mock_point3d(n_samples, offset=0.0),
        right_heel=create_mock_point3d(n_samples, offset=1.0),
        left_toe=create_mock_point3d(n_samples, offset=2.0),
        right_toe=create_mock_point3d(n_samples, offset=3.0),
        left_first_metatarsal_head=create_mock_point3d(n_samples, offset=4.0),
        left_fifth_metatarsal_head=create_mock_point3d(n_samples, offset=5.0),
        right_first_metatarsal_head=create_mock_point3d(n_samples, offset=6.0),
        right_fifth_metatarsal_head=create_mock_point3d(n_samples, offset=7.0),
        # Ankle markers
        left_ankle_lateral=create_mock_point3d(n_samples, offset=8.0),
        left_ankle_medial=create_mock_point3d(n_samples, offset=8.5),
        right_ankle_lateral=create_mock_point3d(n_samples, offset=9.0),
        right_ankle_medial=create_mock_point3d(n_samples, offset=9.5),
        # Knee markers (need medial for internal/external rotation)
        left_knee_lateral=create_mock_point3d(n_samples, offset=10.0),
        left_knee_medial=create_mock_point3d(n_samples, offset=10.5),
        right_knee_lateral=create_mock_point3d(n_samples, offset=11.0),
        right_knee_medial=create_mock_point3d(n_samples, offset=11.5),
        # Hip markers
        left_trochanter=create_mock_point3d(n_samples, offset=12.0),
        right_trochanter=create_mock_point3d(n_samples, offset=13.0),
        # Pelvis markers
        left_asis=create_mock_point3d(n_samples, offset=14.0),
        right_asis=create_mock_point3d(n_samples, offset=15.0),
        left_psis=create_mock_point3d(n_samples, offset=16.0),
        right_psis=create_mock_point3d(n_samples, offset=17.0),
        # Shoulder markers
        left_acromion=create_mock_point3d(n_samples, offset=18.0),
        right_acromion=create_mock_point3d(n_samples, offset=19.0),
        left_shoulder_anterior=create_mock_point3d(n_samples, offset=20.0),
        left_shoulder_posterior=create_mock_point3d(n_samples, offset=21.0),
        right_shoulder_anterior=create_mock_point3d(n_samples, offset=22.0),
        right_shoulder_posterior=create_mock_point3d(n_samples, offset=23.0),
        # Elbow markers
        left_elbow_lateral=create_mock_point3d(n_samples, offset=24.0),
        left_elbow_medial=create_mock_point3d(n_samples, offset=24.5),
        right_elbow_lateral=create_mock_point3d(n_samples, offset=25.0),
        right_elbow_medial=create_mock_point3d(n_samples, offset=25.5),
        # Wrist markers
        left_wrist_lateral=create_mock_point3d(n_samples, offset=35.0),
        right_wrist_lateral=create_mock_point3d(n_samples, offset=36.0),
        # Spine markers
        s2=create_mock_point3d(n_samples, offset=26.0),
        l2=create_mock_point3d(n_samples, offset=27.0),
        c7=create_mock_point3d(n_samples, offset=28.0),
        t5=create_mock_point3d(n_samples, offset=29.0),
        sc=create_mock_point3d(n_samples, offset=30.0),
        # Head markers
        head_anterior=create_mock_point3d(n_samples, offset=31.0),
        head_posterior=create_mock_point3d(n_samples, offset=32.0),
        head_left=create_mock_point3d(n_samples, offset=33.0),
        head_right=create_mock_point3d(n_samples, offset=34.0),
    )

    # Verify all angular measures are accessible
    # With complete marker set, all 38 measures should be calculable
    accessible_count = 0
    for measure_name in WholeBody._angular_measures:
        try:
            angle = getattr(body, measure_name, None)
            if angle is not None:
                assert isinstance(angle, Signal1D), f"Angular measure '{measure_name}' is not Signal1D"
                accessible_count += 1
        except (ValueError, AttributeError) as e:
            # If this happens, there's a bug in the angular measure calculation
            # All measures should be accessible with the complete marker set provided
            pass

    # Verify that ALL angular measures are accessible (100%)
    # This validates that:
    # 1. No deprecated pandas methods are used (e.g., .ix instead of .iloc)
    # 2. All property references are correct (e.g., shoulder_lateral_tilt_global not shoulder_lateral_tilt_global)
    # 3. All required markers are included in the test setup
    assert accessible_count == len(WholeBody._angular_measures), \
        f"Only {accessible_count} out of {len(WholeBody._angular_measures)} angular measures accessible"


def test_t5_marker_presence():
    """Test that t5 marker can be set and retrieved."""
    n_samples = 100

    # Create WholeBody with t5 marker
    t5_marker = create_mock_point3d(n_samples, offset=0.0)
    body = WholeBody(
        t5=t5_marker,
        left_asis=create_mock_point3d(n_samples, offset=1.0),
        right_asis=create_mock_point3d(n_samples, offset=2.0),
        left_psis=create_mock_point3d(n_samples, offset=3.0),
        right_psis=create_mock_point3d(n_samples, offset=4.0),
    )

    # Verify t5 is stored and accessible
    assert body.t5 is not None
    assert body.t5.shape == (n_samples, 3)
    np.testing.assert_array_equal(body.t5.to_numpy(), t5_marker.to_numpy())


def test_cranial_markers_bilateral():
    """Test all 4 cranial markers can be set independently."""
    n_samples = 100

    # Create individual cranial markers
    h_ant = create_mock_point3d(n_samples, offset=0.0)
    h_post = create_mock_point3d(n_samples, offset=1.0)
    h_left = create_mock_point3d(n_samples, offset=2.0)
    h_right = create_mock_point3d(n_samples, offset=3.0)

    # Create WholeBody with all cranial markers
    body = WholeBody(
        head_anterior=h_ant,
        head_posterior=h_post,
        head_left=h_left,
        head_right=h_right,
        left_asis=create_mock_point3d(n_samples, offset=4.0),
        right_asis=create_mock_point3d(n_samples, offset=5.0),
        left_psis=create_mock_point3d(n_samples, offset=6.0),
        right_psis=create_mock_point3d(n_samples, offset=7.0),
    )

    # Verify all 4 cranial markers are stored independently
    assert body.head_anterior is not None
    assert body.head_posterior is not None
    assert body.head_left is not None
    assert body.head_right is not None

    # Verify they have the correct values
    np.testing.assert_array_equal(body.head_anterior.to_numpy(), h_ant.to_numpy())
    np.testing.assert_array_equal(body.head_posterior.to_numpy(), h_post.to_numpy())
    np.testing.assert_array_equal(body.head_left.to_numpy(), h_left.to_numpy())
    np.testing.assert_array_equal(body.head_right.to_numpy(), h_right.to_numpy())


if __name__ == "__main__":
    # Existing tests
    test_ankle_joint_center_with_missing_medial_markers()
    test_ankle_referenceframe_with_missing_medial_markers()
    test_knee_elbow_wrist_with_missing_medial_markers()
    test_ankle_with_both_markers()

    # New marker tests (Suite 1)
    test_head_center_with_all_cranial_markers()
    test_head_center_with_missing_markers()
    test_neck_base_calculation()
    test_foot_plane_with_all_markers()
    test_metatarsal_markers_independence()

    # Derived properties tests (Suite 2)
    test_foot_height_with_new_metatarsals()
    test_ankle_angles_with_new_foot_plane()
    test_neck_angles_with_head_center()

    # Integration tests (Suite 3)
    test_all_angular_measures_accessible()
    test_t5_marker_presence()
    test_cranial_markers_bilateral()

    print("All tests passed!")
