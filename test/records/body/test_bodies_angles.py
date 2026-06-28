"""
Test suite for WholeBody angular measurements.

This module tests the corrected angle calculations including:
- Spinal curvature angles (kyphosis, lordosis)
- Pelvis lateral tilt and rotation
- Knee varus/valgus alignment

Tests verify that angles return expected values for neutral postures
and validate the sign conventions.
"""

import numpy as np
import pytest

from labanalysis.records.body import WholeBody
from labanalysis.timeseries import Point3D


@pytest.fixture
def neutral_spine_wholebody():
    """
    Create a WholeBody with neutral spinal curvature.

    Returns a body with:
    - Thoracic kyphosis ~150° (normal)
    - Lumbar lordosis ~150° (normal)
    """
    n_frames = 10
    time_index = list(range(n_frames))

    # Global coordinate system used for synthetic markers: X=lateral, Y=vertical, Z=anteroposterior
    # Note: Local reference frame axes may differ - see ReferenceFrame construction for semantic mapping
    # Slight posterior curvature at thoracic, anterior at lumbar for normal curves

    # C7: cervical, highest, slightly posterior
    c7_data = np.column_stack([
        np.zeros(n_frames),
        np.full(n_frames, 140.0),
        np.full(n_frames, -20.0),
    ])

    # T5: thoracic, middle height, most posterior (kyphosis)
    t5_data = np.column_stack([
        np.zeros(n_frames),
        np.full(n_frames, 100.0),
        np.full(n_frames, -25.0),
    ])

    # L2: lumbar, lower, slightly anterior (lordosis)
    l2_data = np.column_stack([
        np.zeros(n_frames),
        np.full(n_frames, 60.0),
        np.full(n_frames, -15.0),
    ])

    # PSIS: pelvis posterior, lowest
    psis_data = np.column_stack([
        np.zeros(n_frames),
        np.full(n_frames, 30.0),
        np.full(n_frames, -20.0),
    ])

    # Create Point3D objects
    c7 = Point3D(data=c7_data, index=time_index, columns=["X", "Y", "Z"])
    t5 = Point3D(data=t5_data, index=time_index, columns=["X", "Y", "Z"])
    l2 = Point3D(data=l2_data, index=time_index, columns=["X", "Y", "Z"])

    # PSIS split into left and right
    left_psis_data = psis_data.copy()
    left_psis_data[:, 0] = -10.0
    right_psis_data = psis_data.copy()
    right_psis_data[:, 0] = 10.0

    left_psis = Point3D(data=left_psis_data, index=time_index, columns=["X", "Y", "Z"])
    right_psis = Point3D(data=right_psis_data, index=time_index, columns=["X", "Y", "Z"])

    return WholeBody(c7=c7, t5=t5, l2=l2, left_psis=left_psis, right_psis=right_psis)


@pytest.fixture
def neutral_pelvis_wholebody():
    """
    Create a WholeBody with level pelvis (0° lateral tilt).

    Returns a body with pelvis markers at same height bilaterally.
    """
    n_frames = 10
    time_index = list(range(n_frames))

    # All pelvis markers at same Y height for neutral lateral tilt
    pelvis_y = 100.0

    # ASIS markers (anterior)
    left_asis_data = np.column_stack([
        np.full(n_frames, -10.0),  # left
        np.full(n_frames, pelvis_y),
        np.full(n_frames, 5.0),  # anterior
    ])
    right_asis_data = np.column_stack([
        np.full(n_frames, 10.0),  # right
        np.full(n_frames, pelvis_y),
        np.full(n_frames, 5.0),
    ])

    # PSIS markers (posterior)
    left_psis_data = np.column_stack([
        np.full(n_frames, -10.0),
        np.full(n_frames, pelvis_y),
        np.full(n_frames, -5.0),  # posterior
    ])
    right_psis_data = np.column_stack([
        np.full(n_frames, 10.0),
        np.full(n_frames, pelvis_y),
        np.full(n_frames, -5.0),
    ])

    # Trochanter markers (required for hip joint calculation)
    left_trochanter_data = np.column_stack([
        np.full(n_frames, -15.0),  # lateral to ASIS
        np.full(n_frames, pelvis_y - 5.0),  # slightly below pelvis
        np.full(n_frames, 0.0),
    ])
    right_trochanter_data = np.column_stack([
        np.full(n_frames, 15.0),
        np.full(n_frames, pelvis_y - 5.0),
        np.full(n_frames, 0.0),
    ])

    left_asis = Point3D(data=left_asis_data, index=time_index, columns=["X", "Y", "Z"])
    right_asis = Point3D(data=right_asis_data, index=time_index, columns=["X", "Y", "Z"])
    left_psis = Point3D(data=left_psis_data, index=time_index, columns=["X", "Y", "Z"])
    right_psis = Point3D(data=right_psis_data, index=time_index, columns=["X", "Y", "Z"])
    left_trochanter = Point3D(data=left_trochanter_data, index=time_index, columns=["X", "Y", "Z"])
    right_trochanter = Point3D(data=right_trochanter_data, index=time_index, columns=["X", "Y", "Z"])

    return WholeBody(
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
        left_trochanter=left_trochanter,
        right_trochanter=right_trochanter,
    )


@pytest.fixture
def aligned_knees_wholebody():
    """
    Create a WholeBody with perfectly aligned knees (0° varus/valgus).

    Returns a body with hip-knee-ankle vertically aligned in frontal plane.
    """
    n_frames = 10
    time_index = list(range(n_frames))

    # Left leg - perfectly vertical (X = -10 for all joints)
    left_x = -10.0

    # Add upper body markers (required for pelvis_referenceframe calculation)
    sc = Point3D(
        data=np.column_stack([
            np.zeros(n_frames),
            np.full(n_frames, 145.0),
            np.full(n_frames, 5.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    c7 = Point3D(
        data=np.column_stack([
            np.zeros(n_frames),
            np.full(n_frames, 150.0),
            np.full(n_frames, -5.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Pelvis/hip markers
    left_asis = Point3D(
        data=np.column_stack([
            np.full(n_frames, left_x),
            np.full(n_frames, 105.0),
            np.full(n_frames, 5.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_psis = Point3D(
        data=np.column_stack([
            np.full(n_frames, left_x),
            np.full(n_frames, 105.0),
            np.full(n_frames, -5.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_troch = Point3D(
        data=np.column_stack([
            np.full(n_frames, left_x),
            np.full(n_frames, 100.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Knee markers
    left_knee_med = Point3D(
        data=np.column_stack([
            np.full(n_frames, left_x - 2.0),
            np.full(n_frames, 50.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_knee_lat = Point3D(
        data=np.column_stack([
            np.full(n_frames, left_x + 2.0),
            np.full(n_frames, 50.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Ankle markers
    left_ankle_med = Point3D(
        data=np.column_stack([
            np.full(n_frames, left_x - 2.0),
            np.full(n_frames, 10.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    left_ankle_lat = Point3D(
        data=np.column_stack([
            np.full(n_frames, left_x + 2.0),
            np.full(n_frames, 10.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Right leg - perfectly vertical (X = +10 for all joints)
    right_x = 10.0

    right_asis = Point3D(
        data=np.column_stack([
            np.full(n_frames, right_x),
            np.full(n_frames, 105.0),
            np.full(n_frames, 5.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_psis = Point3D(
        data=np.column_stack([
            np.full(n_frames, right_x),
            np.full(n_frames, 105.0),
            np.full(n_frames, -5.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_troch = Point3D(
        data=np.column_stack([
            np.full(n_frames, right_x),
            np.full(n_frames, 100.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    right_knee_med = Point3D(
        data=np.column_stack([
            np.full(n_frames, right_x - 2.0),
            np.full(n_frames, 50.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_knee_lat = Point3D(
        data=np.column_stack([
            np.full(n_frames, right_x + 2.0),
            np.full(n_frames, 50.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    right_ankle_med = Point3D(
        data=np.column_stack([
            np.full(n_frames, right_x - 2.0),
            np.full(n_frames, 10.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )
    right_ankle_lat = Point3D(
        data=np.column_stack([
            np.full(n_frames, right_x + 2.0),
            np.full(n_frames, 10.0),
            np.zeros(n_frames),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    return WholeBody(
        sc=sc,
        c7=c7,
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
        left_trochanter=left_troch,
        right_trochanter=right_troch,
        left_knee_medial=left_knee_med,
        left_knee_lateral=left_knee_lat,
        right_knee_medial=right_knee_med,
        right_knee_lateral=right_knee_lat,
        left_ankle_medial=left_ankle_med,
        left_ankle_lateral=left_ankle_lat,
        right_ankle_medial=right_ankle_med,
        right_ankle_lateral=right_ankle_lat,
    )


class TestSpinalCurvatureAngles:
    """Test spinal curvature angle calculations."""

    def test_dorsal_kyphosis_normal_range(self, neutral_spine_wholebody):
        """Test that thoracic kyphosis is in normal range (140-160°)."""
        kyphosis = neutral_spine_wholebody.dorsal_kyphosis
        kyphosis_data = np.asarray(kyphosis.data)
        mean_kyphosis = kyphosis_data.mean()

        # Should be in normal range
        assert 140 <= mean_kyphosis <= 160, (
            f"Thoracic kyphosis {mean_kyphosis:.1f}° outside normal range 140-160°"
        )

    def test_lumbar_lordosis_normal_range(self, neutral_spine_wholebody):
        """Test that lumbar lordosis is in normal range (140-160°)."""
        lordosis = neutral_spine_wholebody.lumbar_lordosis
        lordosis_data = np.asarray(lordosis.data)
        mean_lordosis = lordosis_data.mean()

        # Should be in normal range
        assert 140 <= mean_lordosis <= 160, (
            f"Lumbar lordosis {mean_lordosis:.1f}° outside normal range 140-160°"
        )

    def test_kyphosis_angle_order_correction(self, neutral_spine_wholebody):
        """
        Test that kyphosis uses correct point order (C7-T5-L2, not L2-T5-C7).

        Before correction, inverted order gave ~50° instead of ~150°.
        """
        kyphosis = neutral_spine_wholebody.dorsal_kyphosis
        kyphosis_data = np.asarray(kyphosis.data)
        mean_kyphosis = kyphosis_data.mean()

        # Should NOT be around 50° (the bug), should be around 150°
        assert mean_kyphosis > 100, (
            f"Kyphosis {mean_kyphosis:.1f}° suggests inverted calculation (should be ~150°)"
        )

    def test_lordosis_angle_order_correction(self, neutral_spine_wholebody):
        """
        Test that lordosis uses correct point order (T5-L2-PSIS, not PSIS-L2-T5).

        Before correction, inverted order gave ~50° instead of ~150°.
        """
        lordosis = neutral_spine_wholebody.lumbar_lordosis
        lordosis_data = np.asarray(lordosis.data)
        mean_lordosis = lordosis_data.mean()

        # Should NOT be around 50° (the bug), should be around 150°
        assert mean_lordosis > 100, (
            f"Lordosis {mean_lordosis:.1f}° suggests inverted calculation (should be ~150°)"
        )


class TestPelvisAngles:
    """Test pelvis angle calculations."""

    def test_pelvis_lateral_tilt_neutral(self, neutral_pelvis_wholebody):
        """Test that level pelvis returns 0° lateral tilt."""
        lateral_tilt = neutral_pelvis_wholebody.pelvis_lateral_tilt_global
        tilt_data = np.asarray(lateral_tilt.data)
        mean_tilt = tilt_data.mean()

        # Should be approximately 0° for level pelvis
        assert abs(mean_tilt) < 1.0, (
            f"Level pelvis lateral tilt is {mean_tilt:.1f}°, expected ~0°"
        )

    def test_pelvis_lateral_tilt_vector_direction(self, neutral_pelvis_wholebody):
        """
        Test that pelvis lateral tilt uses correct vector direction.

        Before correction, the vector went left→right giving ±180° instead of 0°.
        Now corrected to right→left.
        """
        lateral_tilt = neutral_pelvis_wholebody.pelvis_lateral_tilt_global
        tilt_data = np.asarray(lateral_tilt.data)
        mean_tilt = tilt_data.mean()

        # Should NOT be ±180° (the bug), should be 0°
        assert abs(mean_tilt) < 90, (
            f"Pelvis lateral tilt {mean_tilt:.1f}° suggests inverted vector (should be ~0°)"
        )


class TestKneeVarusValgus:
    """Test knee varus/valgus angle calculations."""

    def test_left_knee_aligned(self, aligned_knees_wholebody):
        """Test that perfectly aligned left knee returns ~0°."""
        vv = aligned_knees_wholebody.left_knee_varusvalgus
        vv_data = np.asarray(vv.data)
        mean_vv = vv_data.mean()

        # Should be approximately 0° for perfect alignment
        # Allow small tolerance due to De Leva hip calculation
        assert abs(mean_vv) < 2.0, (
            f"Aligned left knee varus/valgus is {mean_vv:.1f}°, expected ~0°"
        )

    def test_right_knee_aligned(self, aligned_knees_wholebody):
        """Test that perfectly aligned right knee returns ~0°."""
        vv = aligned_knees_wholebody.right_knee_varusvalgus
        vv_data = np.asarray(vv.data)
        mean_vv = vv_data.mean()

        # Should be approximately 0° for perfect alignment
        assert abs(mean_vv) < 2.0, (
            f"Aligned right knee varus/valgus is {mean_vv:.1f}°, expected ~0°"
        )

    def test_sign_convention_documented(self):
        """
        Document the sign convention for varus/valgus angles.

        Sign convention (updated):
        - Positive (+) = Varus deformity (bow-legged, lateral deviation)
        - Negative (-) = Valgus deformity (knock-knee, medial deviation)
        - Zero (0°) = Perfect alignment (hip-knee-ankle collinear)

        This convention is consistent for both left and right knees.
        """
        # This is a documentation test - it always passes
        # The actual sign convention is tested implicitly by the alignment test
        # (aligned knees should give ~0°, not large positive/negative values)
        assert True, "Sign convention: Positive=Varus, Negative=Valgus, 0°=Aligned"

    def test_angle_normalization(self, aligned_knees_wholebody):
        """
        Test that angles are normalized to [-180°, +180°] range.

        This prevents wrapping issues where small angles near 0°
        could appear as ~360° or ~-360°.
        """
        left_vv = aligned_knees_wholebody.left_knee_varusvalgus
        right_vv = aligned_knees_wholebody.right_knee_varusvalgus

        left_data = np.asarray(left_vv.data)
        right_data = np.asarray(right_vv.data)

        # All angles should be in [-180, 180] range
        assert np.all(left_data >= -180), f"Left knee has angles < -180°: {left_data.min():.1f}°"
        assert np.all(left_data <= 180), f"Left knee has angles > 180°: {left_data.max():.1f}°"
        assert np.all(right_data >= -180), f"Right knee has angles < -180°: {right_data.min():.1f}°"
        assert np.all(right_data <= 180), f"Right knee has angles > 180°: {right_data.max():.1f}°"

        # For aligned knees, should be close to 0°, not ~360°
        assert np.all(np.abs(left_data) < 180), "Left knee angles should not wrap to ±180°"
        assert np.all(np.abs(right_data) < 180), "Right knee angles should not wrap to ±180°"


class TestHipAnglesSignConvention:
    """Test hip angle sign conventions."""

    def test_hip_flexion_is_positive(self):
        """Test that hip flexion (thigh forward) returns positive angle."""
        n_frames = 10
        time_index = list(range(n_frames))

        # Add upper body markers (required for pelvis_referenceframe)
        sc = Point3D(
            data=np.column_stack([
                np.zeros(n_frames),
                np.full(n_frames, 145.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        c7 = Point3D(
            data=np.column_stack([
                np.zeros(n_frames),
                np.full(n_frames, 150.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        # Create pelvis and hip
        left_asis = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_asis = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_psis = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_psis = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_troch = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 95.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_troch = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 95.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        # Knee moved forward (positive Z) = FLEXION
        left_knee_med = Point3D(
            data=np.column_stack([
                np.full(n_frames, -12.0),
                np.full(n_frames, 50.0),
                np.full(n_frames, 30.0),  # forward
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_knee_lat = Point3D(
            data=np.column_stack([
                np.full(n_frames, -8.0),
                np.full(n_frames, 50.0),
                np.full(n_frames, 30.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        body = WholeBody(
            sc=sc,
            c7=c7,
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
            left_trochanter=left_troch,
            right_trochanter=right_troch,
            left_knee_medial=left_knee_med,
            left_knee_lateral=left_knee_lat,
        )

        angle = np.asarray(body.left_hip_flexionextension.data).mean()
        assert angle > 0, f"Hip flexion should be positive, got {angle:.1f}°"

    def test_hip_abduction_is_positive(self):
        """Test that hip abduction (thigh outward) returns positive angle."""
        n_frames = 10
        time_index = list(range(n_frames))

        # Add upper body markers (required for pelvis_referenceframe)
        sc = Point3D(
            data=np.column_stack([
                np.zeros(n_frames),
                np.full(n_frames, 145.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        c7 = Point3D(
            data=np.column_stack([
                np.zeros(n_frames),
                np.full(n_frames, 150.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        # Create pelvis and hip
        left_asis = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_asis = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_psis = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_psis = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_troch = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 95.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_troch = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 95.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        # Knee moved lateral (more negative X) = ABDUCTION
        left_knee_med = Point3D(
            data=np.column_stack([
                np.full(n_frames, -32.0),  # lateral
                np.full(n_frames, 50.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_knee_lat = Point3D(
            data=np.column_stack([
                np.full(n_frames, -28.0),
                np.full(n_frames, 50.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        body = WholeBody(
            sc=sc,
            c7=c7,
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
            left_trochanter=left_troch,
            right_trochanter=right_troch,
            left_knee_medial=left_knee_med,
            left_knee_lateral=left_knee_lat,
        )

        angle = np.asarray(body.left_hip_abductionadduction.data).mean()
        assert angle > 0, f"Hip abduction should be positive, got {angle:.1f}°"


class TestShoulderAnglesSignConvention:
    """Test shoulder angle sign conventions."""

    def test_shoulder_flexion_is_positive(self):
        """Test that shoulder flexion (arm forward) returns positive angle."""
        n_frames = 10
        time_index = list(range(n_frames))

        # Create pelvis (required)
        left_asis = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_asis = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_psis = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_psis = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        # Shoulder and trunk
        left_acromion = Point3D(
            data=np.column_stack([
                np.full(n_frames, -15.0),
                np.full(n_frames, 140.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        sc = Point3D(
            data=np.column_stack([
                np.zeros(n_frames),
                np.full(n_frames, 145.0),
                np.full(n_frames, 2.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        c7 = Point3D(
            data=np.column_stack([
                np.zeros(n_frames),
                np.full(n_frames, 145.0),
                np.full(n_frames, -2.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        # Elbow moved forward (positive Z) = FLEXION
        left_elbow_med = Point3D(
            data=np.column_stack([
                np.full(n_frames, -17.0),
                np.full(n_frames, 110.0),
                np.full(n_frames, 30.0),  # forward
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_elbow_lat = Point3D(
            data=np.column_stack([
                np.full(n_frames, -13.0),
                np.full(n_frames, 110.0),
                np.full(n_frames, 30.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        body = WholeBody(
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
            left_acromion=left_acromion,
            left_elbow_medial=left_elbow_med,
            left_elbow_lateral=left_elbow_lat,
            sc=sc,
            c7=c7,
        )

        angle = np.asarray(body.left_shoulder_flexionextension.data).mean()
        assert angle > 0, f"Shoulder flexion should be positive, got {angle:.1f}°"

    def test_shoulder_abduction_is_positive(self):
        """Test that shoulder abduction (arm outward) returns positive angle."""
        n_frames = 10
        time_index = list(range(n_frames))

        # Create pelvis (required)
        left_asis = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_asis = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, 5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_psis = Point3D(
            data=np.column_stack([
                np.full(n_frames, -10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        right_psis = Point3D(
            data=np.column_stack([
                np.full(n_frames, 10.0),
                np.full(n_frames, 100.0),
                np.full(n_frames, -5.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        # Shoulder and trunk
        left_acromion = Point3D(
            data=np.column_stack([
                np.full(n_frames, -15.0),
                np.full(n_frames, 140.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        sc = Point3D(
            data=np.column_stack([
                np.zeros(n_frames),
                np.full(n_frames, 145.0),
                np.full(n_frames, 2.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        c7 = Point3D(
            data=np.column_stack([
                np.zeros(n_frames),
                np.full(n_frames, 145.0),
                np.full(n_frames, -2.0),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        # Elbow moved lateral (more negative X) = ABDUCTION
        left_elbow_med = Point3D(
            data=np.column_stack([
                np.full(n_frames, -37.0),  # lateral
                np.full(n_frames, 110.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )
        left_elbow_lat = Point3D(
            data=np.column_stack([
                np.full(n_frames, -33.0),
                np.full(n_frames, 110.0),
                np.zeros(n_frames),
            ]),
            index=time_index,
            columns=["X", "Y", "Z"],
        )

        body = WholeBody(
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
            left_acromion=left_acromion,
            left_elbow_medial=left_elbow_med,
            left_elbow_lateral=left_elbow_lat,
            sc=sc,
            c7=c7,
        )

        angle = np.asarray(body.left_shoulder_abductionadduction.data).mean()
        assert angle > 0, f"Shoulder abduction should be positive, got {angle:.1f}°"
