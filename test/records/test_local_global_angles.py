"""
Test suite for local vs global angle measurements.

Tests the sign conventions and basic functionality of global (earth-fixed)
and local (body-segment) reference frame angle measurements.
"""

import numpy as np
import pytest

from labanalysis.records.bodies import WholeBody
from labanalysis.records.timeseries import Point3D


@pytest.fixture
def neutral_wholebody():
    """
    Create a WholeBody with neutral upright posture.

    Note: This creates minimal required markers. Some shoulder calculations
    may not work without additional markers (elbow, etc.).
    """
    n_frames = 10
    time_index = list(range(n_frames))

    # Pelvis markers - level and square (neutral position)
    # Using realistic anatomical spacing
    pelvis_width = 20.0  # cm
    pelvis_depth = 10.0  # cm
    pelvis_height = 100.0  # cm

    left_asis = Point3D(
        data=np.column_stack([
            np.full(n_frames, -pelvis_width/2),  # left
            np.full(n_frames, pelvis_height),
            np.full(n_frames, pelvis_depth/2),   # anterior
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    right_asis = Point3D(
        data=np.column_stack([
            np.full(n_frames, pelvis_width/2),   # right
            np.full(n_frames, pelvis_height),
            np.full(n_frames, pelvis_depth/2),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    left_psis = Point3D(
        data=np.column_stack([
            np.full(n_frames, -pelvis_width/2),
            np.full(n_frames, pelvis_height),
            np.full(n_frames, -pelvis_depth/2),  # posterior
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    right_psis = Point3D(
        data=np.column_stack([
            np.full(n_frames, pelvis_width/2),
            np.full(n_frames, pelvis_height),
            np.full(n_frames, -pelvis_depth/2),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Greater trochanter markers (for hip joint center calculation)
    left_trochanter = Point3D(
        data=np.column_stack([
            np.full(n_frames, -pelvis_width/2 - 2.0),
            np.full(n_frames, pelvis_height - 10.0),
            np.full(n_frames, 0.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    right_trochanter = Point3D(
        data=np.column_stack([
            np.full(n_frames, pelvis_width/2 + 2.0),
            np.full(n_frames, pelvis_height - 10.0),
            np.full(n_frames, 0.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    # Neck markers
    neck_height = 150.0
    c7 = Point3D(
        data=np.column_stack([
            np.zeros(n_frames),
            np.full(n_frames, neck_height),
            np.full(n_frames, -2.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    sc = Point3D(
        data=np.column_stack([
            np.zeros(n_frames),
            np.full(n_frames, neck_height - 5.0),
            np.full(n_frames, 3.0),
        ]),
        index=time_index,
        columns=["X", "Y", "Z"],
    )

    return WholeBody(
        left_asis=left_asis,
        right_asis=right_asis,
        left_psis=left_psis,
        right_psis=right_psis,
        left_trochanter=left_trochanter,
        right_trochanter=right_trochanter,
        c7=c7,
        sc=sc,
    )


class TestPelvisLateralTilt:
    """Test pelvis lateral tilt measurements."""

    def test_sign_convention_left_higher(self, neutral_wholebody):
        """Test that positive angle means left hip higher."""
        body = neutral_wholebody

        # Raise left hip markers by 5 cm
        body.left_asis._data[:, 1] += 5.0
        body.left_psis._data[:, 1] += 5.0
        body.left_trochanter._data[:, 1] += 5.0

        global_tilt = body.pelvis_lateraltilt_global.mean()
        local_tilt = body.pelvis_lateraltilt_local.mean()

        # Both should be positive (left hip higher)
        assert global_tilt > 0, f"Global tilt = {global_tilt:.2f}°, expected > 0° (left hip higher)"
        assert local_tilt > 0, f"Local tilt = {local_tilt:.2f}°, expected > 0° (left hip higher)"

    def test_sign_convention_right_higher(self, neutral_wholebody):
        """Test that negative angle means right hip higher."""
        body = neutral_wholebody

        # Raise right hip markers by 5 cm
        body.right_asis._data[:, 1] += 5.0
        body.right_psis._data[:, 1] += 5.0
        body.right_trochanter._data[:, 1] += 5.0

        global_tilt = body.pelvis_lateraltilt_global.mean()
        local_tilt = body.pelvis_lateraltilt_local.mean()

        # Both should be negative (right hip higher)
        assert global_tilt < 0, f"Global tilt = {global_tilt:.2f}°, expected < 0° (right hip higher)"
        assert local_tilt < 0, f"Local tilt = {local_tilt:.2f}°, expected < 0° (right hip higher)"

    def test_global_and_local_same_sign(self, neutral_wholebody):
        """Test that global and local have the same sign for a given tilt."""
        body = neutral_wholebody

        # Raise left hip
        body.left_asis._data[:, 1] += 5.0
        body.left_psis._data[:, 1] += 5.0
        body.left_trochanter._data[:, 1] += 5.0

        global_tilt = body.pelvis_lateraltilt_global.mean()
        local_tilt = body.pelvis_lateraltilt_local.mean()

        # Should have the same sign (both positive in this case)
        assert global_tilt * local_tilt > 0, \
            f"Global ({global_tilt:.2f}°) and local ({local_tilt:.2f}°) should have the same sign"


class TestPelvisRotation:
    """Test pelvis rotation measurements with dedicated verification."""

    def test_sign_convention_left_forward(self, neutral_wholebody):
        """Test that positive angle means left hip forward."""
        body = neutral_wholebody

        # Move left hip forward by 10 cm (larger displacement for clearer signal)
        body.left_asis._data[:, 2] += 10.0
        body.left_psis._data[:, 2] += 10.0
        body.left_trochanter._data[:, 2] += 10.0

        global_rot = body.pelvis_rotation_global.mean()
        local_rot = body.pelvis_rotation_local.mean()

        # Both should be positive (left hip forward)
        assert global_rot > 0, f"Global rotation = {global_rot:.2f}°, expected > 0° (left hip forward)"
        assert local_rot > 0, f"Local rotation = {local_rot:.2f}°, expected > 0° (left hip forward)"

    def test_sign_convention_right_forward(self, neutral_wholebody):
        """Test that negative angle means right hip forward."""
        body = neutral_wholebody

        # Move right hip forward by 10 cm
        body.right_asis._data[:, 2] += 10.0
        body.right_psis._data[:, 2] += 10.0
        body.right_trochanter._data[:, 2] += 10.0

        global_rot = body.pelvis_rotation_global.mean()
        local_rot = body.pelvis_rotation_local.mean()

        # Both should be negative (right hip forward)
        assert global_rot < 0, f"Global rotation = {global_rot:.2f}°, expected < 0° (right hip forward)"
        assert local_rot < 0, f"Local rotation = {local_rot:.2f}°, expected < 0° (right hip forward)"

    def test_global_and_local_same_sign(self, neutral_wholebody):
        """Test that global and local have the same sign for a given rotation."""
        body = neutral_wholebody

        # Move left hip forward
        body.left_asis._data[:, 2] += 10.0
        body.left_psis._data[:, 2] += 10.0
        body.left_trochanter._data[:, 2] += 10.0

        global_rot = body.pelvis_rotation_global.mean()
        local_rot = body.pelvis_rotation_local.mean()

        # Should have the same sign (both positive in this case)
        assert global_rot * local_rot > 0, \
            f"Global ({global_rot:.2f}°) and local ({local_rot:.2f}°) should have the same sign"


class TestPelvisAnteroposteriorTilt:
    """Test pelvis anteroposterior tilt measurement."""

    def test_sign_convention_anterior_tilt(self, neutral_wholebody):
        """Test that positive angle means anterior tilt (ASIS lower/forward than PSIS)."""
        body = neutral_wholebody

        # Anterior tilt: ASIS lower than PSIS
        # Lower ASIS markers by 10 cm
        body.left_asis._data[:, 1] -= 10.0
        body.right_asis._data[:, 1] -= 10.0

        ap_tilt = body.pelvis_anteroposterior_tilt_global.mean()

        # Should be positive (anterior tilt)
        assert ap_tilt > 0, f"AP tilt = {ap_tilt:.2f}°, expected > 0° (anterior tilt)"

    def test_sign_convention_posterior_tilt(self, neutral_wholebody):
        """Test that negative angle means posterior tilt (PSIS lower/forward than ASIS)."""
        body = neutral_wholebody

        # Posterior tilt: PSIS lower than ASIS
        # Lower PSIS markers by 10 cm
        body.left_psis._data[:, 1] -= 10.0
        body.right_psis._data[:, 1] -= 10.0

        ap_tilt = body.pelvis_anteroposterior_tilt_global.mean()

        # Should be negative (posterior tilt)
        assert ap_tilt < 0, f"AP tilt = {ap_tilt:.2f}°, expected < 0° (posterior tilt)"


class TestPropertyExistence:
    """Test that all required properties exist and are callable."""

    def test_pelvis_properties_exist(self, neutral_wholebody):
        """Test that pelvis angle properties exist and return Signal1D."""
        body = neutral_wholebody

        pelvis_props = [
            'pelvis_anteroposterior_tilt_global',
            'pelvis_lateraltilt_global',
            'pelvis_lateraltilt_local',
            'pelvis_rotation_global',
            'pelvis_rotation_local',
        ]

        for prop_name in pelvis_props:
            assert hasattr(body, prop_name), f"Property {prop_name} does not exist"
            result = getattr(body, prop_name)
            assert result is not None, f"Property {prop_name} returned None"
            assert hasattr(result, 'mean'), f"Property {prop_name} does not return Signal1D"

    def test_shoulder_properties_exist(self):
        """Test that shoulder properties exist (may require additional markers to compute)."""
        # Just verify the properties are defined on the class
        from labanalysis.records.bodies import WholeBody

        shoulder_props = [
            'shoulder_lateraltilt_global',
            'shoulder_lateraltilt_local',
        ]

        for prop_name in shoulder_props:
            assert hasattr(WholeBody, prop_name), f"Property {prop_name} does not exist on WholeBody class"
            prop = getattr(WholeBody, prop_name)
            assert isinstance(prop, property), f"{prop_name} is not a property"


class TestConsistency:
    """Test internal consistency of measurements - covered by other tests."""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
