"""
Extensive tests for pelvis and shoulder lateral tilt and rotation angles.

This test suite validates the implementation of global and local angle measurements
and demonstrates the differences between them in various postural scenarios.
"""

import numpy as np
import pandas as pd
import pytest
from labanalysis.records.bodies import WholeBody
from labanalysis.records.timeseries import Point3D


class TestPelvisAngles:
    """Test pelvis lateral tilt and rotation angles."""

    def create_neutral_body(self, n_frames=10):
        """
        Create a WholeBody with neutral posture.

        Global axes: X=lateral (left), Y=vertical (up), Z=anteroposterior (forward)
        """
        time_index = pd.RangeIndex(n_frames)

        markers = {
            # Pelvis markers - level and square
            'left_asis': Point3D(data=np.array([[0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'right_asis': Point3D(data=np.array([[-0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'left_psis': Point3D(data=np.array([[0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'right_psis': Point3D(data=np.array([[-0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'left_greater_trochanter': Point3D(data=np.array([[0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_greater_trochanter': Point3D(data=np.array([[-0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),

            # Neck markers
            'c7': Point3D(data=np.array([[0.0, 1.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'sc': Point3D(data=np.array([[0.0, 1.45, 0.05]]*n_frames), index=time_index, unit='m'),

            # Shoulders
            'left_shoulder': Point3D(data=np.array([[0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_shoulder': Point3D(data=np.array([[-0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),

            # Lower limb (minimal set)
            'left_knee': Point3D(data=np.array([[0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_knee': Point3D(data=np.array([[-0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_ankle': Point3D(data=np.array([[0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_ankle': Point3D(data=np.array([[-0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
        }

        return WholeBody(**markers)

    def test_neutral_posture(self):
        """Test that neutral posture gives angles close to zero."""
        body = self.create_neutral_body()

        # Pelvis lateral tilt
        lat_tilt_global = body.pelvis_lateraltilt_global.mean()
        lat_tilt_local = body.pelvis_lateraltilt_local.mean()

        # Pelvis rotation
        rot_global = body.pelvis_rotation_global.mean()
        rot_local = body.pelvis_rotation_local.mean()

        # All angles should be close to zero in neutral posture
        tolerance = 5.0  # degrees
        assert abs(lat_tilt_global) < tolerance, f"pelvis_lateraltilt_global = {lat_tilt_global:.2f}°, expected ~0°"
        assert abs(lat_tilt_local) < tolerance, f"pelvis_lateraltilt_local = {lat_tilt_local:.2f}°, expected ~0°"
        assert abs(rot_global) < tolerance, f"pelvis_rotation_global = {rot_global:.2f}°, expected ~0°"
        assert abs(rot_local) < tolerance, f"pelvis_rotation_local = {rot_local:.2f}°, expected ~0°"

    def test_left_hip_raised_global_vs_local(self):
        """
        Test pelvis lateral tilt when left hip is raised.

        Scenario: Left hip raised, trunk vertical.
        Expected: Both global and local should show positive angle (left hip higher).
        """
        n_frames = 10
        time_index = pd.RangeIndex(n_frames)

        # Create body with left hip raised by 0.05m
        markers = {
            'left_asis': Point3D(data=np.array([[0.1, 1.05, 0.05]]*n_frames), index=time_index, unit='m'),
            'right_asis': Point3D(data=np.array([[-0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'left_psis': Point3D(data=np.array([[0.1, 1.05, -0.05]]*n_frames), index=time_index, unit='m'),
            'right_psis': Point3D(data=np.array([[-0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'left_greater_trochanter': Point3D(data=np.array([[0.15, 0.95, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_greater_trochanter': Point3D(data=np.array([[-0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),

            'c7': Point3D(data=np.array([[0.0, 1.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'sc': Point3D(data=np.array([[0.0, 1.45, 0.05]]*n_frames), index=time_index, unit='m'),

            'left_shoulder': Point3D(data=np.array([[0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_shoulder': Point3D(data=np.array([[-0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),

            'left_knee': Point3D(data=np.array([[0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_knee': Point3D(data=np.array([[-0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_ankle': Point3D(data=np.array([[0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_ankle': Point3D(data=np.array([[-0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
        }

        body = WholeBody(**markers)

        lat_tilt_global = body.pelvis_lateraltilt_global.mean()
        lat_tilt_local = body.pelvis_lateraltilt_local.mean()

        # Both should be positive (left hip higher)
        assert lat_tilt_global > 0, f"pelvis_lateraltilt_global = {lat_tilt_global:.2f}°, expected > 0°"
        assert lat_tilt_local > 0, f"pelvis_lateraltilt_local = {lat_tilt_local:.2f}°, expected > 0°"

        # In this case (trunk vertical), they should be similar
        diff = abs(lat_tilt_global - lat_tilt_local)
        assert diff < 10, f"Difference between global and local = {diff:.2f}°, expected small difference"

    def test_trunk_forward_flexion_effect(self):
        """
        Test difference between global and local when trunk is flexed forward.

        Scenario: Pelvis level, trunk flexed forward (C7 moved forward).
        Expected:
        - Global lateral tilt ≈ 0° (pelvis still level relative to gravity)
        - Local lateral tilt ≈ 0° (pelvis level relative to trunk axis too)
        - But if we add pelvis tilt + trunk flexion, we should see differences
        """
        n_frames = 10
        time_index = pd.RangeIndex(n_frames)

        # Pelvis level, but C7 moved forward (trunk flexion)
        markers = {
            'left_asis': Point3D(data=np.array([[0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'right_asis': Point3D(data=np.array([[-0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'left_psis': Point3D(data=np.array([[0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'right_psis': Point3D(data=np.array([[-0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'left_greater_trochanter': Point3D(data=np.array([[0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_greater_trochanter': Point3D(data=np.array([[-0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),

            # C7 moved forward (trunk flexion)
            'c7': Point3D(data=np.array([[0.0, 1.4, 0.3]]*n_frames), index=time_index, unit='m'),
            'sc': Point3D(data=np.array([[0.0, 1.35, 0.35]]*n_frames), index=time_index, unit='m'),

            'left_shoulder': Point3D(data=np.array([[0.2, 1.3, 0.25]]*n_frames), index=time_index, unit='m'),
            'right_shoulder': Point3D(data=np.array([[-0.2, 1.3, 0.25]]*n_frames), index=time_index, unit='m'),

            'left_knee': Point3D(data=np.array([[0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_knee': Point3D(data=np.array([[-0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_ankle': Point3D(data=np.array([[0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_ankle': Point3D(data=np.array([[-0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
        }

        body = WholeBody(**markers)

        lat_tilt_global = body.pelvis_lateraltilt_global.mean()
        lat_tilt_local = body.pelvis_lateraltilt_local.mean()

        # Both should still be close to zero (pelvis is level)
        tolerance = 10.0
        assert abs(lat_tilt_global) < tolerance, f"pelvis_lateraltilt_global = {lat_tilt_global:.2f}°"
        assert abs(lat_tilt_local) < tolerance, f"pelvis_lateraltilt_local = {lat_tilt_local:.2f}°"

    def test_lateral_trunk_flexion_with_pelvis_tilt(self):
        """
        Test that demonstrates the key difference between global and local.

        Scenario: Subject leaning left (trunk + pelvis both tilted left together).
        Expected:
        - Global: shows significant tilt (relative to gravity)
        - Local: shows less tilt (pelvis aligned with trunk axis)
        """
        n_frames = 10
        time_index = pd.RangeIndex(n_frames)

        # Simulate lateral lean: shift everything left and up proportionally
        # This simulates the subject leaning to the left side
        lateral_shift = 0.1  # leftward shift
        vertical_shift_left = 0.03  # left side raised slightly

        markers = {
            # Left pelvis markers raised more
            'left_asis': Point3D(data=np.array([[0.1 + lateral_shift, 1.0 + vertical_shift_left, 0.05]]*n_frames), index=time_index, unit='m'),
            'right_asis': Point3D(data=np.array([[-0.1 + lateral_shift, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'left_psis': Point3D(data=np.array([[0.1 + lateral_shift, 1.0 + vertical_shift_left, -0.05]]*n_frames), index=time_index, unit='m'),
            'right_psis': Point3D(data=np.array([[-0.1 + lateral_shift, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'left_greater_trochanter': Point3D(data=np.array([[0.15 + lateral_shift, 0.9 + vertical_shift_left, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_greater_trochanter': Point3D(data=np.array([[-0.15 + lateral_shift, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),

            # C7 also shifted left and up proportionally (trunk follows pelvis)
            'c7': Point3D(data=np.array([[0.0 + lateral_shift, 1.5 + vertical_shift_left, 0.0]]*n_frames), index=time_index, unit='m'),
            'sc': Point3D(data=np.array([[0.0 + lateral_shift, 1.45 + vertical_shift_left, 0.05]]*n_frames), index=time_index, unit='m'),

            'left_shoulder': Point3D(data=np.array([[0.2 + lateral_shift, 1.4 + vertical_shift_left, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_shoulder': Point3D(data=np.array([[-0.2 + lateral_shift, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),

            'left_knee': Point3D(data=np.array([[0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_knee': Point3D(data=np.array([[-0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_ankle': Point3D(data=np.array([[0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_ankle': Point3D(data=np.array([[-0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
        }

        body = WholeBody(**markers)

        lat_tilt_global = body.pelvis_lateraltilt_global.mean()
        lat_tilt_local = body.pelvis_lateraltilt_local.mean()

        # Global should show tilt (relative to gravity)
        assert lat_tilt_global > 0, f"pelvis_lateraltilt_global = {lat_tilt_global:.2f}°, expected > 0°"

        # Local might be smaller (pelvis more aligned with trunk)
        # The difference demonstrates the compensation for trunk orientation
        print(f"\nLateral trunk flexion test:")
        print(f"  Global: {lat_tilt_global:.2f}°")
        print(f"  Local:  {lat_tilt_local:.2f}°")
        print(f"  Difference: {abs(lat_tilt_global - lat_tilt_local):.2f}°")


class TestShoulderAngles:
    """Test shoulder lateral tilt angles."""

    def create_neutral_body(self, n_frames=10):
        """Create a WholeBody with neutral posture."""
        time_index = pd.RangeIndex(n_frames)

        markers = {
            'left_asis': Point3D(data=np.array([[0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'right_asis': Point3D(data=np.array([[-0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'left_psis': Point3D(data=np.array([[0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'right_psis': Point3D(data=np.array([[-0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'left_greater_trochanter': Point3D(data=np.array([[0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_greater_trochanter': Point3D(data=np.array([[-0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),

            'c7': Point3D(data=np.array([[0.0, 1.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'sc': Point3D(data=np.array([[0.0, 1.45, 0.05]]*n_frames), index=time_index, unit='m'),

            'left_shoulder': Point3D(data=np.array([[0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_shoulder': Point3D(data=np.array([[-0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),

            'left_knee': Point3D(data=np.array([[0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_knee': Point3D(data=np.array([[-0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_ankle': Point3D(data=np.array([[0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_ankle': Point3D(data=np.array([[-0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
        }

        return WholeBody(**markers)

    def test_neutral_posture(self):
        """Test that neutral posture gives angles close to zero."""
        body = self.create_neutral_body()

        shoulder_tilt_global = body.shoulder_lateraltilt_global.mean()
        shoulder_tilt_local = body.shoulder_lateraltilt_local.mean()

        tolerance = 5.0
        assert abs(shoulder_tilt_global) < tolerance, f"shoulder_lateraltilt_global = {shoulder_tilt_global:.2f}°"
        assert abs(shoulder_tilt_local) < tolerance, f"shoulder_lateraltilt_local = {shoulder_tilt_local:.2f}°"

    def test_left_shoulder_raised(self):
        """Test shoulder lateral tilt when left shoulder is raised."""
        n_frames = 10
        time_index = pd.RangeIndex(n_frames)

        markers = {
            'left_asis': Point3D(data=np.array([[0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'right_asis': Point3D(data=np.array([[-0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'left_psis': Point3D(data=np.array([[0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'right_psis': Point3D(data=np.array([[-0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'left_greater_trochanter': Point3D(data=np.array([[0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_greater_trochanter': Point3D(data=np.array([[-0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),

            'c7': Point3D(data=np.array([[0.0, 1.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'sc': Point3D(data=np.array([[0.0, 1.45, 0.05]]*n_frames), index=time_index, unit='m'),

            # Left shoulder raised by 0.05m
            'left_shoulder': Point3D(data=np.array([[0.2, 1.45, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_shoulder': Point3D(data=np.array([[-0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),

            'left_knee': Point3D(data=np.array([[0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_knee': Point3D(data=np.array([[-0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_ankle': Point3D(data=np.array([[0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_ankle': Point3D(data=np.array([[-0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
        }

        body = WholeBody(**markers)

        shoulder_tilt_global = body.shoulder_lateraltilt_global.mean()
        shoulder_tilt_local = body.shoulder_lateraltilt_local.mean()

        # Both should be positive (left shoulder higher)
        assert shoulder_tilt_global > 0, f"shoulder_lateraltilt_global = {shoulder_tilt_global:.2f}°"
        assert shoulder_tilt_local > 0, f"shoulder_lateraltilt_local = {shoulder_tilt_local:.2f}°"


class TestRotationAngles:
    """Test pelvis rotation angles."""

    def test_left_hip_forward(self):
        """Test pelvis rotation when left hip is moved forward."""
        n_frames = 10
        time_index = pd.RangeIndex(n_frames)

        # Left hip moved forward by 0.05m
        markers = {
            'left_asis': Point3D(data=np.array([[0.1, 1.0, 0.10]]*n_frames), index=time_index, unit='m'),
            'right_asis': Point3D(data=np.array([[-0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'left_psis': Point3D(data=np.array([[0.1, 1.0, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_psis': Point3D(data=np.array([[-0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'left_greater_trochanter': Point3D(data=np.array([[0.15, 0.9, 0.05]]*n_frames), index=time_index, unit='m'),
            'right_greater_trochanter': Point3D(data=np.array([[-0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),

            'c7': Point3D(data=np.array([[0.0, 1.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'sc': Point3D(data=np.array([[0.0, 1.45, 0.05]]*n_frames), index=time_index, unit='m'),

            'left_shoulder': Point3D(data=np.array([[0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_shoulder': Point3D(data=np.array([[-0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),

            'left_knee': Point3D(data=np.array([[0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_knee': Point3D(data=np.array([[-0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_ankle': Point3D(data=np.array([[0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_ankle': Point3D(data=np.array([[-0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
        }

        body = WholeBody(**markers)

        rot_global = body.pelvis_rotation_global.mean()
        rot_local = body.pelvis_rotation_local.mean()

        # Both should be positive (left hip forward)
        assert rot_global > 0, f"pelvis_rotation_global = {rot_global:.2f}°, expected > 0°"
        assert rot_local > 0, f"pelvis_rotation_local = {rot_local:.2f}°, expected > 0°"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
