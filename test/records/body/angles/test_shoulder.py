"""Tests for shoulder angles."""

import numpy as np
import pandas as pd
import pytest
from labanalysis.records.body import WholeBody
from labanalysis.timeseries import Point3D


@pytest.mark.unit
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
            'left_trochanter': Point3D(data=np.array([[0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_trochanter': Point3D(data=np.array([[-0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),

            'c7': Point3D(data=np.array([[0.0, 1.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'sc': Point3D(data=np.array([[0.0, 1.45, 0.05]]*n_frames), index=time_index, unit='m'),

            'left_acromion': Point3D(data=np.array([[0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_acromion': Point3D(data=np.array([[-0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),


            # Elbow markers (for shoulder calculation)
            'left_elbow_lateral': Point3D(data=np.array([[0.25, 1.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_elbow_lateral': Point3D(data=np.array([[-0.25, 1.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_knee': Point3D(data=np.array([[0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_knee': Point3D(data=np.array([[-0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_ankle': Point3D(data=np.array([[0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_ankle': Point3D(data=np.array([[-0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
        }

        return WholeBody(**markers)

    def test_neutral_posture(self):
        """Test that neutral posture gives angles close to zero."""
        body = self.create_neutral_body()

        shoulder_tilt_global = body.shoulder_lateral_tilt_global.mean()
        shoulder_tilt_local = body.shoulder_lateral_tilt_local.mean()

        tolerance = 5.0
        assert abs(shoulder_tilt_global) < tolerance, f"shoulder_lateral_tilt_global = {shoulder_tilt_global:.2f}°"
        assert abs(shoulder_tilt_local) < tolerance, f"shoulder_lateral_tilt_local = {shoulder_tilt_local:.2f}°"

    def test_left_shoulder_raised(self):
        """Test shoulder lateral tilt when left shoulder is raised."""
        n_frames = 10
        time_index = pd.RangeIndex(n_frames)

        markers = {
            'left_asis': Point3D(data=np.array([[0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'right_asis': Point3D(data=np.array([[-0.1, 1.0, 0.05]]*n_frames), index=time_index, unit='m'),
            'left_psis': Point3D(data=np.array([[0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'right_psis': Point3D(data=np.array([[-0.1, 1.0, -0.05]]*n_frames), index=time_index, unit='m'),
            'left_trochanter': Point3D(data=np.array([[0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_trochanter': Point3D(data=np.array([[-0.15, 0.9, 0.0]]*n_frames), index=time_index, unit='m'),

            'c7': Point3D(data=np.array([[0.0, 1.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'sc': Point3D(data=np.array([[0.0, 1.45, 0.05]]*n_frames), index=time_index, unit='m'),

            # Left shoulder raised by 0.05m
            'left_acromion': Point3D(data=np.array([[0.2, 1.45, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_acromion': Point3D(data=np.array([[-0.2, 1.4, 0.0]]*n_frames), index=time_index, unit='m'),


            # Elbow markers (for shoulder calculation)
            'left_elbow_lateral': Point3D(data=np.array([[0.25, 1.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_elbow_lateral': Point3D(data=np.array([[-0.25, 1.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_knee': Point3D(data=np.array([[0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_knee': Point3D(data=np.array([[-0.1, 0.5, 0.0]]*n_frames), index=time_index, unit='m'),
            'left_ankle': Point3D(data=np.array([[0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
            'right_ankle': Point3D(data=np.array([[-0.1, 0.1, 0.0]]*n_frames), index=time_index, unit='m'),
        }

        body = WholeBody(**markers)

        shoulder_tilt_global = body.shoulder_lateral_tilt_global.mean()
        shoulder_tilt_local = body.shoulder_lateral_tilt_local.mean()

        # Both should be positive (left shoulder higher)
        assert shoulder_tilt_global > 0, f"shoulder_lateral_tilt_global = {shoulder_tilt_global:.2f}°"
        assert shoulder_tilt_local > 0, f"shoulder_lateral_tilt_local = {shoulder_tilt_local:.2f}°"
