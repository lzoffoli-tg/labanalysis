"""Shared fixtures for body angles tests."""

import numpy as np
import pytest

from labanalysis.timeseries import Point3D


@pytest.fixture
def neutral_stance_markers():
    """Create marker positions for neutral anatomical stance (all angles ~0°)."""
    n_samples = 100
    time = np.linspace(0, 1, n_samples)

    # Neutral standing position markers (simplified)
    markers = {
        "left_asis": Point3D(
            data=np.tile([0.0, 0.1, 1.0], (n_samples, 1)),
            index=time,
            columns=["X", "Y", "Z"],
            unit="m",
        ),
        "right_asis": Point3D(
            data=np.tile([0.0, -0.1, 1.0], (n_samples, 1)),
            index=time,
            columns=["X", "Y", "Z"],
            unit="m",
        ),
        "left_psis": Point3D(
            data=np.tile([-0.1, 0.1, 1.0], (n_samples, 1)),
            index=time,
            columns=["X", "Y", "Z"],
            unit="m",
        ),
        "right_psis": Point3D(
            data=np.tile([-0.1, -0.1, 1.0], (n_samples, 1)),
            index=time,
            columns=["X", "Y", "Z"],
            unit="m",
        ),
    }

    return markers


@pytest.fixture
def flexed_joint_markers():
    """Create markers with joints in flexed positions (known angles)."""
    # Placeholder - would need specific joint configurations
    return neutral_stance_markers()
