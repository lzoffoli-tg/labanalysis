"""Shared fixtures for exercises tests."""

import numpy as np
import pytest

from labanalysis.records import ForcePlatform
from labanalysis.timeseries import Signal3D, Point3D


@pytest.fixture
def sample_force_platform_data():
    """Create sample force platform data: 100Hz, 2s with contact/flight phases."""
    freq = 100  # Hz
    duration = 2.0  # seconds
    n_samples = int(freq * duration)
    time = np.linspace(0, duration, n_samples)

    # Create vertical force with contact (>30N) and flight (<30N) phases
    force_z = np.zeros(n_samples)
    # Contact phase 1: 0.0-0.3s (>30N)
    contact1 = (time >= 0.0) & (time <= 0.3)
    force_z[contact1] = 500 + 200 * np.sin(2 * np.pi * 5 * time[contact1])
    # Flight phase: 0.3-1.2s (<30N)
    flight = (time > 0.3) & (time < 1.2)
    force_z[flight] = 5
    # Contact phase 2: 1.2-2.0s (>30N)
    contact2 = time >= 1.2
    force_z[contact2] = 600 + 150 * np.sin(2 * np.pi * 5 * time[contact2])

    force = Signal3D(
        data=np.column_stack([np.zeros(n_samples), force_z, np.zeros(n_samples)]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="N",
    )

    torque = Signal3D(
        data=np.zeros((n_samples, 3)),
        index=time,
        columns=["X", "Y", "Z"],
        unit="Nm",
    )

    origin = Point3D(
        data=np.zeros((n_samples, 3)),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m",
    )

    return ForcePlatform(force=force, torque=torque, origin=origin)


@pytest.fixture
def mock_wholebody_markers():
    """Create minimal marker set for exercise tests."""
    n_samples = 200
    time = np.linspace(0, 2.0, n_samples)

    markers = {}
    # Create simple marker trajectories (constant positions for simplicity)
    marker_positions = {
        "left_heel": [0.0, 0.05, 0.0],
        "right_heel": [0.0, -0.05, 0.0],
        "left_toe": [0.15, 0.05, 0.0],
        "right_toe": [0.15, -0.05, 0.0],
        "left_ankle_lateral": [0.05, 0.08, 0.05],
        "right_ankle_lateral": [0.05, -0.08, 0.05],
        "left_knee_lateral": [0.0, 0.12, 0.5],
        "right_knee_lateral": [0.0, -0.12, 0.5],
        "left_asis": [-0.1, 0.15, 1.0],
        "right_asis": [-0.1, -0.15, 1.0],
    }

    for name, pos in marker_positions.items():
        data = np.tile(pos, (n_samples, 1))
        markers[name] = Point3D(
            data=data,
            index=time,
            columns=["X", "Y", "Z"],
            unit="m",
        )

    return markers
