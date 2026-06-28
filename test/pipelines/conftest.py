"""Shared fixtures for pipelines tests."""

import numpy as np
import pytest

from labanalysis.timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from labanalysis.records import ForcePlatform


@pytest.fixture
def sample_emgsignal():
    """Create EMG signal: 1000Hz, 1s, sine wave with noise."""
    freq = 1000  # Hz
    duration = 1.0  # seconds
    n_samples = int(freq * duration)
    time = np.linspace(0, duration, n_samples)

    # Sine wave with noise to simulate EMG
    signal = 50 * np.sin(2 * np.pi * 60 * time) + 10 * np.random.randn(n_samples)

    return EMGSignal(
        data=signal.reshape(-1, 1),
        index=time,
        columns=["EMG1"],
        unit="mV",
    )


@pytest.fixture
def sample_point3d_with_gaps():
    """Create Point3D trajectory with 10% NaN values."""
    n_samples = 200
    time = np.linspace(0, 2.0, n_samples)

    # Create trajectory with some gaps
    x = np.sin(2 * np.pi * time)
    y = np.cos(2 * np.pi * time)
    z = 0.5 * time

    data = np.column_stack([x, y, z])

    # Add 10% gaps
    gap_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    data[gap_indices, :] = np.nan

    return Point3D(
        data=data,
        index=time,
        columns=["X", "Y", "Z"],
        unit="m",
    )


@pytest.fixture
def sample_forceplatform_contact():
    """Create force platform with contact (>30N) and flight (<30N) phases."""
    freq = 100  # Hz
    duration = 2.0  # seconds
    n_samples = int(freq * duration)
    time = np.linspace(0, duration, n_samples)

    # Vertical force with contact and flight phases
    force_z = np.zeros(n_samples)
    # Contact phase: 0.0-0.8s
    contact = (time >= 0.0) & (time <= 0.8)
    force_z[contact] = 500 + 100 * np.sin(2 * np.pi * 5 * time[contact])
    # Flight phase: 0.8-1.2s
    flight = (time > 0.8) & (time < 1.2)
    force_z[flight] = 10  # Below threshold
    # Contact phase: 1.2-2.0s
    contact2 = time >= 1.2
    force_z[contact2] = 600 + 80 * np.sin(2 * np.pi * 5 * time[contact2])

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
