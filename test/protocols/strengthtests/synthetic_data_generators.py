"""
Synthetic signal generators for strength testing.

This module provides utilities to generate synthetic force and position signals
for testing IsometricTest and IsokineticTest protocols.
"""

import numpy as np

from labanalysis.timeseries import Signal1D


def generate_force_signal(
    duration=5.0,
    fsamp=100.0,
    peak_force=500.0,
    baseline=10.0,
    noise_level=5.0,
    plateau_duration=3.0
):
    """
    Generate synthetic force signal with plateau at peak.

    Parameters
    ----------
    duration : float, optional
        Total duration in seconds (default: 5.0)
    fsamp : float, optional
        Sampling frequency in Hz (default: 100.0)
    peak_force : float, optional
        Peak force value in N (default: 500.0)
    baseline : float, optional
        Baseline force in N (default: 10.0)
    noise_level : float, optional
        Standard deviation of Gaussian noise in N (default: 5.0)
    plateau_duration : float, optional
        Duration of plateau at peak in seconds (default: 3.0)
        Must be >= 3.0 for isometric repetition detection

    Returns
    -------
    Signal1D
        Force signal in Newtons
    """
    time = np.arange(0, duration, 1/fsamp)
    force = np.zeros_like(time)

    # Calculate phase durations
    ramp_up_duration = (duration - plateau_duration) / 2
    ramp_down_duration = (duration - plateau_duration) / 2

    ramp_up_samples = int(ramp_up_duration * fsamp)
    plateau_samples = int(plateau_duration * fsamp)
    ramp_down_samples = len(time) - ramp_up_samples - plateau_samples

    # Ramp up (quadratic)
    force[:ramp_up_samples] = baseline + (peak_force - baseline) * np.linspace(0, 1, ramp_up_samples)**2

    # Plateau at peak
    force[ramp_up_samples:ramp_up_samples + plateau_samples] = peak_force

    # Ramp down (quadratic)
    force[ramp_up_samples + plateau_samples:] = baseline + (peak_force - baseline) * np.linspace(1, 0, ramp_down_samples)**2

    # Add Gaussian noise
    force += np.random.normal(0, noise_level, len(time))

    return Signal1D(data=force, index=time, unit="N")


def generate_position_signal_isometric(
    duration=5.0,
    fsamp=100.0,
    position=0.5,
    noise_level=0.001
):
    """
    Generate synthetic position signal for isometric test (constant position).

    Parameters
    ----------
    duration : float, optional
        Total duration in seconds (default: 5.0)
    fsamp : float, optional
        Sampling frequency in Hz (default: 100.0)
    position : float, optional
        Constant position value in meters (default: 0.5)
    noise_level : float, optional
        Standard deviation of Gaussian noise in meters (default: 0.001)

    Returns
    -------
    Signal1D
        Position signal in meters
    """
    time = np.arange(0, duration, 1/fsamp)
    position_data = np.ones_like(time) * position
    position_data += np.random.normal(0, noise_level, len(time))

    return Signal1D(data=position_data, index=time, unit="m")


def generate_position_signal_isokinetic(
    duration=5.0,
    fsamp=100.0,
    rom_start=0.1,
    rom_end=0.9,
    noise_level=0.001
):
    """
    Generate synthetic position signal for isokinetic test (constant velocity movement).

    Creates a position signal with three phases:
    - Acceleration (0-20% of duration)
    - Constant velocity (20-80% of duration)
    - Deceleration (80-100% of duration)

    Parameters
    ----------
    duration : float, optional
        Total duration in seconds (default: 5.0)
    fsamp : float, optional
        Sampling frequency in Hz (default: 100.0)
    rom_start : float, optional
        Starting position in meters (default: 0.1)
    rom_end : float, optional
        Ending position in meters (default: 0.9)
    noise_level : float, optional
        Standard deviation of Gaussian noise in meters (default: 0.001)

    Returns
    -------
    Signal1D
        Position signal in meters
    """
    time = np.arange(0, duration, 1/fsamp)
    position_data = np.zeros_like(time)

    # Phase 1: acceleration (0-20%)
    phase1_end = int(0.2 * len(time))
    position_data[:phase1_end] = rom_start + (rom_end - rom_start) * 0.2 * (np.linspace(0, 1, phase1_end)**2)

    # Phase 2: constant velocity (20-80%)
    phase2_end = int(0.8 * len(time))
    position_data[phase1_end:phase2_end] = rom_start + (rom_end - rom_start) * np.linspace(0.2, 0.8, phase2_end - phase1_end)

    # Phase 3: deceleration (80-100%)
    position_data[phase2_end:] = rom_start + (rom_end - rom_start) * (0.8 + 0.2 * (1 - (1 - np.linspace(0, 1, len(time) - phase2_end))**2))

    # Add Gaussian noise
    position_data += np.random.normal(0, noise_level, len(time))

    return Signal1D(data=position_data, index=time, unit="m")


__all__ = [
    "generate_force_signal",
    "generate_position_signal_isometric",
    "generate_position_signal_isokinetic",
]
