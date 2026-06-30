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


def generate_isometric_exercise(
    side="bilateral",
    duration=5.0,
    fsamp=100.0,
    peak_force=800.0,
    baseline=10.0,
    noise_level=8.0,
    plateau_duration=3.0,
    max_time_s=None,
    time_points=None,
):
    """
    Generate a complete synthetic IsometricExercise.

    Parameters
    ----------
    side : str, optional
        Side of the exercise: 'left', 'right', or 'bilateral' (default: 'bilateral')
    duration : float, optional
        Total duration in seconds (default: 5.0)
    fsamp : float, optional
        Sampling frequency in Hz (default: 100.0)
    peak_force : float, optional
        Peak force value in N (default: 800.0)
    baseline : float, optional
        Baseline force in N (default: 10.0)
    noise_level : float, optional
        Standard deviation of Gaussian noise in N (default: 8.0)
    plateau_duration : float, optional
        Duration of plateau at peak in seconds (default: 3.0)
    max_time_s : float or None, optional
        Maximum analysis time in seconds (default: None)
    time_points : list[int] or None, optional
        Time points for force measurement in ms (default: None uses [100, 200, 500, 1000])

    Returns
    -------
    IsometricExercise
        Complete isometric exercise with force and position signals
    """
    from labanalysis.exercises.strength import IsometricExercise

    force = generate_force_signal(
        duration=duration,
        fsamp=fsamp,
        peak_force=peak_force,
        baseline=baseline,
        noise_level=noise_level,
        plateau_duration=plateau_duration,
    )
    force._name = 'force'

    position = generate_position_signal_isometric(
        duration=duration,
        fsamp=fsamp,
    )
    position._name = 'position'

    kwargs = {'side': side, 'force': force, 'position': position, 'synchronize_signals': False}
    if max_time_s is not None:
        kwargs['max_time_s'] = max_time_s
    if time_points is not None:
        kwargs['time_points'] = time_points

    return IsometricExercise(**kwargs)


def generate_isokinetic_exercise(
    side="bilateral",
    num_reps=3,
    duration_per_rep=3.0,
    gap_duration=0.5,
    fsamp=100.0,
    base_peak_force=600.0,
    peak_force_increment=100.0,
    noise_level=5.0,
    rom_start=0.0,
    rom_end=0.45,
):
    """
    Generate a complete synthetic IsokineticExercise with multiple repetitions.

    Parameters
    ----------
    side : str, optional
        Side of the exercise: 'left', 'right', or 'bilateral' (default: 'bilateral')
    num_reps : int, optional
        Number of repetitions (default: 3)
    duration_per_rep : float, optional
        Duration of each repetition in seconds (default: 3.0)
    gap_duration : float, optional
        Gap between repetitions in seconds (default: 0.5)
    fsamp : float, optional
        Sampling frequency in Hz (default: 100.0)
    base_peak_force : float, optional
        Peak force for first repetition in N (default: 600.0)
    peak_force_increment : float, optional
        Force increment per repetition in N (default: 100.0)
    noise_level : float, optional
        Standard deviation of Gaussian noise in N (default: 5.0)
    rom_start : float, optional
        Starting position in meters (default: 0.0)
    rom_end : float, optional
        Ending position in meters (default: 0.45)

    Returns
    -------
    IsokineticExercise
        Complete isokinetic exercise with force and position signals
    """
    from labanalysis.exercises.strength import IsokineticExercise

    # Create a single long signal with multiple repetitions
    total_duration = num_reps * duration_per_rep + (num_reps - 1) * gap_duration
    time = np.arange(0, total_duration, 1/fsamp)

    force_data = np.zeros_like(time)
    position_data = np.zeros_like(time)

    for rep_idx in range(num_reps):
        start_time = rep_idx * (duration_per_rep + gap_duration)

        # Variable peak force (increases with each rep)
        peak_force = base_peak_force + rep_idx * peak_force_increment + np.random.uniform(-20, 20)

        for i, t in enumerate(time):
            t_local = t - start_time

            if 0 <= t_local < duration_per_rep:
                # Phase 1: Rest (0-0.5s)
                if t_local < 0.5:
                    force_data[i] = 10.0
                    position_data[i] = rom_start
                # Phase 2: Ramp up (0.5-1.5s)
                elif t_local < 1.5:
                    progress = (t_local - 0.5) / 1.0
                    force_data[i] = 10.0 + (peak_force - 10.0) * np.sin(progress * np.pi / 2) ** 2
                    position_data[i] = rom_start + (rom_end - rom_start) * progress * 0.67
                # Phase 3: Hold peak (1.5-2.0s)
                elif t_local < 2.0:
                    force_data[i] = peak_force + np.sin(t_local * 20) * 20
                    position_data[i] = rom_start + (rom_end - rom_start) * (0.67 + (t_local - 1.5) * 0.33 / 0.5)
                # Phase 4: Ramp down (2.0-3.0s)
                else:
                    progress = (t_local - 2.0) / 1.0
                    force_data[i] = peak_force * (1 - progress)
                    position_data[i] = rom_end - progress * rom_end

    force_data += np.random.normal(0, noise_level, len(time))
    position_data += np.random.normal(0, 0.001, len(time))

    force = Signal1D(data=force_data, index=time, unit='N')
    force._name = 'force'
    position = Signal1D(data=position_data, index=time, unit='m')
    position._name = 'position'

    return IsokineticExercise(
        side=side,
        force=force,
        position=position,
        synchronize_signals=False,
    )


def generate_isometric_test(
    participant,
    include_left=True,
    include_right=True,
    include_bilateral=False,
    left_peak_force=750.0,
    right_peak_force=820.0,
    bilateral_peak_force=1500.0,
    **exercise_kwargs
):
    """
    Generate a complete synthetic IsometricTest.

    Parameters
    ----------
    participant : Participant
        Participant information
    include_left : bool, optional
        Include left limb test (default: True)
    include_right : bool, optional
        Include right limb test (default: True)
    include_bilateral : bool, optional
        Include bilateral test (default: False)
    left_peak_force : float, optional
        Peak force for left limb in N (default: 750.0)
    right_peak_force : float, optional
        Peak force for right limb in N (default: 820.0)
    bilateral_peak_force : float, optional
        Peak force for bilateral in N (default: 1500.0)
    **exercise_kwargs
        Additional kwargs passed to generate_isometric_exercise

    Returns
    -------
    IsometricTest
        Complete isometric test protocol
    """
    from labanalysis.protocols import IsometricTest

    left = generate_isometric_exercise(side="left", peak_force=left_peak_force, **exercise_kwargs) if include_left else None
    right = generate_isometric_exercise(side="right", peak_force=right_peak_force, **exercise_kwargs) if include_right else None
    bilateral = generate_isometric_exercise(side="bilateral", peak_force=bilateral_peak_force, **exercise_kwargs) if include_bilateral else None

    return IsometricTest(
        left=left,
        right=right,
        bilateral=bilateral,
        participant=participant,
    )


def generate_isokinetic_test(
    participant,
    include_left=False,
    include_right=False,
    include_bilateral=True,
    rm1_coefs=None,
    left_base_peak=600.0,
    right_base_peak=650.0,
    bilateral_base_peak=1200.0,
    **exercise_kwargs
):
    """
    Generate a complete synthetic Isokinetic1RMTest.

    Parameters
    ----------
    participant : Participant
        Participant information
    include_left : bool, optional
        Include left limb test (default: False)
    include_right : bool, optional
        Include right limb test (default: False)
    include_bilateral : bool, optional
        Include bilateral test (default: True)
    rm1_coefs : dict or None, optional
        1RM coefficients {'beta0': float, 'beta1': float} (default: None uses {'beta0': 50.0, 'beta1': 1.2})
    left_base_peak : float, optional
        Base peak force for left limb in N (default: 600.0)
    right_base_peak : float, optional
        Base peak force for right limb in N (default: 650.0)
    bilateral_base_peak : float, optional
        Base peak force for bilateral in N (default: 1200.0)
    **exercise_kwargs
        Additional kwargs passed to generate_isokinetic_exercise

    Returns
    -------
    Isokinetic1RMTest
        Complete isokinetic 1RM test protocol
    """
    from labanalysis.protocols import Isokinetic1RMTest

    if rm1_coefs is None:
        rm1_coefs = {'beta0': 50.0, 'beta1': 1.2}

    left = generate_isokinetic_exercise(side="left", base_peak_force=left_base_peak, **exercise_kwargs) if include_left else None
    right = generate_isokinetic_exercise(side="right", base_peak_force=right_base_peak, **exercise_kwargs) if include_right else None
    bilateral = generate_isokinetic_exercise(side="bilateral", base_peak_force=bilateral_base_peak, **exercise_kwargs) if include_bilateral else None

    return Isokinetic1RMTest(
        rm1_coefs=rm1_coefs,
        left=left,
        right=right,
        bilateral=bilateral,
        participant=participant,
    )


__all__ = [
    "generate_force_signal",
    "generate_position_signal_isometric",
    "generate_position_signal_isokinetic",
    "generate_isometric_exercise",
    "generate_isokinetic_exercise",
    "generate_isometric_test",
    "generate_isokinetic_test",
]
