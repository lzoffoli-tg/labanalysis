"""
Tests for synthetic data generators.

This module verifies that all synthetic data generators work correctly
and produce valid strength test data.
"""

import pytest

from labanalysis.protocols import Participant
from .synthetic_data_generators import (
    generate_force_signal,
    generate_position_signal_isometric,
    generate_position_signal_isokinetic,
    generate_isometric_exercise,
    generate_isokinetic_exercise,
    generate_isometric_test,
    generate_isokinetic_test,
)


@pytest.fixture
def participant():
    """Create a test participant."""
    return Participant(
        surname="Rossi",
        name="Mario",
        gender="M",
        height=175,
        weight=75.0,
        age=30
    )


def test_generate_force_signal():
    """Test force signal generation."""
    signal = generate_force_signal(
        duration=5.0,
        fsamp=100.0,
        peak_force=800.0,
        baseline=10.0,
        noise_level=8.0,
        plateau_duration=3.0
    )

    assert signal is not None
    assert len(signal.data) == 500  # 5s * 100Hz
    assert signal.unit == "N"
    assert signal.to_numpy().max() > 750  # Should be around peak_force


def test_generate_position_signal_isometric():
    """Test isometric position signal generation."""
    signal = generate_position_signal_isometric(
        duration=5.0,
        fsamp=100.0,
        position=0.5,
        noise_level=0.001
    )

    assert signal is not None
    assert len(signal.data) == 500
    assert signal.unit == "m"
    # Should be approximately constant around 0.5m
    assert 0.49 < signal.to_numpy().mean() < 0.51


def test_generate_position_signal_isokinetic():
    """Test isokinetic position signal generation."""
    signal = generate_position_signal_isokinetic(
        duration=5.0,
        fsamp=100.0,
        rom_start=0.1,
        rom_end=0.9,
        noise_level=0.001
    )

    assert signal is not None
    assert len(signal.data) == 500
    assert signal.unit == "m"
    # Should start near rom_start and end near rom_end
    assert signal.to_numpy()[0] < 0.15
    assert signal.to_numpy()[-1] > 0.85


def test_generate_isometric_exercise():
    """Test isometric exercise generation."""
    exercise = generate_isometric_exercise(
        side="left",
        duration=5.0,
        fsamp=100.0,
        peak_force=800.0,
        max_time_s=2,
        time_points=[100, 200, 500, 1000]
    )

    assert exercise is not None
    assert exercise.side == "left"
    assert hasattr(exercise, 'force')
    assert hasattr(exercise, 'position')
    assert exercise.max_time_s == 2
    assert exercise.time_points == [100, 200, 500, 1000]
    assert len(exercise.repetitions) > 0


def test_generate_isokinetic_exercise():
    """Test isokinetic exercise generation."""
    exercise = generate_isokinetic_exercise(
        side="bilateral",
        num_reps=3,
        duration_per_rep=3.0,
        fsamp=100.0,
        base_peak_force=600.0,
    )

    assert exercise is not None
    assert exercise.side == "bilateral"
    assert hasattr(exercise, 'force')
    assert hasattr(exercise, 'position')
    # Should have detected at least 2 repetitions
    assert len(exercise.repetitions) >= 2


def test_generate_isometric_test(participant):
    """Test isometric test generation."""
    test = generate_isometric_test(
        participant=participant,
        include_left=True,
        include_right=True,
        include_bilateral=False,
        left_peak_force=750.0,
        right_peak_force=820.0,
    )

    assert test is not None
    assert test.left is not None
    assert test.right is not None
    assert test.bilateral is None
    assert test.participant == participant


def test_generate_isokinetic_test(participant):
    """Test isokinetic test generation."""
    test = generate_isokinetic_test(
        participant=participant,
        include_left=False,
        include_right=False,
        include_bilateral=True,
        rm1_coefs={'beta0': 50.0, 'beta1': 1.2},
        bilateral_base_peak=1200.0,
    )

    assert test is not None
    assert test.left is None
    assert test.right is None
    assert test.bilateral is not None
    assert test.participant == participant
    assert test.rm1_coefs == {'beta0': 50.0, 'beta1': 1.2}


def test_isometric_test_get_results(participant):
    """Test that generated isometric test produces valid results."""
    test = generate_isometric_test(
        participant=participant,
        include_bilateral=True,
        bilateral_peak_force=800.0,
    )

    results = test.get_results(include_emg=False)

    assert results is not None
    assert hasattr(results, 'summary')
    assert hasattr(results, 'figures')

    # Check summary has expected parameters
    params = results.summary['parameter'].tolist()
    assert 'peak force (kN)' in params
    assert any('RFD 0-' in p and 'ms (kN/s)' in p for p in params)


def test_isokinetic_test_get_results(participant):
    """Test that generated isokinetic test produces valid results."""
    test = generate_isokinetic_test(
        participant=participant,
        include_bilateral=True,
        bilateral_base_peak=800.0,
        num_reps=3,
    )

    results = test.get_results(include_emg=False, estimate_1rm=True)

    assert results is not None
    assert hasattr(results, 'summary')
    assert hasattr(results, 'figures')

    # Check summary has expected parameters
    params = results.summary['parameter'].tolist()
    assert 'peak force (N)' in params
    assert 'estimated 1RM (kg)' in params
