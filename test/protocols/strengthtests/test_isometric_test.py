"""
Tests for IsometricTest protocol with synthetic data.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from labanalysis.exercises.strength import IsometricExercise
from labanalysis.protocols import Participant, IsometricTest

from .synthetic_data_generators import (
    generate_force_signal,
    generate_position_signal_isometric,
)


@pytest.fixture
def participant():
    """Create a test participant."""
    return Participant(
        surname="Rossi",
        name="Mario",
        gender="M",
        height=175,  # cm
        weight=75.0,  # kg
        age=30
    )


@pytest.fixture
def synthetic_isometric_test(participant):
    """Create a synthetic IsometricTest with bilateral data."""
    duration = 8.0
    fsamp = 100.0

    force = generate_force_signal(
        duration=duration,
        fsamp=fsamp,
        peak_force=800.0,
        baseline=10.0,
        noise_level=8.0,
        plateau_duration=3.0  # Must be >= 3s for repetition detection
    )

    position = generate_position_signal_isometric(
        duration=duration,
        fsamp=fsamp,
        position=0.5,
        noise_level=0.002
    )

    bilateral_exercise = IsometricExercise(
        side="bilateral",
        force=force,
        position=position,
        synchronize_signals=False
    )

    test = IsometricTest(
        left=None,
        right=None,
        bilateral=bilateral_exercise,
        participant=participant
    )

    return test


def test_create_isometric_test(synthetic_isometric_test, participant):
    """Test IsometricTest creation with synthetic data."""
    test = synthetic_isometric_test

    assert test.participant.fullname == participant.fullname
    assert test.bilateral is not None
    assert test.left is None
    assert test.right is None
    assert isinstance(test.bilateral, IsometricExercise)


def test_save_load_isometric_test(synthetic_isometric_test, tmp_path):
    """Test IsometricTest serialization with pickle."""
    test = synthetic_isometric_test

    # Save
    filepath = tmp_path / "test_isometric.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(test, f)

    assert filepath.exists()
    assert filepath.stat().st_size > 0

    # Load
    with open(filepath, 'rb') as f:
        loaded_test = pickle.load(f)

    assert loaded_test.participant.fullname == test.participant.fullname
    assert loaded_test.bilateral is not None
    assert isinstance(loaded_test.bilateral, IsometricExercise)


def test_isometric_test_get_results(synthetic_isometric_test):
    """Test IsometricTest.get_results() method."""
    test = synthetic_isometric_test

    # Get results without EMG
    results = test.get_results(include_emg=False)

    assert results is not None
    assert hasattr(results, 'summary')
    assert hasattr(results, 'analytics')
    assert hasattr(results, 'figures')

    # Check summary structure
    summary = results.summary
    assert 'parameter' in summary.columns
    assert 'bilateral' in summary.columns

    # Check expected metrics
    params = summary['parameter'].tolist()
    assert 'peak force (N)' in params
    assert 'rate of force development (kN/s)' in params
    assert 'time to peak force (ms)' in params

    # Verify peak force is reasonable
    peak_force_row = summary[summary['parameter'] == 'peak force (N)']
    peak_force = float(peak_force_row['bilateral'].iloc[0])
    assert 700 < peak_force < 900  # Should be around 800 N


def test_isometric_test_figures_generation(synthetic_isometric_test):
    """Test that IsometricTest generates figures correctly."""
    test = synthetic_isometric_test

    results = test.get_results(include_emg=False)

    assert results.figures is not None
    assert isinstance(results.figures, dict)
    assert len(results.figures) > 0

    # Check figure structure
    for fig_name, fig in results.figures.items():
        assert fig is not None
        assert hasattr(fig, 'data')  # Plotly figure attribute
