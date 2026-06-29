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


def test_isometric_test_force_at_time_metrics(synthetic_isometric_test):
    """Test that force at specific time points is calculated correctly."""
    test = synthetic_isometric_test
    results = test.get_results(include_emg=False)
    summary = results.summary

    # Check that force at time metrics exist
    params = summary['parameter'].tolist()
    assert 'force at 100 ms (N)' in params
    assert 'force at 200 ms (N)' in params
    assert 'force at 500 ms (N)' in params
    assert 'force at 1000 ms (N)' in params

    # Extract force values
    f100 = float(summary[summary['parameter'] == 'force at 100 ms (N)']['bilateral'].iloc[0])
    f200 = float(summary[summary['parameter'] == 'force at 200 ms (N)']['bilateral'].iloc[0])
    f500 = float(summary[summary['parameter'] == 'force at 500 ms (N)']['bilateral'].iloc[0])
    f1000 = float(summary[summary['parameter'] == 'force at 1000 ms (N)']['bilateral'].iloc[0])
    peak = float(summary[summary['parameter'] == 'peak force (N)']['bilateral'].iloc[0])

    # Force should increase over time during ramp-up
    assert f100 < f200 < f500
    # Force at 1000ms should be close to peak (plateau phase)
    assert f1000 >= f500
    # All forces should be positive and less than or equal to peak
    assert 0 < f100 < peak
    assert 0 < f200 < peak
    assert 0 < f500 <= peak
    assert 0 < f1000 <= peak


def test_isometric_test_max_time_s_parameter(participant):
    """Test that max_time_s parameter correctly limits analysis duration."""
    duration = 8.0
    fsamp = 100.0

    force = generate_force_signal(
        duration=duration,
        fsamp=fsamp,
        peak_force=800.0,
        baseline=10.0,
        noise_level=8.0,
        plateau_duration=3.0
    )

    position = generate_position_signal_isometric(
        duration=duration,
        fsamp=fsamp,
        position=0.5,
        noise_level=0.002
    )

    # Create exercise with max_time_s = 2
    bilateral_exercise = IsometricExercise(
        side="bilateral",
        force=force,
        position=position,
        synchronize_signals=False,
        max_time_s=2  # Set on exercise, not test
    )

    # Create test
    test = IsometricTest(
        left=None,
        right=None,
        bilateral=bilateral_exercise,
        participant=participant
    )

    # Verify exercise has max_time_s
    assert hasattr(bilateral_exercise, '_max_time_s')
    assert bilateral_exercise._max_time_s == 2

    # Get processed data
    processed = test.processed_data

    # Check that repetitions are trimmed to 2 seconds
    for rep in processed.bilateral.repetitions:
        duration_s = rep.index[-1] - rep.index[0]
        assert duration_s <= 2.0
        # Should be very close to 2 seconds (within sampling tolerance)
        assert duration_s >= 1.9


def test_isometric_test_max_time_s_preserves_on_copy(participant):
    """Test that max_time_s is preserved when copying IsometricExercise."""
    duration = 8.0
    fsamp = 100.0

    force = generate_force_signal(
        duration=duration,
        fsamp=fsamp,
        peak_force=800.0,
        baseline=10.0,
        noise_level=8.0,
        plateau_duration=3.0
    )

    position = generate_position_signal_isometric(
        duration=duration,
        fsamp=fsamp,
        position=0.5,
        noise_level=0.002
    )

    # Create exercise with max_time_s
    bilateral_exercise = IsometricExercise(
        side="bilateral",
        force=force,
        position=position,
        synchronize_signals=False,
        max_time_s=3
    )

    # Copy the exercise
    copied_exercise = bilateral_exercise.copy()

    # Verify max_time_s is preserved
    assert hasattr(copied_exercise, '_max_time_s')
    assert copied_exercise._max_time_s == bilateral_exercise._max_time_s
    assert copied_exercise._max_time_s == 3


def test_isometric_test_max_time_s_pickle_compatibility(participant, tmp_path):
    """Test that max_time_s is preserved when saving/loading IsometricExercise with pickle."""
    duration = 8.0
    fsamp = 100.0

    force = generate_force_signal(
        duration=duration,
        fsamp=fsamp,
        peak_force=800.0,
        baseline=10.0,
        noise_level=8.0,
        plateau_duration=3.0
    )

    position = generate_position_signal_isometric(
        duration=duration,
        fsamp=fsamp,
        position=0.5,
        noise_level=0.002
    )

    # Create exercise with max_time_s
    bilateral_exercise = IsometricExercise(
        side="bilateral",
        force=force,
        position=position,
        synchronize_signals=False,
        max_time_s=4
    )

    # Save exercise
    filepath = tmp_path / "exercise_with_max_time.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(bilateral_exercise, f)

    # Load exercise
    with open(filepath, 'rb') as f:
        loaded_exercise = pickle.load(f)

    # Verify max_time_s is preserved
    assert hasattr(loaded_exercise, '_max_time_s')
    assert loaded_exercise._max_time_s == 4
    assert loaded_exercise._max_time_s == bilateral_exercise._max_time_s


def test_isometric_test_max_time_s_none_default(participant):
    """Test that max_time_s defaults to None when not specified."""
    duration = 8.0
    fsamp = 100.0

    force = generate_force_signal(
        duration=duration,
        fsamp=fsamp,
        peak_force=800.0,
        baseline=10.0,
        noise_level=8.0,
        plateau_duration=3.0
    )

    position = generate_position_signal_isometric(
        duration=duration,
        fsamp=fsamp,
        position=0.5,
        noise_level=0.002
    )

    # Create exercise without max_time_s
    bilateral_exercise = IsometricExercise(
        side="bilateral",
        force=force,
        position=position,
        synchronize_signals=False
    )

    # Verify max_time_s defaults to None
    assert hasattr(bilateral_exercise, '_max_time_s')
    assert bilateral_exercise._max_time_s is None


def test_isometric_test_max_time_s_validation(participant):
    """Test that max_time_s validation rejects invalid values."""
    duration = 8.0
    fsamp = 100.0

    force = generate_force_signal(
        duration=duration,
        fsamp=fsamp,
        peak_force=800.0,
        baseline=10.0,
        noise_level=8.0,
        plateau_duration=3.0
    )

    position = generate_position_signal_isometric(
        duration=duration,
        fsamp=fsamp,
        position=0.5,
        noise_level=0.002
    )

    # Should raise ValueError for max_time_s < 1
    with pytest.raises(ValueError, match="max_time_s must be >= 1 or None"):
        bilateral_exercise = IsometricExercise(
            side="bilateral",
            force=force,
            position=position,
            synchronize_signals=False,
            max_time_s=0
        )

    with pytest.raises(ValueError, match="max_time_s must be >= 1 or None"):
        bilateral_exercise = IsometricExercise(
            side="bilateral",
            force=force,
            position=position,
            synchronize_signals=False,
            max_time_s=-1
        )


@pytest.fixture
def synthetic_isometric_test_multiple_reps(participant):
    """Create a synthetic IsometricTest with multiple repetitions."""
    duration = 25.0  # Long enough for 3 repetitions
    fsamp = 100.0

    # Generate force signal with 3 distinct contractions
    time = np.arange(0, duration, 1/fsamp)
    force_data = np.ones_like(time) * 10.0  # baseline

    # Add 3 repetitions with slightly different peak forces
    rep_peaks = [800.0, 820.0, 790.0]
    rep_starts = [1.0, 9.0, 17.0]
    rep_duration = 6.0

    for peak, start in zip(rep_peaks, rep_starts):
        start_idx = int(start * fsamp)
        rep_samples = int(rep_duration * fsamp)

        # Ramp up (2s)
        ramp_up_samples = int(2.0 * fsamp)
        force_data[start_idx:start_idx + ramp_up_samples] = (
            10.0 + (peak - 10.0) * np.linspace(0, 1, ramp_up_samples)**2
        )

        # Plateau (3s) - must be >= 3s for detection
        plateau_samples = int(3.0 * fsamp)
        force_data[start_idx + ramp_up_samples:start_idx + ramp_up_samples + plateau_samples] = peak

        # Ramp down (1s)
        ramp_down_samples = int(1.0 * fsamp)
        end_idx = start_idx + ramp_up_samples + plateau_samples
        if end_idx + ramp_down_samples <= len(force_data):
            force_data[end_idx:end_idx + ramp_down_samples] = (
                10.0 + (peak - 10.0) * np.linspace(1, 0, ramp_down_samples)**2
            )

    # Add noise
    force_data += np.random.normal(0, 8.0, len(time))

    from labanalysis.timeseries import Signal1D
    force = Signal1D(data=force_data, index=time, unit="N")

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


def test_isometric_test_multiple_repetitions_detected(synthetic_isometric_test_multiple_reps):
    """Test that multiple repetitions are correctly detected and processed."""
    test = synthetic_isometric_test_multiple_reps

    # Check that multiple repetitions are detected
    assert len(test.bilateral.repetitions) >= 2, "Should detect at least 2 repetitions"

    # Get results
    results = test.get_results(include_emg=False)
    summary = results.summary

    # Verify that metrics are computed
    assert 'peak force (N)' in summary['parameter'].tolist()

    # The summary should contain averaged values across all repetitions
    peak_force = float(summary[summary['parameter'] == 'peak force (N)']['bilateral'].iloc[0])
    assert 750 < peak_force < 850  # Should be around average of [800, 820, 790]


def test_isometric_test_multiple_reps_all_contribute_to_summary(synthetic_isometric_test_multiple_reps):
    """Test that all repetitions contribute to the summary metrics."""
    test = synthetic_isometric_test_multiple_reps
    processed = test.processed_data

    # Get number of repetitions
    n_reps = len(processed.bilateral.repetitions)
    assert n_reps >= 2, "Need multiple repetitions for this test"

    # Get results
    results = test.get_results(include_emg=False)

    # The summary should represent data from all repetitions
    # We can verify this indirectly by checking that analytics contains all reps
    analytics = results.analytics
    unique_reps = analytics['repetition'].unique()
    assert len(unique_reps) == n_reps, "Analytics should include all repetitions"


def test_isometric_test_max_time_s_applies_to_all_repetitions(participant):
    """Test that max_time_s is applied to all repetitions, not just the first."""
    duration = 25.0
    fsamp = 100.0

    # Generate force signal with 3 repetitions
    time = np.arange(0, duration, 1/fsamp)
    force_data = np.ones_like(time) * 10.0

    rep_peaks = [800.0, 820.0, 790.0]
    rep_starts = [1.0, 9.0, 17.0]

    for peak, start in zip(rep_peaks, rep_starts):
        start_idx = int(start * fsamp)
        ramp_up_samples = int(2.0 * fsamp)
        plateau_samples = int(3.0 * fsamp)
        ramp_down_samples = int(1.0 * fsamp)

        force_data[start_idx:start_idx + ramp_up_samples] = (
            10.0 + (peak - 10.0) * np.linspace(0, 1, ramp_up_samples)**2
        )
        force_data[start_idx + ramp_up_samples:start_idx + ramp_up_samples + plateau_samples] = peak
        end_idx = start_idx + ramp_up_samples + plateau_samples
        if end_idx + ramp_down_samples <= len(force_data):
            force_data[end_idx:end_idx + ramp_down_samples] = (
                10.0 + (peak - 10.0) * np.linspace(1, 0, ramp_down_samples)**2
            )

    force_data += np.random.normal(0, 8.0, len(time))

    from labanalysis.timeseries import Signal1D
    force = Signal1D(data=force_data, index=time, unit="N")

    position = generate_position_signal_isometric(
        duration=duration,
        fsamp=fsamp,
        position=0.5,
        noise_level=0.002
    )

    # Create exercise with max_time_s = 2
    bilateral_exercise = IsometricExercise(
        side="bilateral",
        force=force,
        position=position,
        synchronize_signals=False,
        max_time_s=2
    )

    # Create test
    test = IsometricTest(
        left=None,
        right=None,
        bilateral=bilateral_exercise,
        participant=participant
    )

    # Get processed data
    processed = test.processed_data

    # Verify all repetitions are trimmed to 2 seconds
    for i, rep in enumerate(processed.bilateral.repetitions):
        duration_s = rep.index[-1] - rep.index[0]
        assert duration_s <= 2.0, f"Repetition {i} should be trimmed to 2s"
        assert duration_s >= 1.9, f"Repetition {i} should be close to 2s"


def test_isometric_test_figure_uses_max_time_s(participant):
    """Test that generated figures respect max_time_s setting."""
    duration = 8.0
    fsamp = 100.0

    force = generate_force_signal(
        duration=duration,
        fsamp=fsamp,
        peak_force=800.0,
        baseline=10.0,
        noise_level=8.0,
        plateau_duration=3.0
    )

    position = generate_position_signal_isometric(
        duration=duration,
        fsamp=fsamp,
        position=0.5,
        noise_level=0.002
    )

    # Create exercise with max_time_s = 2
    bilateral_exercise = IsometricExercise(
        side="bilateral",
        force=force,
        position=position,
        synchronize_signals=False,
        max_time_s=2
    )

    # Create test
    test = IsometricTest(
        left=None,
        right=None,
        bilateral=bilateral_exercise,
        participant=participant
    )

    results = test.get_results(include_emg=False)

    # Check that figure exists
    assert 'force_profiles_with_muscle_balance' in results.figures

    # Get the figure
    fig = results.figures['force_profiles_with_muscle_balance']

    # The x-axis should show time in ms up to 2000 ms
    # We can check this by examining the figure data
    assert fig is not None
    assert hasattr(fig, 'data')

    # Find the force trace
    for trace in fig.data:
        if hasattr(trace, 'x') and trace.x is not None:
            max_x = max(trace.x)
            # X-axis should not exceed 2000 ms (with some tolerance)
            assert max_x <= 2100, f"Figure x-axis should not exceed 2000ms, got {max_x}"
