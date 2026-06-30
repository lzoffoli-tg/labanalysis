"""
Tests for Isokinetic1RMTest protocol with synthetic data.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from labanalysis.exercises.strength import IsokineticExercise
from labanalysis.protocols import Isokinetic1RMTest, Participant
from labanalysis.timeseries import Signal1D

from .synthetic_data_generators import (
    generate_force_signal,
    generate_position_signal_isokinetic,
)


@pytest.fixture
def participant():
    """Create a test participant."""
    return Participant(
        surname="Bianchi",
        name="Luca",
        gender="M",
        height=180,  # cm
        weight=80.0,  # kg
        age=28
    )


@pytest.fixture
def synthetic_isokinetic_test(participant):
    """Create a synthetic Isokinetic1RMTest with bilateral data (3 repetitions)."""
    fsamp = 100.0
    combined_force = []
    combined_position = []
    combined_time = []

    for rep_num in range(3):
        duration = 3.0
        peak_force = 600.0 - rep_num * 50  # Decreasing force across reps

        force = generate_force_signal(
            duration=duration,
            fsamp=fsamp,
            peak_force=peak_force,
            baseline=20.0,
            noise_level=10.0,
            plateau_duration=1.0
        )

        position = generate_position_signal_isokinetic(
            duration=duration,
            fsamp=fsamp,
            rom_start=0.1,
            rom_end=0.8,
            noise_level=0.003
        )

        if rep_num == 0:
            combined_force = force.to_numpy().flatten()
            combined_position = position.to_numpy().flatten()
            combined_time = force.index
        else:
            # Add 1 second gap between repetitions
            gap_samples = int(fsamp * 1.0)
            gap_force = np.ones(gap_samples) * 20.0
            gap_position = np.ones(gap_samples) * 0.1
            gap_time = combined_time[-1] + np.arange(1, gap_samples + 1) / fsamp

            combined_force = np.concatenate([combined_force, gap_force, force.to_numpy().flatten()])
            combined_position = np.concatenate([combined_position, gap_position, position.to_numpy().flatten()])
            combined_time = np.concatenate([combined_time, gap_time, gap_time[-1] + force.index])

    # Create combined signals
    force_signal = Signal1D(data=combined_force, index=combined_time, unit="N")
    position_signal = Signal1D(data=combined_position, index=combined_time, unit="m")

    bilateral_exercise = IsokineticExercise(
        side="bilateral",
        force=force_signal,
        position=position_signal,
        synchronize_signals=False
    )

    # 1RM coefficients
    rm1_coefs = {"beta0": 50.0, "beta1": 200.0}

    test = Isokinetic1RMTest(
        rm1_coefs=rm1_coefs,
        left=None,
        right=None,
        bilateral=bilateral_exercise,
        participant=participant
    )

    return test


def test_create_isokinetic_test(synthetic_isokinetic_test, participant):
    """Test Isokinetic1RMTest creation with synthetic data."""
    test = synthetic_isokinetic_test

    assert test.participant.fullname == participant.fullname
    assert test.bilateral is not None
    assert test.left is None
    assert test.right is None
    assert isinstance(test.bilateral, IsokineticExercise)
    assert test.rm1_coefs == {"beta0": 50.0, "beta1": 200.0}


def test_save_load_isokinetic_test(synthetic_isokinetic_test, tmp_path):
    """Test Isokinetic1RMTest serialization with pickle."""
    test = synthetic_isokinetic_test

    # Save
    filepath = tmp_path / "test_isokinetic.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(test, f)

    assert filepath.exists()
    assert filepath.stat().st_size > 0

    # Load
    with open(filepath, 'rb') as f:
        loaded_test = pickle.load(f)

    assert loaded_test.participant.fullname == test.participant.fullname
    assert loaded_test.bilateral is not None
    assert isinstance(loaded_test.bilateral, IsokineticExercise)
    assert loaded_test.rm1_coefs == test.rm1_coefs


def test_isokinetic_test_get_results(synthetic_isokinetic_test):
    """Test Isokinetic1RMTest.get_results() method."""
    test = synthetic_isokinetic_test

    # Get results with 1RM estimation, without EMG
    results = test.get_results(
        include_emg=False,
        estimate_1rm=True,
        include_force_balance=False
    )

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
    assert 'estimated 1RM (kg)' in params
    assert 'rom (m)' in params

    # Verify peak force is reasonable
    peak_force_row = summary[summary['parameter'] == 'peak force (N)']
    peak_force = float(peak_force_row['bilateral'].iloc[0])
    assert 500 < peak_force < 700  # Should be around 600 N

    # Verify 1RM estimation
    rm1_row = summary[summary['parameter'] == 'estimated 1RM (kg)']
    rm1 = float(rm1_row['bilateral'].iloc[0])
    assert rm1 > 0  # Should be positive


def test_isokinetic_test_figures_generation(synthetic_isokinetic_test):
    """Test that Isokinetic1RMTest generates figures correctly."""
    test = synthetic_isokinetic_test

    results = test.get_results(
        include_emg=False,
        estimate_1rm=True,
        include_force_balance=False
    )

    assert results.figures is not None
    assert isinstance(results.figures, dict)
    assert len(results.figures) > 0

    # Check figure structure
    for fig_name, fig in results.figures.items():
        assert fig is not None
        assert hasattr(fig, 'data')  # Plotly figure attribute


def test_isokinetic_test_analytics(synthetic_isokinetic_test):
    """Test analytics dataframe structure."""
    test = synthetic_isokinetic_test

    results = test.get_results(include_emg=False)

    analytics = results.analytics
    assert 'side' in analytics.columns
    assert 'repetition' in analytics.columns
    assert 'time_s' in analytics.columns

    # Should have 3 repetitions
    repetitions = analytics['repetition'].unique()
    assert len(repetitions) >= 1  # At least one repetition detected


def test_isokinetic_test_from_synthetic_biostrength_files_end_to_end(participant, tmp_path):
    """
    End-to-end integration test: create test from synthetic Biostrength files,
    generate results, save, and load.

    This test verifies the complete workflow used in real data analysis.
    """
    import tempfile

    # Import the synthetic data generator from biostrength tests
    import sys
    from pathlib import Path
    biostrength_test_path = Path(__file__).parent.parent.parent / "io" / "read"
    sys.path.insert(0, str(biostrength_test_path))

    try:
        from test_biostrength import _generate_synthetic_biostrength_txt
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate synthetic Biostrength files (left and right)
        left_file = tmpdir / "legcurl_left.txt"
        right_file = tmpdir / "legcurl_right.txt"

        _generate_synthetic_biostrength_txt(
            str(left_file),
            n_repetitions=3,
            rep_duration_s=6.5,
            torque_peak_nm=30.0,
            position_range_rad=(1.1, 7.5),
            roller_position=11
        )

        _generate_synthetic_biostrength_txt(
            str(right_file),
            n_repetitions=3,
            rep_duration_s=6.5,
            torque_peak_nm=35.0,
            position_range_rad=(1.1, 7.5),
            roller_position=11
        )

        # Create test from files (mimics real workflow from analisi_dati.py)
        test = Isokinetic1RMTest.from_files(
            participant=participant,
            product="LEG CURL",
            left_biostrength_filename=str(left_file),
            right_biostrength_filename=str(right_file),
        )

        # Verify test creation
        assert test is not None
        assert test.participant.fullname == participant.fullname
        assert test.left is not None
        assert test.right is not None
        assert test.bilateral is None  # Bilateral not created when left/right provided

        # Verify _rm1_coefs was loaded correctly
        assert test.rm1_coefs is not None
        assert 'beta0' in test.rm1_coefs
        assert 'beta1' in test.rm1_coefs

        # Generate results
        results = test.get_results(include_emg=False, estimate_1rm=True)

        assert results is not None
        assert results.summary is not None
        assert results.analytics is not None
        assert results.figures is not None

        # Verify summary has data for both sides
        summary = results.summary
        assert 'left' in summary.columns
        assert 'right' in summary.columns

        # Save test
        test_save_path = tmp_path / "test.isokinetic1rmtest"
        test.save(str(test_save_path), force_overwrite=True)
        assert test_save_path.exists()

        # Load test
        loaded_test = Isokinetic1RMTest.load(str(test_save_path))
        assert loaded_test is not None
        assert loaded_test.participant.fullname == test.participant.fullname
        assert loaded_test.left is not None
        assert loaded_test.right is not None

        # Save results
        results_dir = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)
        results.save_all(str(results_dir), force_overwrite=True)

        # Verify results files were created
        import os
        created_files = list(os.listdir(results_dir))
        assert len(created_files) > 0, f"No result files created in {results_dir}"

        # Check for expected result files
        assert (results_dir / 'summary.csv').exists(), "summary.csv not found"
        assert (results_dir / 'analytics.csv').exists(), "analytics.csv not found"
        assert (results_dir / 'figures').exists(), "figures directory not found"
        assert (results_dir / 'figures').is_dir(), "figures should be a directory"

        # Verify figures directory has content
        figures_dir = results_dir / 'figures'
        figure_files = list(os.listdir(figures_dir))
        assert len(figure_files) > 0, f"No figures generated"
        # Figures can be in various formats (SVG, HTML, PNG, PDF, etc.)
        assert any(f.endswith(('.html', '.png', '.pdf', '.svg')) for f in figure_files), \
            f"No figure files found. Found: {figure_files}"
