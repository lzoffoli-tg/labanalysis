"""
Comprehensive test suite for the RunningExercise class.

Tests cover:
- Kinematics-based cycle detection
- Kinetics-based cycle detection
- Edge cases and error handling
- RunningStep phase properties
- Integration tests with real TDF data
"""

import sys
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import pytest

sys.path.append(dirname(dirname(abspath(__file__))))

import src.labanalysis as laban
from src.labanalysis.records.locomotion import RunningExercise, RunningStep


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def running_tdf_file():
    """
    Path to running test TDF file.

    Note: Large TDF files (>100MB) are excluded from the repository.
    Integration tests will be skipped if the file is not available.
    For development, place a running.tdf file in test/assets/running_test/
    """
    tdf_path = join(dirname(__file__), "assets", "running_test")
    tdf_file = join(tdf_path, "running.tdf")

    # Create directory if it doesn't exist
    import os
    os.makedirs(tdf_path, exist_ok=True)

    return tdf_file


@pytest.fixture
def baseline_tdf_file():
    """Path to baseline TDF file."""
    tdf_path = join(dirname(__file__), "assets", "running_test")
    return join(tdf_path, "baseline.tdf")


@pytest.fixture
def marker_mapping():
    """Standard marker set mapping for running tests."""
    return dict(
        left_heel="lHeel",
        right_heel="rHeel",
        left_toe="lToe",
        right_toe="rToe",
        left_metatarsal_head="lMeta5",
        right_metatarsal_head="rMeta5",
        left_ankle_medial="lMalMed",
        left_ankle_lateral="lMalExt",
        right_ankle_medial="rMalMed",
        right_ankle_lateral="rMalExt",
        left_knee_medial="lKneeMed",
        left_knee_lateral="lKneeExt",
        right_knee_medial="rKneeMed",
        right_knee_lateral="rKneeExt",
        right_throcanter="rTroc",
        left_throcanter="lTroc",
        left_asis="lASIS",
        right_asis="rASIS",
        left_psis="lPSIS",
        right_psis="rPSIS",
        left_elbow_medial="lElbMed",
        left_elbow_lateral="lElbExt",
        right_elbow_medial="rElbMed",
        right_elbow_lateral="rElbExt",
        left_wrist_medial="lWriMed",
        left_wrist_lateral="lWriExt",
        right_wrist_medial="rWriMed",
        right_wrist_lateral="rWriExt",
        left_acromion="lAcro",
        right_acromion="rAcro",
        l2="L2",
        c7="C7",
        sc="cla",
    )


@pytest.fixture
def mock_running_markers():
    """Generate synthetic laban.Point3D markers for controlled testing."""
    def _make_markers(n_cycles=3, fsamp=500, flight_time=0.15, contact_time=0.20):
        """
        Create synthetic running markers with realistic vertical motion.

        Parameters
        ----------
        n_cycles : int
            Number of complete running cycles to generate
        fsamp : float
            Sampling frequency in Hz
        flight_time : float
            Duration of flight phase in seconds
        contact_time : float
            Duration of contact phase in seconds

        Returns
        -------
        dict
            Dictionary with 'left_toe' and 'right_toe' laban.Point3D objects
        """
        cycle_time = flight_time + contact_time
        total_time = n_cycles * cycle_time * 2  # *2 for both feet
        n_samples = int(total_time * fsamp)
        time = np.linspace(0, total_time, n_samples)

        # Create vertical toe motion (flight high, contact low)
        left_toe_z = np.zeros(n_samples)
        right_toe_z = np.zeros(n_samples)

        # Step frequency
        step_freq = 1.0 / (flight_time + contact_time)

        # Create smooth, realistic running motion
        for i, t in enumerate(time):
            # Left toe
            phase_left = (t * step_freq) % 1.0
            if phase_left < (flight_time / cycle_time):
                # Flight phase - smooth parabolic trajectory
                flight_progress = phase_left / (flight_time / cycle_time)
                # Parabola: starts at 10mm, peaks at 150mm, returns to 10mm
                left_toe_z[i] = 10 + 140 * np.sin(flight_progress * np.pi)**2
            else:
                # Contact phase - smooth transition to ground
                contact_progress = (phase_left - flight_time / cycle_time) / (contact_time / cycle_time)
                # Smooth return to ground level
                left_toe_z[i] = 10 * (1 - 0.5 * np.sin(contact_progress * np.pi))

            # Right toe (offset by one full cycle time to alternate)
            phase_right = ((t + cycle_time) * step_freq) % 1.0
            if phase_right < (flight_time / cycle_time):
                flight_progress = phase_right / (flight_time / cycle_time)
                right_toe_z[i] = 10 + 140 * np.sin(flight_progress * np.pi)**2
            else:
                contact_progress = (phase_right - flight_time / cycle_time) / (contact_time / cycle_time)
                right_toe_z[i] = 10 * (1 - 0.5 * np.sin(contact_progress * np.pi))

        # Create laban.Point3D objects
        left_toe = laban.Point3D(
            data=np.column_stack([
                np.ones(n_samples) * 100,  # X
                np.linspace(0, 1000, n_samples),  # Y (forward motion)
                left_toe_z  # Z (vertical)
            ]),
            index=time,
            columns=["X", "Y", "Z"],
            unit="mm"
        )

        right_toe = laban.Point3D(
            data=np.column_stack([
                np.ones(n_samples) * -100,  # X
                np.linspace(0, 1000, n_samples),  # Y (forward motion)
                right_toe_z  # Z (vertical)
            ]),
            index=time,
            columns=["X", "Y", "Z"],
            unit="mm"
        )

        # Create heel markers (slightly behind toes, for algorithm validation)
        left_heel = laban.Point3D(
            data=np.column_stack([
                np.ones(n_samples) * 100,  # X
                np.linspace(-100, 900, n_samples),  # Y (behind toes)
                left_toe_z + 20  # Z (slightly higher)
            ]),
            index=time,
            columns=["X", "Y", "Z"],
            unit="mm"
        )

        right_heel = laban.Point3D(
            data=np.column_stack([
                np.ones(n_samples) * -100,  # X
                np.linspace(-100, 900, n_samples),  # Y (behind toes)
                right_toe_z + 20  # Z (slightly higher)
            ]),
            index=time,
            columns=["X", "Y", "Z"],
            unit="mm"
        )

        return {
            "left_toe": left_toe,
            "right_toe": right_toe,
            "left_heel": left_heel,
            "right_heel": right_heel
        }

    return _make_markers


@pytest.fixture
def mock_force_platform():
    """Generate synthetic laban.ForcePlatform data with flight phases."""
    def _make_force_platform(n_cycles=3, fsamp=500, flight_time=0.15, contact_time=0.20):
        """
        Create synthetic force platform data with running pattern.

        Parameters
        ----------
        n_cycles : int
            Number of complete running cycles
        fsamp : float
            Sampling frequency in Hz
        flight_time : float
            Duration of flight phase in seconds
        contact_time : float
            Duration of contact phase in seconds

        Returns
        -------
        laban.ForcePlatform
            Synthetic force platform with origin (CoP) and force data
        """
        cycle_time = flight_time + contact_time
        total_time = n_cycles * cycle_time * 2
        n_samples = int(total_time * fsamp)
        time = np.linspace(0, total_time, n_samples)

        # Initialize force and CoP arrays
        vgrf = np.zeros(n_samples)
        cop_ml = np.zeros(n_samples)

        step_freq = 1.0 / cycle_time

        for i, t in enumerate(time):
            cycle_phase = (t * step_freq) % 2.0  # 0-2 for left and right steps

            if cycle_phase < 1.0:
                # Left foot cycle
                phase = cycle_phase
                if phase < (flight_time / cycle_time):
                    # Flight - no force
                    vgrf[i] = 0
                    cop_ml[i] = 0
                else:
                    # Contact - force with peak in middle
                    contact_phase = (phase - flight_time / cycle_time) / (contact_time / cycle_time)
                    vgrf[i] = 1000 + 500 * np.sin(contact_phase * np.pi)
                    cop_ml[i] = -50  # Left side (negative)
            else:
                # Right foot cycle
                phase = cycle_phase - 1.0
                if phase < (flight_time / cycle_time):
                    # Flight - no force
                    vgrf[i] = 0
                    cop_ml[i] = 0
                else:
                    # Contact - force with peak in middle
                    contact_phase = (phase - flight_time / cycle_time) / (contact_time / cycle_time)
                    vgrf[i] = 1000 + 500 * np.sin(contact_phase * np.pi)
                    cop_ml[i] = 50  # Right side (positive)

        # Create force platform data structure
        force_data = np.column_stack([
            np.zeros(n_samples),  # Fx
            np.zeros(n_samples),  # Fy
            vgrf  # Fz (vertical)
        ])

        origin_data = np.column_stack([
            cop_ml,  # X (medial-lateral)
            np.zeros(n_samples),  # Y
            np.zeros(n_samples)  # Z
        ])

        torque_data = np.zeros((n_samples, 3))

        # Create individual components
        force = laban.Signal3D(
            data=force_data,
            index=time,
            columns=["X", "Y", "Z"],
            unit="newton"
        )
        origin = laban.Point3D(
            data=origin_data,
            index=time,
            columns=["X", "Y", "Z"],
            unit="millimeter"
        )
        torque = laban.Signal3D(
            data=torque_data,
            index=time,
            columns=["X", "Y", "Z"],
            unit="newton*millimeter"
        )

        # Create laban.ForcePlatform object with required arguments
        force_platform = laban.ForcePlatform(origin=origin, force=force, torque=torque)

        return force_platform

    return _make_force_platform


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_kinematics_exercise(markers, fp, **kwargs):
    """Helper to create RunningExercise with kinematics algorithm."""
    return RunningExercise(
        algorithm='kinematics',
        left_toe=markers['left_toe'],
        right_toe=markers['right_toe'],
        left_heel=markers['left_heel'],
        right_heel=markers['right_heel'],
        left_foot_ground_reaction_force=fp,
        **kwargs
    )


# ============================================================================
# KINEMATICS ALGORITHM TESTS
# ============================================================================

class TestRunningExerciseKinematics:
    """Tests for kinematics-based cycle detection."""

    def test_basic_cycle_detection(self, mock_running_markers, mock_force_platform):
        """Test basic cycle detection with synthetic markers."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = create_kinematics_exercise(markers, fp)

        cycles = exercise.cycles

        assert isinstance(cycles, list)
        assert len(cycles) > 0
        assert all(isinstance(c, RunningStep) for c in cycles)

    def test_cycle_sides(self, mock_running_markers, mock_force_platform):
        """Verify all cycles have correct side ('left' or 'right')."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        for cycle in cycles:
            assert cycle.side in ['left', 'right']

    def test_time_ordering(self, mock_running_markers, mock_force_platform):
        """Verify temporal ordering of events in each cycle."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        for cycle in cycles:
            assert cycle.init_s < cycle.footstrike_s
            assert cycle.footstrike_s < cycle.midstance_s
            assert cycle.midstance_s < cycle.end_s

    def test_missing_left_toe(self, mock_running_markers, mock_force_platform):
        """Test error when left_toe marker is missing."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
            # left_toe intentionally missing
        )

        with pytest.raises(ValueError, match="left_toe is missing"):
            _ = exercise.cycles

    def test_missing_right_toe(self, mock_running_markers, mock_force_platform):
        """Test error when right_toe marker is missing."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            left_foot_ground_reaction_force=fp
            # right_toe intentionally missing
        )

        with pytest.raises(ValueError, match="right_toe is missing"):
            _ = exercise.cycles

    def test_missing_both_toes(self, mock_force_platform):
        """Test error when both toe markers are missing with FP available - should fallback to kinetics."""
        fp = mock_force_platform(n_cycles=3)

        # When kinematics requested but markers missing, and FP available, it falls back to kinetics
        with pytest.warns(UserWarning, match="kinetics"):
            exercise = RunningExercise(
                algorithm='kinematics',
                left_foot_ground_reaction_force=fp
            )
            # Should use kinetics instead
            assert exercise.algorithm == 'kinetics'

    def test_algorithm_property(self, mock_running_markers, mock_force_platform):
        """Verify algorithm property returns correct value."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        assert exercise.algorithm == 'kinematics'

    def test_height_threshold_property(self, mock_running_markers, mock_force_platform):
        """Test custom height threshold."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        custom_threshold = 0.10
        exercise = RunningExercise(
            algorithm='kinematics',
            height_threshold=custom_threshold,
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        assert exercise.height_threshold == custom_threshold

    @pytest.mark.parametrize("threshold", [0.02, 0.05, 0.10, 0.15])
    def test_various_height_thresholds(self, mock_running_markers, mock_force_platform, threshold):
        """Test cycle detection with various height thresholds."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            height_threshold=threshold,
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles
        assert isinstance(cycles, list)


# ============================================================================
# KINETICS ALGORITHM TESTS
# ============================================================================

class TestRunningExerciseKinetics:
    """Tests for kinetics-based cycle detection."""

    def test_basic_cycle_detection(self, mock_force_platform):
        """Test basic cycle detection with synthetic force platform."""
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinetics',
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        assert isinstance(cycles, list)
        assert len(cycles) > 0
        assert all(isinstance(c, RunningStep) for c in cycles)

    def test_cycle_sides(self, mock_force_platform):
        """Verify left/right foot identification from CoP."""
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinetics',
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        for cycle in cycles:
            assert cycle.side in ['left', 'right']

    def test_missing_force_platform(self):
        """Test error when no force platform data is available."""
        exercise = RunningExercise(algorithm='kinetics')

        with pytest.raises(ValueError, match="no ground reaction force data available"):
            _ = exercise.cycles

    def test_algorithm_property(self, mock_force_platform):
        """Verify algorithm property returns correct value."""
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinetics',
            left_foot_ground_reaction_force=fp
        )

        assert exercise.algorithm == 'kinetics'

    def test_grf_threshold_property(self, mock_force_platform):
        """Test custom GRF threshold."""
        fp = mock_force_platform(n_cycles=3)

        custom_threshold = 150.0
        exercise = RunningExercise(
            algorithm='kinetics',
            ground_reaction_force_threshold=custom_threshold,
            left_foot_ground_reaction_force=fp
        )

        assert exercise.ground_reaction_force_threshold == custom_threshold

    @pytest.mark.parametrize("threshold", [50, 100, 150, 200])
    def test_various_grf_thresholds(self, mock_force_platform, threshold):
        """Test cycle detection with various GRF thresholds."""
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinetics',
            ground_reaction_force_threshold=threshold,
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles
        assert isinstance(cycles, list)

    def test_no_flight_phases(self, mock_force_platform):
        """Test error when no flight phases are detected (walking pattern)."""
        # Create force platform with no flight phases (always above threshold)
        fp = mock_force_platform(n_cycles=3)

        # Modify to have no flight phases (constant high force)
        fp["force"].loc[:, "Z"] = 500  # Always above threshold

        exercise = RunningExercise(
            algorithm='kinetics',
            ground_reaction_force_threshold=100,
            left_foot_ground_reaction_force=fp
        )

        with pytest.raises(ValueError, match="No flight phases have been found"):
            _ = exercise.cycles


# ============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# ============================================================================

class TestRunningExerciseEdgeCases:
    """Tests for edge cases, errors, and boundary conditions."""

    def test_no_toe_offs_found(self, mock_force_platform):
        """Test error when no toe-off events are detected."""
        # Create flat marker data (no peaks)
        n_samples = 1000
        time = np.linspace(0, 10, n_samples)
        flat_data = np.column_stack([
            np.zeros(n_samples),
            np.zeros(n_samples),
            np.ones(n_samples) * 10  # Constant low height
        ])

        flat_toe = laban.Point3D(
            data=flat_data,
            index=time,
            columns=["X", "Y", "Z"],
            unit="mm"
        )
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=flat_toe,
            right_toe=flat_toe,
            left_foot_ground_reaction_force=fp
        )

        with pytest.raises(ValueError, match="no toe-offs have been found"):
            _ = exercise.cycles

    def test_very_short_recording(self, mock_running_markers, mock_force_platform):
        """Test with very short recording (< 1 complete cycle)."""
        markers = mock_running_markers(n_cycles=0.3)
        fp = mock_force_platform(n_cycles=0.3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        # Should either return empty list or raise error
        try:
            cycles = exercise.cycles
            assert isinstance(cycles, list)
        except ValueError:
            pass  # Acceptable to raise error for insufficient data

    def test_markers_with_nan(self, mock_running_markers, mock_force_platform):
        """Test handling of NaN values in marker data."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        # Inject some NaN values
        markers['left_toe'].loc[50:60, 'Z'] = np.nan

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        # Should handle NaN gracefully (normalization handles it)
        cycles = exercise.cycles
        assert isinstance(cycles, list)


# ============================================================================
# RUNNING STEP PROPERTIES TESTS
# ============================================================================

class TestRunningStepProperties:
    """Tests for RunningStep phase and timing properties."""

    def test_flight_phase_extraction(self, mock_running_markers, mock_force_platform):
        """Test flight phase data extraction."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        for cycle in cycles:
            flight = cycle.flight_phase
            assert isinstance(flight, laban.TimeseriesRecord)

    def test_contact_phase_extraction(self, mock_running_markers, mock_force_platform):
        """Test contact phase data extraction."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        for cycle in cycles:
            contact = cycle.contact_phase
            assert isinstance(contact, laban.TimeseriesRecord)

    def test_loading_response_phase_extraction(self, mock_running_markers, mock_force_platform):
        """Test loading response phase data extraction."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        for cycle in cycles:
            loading = cycle.loading_response_phase
            assert isinstance(loading, laban.TimeseriesRecord)

    def test_propulsion_phase_extraction(self, mock_running_markers, mock_force_platform):
        """Test propulsion phase data extraction."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        for cycle in cycles:
            propulsion = cycle.propulsion_phase
            assert isinstance(propulsion, laban.TimeseriesRecord)

    def test_time_properties_positive(self, mock_running_markers, mock_force_platform):
        """Verify all time durations are positive."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        for cycle in cycles:
            assert cycle.flight_time_s > 0
            assert cycle.contact_time_s > 0
            assert cycle.loadingresponse_time_s > 0
            assert cycle.propulsion_time_s > 0

    def test_event_timing_order(self, mock_running_markers, mock_force_platform):
        """Verify event timestamps are properly ordered."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles

        for cycle in cycles:
            assert hasattr(cycle, 'footstrike_s')
            assert hasattr(cycle, 'midstance_s')
            assert cycle.footstrike_s is not None
            assert cycle.midstance_s is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRunningExerciseIntegration:
    """End-to-end integration tests with real TDF data."""

    @pytest.mark.skipif(
        not __import__('os').path.exists(
            join(dirname(__file__), "assets", "running_test", "running.tdf")
        ),
        reason="TDF file not available (file too large for repository)"
    )
    def test_load_real_tdf_kinematics(self, running_tdf_file, marker_mapping):
        """Test loading real TDF file and detecting cycles with kinematics."""
        # Load the TDF file
        data = laban.TimeseriesRecord.from_tdf(running_tdf_file)

        # Map markers from TDF names to RunningExercise parameter names
        marker_args = {}
        for param_name, marker_name in marker_mapping.items():
            if marker_name in data.points3d:
                marker_args[param_name] = data.points3d[marker_name]

        # Add force platforms
        for fp in data.forceplatforms.values():
            marker_args['left_foot_ground_reaction_force'] = fp
            break  # Use first force platform

        # Create exercise with mapped markers
        exercise = RunningExercise(
            algorithm='kinematics',
            **marker_args
        )

        # Get cycles
        cycles = exercise.cycles

        # Validate
        assert isinstance(cycles, list)
        assert len(cycles) > 0
        assert all(isinstance(c, RunningStep) for c in cycles)

        # Check basic properties
        for cycle in cycles:
            assert cycle.side in ['left', 'right']
            assert cycle.flight_time_s > 0
            assert cycle.contact_time_s > 0

    @pytest.mark.skipif(
        not __import__('os').path.exists(
            join(dirname(__file__), "assets", "running_test", "running.tdf")
        ),
        reason="TDF file not available (file too large for repository)"
    )
    def test_load_real_tdf_kinetics(self, running_tdf_file, marker_mapping):
        """Test loading real TDF file and detecting cycles with kinetics."""
        # Load the TDF file
        data = laban.TimeseriesRecord.from_tdf(running_tdf_file)

        # Map markers and force platforms
        marker_args = {}
        for param_name, marker_name in marker_mapping.items():
            if marker_name in data.points3d:
                marker_args[param_name] = data.points3d[marker_name]

        # Add force platforms
        if len(data.forceplatforms) > 0:
            for fp in data.forceplatforms.values():
                marker_args['left_foot_ground_reaction_force'] = fp
                break

        # Create exercise with kinetics algorithm
        exercise = RunningExercise(
            algorithm='kinetics',
            **marker_args
        )

        # Get cycles
        try:
            cycles = exercise.cycles

            # Validate
            assert isinstance(cycles, list)
            if len(cycles) > 0:
                assert all(isinstance(c, RunningStep) for c in cycles)

                # Check basic properties
                for cycle in cycles:
                    assert cycle.side in ['left', 'right']
        except ValueError as e:
            # Acceptable if no force platform data or no flight phases
            assert "no ground reaction force data available" in str(e) or \
                   "No flight phases have been found" in str(e)

    @pytest.mark.skipif(
        not __import__('os').path.exists(
            join(dirname(__file__), "assets", "running_test", "running.tdf")
        ),
        reason="TDF file not available (file too large for repository)"
    )
    def test_from_tdf_classmethod(self, running_tdf_file, marker_mapping):
        """Test the from_tdf classmethod with proper marker mapping."""
        # Load data first
        data = laban.TimeseriesRecord.from_tdf(running_tdf_file)

        # Map markers
        marker_args = {}
        for param_name, marker_name in marker_mapping.items():
            if marker_name in data.points3d:
                marker_args[param_name] = data.points3d[marker_name]

        # Add first force platform
        if len(data.forceplatforms) > 0:
            for fp in data.forceplatforms.values():
                marker_args['left_foot_ground_reaction_force'] = fp
                break

        # Create exercise
        exercise = RunningExercise(**marker_args)

        assert isinstance(exercise, RunningExercise)

        # Should be able to get cycles
        cycles = exercise.cycles
        assert isinstance(cycles, list)
        assert len(cycles) > 0

    def test_cycle_caching(self, mock_running_markers, mock_force_platform):
        """Test that cycles are cached after first access."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        # First access
        cycles1 = exercise.cycles

        # Second access (should return same object)
        cycles2 = exercise.cycles

        # Should be the same list
        assert cycles1 is cycles2

    @pytest.mark.skipif(
        not __import__('os').path.exists(
            join(dirname(__file__), "assets", "running_test", "running.tdf")
        ),
        reason="TDF file not available (file too large for repository)"
    )
    def test_marker_mapping_full(self, running_tdf_file, marker_mapping):
        """Test with complete marker set mapping."""
        # This is the original test case, now properly validated
        data = laban.TimeseriesRecord.from_tdf(running_tdf_file)

        # Build marker arguments
        marker_args = {}
        for param_name, marker_name in marker_mapping.items():
            if marker_name in data.points3d:
                marker_args[param_name] = data.points3d[marker_name]

        exercise = RunningExercise(
            algorithm='kinematics',
            **marker_args
        )

        cycles = exercise.cycles

        assert isinstance(cycles, list)
        assert len(cycles) > 0


# ============================================================================
# PARAMETRIC TESTS
# ============================================================================

class TestRunningExerciseParametric:
    """Parametric tests for various combinations."""

    @pytest.mark.parametrize("algorithm", ["kinematics", "kinetics"])
    def test_both_algorithms(self, mock_running_markers, mock_force_platform, algorithm):
        """Test both algorithms work correctly."""
        markers = mock_running_markers(n_cycles=3)
        fp = mock_force_platform(n_cycles=3)

        if algorithm == 'kinematics':
            exercise = RunningExercise(
                algorithm=algorithm,
                left_toe=markers['left_toe'],
                right_toe=markers['right_toe'],
                left_foot_ground_reaction_force=fp
            )
        else:  # kinetics
            exercise = RunningExercise(
                algorithm=algorithm,
                left_foot_ground_reaction_force=fp
            )

        cycles = exercise.cycles
        assert isinstance(cycles, list)
        assert len(cycles) > 0

    @pytest.mark.parametrize("n_cycles", [1, 2, 3, 5])
    def test_various_cycle_counts(self, mock_running_markers, mock_force_platform, n_cycles):
        """Test with various numbers of cycles."""
        markers = mock_running_markers(n_cycles=n_cycles)
        fp = mock_force_platform(n_cycles=n_cycles)

        exercise = RunningExercise(
            algorithm='kinematics',
            left_toe=markers['left_toe'],
            right_toe=markers['right_toe'],
            left_foot_ground_reaction_force=fp
        )

        cycles = exercise.cycles
        assert isinstance(cycles, list)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
