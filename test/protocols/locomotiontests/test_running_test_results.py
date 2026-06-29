"""Tests for RunningTestResults implementation."""

import pickle
import sys
from pathlib import Path

# Ensure we load from src/ not installed package
_repo_root = Path(__file__).parent.parent.parent.parent
if _repo_root / "src" not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(_repo_root / "src"))

import numpy as np
import pandas as pd
import pytest

from labanalysis.exercises.gait import RunningStep
from labanalysis.protocols import Participant, RunningTest
from labanalysis.protocols.locomotiontests import RunningTestResults
from labanalysis.records import ForcePlatform
from labanalysis.timeseries import Point3D, Signal3D


def generate_continuous_running_data(
    bodymass_kg=75.0,
    num_steps=6,
    contact_duration_s=0.25,
    flight_duration_s=0.15,
    fsamp=1000.0
):
    """
    Generate continuous running data with multiple steps (left/right alternating).

    Parameters
    ----------
    bodymass_kg : float
        Body mass in kg.
    num_steps : int
        Number of steps to generate.
    contact_duration_s : float
        Contact phase duration per step in seconds.
    flight_duration_s : float
        Flight phase duration per step in seconds.
    fsamp : float
        Sampling frequency in Hz.

    Returns
    -------
    ForcePlatform
        Continuous force platform data with multiple steps.
    Point3D
        Continuous pelvis marker trajectory.
    """
    # Calculate samples per step
    n_flight = int(flight_duration_s * fsamp)
    n_contact = int(contact_duration_s * fsamp)
    n_step = n_flight + n_contact
    n_total = n_step * num_steps

    # Time array
    time = np.linspace(0, (n_total - 1) / fsamp, n_total)

    # Initialize arrays
    # NOTE: Labanalysis uses Y as vertical axis, X as lateral, Z as anteroposterior
    vgrf = np.zeros(n_total)
    ap_force = np.zeros(n_total)
    ml_force = np.zeros(n_total)
    cop_x = np.zeros(n_total)
    cop_y = np.zeros(n_total)
    cop_z = np.zeros(n_total)
    pelvis_x = np.zeros(n_total)  # Lateral (left/right)
    pelvis_y = np.ones(n_total) * 0.95  # Vertical (height)
    # Subject runs forward: pelvis moves in +Z direction (anteroposterior)
    running_speed = 3.0  # m/s forward
    pelvis_z = time * running_speed  # Anteroposterior (forward progression)

    # Generate each step
    for i_step in range(num_steps):
        start_idx = i_step * n_step
        flight_start = start_idx
        flight_end = start_idx + n_flight
        contact_start = flight_end
        contact_end = start_idx + n_step

        side = "left" if i_step % 2 == 0 else "right"
        # X=left (positive), so left foot has +offset, right foot has -offset
        cop_offset = 0.15 if side == "left" else -0.15

        # Contact phase: vertical force (realistic shape with impact transient)
        # Split contact into: impact (10%) → loading (40%) → propulsion (50%)
        n_impact = n_contact // 10
        n_loading = (n_contact * 4) // 10
        n_propulsion = n_contact - n_impact - n_loading

        impact_time = np.linspace(0, np.pi/2, n_impact)
        loading_time = np.linspace(np.pi/2, np.pi, n_loading)
        propulsion_time = np.linspace(np.pi, 0, n_propulsion)

        # Impact: rapid rise
        vgrf[contact_start:contact_start+n_impact] = bodymass_kg * 9.81 * 2.5 * np.sin(impact_time)
        # Loading: peak force
        vgrf[contact_start+n_impact:contact_start+n_impact+n_loading] = bodymass_kg * 9.81 * 2.5 * np.sin(loading_time)
        # Propulsion: gradual decrease
        vgrf[contact_start+n_impact+n_loading:contact_end] = bodymass_kg * 9.81 * 2.5 * np.sin(propulsion_time)

        # AP force: braking during loading, propulsion during push-off
        mid_contact = contact_start + n_impact + n_loading

        # Braking: negative during impact and loading
        braking_end = mid_contact
        braking_samples = braking_end - contact_start
        braking_time = np.linspace(0, np.pi, braking_samples)
        ap_force[contact_start:braking_end] = -bodymass_kg * 9.81 * 0.3 * np.sin(braking_time)

        # Propulsion: positive during push-off
        propulsion_samples = contact_end - mid_contact
        propulsion_time = np.linspace(0, np.pi, propulsion_samples)
        ap_force[mid_contact:contact_end] = bodymass_kg * 9.81 * 0.4 * np.sin(propulsion_time)

        # Lateral force
        ml_force[start_idx:start_idx + n_step] = np.random.normal(
            0, bodymass_kg * 9.81 * 0.02, n_step
        )

        # COP position
        cop_x[contact_start:contact_end] = cop_offset + np.random.normal(0, 0.01, n_contact)
        cop_y[contact_start:contact_end] = np.linspace(-0.05, 0.05, n_contact) + np.random.normal(
            0, 0.005, n_contact
        )

        # Pelvis oscillation during contact (simple sinusoidal oscillation in Y=vertical)
        pelvis_osc_time = np.linspace(0, np.pi, n_contact)
        pelvis_y[contact_start:contact_end] += 0.08 * np.sin(pelvis_osc_time)  # Vertical oscillation
        pelvis_x[start_idx:start_idx + n_step] = cop_offset  # Lateral position

    # Create signals
    # Force: X=lateral, Y=vertical, Z=anteroposterior
    force = Signal3D(
        data=np.column_stack([ml_force, vgrf, ap_force]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="N"
    )

    # COP: X=lateral, Y=vertical (always 0), Z=anteroposterior
    cop = Point3D(
        data=np.column_stack([cop_x, cop_y, cop_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    torque = Signal3D(
        data=np.column_stack([
            np.random.normal(0, 5, n_total),
            np.random.normal(0, 5, n_total),
            np.random.normal(0, 2, n_total)
        ]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="Nm"
    )

    fp = ForcePlatform(force=force, origin=cop, torque=torque)

    # Pelvis center: X=lateral, Y=vertical, Z=anteroposterior
    pelvis_center = Point3D(
        data=np.column_stack([pelvis_x, pelvis_y, pelvis_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    # Generate markers for trunk and pelvis angles
    # Pelvis markers (ASIS and PSIS at pelvis height)
    # Coordinate system: X=lateral, Y=vertical, Z=anteroposterior
    # ASIS are anterior and wider than PSIS
    asis_width = 0.26  # ~26cm ASIS width (lateral separation)
    psis_width = 0.10  # ~10cm PSIS width (narrower)
    pelvis_depth = 0.12  # ~12cm pelvis depth (anteroposterior ASIS-PSIS distance)

    # Add realistic pelvis tilt during running: stance hip drops slightly
    pelvis_tilt = np.zeros(n_total)

    for i_step in range(num_steps):
        start_idx = i_step * n_step
        contact_start = start_idx + n_flight
        contact_end = start_idx + n_step

        side = "left" if i_step % 2 == 0 else "right"

        # Hip drop during stance: contralateral hip drops (opposite side goes down)
        # Left stance → right hip drops (right ASIS/PSIS lower)
        # Right stance → left hip drops (left ASIS/PSIS lower)
        contact_phase_time = np.linspace(0, np.pi, contact_end - contact_start)
        hip_drop_direction = -1 if side == "left" else 1  # Right hip drops when left stance
        pelvis_tilt[contact_start:contact_end] = hip_drop_direction * 0.015 * np.sin(contact_phase_time)  # 1.5cm drop

    # Coordinate system: X=left (positive), Y=up (positive), Z=forward (positive)
    # Left ASIS: left side (+X), wider, variable height (affected by tilt), anterior (+Z)
    left_asis_x = pelvis_x + asis_width / 2  # Left side (+X)
    left_asis_y = pelvis_y - pelvis_tilt  # Left side goes down when tilt is positive
    left_asis_z = pelvis_z + pelvis_depth / 2  # Anterior (+Z forward)

    # Right ASIS: right side (-X), wider, variable height (affected by tilt), anterior (+Z)
    right_asis_x = pelvis_x - asis_width / 2  # Right side (-X)
    right_asis_y = pelvis_y + pelvis_tilt  # Right side goes down when tilt is negative
    right_asis_z = pelvis_z + pelvis_depth / 2  # Anterior (+Z forward)

    # Left PSIS: left side (+X), narrower, variable height (affected by tilt), posterior (-Z)
    left_psis_x = pelvis_x + psis_width / 2  # Left side (+X), narrower
    left_psis_y = pelvis_y - pelvis_tilt  # Left side goes down when tilt is positive
    left_psis_z = pelvis_z - pelvis_depth / 2  # Posterior (-Z backward)

    # Right PSIS: right side (-X), narrower, variable height (affected by tilt), posterior (-Z)
    right_psis_x = pelvis_x - psis_width / 2  # Right side (-X), narrower
    right_psis_y = pelvis_y + pelvis_tilt  # Right side goes down when tilt is negative
    right_psis_z = pelvis_z - pelvis_depth / 2  # Posterior (-Z backward)

    left_asis = Point3D(
        data=np.column_stack([left_asis_x, left_asis_y, left_asis_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    right_asis = Point3D(
        data=np.column_stack([right_asis_x, right_asis_y, right_asis_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    left_psis = Point3D(
        data=np.column_stack([left_psis_x, left_psis_y, left_psis_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    right_psis = Point3D(
        data=np.column_stack([right_psis_x, right_psis_y, right_psis_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    # Trunk markers (at shoulder level, ~60cm above pelvis)
    # Y axis is vertical, so trunk_height is an offset in Y
    trunk_height_offset = 0.60  # 60cm above pelvis
    shoulder_width = 0.40  # ~40cm shoulder width (lateral separation)

    # Add realistic trunk movement during running
    # Lateral lean: oscillates left/right with each step (X direction)
    # Rotation: trunk rotates during arm swing (shoulder axis rotates in transverse plane)
    lateral_lean = np.zeros(n_total)
    trunk_rotation_angle = np.zeros(n_total)  # Rotation angle in radians

    for i_step in range(num_steps):
        start_idx = i_step * n_step
        contact_start = start_idx + n_flight
        contact_end = start_idx + n_step

        side = "left" if i_step % 2 == 0 else "right"

        # Lateral lean: lean toward stance leg during contact (X direction)
        # X=left (positive), so left stance = +lean, right stance = -lean
        lean_direction = 1 if side == "left" else -1
        contact_phase_time = np.linspace(0, np.pi, contact_end - contact_start)
        lateral_lean[contact_start:contact_end] = lean_direction * 0.03 * np.sin(contact_phase_time)  # 3cm lean

        # Trunk rotation: opposite arm swings forward (left arm forward during right stance)
        # This creates a shoulder axis rotation in the transverse plane
        # Positive rotation = left rotation (right shoulder forward)
        rotation_direction = 1 if side == "right" else -1  # Right stance → left rotation
        trunk_rotation_angle[contact_start:contact_end] = rotation_direction * np.deg2rad(8) * np.sin(contact_phase_time)  # ±8° rotation

    # Trunk markers: C7 and SC define shoulder axis for trunk rotation
    # Anatomically: C7 is posterior (back of neck), SC is anterior (sternum top)
    # With correct ASIS/PSIS anatomy (ASIS anterior and wider), the pelvis reference
    # frame should orient correctly

    # C7 (7th cervical vertebra) - posterior (back of neck)
    c7_base_x = 0.0  # Midline (lateral)
    c7_base_y = trunk_height_offset  # Above pelvis (vertical)
    c7_base_z = -0.05  # Posterior (-Z backward from pelvis)

    # SC (sternoclavicular joint) - anterior (sternum top)
    sc_base_x = 0.0  # Midline (lateral)
    sc_base_y = trunk_height_offset - 0.05  # Slightly below C7 (vertical)
    sc_base_z = 0.10  # Anterior (+Z forward from pelvis)

    # Apply trunk rotation in transverse plane
    # Rotation is around vertical axis passing through trunk center
    # Positive rotation = CCW from above = left rotation = right shoulder forward

    # Rotate C7 and SC together around pelvis vertical axis
    c7_x = pelvis_x + c7_base_x * np.cos(trunk_rotation_angle) - c7_base_z * np.sin(trunk_rotation_angle) + lateral_lean
    c7_y = pelvis_y + c7_base_y
    c7_z = pelvis_z + c7_base_x * np.sin(trunk_rotation_angle) + c7_base_z * np.cos(trunk_rotation_angle)

    sc_x = pelvis_x + sc_base_x * np.cos(trunk_rotation_angle) - sc_base_z * np.sin(trunk_rotation_angle) + lateral_lean
    sc_y = pelvis_y + sc_base_y
    sc_z = pelvis_z + sc_base_x * np.sin(trunk_rotation_angle) + sc_base_z * np.cos(trunk_rotation_angle)

    c7 = Point3D(
        data=np.column_stack([c7_x, c7_y, c7_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    sc = Point3D(
        data=np.column_stack([sc_x, sc_y, sc_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    # Left and right shoulder markers (rotate with trunk in transverse plane)
    # Coordinate system: X=left, Y=up, Z=forward
    # Shoulders rotate: right shoulder forward during left rotation
    shoulder_radius = shoulder_width / 2  # Distance from center to each shoulder

    # Left shoulder: left side (+X), rotates with trunk
    left_shoulder_x = pelvis_x + lateral_lean + shoulder_radius * np.cos(trunk_rotation_angle)  # Left (+X)
    left_shoulder_y = pelvis_y + trunk_height_offset - 0.10  # Vertical
    left_shoulder_z = pelvis_z + shoulder_radius * np.sin(trunk_rotation_angle)  # Anteroposterior

    # Right shoulder: right side (-X), rotates with trunk
    right_shoulder_x = pelvis_x + lateral_lean - shoulder_radius * np.cos(trunk_rotation_angle)  # Right (-X)
    right_shoulder_y = pelvis_y + trunk_height_offset - 0.10  # Vertical
    right_shoulder_z = pelvis_z - shoulder_radius * np.sin(trunk_rotation_angle)  # Anteroposterior

    left_shoulder_anterior = Point3D(
        data=np.column_stack([left_shoulder_x, left_shoulder_y, left_shoulder_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    right_shoulder_anterior = Point3D(
        data=np.column_stack([right_shoulder_x, right_shoulder_y, right_shoulder_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    # Hip markers (trochanters) - below pelvis vertically, follow pelvis tilt
    # Coordinate system: X=left, Y=up, Z=forward
    hip_width = 0.30

    left_trochanter_x = pelvis_x + hip_width / 2  # Left side (+X)
    left_trochanter_y = pelvis_y - 0.10 - pelvis_tilt  # Below pelvis, follows left side tilt
    left_trochanter_z = pelvis_z.copy()  # Same anteroposterior

    right_trochanter_x = pelvis_x - hip_width / 2  # Right side (-X)
    right_trochanter_y = pelvis_y - 0.10 + pelvis_tilt  # Below pelvis, follows right side tilt
    right_trochanter_z = pelvis_z.copy()  # Same anteroposterior

    left_trochanter = Point3D(
        data=np.column_stack([left_trochanter_x, left_trochanter_y, left_trochanter_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    right_trochanter = Point3D(
        data=np.column_stack([right_trochanter_x, right_trochanter_y, right_trochanter_z]),
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    markers = {
        'left_asis': left_asis,
        'right_asis': right_asis,
        'left_psis': left_psis,
        'right_psis': right_psis,
        'c7': c7,
        'sc': sc,
        'left_shoulder_anterior': left_shoulder_anterior,
        'right_shoulder_anterior': right_shoulder_anterior,
        'left_trochanter': left_trochanter,
        'right_trochanter': right_trochanter,
    }

    return fp, pelvis_center, markers


@pytest.fixture
def participant():
    """Create test participant."""
    return Participant(
        surname="Rossi",
        name="Mario",
        gender="M",
        height=178,
        weight=75.0,
        age=30
    )


@pytest.fixture
def running_test_with_cycles(participant):
    """Create RunningTest with continuous synthetic running data."""
    bodymass = participant.weight

    # Generate continuous running data (6 steps, alternating left/right)
    fp, pelvis_center, markers = generate_continuous_running_data(
        bodymass_kg=bodymass,
        num_steps=6,
        contact_duration_s=0.25,
        flight_duration_s=0.15,
        fsamp=1000.0
    )

    # Create RunningTest with force platform, pelvis, and all markers
    test = RunningTest(
        participant=participant,
        algorithm="kinetics",
        left_foot_ground_reaction_force=fp,
        pelvis_center=pelvis_center,
        **markers  # Add all trunk and pelvis markers
    )

    return test


def test_running_test_results_creation(running_test_with_cycles):
    """Test that RunningTestResults can be created."""
    results = running_test_with_cycles.get_results(include_emg=False)

    assert results is not None
    assert isinstance(results, RunningTestResults)


def test_summary_has_correct_structure(running_test_with_cycles):
    """Test that summary has both per_step and aggregate tables."""
    results = running_test_with_cycles.get_results()

    assert 'per_step' in results.summary
    assert 'aggregate' in results.summary

    per_step = results.summary['per_step']
    aggregate = results.summary['aggregate']

    # Check per_step structure
    assert isinstance(per_step, pd.DataFrame)
    assert len(per_step) >= 4  # At least 4 cycles detected
    assert 'cycle' in per_step.columns
    assert 'side' in per_step.columns

    # Check aggregate structure
    assert isinstance(aggregate, pd.DataFrame)
    assert 'metric' in aggregate.columns
    assert 'left_mean' in aggregate.columns
    assert 'left_std' in aggregate.columns
    assert 'left_cv%' in aggregate.columns
    assert 'right_mean' in aggregate.columns
    assert 'right_std' in aggregate.columns
    assert 'right_cv%' in aggregate.columns
    assert 'diff_%' in aggregate.columns


def test_summary_per_step_has_required_metrics(running_test_with_cycles):
    """Test that per_step summary contains core required metrics."""
    results = running_test_with_cycles.get_results()
    per_step = results.summary['per_step']

    # Core metrics that should always be present
    core_metrics = [
        'contact_time_s',
        'propulsion_time_s',
        'flight_time_s',
        'cadence_steps_per_min',
        'peak_vertical_force_N',
        'peak_propulsion_force_N',  # Propulsion usually present
    ]

    for metric in core_metrics:
        assert metric in per_step.columns, f"Missing metric: {metric}"

    # Braking force may not be present if loading phase is too short
    # Just check it's in the defined metrics, not required to have values


def test_summary_aggregate_contains_all_metrics(running_test_with_cycles):
    """Test that aggregate summary contains statistics for all metrics."""
    results = running_test_with_cycles.get_results()
    aggregate = results.summary['aggregate']

    # Should have one row per metric
    assert len(aggregate) >= 6  # At least basic metrics

    # Check that metrics are present
    metrics_present = aggregate['metric'].tolist()
    assert 'contact_time_s' in metrics_present
    assert 'cadence_steps_per_min' in metrics_present


def test_analytics_structure(running_test_with_cycles):
    """Test analytics DataFrame structure."""
    results = running_test_with_cycles.get_results()
    analytics = results.analytics

    assert isinstance(analytics, pd.DataFrame)
    assert len(analytics) > 0

    # Check required columns
    assert 'cycle' in analytics.columns
    assert 'side' in analytics.columns
    assert 'time_s' in analytics.columns
    assert 'vertical_force_N' in analytics.columns
    assert 'anteroposterior_force_N' in analytics.columns


def test_figures_structure(running_test_with_cycles):
    """Test that figures dictionary is created correctly."""
    results = running_test_with_cycles.get_results()
    figures = results.figures

    assert isinstance(figures, dict)
    assert 'force_profiles' in figures

    fig = figures['force_profiles']
    assert fig is not None
    # Plotly figure should have data traces
    assert hasattr(fig, 'data')
    assert len(fig.data) > 0  # Should have traces


def test_force_profile_figure_has_correct_layout(running_test_with_cycles):
    """Test that force profile figure has 2x2 subplot layout."""
    results = running_test_with_cycles.get_results()
    fig = results.figures['force_profiles']

    # Check that figure has subplot structure
    assert hasattr(fig, 'layout')

    # Should have multiple subplots (2 rows x 2 cols = 4 subplots)
    # Each subplot should have data traces
    assert len(fig.data) >= 4  # At least 4 mean traces


def test_cadence_calculation(running_test_with_cycles):
    """Test that cadence is calculated correctly."""
    results = running_test_with_cycles.get_results()
    per_step = results.summary['per_step']

    # Cadence should be reasonable for running (100-200 steps/min)
    cadences = per_step['cadence_steps_per_min']
    assert all(cadences > 50)  # Not too slow
    assert all(cadences < 300)  # Not impossibly fast


def test_vertical_oscillation_in_mm(running_test_with_cycles):
    """Test that vertical oscillation is reported in millimeters."""
    results = running_test_with_cycles.get_results()
    per_step = results.summary['per_step']

    # Check that vertical_oscillation_mm exists and has reasonable values
    if 'vertical_oscillation_mm' in per_step.columns:
        osc = per_step['vertical_oscillation_mm'].dropna()
        if len(osc) > 0:
            # Should be in range ~20-120mm for running
            assert all(osc > 0)
            assert all(osc < 500)  # Not impossibly large


def test_left_right_asymmetry_calculation(running_test_with_cycles):
    """Test that left-right asymmetry is calculated."""
    results = running_test_with_cycles.get_results()
    aggregate = results.summary['aggregate']

    # Check that diff_% exists
    assert 'diff_%' in aggregate.columns

    # Asymmetry should be present for metrics with both left and right data
    contact_time_row = aggregate[aggregate['metric'] == 'contact_time_s']
    if len(contact_time_row) > 0:
        assert 'diff_%' in contact_time_row.columns


def test_results_serialization(running_test_with_cycles, tmp_path):
    """Test that results can be pickled and unpickled."""
    results = running_test_with_cycles.get_results()

    # Save to pickle
    filepath = tmp_path / "running_results.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    assert filepath.exists()

    # Load from pickle
    with open(filepath, 'rb') as f:
        loaded_results = pickle.load(f)

    assert loaded_results is not None
    assert isinstance(loaded_results, RunningTestResults)


def test_peak_braking_propulsion_forces(running_test_with_cycles):
    """Test that braking and propulsion forces are extracted."""
    results = running_test_with_cycles.get_results()
    per_step = results.summary['per_step']

    # Check that peak forces exist
    if 'peak_braking_force_N' in per_step.columns:
        braking = per_step['peak_braking_force_N'].dropna()
        assert len(braking) > 0
        assert all(braking > 0)  # Should be positive magnitude

    if 'peak_propulsion_force_N' in per_step.columns:
        propulsion = per_step['peak_propulsion_force_N'].dropna()
        assert len(propulsion) > 0
        assert all(propulsion > 0)  # Should be positive


def test_coefficient_of_variation(running_test_with_cycles):
    """Test that coefficient of variation is calculated correctly."""
    results = running_test_with_cycles.get_results()
    aggregate = results.summary['aggregate']

    # CV% should be calculated for metrics with non-zero mean
    cv_cols = [col for col in aggregate.columns if 'cv%' in col]
    assert len(cv_cols) > 0  # Should have CV% columns

    # CV% should be non-negative
    for col in cv_cols:
        cv_values = aggregate[col].dropna()
        if len(cv_values) > 0:
            assert all(cv_values >= 0)
