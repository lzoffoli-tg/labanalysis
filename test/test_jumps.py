"""jump analysis testing for SingleJump, JumpExercise, and DropJump"""
import sys
from os.path import abspath, dirname
import pytest
import pandas as pd

# add project root to path like other tests do
sys.path.append(dirname(dirname(abspath(__file__))))

# import classes under test
from src.labanalysis import SingleJump, JumpExercise, DropJump

@pytest.fixture(scope="session")
def tdf_path():
    return "C:\\Users\\smontanari\\Technogym SPA\\SCIENTIFIC RESEARCH - Documents\\t-lab\\test\\musetti_lorenzo\\2025_11_27\\jump_tests\\tracked_data\\dropjump_sx_1.tdf"

@pytest.fixture(scope="session")
def bodymass_kg():
    return float(70)

@pytest.fixture(scope="session")
def box_height_cm():
    return float(30)

@pytest.fixture(scope="session")
def left_foot_ground_reaction_force():
    return "left_frz"

@pytest.fixture(scope="session")
def right_foot_ground_reaction_force():
    return None
# ------------------------------
# SingleJump tests
# ------------------------------

def test_single_jump_from_tdf(tdf_path, left_foot_ground_reaction_force, right_foot_ground_reaction_force, bodymass_kg):
    jump = SingleJump.from_tdf(file=tdf_path, left_foot_ground_reaction_force= left_foot_ground_reaction_force,  
                               right_foot_ground_reaction_force= right_foot_ground_reaction_force, 
                               bodymass_kg=bodymass_kg)
    assert isinstance(jump, SingleJump)
    # basic properties should be numeric and finite
    assert isinstance(jump.elevation_cm, float)
    assert isinstance(jump.takeoff_velocity_ms, float)
    assert isinstance(jump.output_metrics, pd.DataFrame)
    cols = set(jump.output_metrics.columns)
    assert {"elevation_cm", "takeoff_velocity_m/s", "contact_time_ms", "flight_time_ms"}.issubset(cols)
    # flight time should be positive if a valid jump is present
    assert jump.flight_time_s > 0

# ------------------------------
# JumpExercise tests
# ------------------------------

def test_jump_exercise_from_tdf(tdf_path, left_foot_ground_reaction_force, right_foot_ground_reaction_force, bodymass_kg):
    exercise = JumpExercise.from_tdf(file=tdf_path, left_foot_ground_reaction_force= left_foot_ground_reaction_force,  
                               right_foot_ground_reaction_force= right_foot_ground_reaction_force, 
                               bodymass_kg=bodymass_kg)
    assert isinstance(exercise, JumpExercise)
    jumps = exercise.jumps
    assert isinstance(jumps, list)
    assert len(jumps) > 0, "At least one jump should be detected"
    assert all(isinstance(j, SingleJump) for j in jumps)
    for j in jumps:
        assert isinstance(j.elevation_cm, float)
        assert j.flight_time_s > 0

# ------------------------------
# DropJump tests
# ------------------------------

def test_drop_jump_from_tdf(tdf_path, left_foot_ground_reaction_force, right_foot_ground_reaction_force, bodymass_kg, box_height_cm):
    # Provide optional EMG thresholds via defaults; not required for object creation
    dj = DropJump.from_tdf(
        file=tdf_path,
        box_height_cm=box_height_cm,
        bodymass_kg=bodymass_kg,
        left_foot_ground_reaction_force= left_foot_ground_reaction_force,  
                               right_foot_ground_reaction_force= right_foot_ground_reaction_force, 
    )
    assert isinstance(dj, DropJump)
    # Check box height propagation
    assert isinstance(dj.box_height_cm, float)
    assert dj.box_height_cm == pytest.approx(float(box_height_cm))
    # Output metrics should include box height and standard jump fields
    out = dj.output_metrics
    assert isinstance(out, pd.DataFrame)
    cols = set(out.columns)
    assert {"box_height_cm", "elevation_cm", "takeoff_velocity_m/s"}.issubset(cols)
    # Landing and flight phases should be sliceable
    landing = dj.landing_phase
    assert hasattr(landing, 'index')
    flight = dj.flight_phase
    assert hasattr(flight, 'index')
    # Activation times dict (may be empty if no EMG available) should be accessible via property
    _ = dj.activation_time_ms

# ------------------------------
# Error handling tests (label missing)
# ------------------------------

def test_single_jump_missing_labels_raises(tmp_path, left_foot_ground_reaction_force, right_foot_ground_reaction_force,bodymass_kg):
    # Create a fake path to trigger ValueError in from_tdf when mandatory labels not found
    # We don't have I/O helpers here, but from_tdf will attempt to parse; we expect a ValueError
    # when neither left/right labels are provided (both set to None).
    with pytest.raises(ValueError):
        SingleJump.from_tdf(file=str(tmp_path / "empty.tdf"), bodymass_kg=bodymass_kg,
                            left_foot_ground_reaction_force=None,
                            right_foot_ground_reaction_force=None)

