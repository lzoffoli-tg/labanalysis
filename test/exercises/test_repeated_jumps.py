"""Tests for repeated_jumps module."""

import numpy as np
import pytest


@pytest.mark.integration
class TestRepeatedJumps:
    """Test RepeatedJumps class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps
        assert RepeatedJumps is not None

    def test_repeated_jumps_inheritance(self):
        """Test RepeatedJumps inherits from WholeBody."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps
        from labanalysis.records.body import WholeBody

        assert issubclass(RepeatedJumps, WholeBody)

    def test_repeated_jumps_has_bodymass_property(self):
        """Test RepeatedJumps has bodymass_kg property."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps

        assert hasattr(RepeatedJumps, 'bodymass_kg')
        assert isinstance(getattr(RepeatedJumps, 'bodymass_kg'), property)

    def test_repeated_jumps_has_jumps_property(self):
        """Test RepeatedJumps has jumps property."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps

        assert hasattr(RepeatedJumps, 'jumps')

    def test_repeated_jumps_docstring_exists(self):
        """Test RepeatedJumps has comprehensive docstring."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps

        assert RepeatedJumps.__doc__ is not None
        assert len(RepeatedJumps.__doc__) > 100
        assert 'fatigue' in RepeatedJumps.__doc__.lower() or 'endurance' in RepeatedJumps.__doc__.lower()


@pytest.mark.integration
class TestRepeatedJumpsLocIloc:
    """Test loc/iloc indexing for RepeatedJumps preserves custom attributes."""

    @pytest.fixture
    def repeatedjumps(self):
        """Create a minimal RepeatedJumps for testing."""
        from labanalysis.exercises.repeated_jumps import RepeatedJumps
        from labanalysis.records import ForcePlatform
        from labanalysis.timeseries import Signal3D, Point3D

        # Create synthetic force platform data for repeated jumps
        fsamp = 1000.0
        duration = 3.0  # 3 seconds
        n_samples = int(duration * fsamp)
        index = np.linspace(0, duration, n_samples)

        # Simple sinusoidal force pattern simulating multiple jumps
        force_data = np.zeros((n_samples, 3))
        force_data[:, 1] = 400 + 300 * np.abs(np.sin(2 * np.pi * 2 * index))  # Vertical force oscillating

        origin_data = np.random.randn(n_samples, 3) * 0.01  # Small COP movement
        torque_data = np.random.randn(n_samples, 3) * 5.0

        origin = Point3D(origin_data, index, unit='m')
        force = Signal3D(force_data, index, unit='N', vertical_axis='Y')
        torque = Signal3D(torque_data, index, unit='Nm')

        fp_left = ForcePlatform(origin=origin, force=force, torque=torque)

        return RepeatedJumps(
            bodymass_kg=75.0,
            left_foot_ground_reaction_force=fp_left,
            exclude_jumps=[],
            straight_legs=True,
            free_hands=False
        )

    def test_loc_preserves_all_attributes(self, repeatedjumps):
        """Test loc[] preserves bodymass_kg, excluded_jumps, straight_legs, free_hands."""
        start_idx = repeatedjumps.index[100]
        end_idx = repeatedjumps.index[200]
        sliced = repeatedjumps.loc[start_idx:end_idx, :]

        assert isinstance(sliced, type(repeatedjumps))
        assert sliced.bodymass_kg == 75.0
        assert sliced.excluded_jumps == []
        assert sliced.straight_legs == True
        assert sliced.free_hands == False

    def test_iloc_preserves_all_attributes(self, repeatedjumps):
        """Test iloc[] preserves all custom attributes."""
        sliced = repeatedjumps.iloc[100:200, :]

        assert isinstance(sliced, type(repeatedjumps))
        assert sliced.bodymass_kg == 75.0
        assert sliced.excluded_jumps == []
        assert sliced.straight_legs == True
        assert sliced.free_hands == False

    def test_loc_setter_preserves_type(self, repeatedjumps):
        """Test loc[] setter works without breaking type."""
        repeatedjumps.loc[repeatedjumps.index[50], 'left_foot_ground_reaction_force'] = 99.0
        assert isinstance(repeatedjumps, type(repeatedjumps))
        assert repeatedjumps.bodymass_kg == 75.0

    def test_iloc_setter_preserves_type(self, repeatedjumps):
        """Test iloc[] setter works without breaking type."""
        repeatedjumps.iloc[50, 0] = 88.0
        assert isinstance(repeatedjumps, type(repeatedjumps))
        assert repeatedjumps.bodymass_kg == 75.0
