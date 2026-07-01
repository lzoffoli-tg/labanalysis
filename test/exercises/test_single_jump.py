"""Tests for single_jump module."""

import numpy as np
import pytest


@pytest.mark.integration
class TestSingleJump:
    """Test SingleJump class - requires real data for proper testing."""

    def test_module_imports(self):
        """Test that SingleJump can be imported."""
        from labanalysis.exercises.jumps.single_jump import SingleJump

        assert SingleJump is not None

    def test_single_jump_inheritance(self):
        """Test SingleJump inherits from WholeBody."""
        from labanalysis.exercises.jumps.single_jump import SingleJump
        from labanalysis.records.body import WholeBody

        assert issubclass(SingleJump, WholeBody)

    def test_single_jump_has_flight_time_property(self):
        """Test SingleJump has flight_time property."""
        from labanalysis.exercises.jumps.single_jump import SingleJump

        assert hasattr(SingleJump, "flight_time")
        assert isinstance(getattr(SingleJump, "flight_time"), property)

    def test_single_jump_has_contact_time_property(self):
        """Test SingleJump has contact_time property."""
        from labanalysis.exercises.jumps.single_jump import SingleJump

        assert hasattr(SingleJump, "contact_time")
        assert isinstance(getattr(SingleJump, "contact_time"), property)

    def test_single_jump_docstring_exists(self):
        """Test SingleJump has comprehensive docstring."""
        from labanalysis.exercises.jumps.single_jump import SingleJump

        assert SingleJump.__doc__ is not None
        assert len(SingleJump.__doc__) > 100
        assert "jump" in SingleJump.__doc__.lower()


@pytest.mark.integration
class TestSingleJumpLocIloc:
    """Test loc/iloc indexing for SingleJump preserves custom attributes."""

    @pytest.fixture
    def singlejump(self):
        """Create a minimal SingleJump for testing."""
        from labanalysis.exercises.jumps.single_jump import SingleJump
        from test.protocols.jumptests.synthetic_jump_generators import (
            generate_squat_jump_force_platform,
        )

        bodymass = 75.0
        fp_left = generate_squat_jump_force_platform(
            bodymass_kg=bodymass / 2, jump_height_m=0.30, side="left", fsamp=1000.0
        )
        fp_right = generate_squat_jump_force_platform(
            bodymass_kg=bodymass / 2, jump_height_m=0.30, side="right", fsamp=1000.0
        )

        return SingleJump(
            bodymass_kg=bodymass,
            free_hands=True,
            straight_legs=False,
            left_foot_ground_reaction_force=fp_left,
            right_foot_ground_reaction_force=fp_right,
        )

    def test_loc_preserves_bodymass_and_flags(self, singlejump):
        """Test loc[] preserves bodymass_kg, free_hands, straight_legs."""
        start_idx = singlejump.index[100]
        end_idx = singlejump.index[200]
        sliced = singlejump.loc[start_idx:end_idx, :]

        assert isinstance(sliced, type(singlejump))
        assert sliced.bodymass_kg == 75.0
        assert sliced.free_hands == True
        assert sliced.straight_legs == False

    def test_iloc_preserves_custom_attributes(self, singlejump):
        """Test iloc[] preserves all custom attributes."""
        sliced = singlejump.iloc[100:200, :]

        assert isinstance(sliced, type(singlejump))
        assert sliced.bodymass_kg == 75.0
        assert sliced.free_hands == True
        assert sliced.straight_legs == False

    def test_loc_setter_preserves_type(self, singlejump):
        """Test loc[] setter works without breaking type."""
        singlejump.loc[singlejump.index[50], "left_foot_ground_reaction_force"] = 99.0
        assert isinstance(singlejump, type(singlejump))
        assert singlejump.bodymass_kg == 75.0

    def test_iloc_setter_preserves_type(self, singlejump):
        """Test iloc[] setter works without breaking type."""
        singlejump.iloc[50, 0] = 88.0
        assert isinstance(singlejump, type(singlejump))
        assert singlejump.bodymass_kg == 75.0
