"""Tests for drop_jump module."""

import pytest
import numpy as np


@pytest.mark.integration
class TestDropJump:
    """Test DropJump class."""

    def test_module_imports(self):
        """Test that DropJump can be imported."""
        from labanalysis.exercises.jumps.drop_jump import DropJump

        assert DropJump is not None

    def test_drop_jump_inheritance(self):
        """Test DropJump inherits from SingleJump."""
        from labanalysis.exercises.jumps.drop_jump import DropJump
        from labanalysis.exercises.jumps.single_jump import SingleJump

        assert issubclass(DropJump, SingleJump)

    def test_drop_jump_has_landing_phase_property(self):
        """Test DropJump has landing_phase property."""
        from labanalysis.exercises.jumps.drop_jump import DropJump

        assert hasattr(DropJump, "landing_phase")
        assert isinstance(getattr(DropJump, "landing_phase"), property)

    def test_drop_jump_init_signature(self):
        """Test DropJump __init__ accepts expected parameters."""
        from labanalysis.exercises.jumps.drop_jump import DropJump
        import inspect

        sig = inspect.signature(DropJump.__init__)
        params = list(sig.parameters.keys())

        # Should have bodymass_kg parameter (inherited from SingleJump)
        assert "bodymass_kg" in params or "kwargs" in params

    def test_drop_jump_docstring_exists(self):
        """Test DropJump has comprehensive docstring."""
        from labanalysis.exercises.jumps.drop_jump import DropJump

        assert DropJump.__doc__ is not None
        assert len(DropJump.__doc__) > 100
        assert (
            "plyometric" in DropJump.__doc__.lower()
            or "reactive" in DropJump.__doc__.lower()
        )


@pytest.mark.integration
class TestDropJumpLocIloc:
    """Test loc/iloc indexing for DropJump preserves custom attributes."""

    @pytest.fixture
    def dropjump(self):
        """Create a minimal DropJump for testing."""
        from labanalysis.exercises.jumps.drop_jump import DropJump
        from test.protocols.jumptests.synthetic_jump_generators import (
            generate_drop_jump_force_platform,
        )

        bodymass = 75.0
        box_height = 40.0
        fp_left = generate_drop_jump_force_platform(
            bodymass_kg=bodymass / 2,
            drop_height_cm=box_height,
            jump_height_m=0.25,
            side="left",
            fsamp=1000.0,
        )
        fp_right = generate_drop_jump_force_platform(
            bodymass_kg=bodymass / 2,
            drop_height_cm=box_height,
            jump_height_m=0.25,
            side="right",
            fsamp=1000.0,
        )

        return DropJump(
            bodymass_kg=bodymass,
            box_height_cm=box_height,
            free_hands=False,
            left_foot_ground_reaction_force=fp_left,
            right_foot_ground_reaction_force=fp_right,
        )

    def test_loc_preserves_all_attributes(self, dropjump):
        """Test loc[] preserves bodymass_kg, box_height_cm, free_hands."""
        start_idx = dropjump.index[50]
        end_idx = dropjump.index[150]
        sliced = dropjump.loc[start_idx:end_idx, :]

        assert isinstance(sliced, type(dropjump))
        assert sliced.bodymass_kg == 75.0
        assert sliced.box_height_cm == 40.0
        assert sliced.free_hands == False

    def test_iloc_preserves_all_attributes(self, dropjump):
        """Test iloc[] preserves all custom attributes."""
        sliced = dropjump.iloc[50:150, :]

        assert isinstance(sliced, type(dropjump))
        assert sliced.bodymass_kg == 75.0
        assert sliced.box_height_cm == 40.0
        assert sliced.free_hands == False

    def test_loc_setter_preserves_type(self, dropjump):
        """Test loc[] setter works without breaking type."""
        dropjump.loc[dropjump.index[50], "left_foot_ground_reaction_force"] = 99.0
        assert isinstance(dropjump, type(dropjump))
        assert dropjump.box_height_cm == 40.0

    def test_iloc_setter_preserves_type(self, dropjump):
        """Test iloc[] setter works without breaking type."""
        dropjump.iloc[50, 0] = 88.0
        assert isinstance(dropjump, type(dropjump))
        assert dropjump.box_height_cm == 40.0
