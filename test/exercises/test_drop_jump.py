"""Tests for drop_jump module."""

import pytest
import numpy as np


@pytest.mark.integration
class TestDropJump:
    """Test DropJump class."""

    def test_module_imports(self):
        """Test that DropJump can be imported."""
        from labanalysis.exercises.drop_jump import DropJump
        assert DropJump is not None

    def test_drop_jump_inheritance(self):
        """Test DropJump inherits from SingleJump."""
        from labanalysis.exercises.drop_jump import DropJump
        from labanalysis.exercises.single_jump import SingleJump

        assert issubclass(DropJump, SingleJump)

    def test_drop_jump_has_landing_phase_property(self):
        """Test DropJump has landing_phase property."""
        from labanalysis.exercises.drop_jump import DropJump

        assert hasattr(DropJump, 'landing_phase')
        assert isinstance(getattr(DropJump, 'landing_phase'), property)

    def test_drop_jump_init_signature(self):
        """Test DropJump __init__ accepts expected parameters."""
        from labanalysis.exercises.drop_jump import DropJump
        import inspect

        sig = inspect.signature(DropJump.__init__)
        params = list(sig.parameters.keys())

        # Should have bodymass_kg parameter (inherited from SingleJump)
        assert 'bodymass_kg' in params or 'kwargs' in params

    def test_drop_jump_docstring_exists(self):
        """Test DropJump has comprehensive docstring."""
        from labanalysis.exercises.drop_jump import DropJump

        assert DropJump.__doc__ is not None
        assert len(DropJump.__doc__) > 100
        assert 'plyometric' in DropJump.__doc__.lower() or 'reactive' in DropJump.__doc__.lower()
