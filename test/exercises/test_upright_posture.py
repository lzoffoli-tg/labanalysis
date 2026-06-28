"""Tests for upright_posture module."""

import pytest


@pytest.mark.integration
class TestUprightPosture:
    """Test UprightPosture class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.exercises.upright_posture import UprightPosture
        assert UprightPosture is not None

    def test_upright_posture_inheritance(self):
        """Test UprightPosture inherits from WholeBody."""
        from labanalysis.exercises.upright_posture import UprightPosture
        from labanalysis.records.body import WholeBody

        assert issubclass(UprightPosture, WholeBody)

    def test_upright_posture_has_side_property(self):
        """Test UprightPosture has side property."""
        from labanalysis.exercises.upright_posture import UprightPosture

        assert hasattr(UprightPosture, 'side')
        assert isinstance(getattr(UprightPosture, 'side'), property)

    def test_upright_posture_init_signature(self):
        """Test UprightPosture __init__ accepts expected parameters."""
        from labanalysis.exercises.upright_posture import UprightPosture
        import inspect

        sig = inspect.signature(UprightPosture.__init__)
        params = list(sig.parameters.keys())

        # Should have foot force platform parameters
        assert 'left_foot_ground_reaction_force' in params
        assert 'right_foot_ground_reaction_force' in params

    def test_upright_posture_docstring_exists(self):
        """Test UprightPosture has comprehensive docstring."""
        from labanalysis.exercises.upright_posture import UprightPosture

        assert UprightPosture.__doc__ is not None
        assert len(UprightPosture.__doc__) > 100
        assert 'balance' in UprightPosture.__doc__.lower() or 'posture' in UprightPosture.__doc__.lower()
