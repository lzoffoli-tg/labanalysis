"""Tests for prone_posture module."""

import pytest


@pytest.mark.integration
class TestPronePosture:
    """Test PronePosture class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.exercises.prone_posture import PronePosture
        assert PronePosture is not None

    def test_prone_posture_inheritance(self):
        """Test PronePosture inherits from WholeBody."""
        from labanalysis.exercises.prone_posture import PronePosture
        from labanalysis.records.body import WholeBody

        assert issubclass(PronePosture, WholeBody)

    def test_prone_posture_init_signature(self):
        """Test PronePosture __init__ accepts expected parameters."""
        from labanalysis.exercises.prone_posture import PronePosture
        import inspect

        sig = inspect.signature(PronePosture.__init__)
        params = list(sig.parameters.keys())

        # Should have all four force platform parameters
        assert 'left_foot_ground_reaction_force' in params
        assert 'right_foot_ground_reaction_force' in params
        assert 'left_hand_ground_reaction_force' in params
        assert 'right_hand_ground_reaction_force' in params

    def test_prone_posture_docstring_exists(self):
        """Test PronePosture has comprehensive docstring."""
        from labanalysis.exercises.prone_posture import PronePosture

        assert PronePosture.__doc__ is not None
        assert len(PronePosture.__doc__) > 100
        assert 'prone' in PronePosture.__doc__.lower() or 'plank' in PronePosture.__doc__.lower()
