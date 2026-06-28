"""Tests for joint_centers module."""

import pytest


@pytest.mark.unit
class TestHipJointsMixin:
    """Test HipJointsMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.joint_centers.hip import HipJointsMixin
        assert HipJointsMixin is not None

    def test_hip_joints_mixin_has_hip_properties(self):
        """Test HipJointsMixin has hip joint properties."""
        from labanalysis.records.body.joint_centers.hip import HipJointsMixin

        assert hasattr(HipJointsMixin, 'left_hip')
        assert isinstance(getattr(HipJointsMixin, 'left_hip'), property)
        assert hasattr(HipJointsMixin, 'right_hip')
        assert isinstance(getattr(HipJointsMixin, 'right_hip'), property)

    def test_hip_joints_mixin_has_referenceframe_properties(self):
        """Test HipJointsMixin has reference frame properties."""
        from labanalysis.records.body.joint_centers.hip import HipJointsMixin

        assert hasattr(HipJointsMixin, 'left_hip_referenceframe')
        assert isinstance(getattr(HipJointsMixin, 'left_hip_referenceframe'), property)

    def test_hip_joints_mixin_docstring_exists(self):
        """Test HipJointsMixin has docstring."""
        from labanalysis.records.body.joint_centers.hip import HipJointsMixin

        assert HipJointsMixin.__doc__ is not None
        assert 'hip' in HipJointsMixin.__doc__.lower()

