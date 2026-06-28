"""Tests for joint_centers module."""

import pytest


@pytest.mark.unit
class TestShoulderJointsMixin:
    """Test ShoulderJointsMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.joint_centers.shoulder import ShoulderJointsMixin
        assert ShoulderJointsMixin is not None

    def test_shoulder_joints_mixin_has_shoulder_properties(self):
        """Test ShoulderJointsMixin has shoulder joint properties."""
        from labanalysis.records.body.joint_centers.shoulder import ShoulderJointsMixin

        assert hasattr(ShoulderJointsMixin, 'left_shoulder')
        assert isinstance(getattr(ShoulderJointsMixin, 'left_shoulder'), property)
        assert hasattr(ShoulderJointsMixin, 'right_shoulder')
        assert isinstance(getattr(ShoulderJointsMixin, 'right_shoulder'), property)

    def test_shoulder_joints_mixin_has_referenceframe_properties(self):
        """Test ShoulderJointsMixin has reference frame properties."""
        from labanalysis.records.body.joint_centers.shoulder import ShoulderJointsMixin

        assert hasattr(ShoulderJointsMixin, 'left_shoulder_referenceframe')
        assert isinstance(getattr(ShoulderJointsMixin, 'left_shoulder_referenceframe'), property)

    def test_shoulder_joints_mixin_docstring_exists(self):
        """Test ShoulderJointsMixin has docstring."""
        from labanalysis.records.body.joint_centers.shoulder import ShoulderJointsMixin

        assert ShoulderJointsMixin.__doc__ is not None
        assert 'shoulder' in ShoulderJointsMixin.__doc__.lower()

