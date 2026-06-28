"""Tests for joint_centers module."""

import pytest


@pytest.mark.unit
class TestWristJointsMixin:
    """Test WristJointsMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.joint_centers.wrist import WristJointsMixin
        assert WristJointsMixin is not None

    def test_wrist_joints_mixin_has_wrist_properties(self):
        """Test WristJointsMixin has wrist joint properties."""
        from labanalysis.records.body.joint_centers.wrist import WristJointsMixin

        assert hasattr(WristJointsMixin, 'left_wrist')
        assert isinstance(getattr(WristJointsMixin, 'left_wrist'), property)
        assert hasattr(WristJointsMixin, 'right_wrist')
        assert isinstance(getattr(WristJointsMixin, 'right_wrist'), property)

    def test_wrist_joints_mixin_has_referenceframe_properties(self):
        """Test WristJointsMixin has reference frame properties."""
        from labanalysis.records.body.joint_centers.wrist import WristJointsMixin

        assert hasattr(WristJointsMixin, 'left_wrist_referenceframe')
        assert isinstance(getattr(WristJointsMixin, 'left_wrist_referenceframe'), property)

    def test_wrist_joints_mixin_docstring_exists(self):
        """Test WristJointsMixin has docstring."""
        from labanalysis.records.body.joint_centers.wrist import WristJointsMixin

        assert WristJointsMixin.__doc__ is not None
        assert 'wrist' in WristJointsMixin.__doc__.lower()

