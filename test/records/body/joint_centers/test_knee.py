"""Tests for joint_centers module."""

import pytest


@pytest.mark.unit
class TestKneeJointsMixin:
    """Test KneeJointsMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.joint_centers.knee import KneeJointsMixin
        assert KneeJointsMixin is not None

    def test_knee_joints_mixin_has_knee_properties(self):
        """Test KneeJointsMixin has knee joint properties."""
        from labanalysis.records.body.joint_centers.knee import KneeJointsMixin

        assert hasattr(KneeJointsMixin, 'left_knee')
        assert isinstance(getattr(KneeJointsMixin, 'left_knee'), property)
        assert hasattr(KneeJointsMixin, 'right_knee')
        assert isinstance(getattr(KneeJointsMixin, 'right_knee'), property)

    def test_knee_joints_mixin_has_referenceframe_properties(self):
        """Test KneeJointsMixin has reference frame properties."""
        from labanalysis.records.body.joint_centers.knee import KneeJointsMixin

        assert hasattr(KneeJointsMixin, 'left_knee_referenceframe')
        assert isinstance(getattr(KneeJointsMixin, 'left_knee_referenceframe'), property)

    def test_knee_joints_mixin_docstring_exists(self):
        """Test KneeJointsMixin has docstring."""
        from labanalysis.records.body.joint_centers.knee import KneeJointsMixin

        assert KneeJointsMixin.__doc__ is not None
        assert 'knee' in KneeJointsMixin.__doc__.lower()

