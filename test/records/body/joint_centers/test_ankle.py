"""Tests for joint_centers module."""

import pytest


@pytest.mark.unit
class TestAnkleJointsMixin:
    """Test AnkleJointsMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.joint_centers.ankle import AnkleJointsMixin
        assert AnkleJointsMixin is not None

    def test_ankle_joints_mixin_has_ankle_properties(self):
        """Test AnkleJointsMixin has ankle joint properties."""
        from labanalysis.records.body.joint_centers.ankle import AnkleJointsMixin

        assert hasattr(AnkleJointsMixin, 'left_ankle')
        assert isinstance(getattr(AnkleJointsMixin, 'left_ankle'), property)
        assert hasattr(AnkleJointsMixin, 'right_ankle')
        assert isinstance(getattr(AnkleJointsMixin, 'right_ankle'), property)

    def test_ankle_joints_mixin_has_referenceframe_properties(self):
        """Test AnkleJointsMixin has reference frame properties."""
        from labanalysis.records.body.joint_centers.ankle import AnkleJointsMixin

        assert hasattr(AnkleJointsMixin, 'left_ankle_referenceframe')
        assert isinstance(getattr(AnkleJointsMixin, 'left_ankle_referenceframe'), property)

    def test_ankle_joints_mixin_docstring_exists(self):
        """Test AnkleJointsMixin has docstring."""
        from labanalysis.records.body.joint_centers.ankle import AnkleJointsMixin

        assert AnkleJointsMixin.__doc__ is not None
        assert 'ankle' in AnkleJointsMixin.__doc__.lower()

