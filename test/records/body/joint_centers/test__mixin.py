"""Tests for joint_centers module."""

import pytest


@pytest.mark.unit
class TestJointCentersMixin:
    """Test JointCentersMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.joint_centers._mixin import JointCentersMixin
        assert JointCentersMixin is not None

    def test_joint_centers_mixin_inheritance(self):
        """Test JointCentersMixin inherits from all joint center mixins."""
        from labanalysis.records.body.joint_centers._mixin import JointCentersMixin
        from labanalysis.records.body.joint_centers.ankle import AnkleJointsMixin
        from labanalysis.records.body.joint_centers.knee import KneeJointsMixin
        from labanalysis.records.body.joint_centers.hip import HipJointsMixin

        assert issubclass(JointCentersMixin, AnkleJointsMixin)
        assert issubclass(JointCentersMixin, KneeJointsMixin)
        assert issubclass(JointCentersMixin, HipJointsMixin)

    def test_joint_centers_mixin_docstring_exists(self):
        """Test JointCentersMixin has comprehensive docstring."""
        from labanalysis.records.body.joint_centers._mixin import JointCentersMixin

        assert JointCentersMixin.__doc__ is not None
        assert len(JointCentersMixin.__doc__) > 100
        assert 'joint' in JointCentersMixin.__doc__.lower()

