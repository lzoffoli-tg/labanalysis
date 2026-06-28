"""Tests for joint_centers module."""

import pytest


@pytest.mark.unit
class TestElbowJointsMixin:
    """Test ElbowJointsMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.joint_centers.elbow import ElbowJointsMixin
        assert ElbowJointsMixin is not None

    def test_elbow_joints_mixin_has_elbow_properties(self):
        """Test ElbowJointsMixin has elbow joint properties."""
        from labanalysis.records.body.joint_centers.elbow import ElbowJointsMixin

        assert hasattr(ElbowJointsMixin, 'left_elbow')
        assert isinstance(getattr(ElbowJointsMixin, 'left_elbow'), property)
        assert hasattr(ElbowJointsMixin, 'right_elbow')
        assert isinstance(getattr(ElbowJointsMixin, 'right_elbow'), property)

    def test_elbow_joints_mixin_has_referenceframe_properties(self):
        """Test ElbowJointsMixin has reference frame properties."""
        from labanalysis.records.body.joint_centers.elbow import ElbowJointsMixin

        assert hasattr(ElbowJointsMixin, 'left_elbow_referenceframe')
        assert isinstance(getattr(ElbowJointsMixin, 'left_elbow_referenceframe'), property)

    def test_elbow_joints_mixin_docstring_exists(self):
        """Test ElbowJointsMixin has docstring."""
        from labanalysis.records.body.joint_centers.elbow import ElbowJointsMixin

        assert ElbowJointsMixin.__doc__ is not None
        assert 'elbow' in ElbowJointsMixin.__doc__.lower()

