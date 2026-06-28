"""Tests for joint_centers module."""

import pytest


@pytest.mark.unit
class TestAxialJointsMixin:
    """Test AxialJointsMixin class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.records.body.joint_centers.axial import AxialJointsMixin
        assert AxialJointsMixin is not None

    def test_axial_joints_mixin_has_head_property(self):
        """Test AxialJointsMixin has head center property."""
        from labanalysis.records.body.joint_centers.axial import AxialJointsMixin

        assert hasattr(AxialJointsMixin, 'head_center')
        assert isinstance(getattr(AxialJointsMixin, 'head_center'), property)

    def test_axial_joints_mixin_has_neck_base_property(self):
        """Test AxialJointsMixin has neck base property."""
        from labanalysis.records.body.joint_centers.axial import AxialJointsMixin

        assert hasattr(AxialJointsMixin, 'neck_base')
        assert isinstance(getattr(AxialJointsMixin, 'neck_base'), property)

    def test_axial_joints_mixin_docstring_exists(self):
        """Test AxialJointsMixin has docstring."""
        from labanalysis.records.body.joint_centers.axial import AxialJointsMixin

        assert AxialJointsMixin.__doc__ is not None
        assert 'axial' in AxialJointsMixin.__doc__.lower() or 'head' in AxialJointsMixin.__doc__.lower()

