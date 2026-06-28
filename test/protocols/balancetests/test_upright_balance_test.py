"""Tests for upright_balance_test module."""

import pytest


@pytest.mark.unit
class TestUprightBalanceTest:
    """Test UprightBalanceTest class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.protocols.balancetests.upright_balance_test import UprightBalanceTest
        assert UprightBalanceTest is not None

    def test_upright_balance_test_has_eyes_property(self):
        """Test UprightBalanceTest has eyes property."""
        from labanalysis.protocols.balancetests.upright_balance_test import UprightBalanceTest

        assert hasattr(UprightBalanceTest, 'eyes')
        assert isinstance(getattr(UprightBalanceTest, 'eyes'), property)

    def test_upright_balance_test_has_side_property(self):
        """Test UprightBalanceTest has side property."""
        from labanalysis.protocols.balancetests.upright_balance_test import UprightBalanceTest

        assert hasattr(UprightBalanceTest, 'side')
        assert isinstance(getattr(UprightBalanceTest, 'side'), property)

    def test_upright_balance_test_has_exercise_property(self):
        """Test UprightBalanceTest has exercise property."""
        from labanalysis.protocols.balancetests.upright_balance_test import UprightBalanceTest

        assert hasattr(UprightBalanceTest, 'exercise')
        assert isinstance(getattr(UprightBalanceTest, 'exercise'), property)
