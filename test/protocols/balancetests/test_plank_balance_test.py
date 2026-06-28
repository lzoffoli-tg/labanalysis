"""Tests for plank_balance_test module."""

import pytest


@pytest.mark.unit
class TestPlankBalanceTest:
    """Test PlankBalanceTest class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.protocols.balancetests.plank_balance_test import PlankBalanceTest
        assert PlankBalanceTest is not None

    def test_plank_balance_test_has_eyes_property(self):
        """Test PlankBalanceTest has eyes property."""
        from labanalysis.protocols.balancetests.plank_balance_test import PlankBalanceTest

        assert hasattr(PlankBalanceTest, 'eyes')
        assert isinstance(getattr(PlankBalanceTest, 'eyes'), property)

    def test_plank_balance_test_has_exercise_property(self):
        """Test PlankBalanceTest has exercise property."""
        from labanalysis.protocols.balancetests.plank_balance_test import PlankBalanceTest

        assert hasattr(PlankBalanceTest, 'exercise')
        assert isinstance(getattr(PlankBalanceTest, 'exercise'), property)

    def test_plank_balance_test_docstring_exists(self):
        """Test PlankBalanceTest has comprehensive docstring."""
        from labanalysis.protocols.balancetests.plank_balance_test import PlankBalanceTest

        assert PlankBalanceTest.__doc__ is not None
        assert len(PlankBalanceTest.__doc__) > 100
        assert 'plank' in PlankBalanceTest.__doc__.lower() or 'prone' in PlankBalanceTest.__doc__.lower()
