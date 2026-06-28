"""Tests for plank_balance_test_results module."""

import pytest


@pytest.mark.unit
class TestPlankBalanceTestResults:
    """Test PlankBalanceTestResults class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.protocols.balancetests.plank_balance_test_results import PlankBalanceTestResults
        assert PlankBalanceTestResults is not None

    def test_plank_balance_test_results_init_signature(self):
        """Test PlankBalanceTestResults __init__ accepts expected parameters."""
        from labanalysis.protocols.balancetests.plank_balance_test_results import PlankBalanceTestResults
        import inspect

        sig = inspect.signature(PlankBalanceTestResults.__init__)
        params = list(sig.parameters.keys())

        # Should have test and include_emg parameters
        assert 'test' in params
        assert 'include_emg' in params
