"""Tests for upright_balance_test_results module."""

import pytest


@pytest.mark.unit
class TestUprightBalanceTestResults:
    """Test UprightBalanceTestResults class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.protocols.balancetests.upright_balance_test_results import UprightBalanceTestResults
        assert UprightBalanceTestResults is not None

    def test_upright_balance_test_results_init_signature(self):
        """Test UprightBalanceTestResults __init__ accepts expected parameters."""
        from labanalysis.protocols.balancetests.upright_balance_test_results import UprightBalanceTestResults
        import inspect

        sig = inspect.signature(UprightBalanceTestResults.__init__)
        params = list(sig.parameters.keys())

        # Should have test and include_emg parameters
        assert 'test' in params
        assert 'include_emg' in params
