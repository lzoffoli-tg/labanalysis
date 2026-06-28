"""Tests for agilitytests.shuttle_test_results module."""

import pytest


@pytest.mark.unit
class TestShuttleTestResults:
    """Test ShuttleTestResults class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.protocols.agilitytests.shuttle_test_results import ShuttleTestResults
        assert ShuttleTestResults is not None

    def test_shuttle_test_results_init_signature(self):
        """Test ShuttleTestResults __init__ accepts expected parameters."""
        from labanalysis.protocols.agilitytests.shuttle_test_results import ShuttleTestResults
        import inspect

        sig = inspect.signature(ShuttleTestResults.__init__)
        params = list(sig.parameters.keys())

        # Should have test parameter
        assert 'test' in params
