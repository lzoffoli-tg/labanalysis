"""Tests for strengthtests._plotting module."""

import pytest


@pytest.mark.unit
class TestStrengthPlotting:
    """Test strength plotting utilities."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.protocols.strengthtests import _plotting
        assert _plotting is not None

    def test_has_plotting_functions(self):
        """Test module has plotting functions."""
        from labanalysis.protocols.strengthtests import _plotting

        # Check that module has some callable attributes
        callables = [name for name in dir(_plotting) if callable(getattr(_plotting, name)) and not name.startswith('_')]
        assert len(callables) > 0
