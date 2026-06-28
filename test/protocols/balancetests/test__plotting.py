"""Tests for balancetests._plotting module."""

import pytest


@pytest.mark.unit
class TestBalanceTestPlotting:
    """Test balance test plotting functions."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.protocols.balancetests import _plotting
        assert _plotting is not None

    def test_has_get_sway_figure_function(self):
        """Test module has _get_sway_figure function."""
        from labanalysis.protocols.balancetests._plotting import _get_sway_figure

        assert _get_sway_figure is not None
        assert callable(_get_sway_figure)
