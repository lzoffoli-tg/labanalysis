"""Tests for protocols._plotting_utils module."""

import pytest


@pytest.mark.unit
class TestPlottingUtils:
    """Test plotting utilities."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.protocols import _plotting_utils
        assert _plotting_utils is not None

    def test_has_utility_functions(self):
        """Test module has utility functions."""
        from labanalysis.protocols import _plotting_utils

        # Check that module has some callable attributes
        callables = [name for name in dir(_plotting_utils) if callable(getattr(_plotting_utils, name)) and not name.startswith('_')]
        assert len(callables) > 0
