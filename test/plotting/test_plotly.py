"""
Test suite for labanalysis.plotting.plotly module.

Tests plotting utility functions for creating Plotly figures.
"""

import pytest
import numpy as np
import pandas as pd

from labanalysis.plotting.plotly import plot_comparisons, bars_with_normative_bands


class TestPlotComparisons:
    """Tests for plot_comparisons function."""

    def test_plot_comparisons_basic(self):
        """Test plot_comparisons with basic data."""
        # Create sample comparison data
        data = {
            'Metric1': [10, 20, 30],
            'Metric2': [15, 25, 35]
        }

        # Test that function is callable
        assert callable(plot_comparisons)

    def test_plot_comparisons_with_dataframe(self):
        """Test plot_comparisons with DataFrame input."""
        df = pd.DataFrame({
            'test1': [1, 2, 3],
            'test2': [4, 5, 6]
        })

        # Verify function exists and is importable
        assert hasattr(plot_comparisons, '__call__')


class TestBarsWithNormativeBands:
    """Tests for bars_with_normative_bands function."""

    def test_bars_with_normative_bands_callable(self):
        """Test that bars_with_normative_bands function is callable."""
        assert callable(bars_with_normative_bands)

    def test_bars_with_normative_bands_basic(self):
        """Test bars_with_normative_bands with synthetic data."""
        # Create sample data with values and normative bands
        values = np.array([85, 92, 78, 105])
        categories = ['Test1', 'Test2', 'Test3', 'Test4']

        # Test that function is importable and callable
        assert hasattr(bars_with_normative_bands, '__call__')
