"""Tests for protocols.test_results module."""

import pandas as pd
import plotly.graph_objects as go
import pytest


@pytest.mark.unit
class TestTestResults:
    """Test TestResults protocol/class."""

    def test_module_imports(self):
        """Test TestResults can be imported."""
        from labanalysis.protocols.test_results import TestResults
        assert TestResults is not None

    def test_copy_creates_new_instance(self):
        """copy() creates a new TestResults instance with independent data."""
        from labanalysis.protocols.test_results import TestResults

        # Create a mock TestResults instance
        class MockTestResults:
            def __init__(self):
                self._summary = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
                self._analytics = pd.DataFrame({'metric': [10, 20]})
                self._figures = {'fig1': go.Figure()}
                self._include_emg = False

            # Borrow copy() from TestResults
            copy = TestResults.copy

        results = MockTestResults()
        results_copy = results.copy()

        # Verify new instance with independent data
        assert results_copy is not results
        assert results_copy._summary is not results._summary
        assert results_copy._analytics is not results._analytics

    def test_copy_deep_copies_dataframes(self):
        """copy() deep copies summary and analytics DataFrames."""
        from labanalysis.protocols.test_results import TestResults

        class MockTestResults:
            def __init__(self):
                self._summary = pd.DataFrame({'A': [1, 2, 3]})
                self._analytics = pd.DataFrame({'B': [4, 5, 6]})
                self._figures = {}
                self._include_emg = False

            copy = TestResults.copy

        results = MockTestResults()
        results_copy = results.copy()

        # Verify DataFrames are independent
        assert results_copy._summary is not results._summary
        assert results_copy._analytics is not results._analytics

        # Verify content is equal
        pd.testing.assert_frame_equal(results_copy._summary, results._summary)
        pd.testing.assert_frame_equal(results_copy._analytics, results._analytics)

        # Modify copy and verify original unchanged
        results_copy._summary.loc[0, 'A'] = 999
        assert results._summary.loc[0, 'A'] == 1

    def test_copy_deep_copies_figures(self):
        """copy() deep copies figures dictionary."""
        from labanalysis.protocols.test_results import TestResults

        class MockTestResults:
            def __init__(self):
                self._summary = pd.DataFrame()
                self._analytics = pd.DataFrame()
                self._figures = {'plot1': go.Figure(), 'plot2': go.Figure()}
                self._include_emg = False

            copy = TestResults.copy

        results = MockTestResults()
        results_copy = results.copy()

        # Verify figures dict is independent
        assert results_copy._figures is not results._figures

    def test_copy_preserves_include_emg(self):
        """copy() preserves include_emg flag."""
        from labanalysis.protocols.test_results import TestResults

        class MockTestResults:
            def __init__(self, include_emg):
                self._summary = pd.DataFrame()
                self._analytics = pd.DataFrame()
                self._figures = {}
                self._include_emg = include_emg

            copy = TestResults.copy

        # Test with True
        results_true = MockTestResults(include_emg=True)
        results_true_copy = results_true.copy()
        assert results_true_copy._include_emg is True

        # Test with False
        results_false = MockTestResults(include_emg=False)
        results_false_copy = results_false.copy()
        assert results_false_copy._include_emg is False
