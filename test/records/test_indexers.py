"""Test suite for labanalysis indexers."""

import pytest
from labanalysis.records import RecordLocIndexer, RecordILocIndexer
from labanalysis.timeseries import TimeseriesLocIndexer, TimeseriesILocIndexer


def test_record_indexers_importable():
    """Test that record indexers import successfully."""
    assert RecordLocIndexer is not None
    assert RecordILocIndexer is not None


def test_timeseries_indexers_importable():
    """Test that timeseries indexers import successfully."""
    assert TimeseriesLocIndexer is not None
    assert TimeseriesILocIndexer is not None
