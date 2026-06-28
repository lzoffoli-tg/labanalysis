"""Tests for cardio.run module."""

import pytest
from labanalysis.equations.cardio.run import Run


@pytest.mark.unit
class TestRun:
    """Test Run equations."""
    
    def test_run_import(self):
        """Test Run class can be instantiated."""
        run = Run()
        assert run is not None
