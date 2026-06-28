"""Tests for cardio.bike module."""

import pytest
from labanalysis.equations.cardio.bike import Bike


@pytest.mark.unit
class TestBike:
    """Test Bike equations."""
    
    def test_bike_import(self):
        """Test Bike class can be instantiated."""
        bike = Bike()
        assert bike is not None
