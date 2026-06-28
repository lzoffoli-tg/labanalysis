"""Tests for strength.brzycki module."""

import pytest
from labanalysis.equations.strength.brzycki import Brzycki1RM


@pytest.mark.unit
class TestBrzycki1RM:
    """Test Brzycki1RM equations."""
    
    def test_brzycki_import(self):
        """Test Brzycki1RM class can be instantiated."""
        brzycki = Brzycki1RM()
        assert brzycki is not None
