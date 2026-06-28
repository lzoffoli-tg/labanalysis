"""
Test suite for labanalysis.equations.strength module.

Tests Brzycki equation for 1RM prediction.
"""

import pytest

from labanalysis.equations.strength import Brzycki1RM


class TestBrzycki1RM:
    """Tests for Brzycki1RM class."""

    def test_predict_1rm_single_rep(self):
        """Test 1RM prediction with 1 rep (should equal load)."""
        brzycki = Brzycki1RM()

        # 1 rep at 100 kg should give 1RM = 100 kg
        rm1 = brzycki.predict_1rm(reps=1, load=100.0)

        assert rm1 == pytest.approx(100.0, rel=0.01)

    def test_predict_1rm_multiple_reps(self):
        """Test 1RM prediction with multiple reps."""
        brzycki = Brzycki1RM()

        # 10 reps at 75 kg
        # Formula: 1RM = 75 × 36 / (37 - 10) = 75 × 36 / 27 = 100 kg
        rm1 = brzycki.predict_1rm(reps=10, load=75.0)

        expected = 75.0 * 36 / (37 - 10)
        assert rm1 == pytest.approx(expected, rel=0.01)

    def test_predict_load_from_1rm(self):
        """Test load prediction from 1RM and reps."""
        brzycki = Brzycki1RM()

        # For 1RM=100 kg, 10 reps
        # Load = 100 × (37 - 10) / 36 = 75 kg
        load = brzycki.predict_load(reps=10, rm1=100.0)

        expected = 100.0 * (37 - 10) / 36
        assert load == pytest.approx(expected, rel=0.01)

    def test_predict_reps_from_load_and_1rm(self):
        """Test reps prediction from load and 1RM."""
        brzycki = Brzycki1RM()

        # For load=75 kg, 1RM=100 kg
        # Reps = 37 - 36 × 75 / 100 = 10
        reps = brzycki.predict_reps(load=75.0, rm1=100.0)

        expected = 37 - 36 * 75.0 / 100.0
        assert reps == pytest.approx(expected, rel=0.01)

    def test_roundtrip_consistency(self):
        """Test that predict functions are consistent (roundtrip)."""
        brzycki = Brzycki1RM()

        # Start with known values
        original_load = 80.0
        original_reps = 8

        # Predict 1RM
        rm1 = brzycki.predict_1rm(reps=original_reps, load=original_load)

        # Predict load back from 1RM
        load_back = brzycki.predict_load(reps=original_reps, rm1=rm1)

        assert load_back == pytest.approx(original_load, rel=0.01)

    def test_invalid_reps_type(self):
        """Test that non-integer reps raises ValueError."""
        brzycki = Brzycki1RM()

        with pytest.raises(ValueError, match="reps must be an int"):
            brzycki.predict_1rm(reps=10.5, load=75.0)  # type: ignore

    def test_invalid_reps_range_negative(self):
        """Test that negative reps raises ValueError."""
        brzycki = Brzycki1RM()

        with pytest.raises(ValueError, match="reps must be an int within the"):
            brzycki.predict_1rm(reps=-1, load=75.0)

    def test_invalid_reps_range_too_high(self):
        """Test that reps > 36 raises ValueError."""
        brzycki = Brzycki1RM()

        with pytest.raises(ValueError, match="reps must be an int within the"):
            brzycki.predict_1rm(reps=37, load=75.0)

    def test_invalid_load_type(self):
        """Test that non-numeric load raises ValueError."""
        brzycki = Brzycki1RM()

        with pytest.raises(ValueError, match="load must be a float or int"):
            brzycki.predict_1rm(reps=10, load="75")  # type: ignore

    def test_invalid_load_negative(self):
        """Test that negative load raises ValueError."""
        brzycki = Brzycki1RM()

        with pytest.raises(ValueError, match="load must be >= 0"):
            brzycki.predict_1rm(reps=10, load=-75.0)

    def test_zero_load(self):
        """Test that zero load gives zero 1RM."""
        brzycki = Brzycki1RM()

        rm1 = brzycki.predict_1rm(reps=10, load=0.0)

        assert rm1 == 0.0

    def test_edge_case_max_reps(self):
        """Test at maximum valid reps (36)."""
        brzycki = Brzycki1RM()

        # 36 reps at 50 kg
        rm1 = brzycki.predict_1rm(reps=36, load=50.0)

        # Formula: 50 × 36 / (37 - 36) = 50 × 36 = 1800 kg
        expected = 50.0 * 36 / (37 - 36)
        assert rm1 == pytest.approx(expected, rel=0.01)

    def test_copy_creates_new_instance(self):
        """copy() creates a new Brzycki1RM instance."""
        brzycki = Brzycki1RM()
        brzycki_copy = brzycki.copy()

        assert isinstance(brzycki_copy, Brzycki1RM)
        assert brzycki_copy is not brzycki

    def test_copy_preserves_functionality(self):
        """copy() preserves functionality (stateless class)."""
        brzycki = Brzycki1RM()
        brzycki_copy = brzycki.copy()

        # Both instances should produce same results
        rm1_original = brzycki.predict_1rm(reps=10, load=75.0)
        rm1_copy = brzycki_copy.predict_1rm(reps=10, load=75.0)

        assert rm1_original == rm1_copy
