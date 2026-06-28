"""
Test suite for labanalysis.equations.cardio module.

Tests ACSM metabolic equations for VO2 prediction during walking and running.
"""

import numpy as np
import pytest

from labanalysis.equations.cardio import Run, Bike


class TestRunClass:
    """Tests for Run ACSM equations."""

    def test_predict_vo2_walking(self):
        """Test VO2 prediction at walking speed (≤5 km/h)."""
        run = Run()

        # Test at 4 km/h, 0% grade
        vo2 = run.predict_vo2(speed=4.0, grade=0.0)
        # Expected: 3.5 + 5/3*4 + 3/10*4*0 = 3.5 + 6.67 = 10.17
        expected = 3.5 + 5/3*4.0
        np.testing.assert_almost_equal(vo2[0], expected, decimal=2)

    def test_predict_vo2_running(self):
        """Test VO2 prediction at running speed (≥7 km/h)."""
        run = Run()

        # Test at 10 km/h, 0% grade
        vo2 = run.predict_vo2(speed=10.0, grade=0.0)
        # Expected: 3.5 + 10/3*10 + 3/20*10*0 = 3.5 + 33.33 = 36.83
        expected = 3.5 + 10/3*10.0
        np.testing.assert_almost_equal(vo2[0], expected, decimal=2)

    def test_predict_vo2_with_grade(self):
        """Test VO2 prediction with positive grade."""
        run = Run()

        # Test at 10 km/h, 5% grade
        vo2 = run.predict_vo2(speed=10.0, grade=5.0)
        # Expected: 3.5 + 10/3*10 + 3/20*10*5 = 3.5 + 33.33 + 7.5 = 44.33
        expected = 3.5 + 10/3*10.0 + 3/20*10.0*5.0
        np.testing.assert_almost_equal(vo2[0], expected, decimal=2)

    def test_predict_vo2_transition_zone(self):
        """Test VO2 prediction in transition zone (5-7 km/h)."""
        run = Run()

        # Test at 6 km/h (midpoint of transition)
        vo2 = run.predict_vo2(speed=6.0, grade=0.0)

        # Should be between walking and running equations
        vo2_walk = run.predict_vo2(speed=5.0, grade=0.0)[0]
        vo2_run = run.predict_vo2(speed=7.0, grade=0.0)[0]

        assert vo2_walk < vo2[0] < vo2_run, \
            f"Transition VO2 {vo2[0]} not between walking {vo2_walk} and running {vo2_run}"

    def test_predict_vo2_array_input(self):
        """Test VO2 prediction with array inputs."""
        run = Run()

        speeds = np.array([4.0, 6.0, 10.0])
        grades = np.array([0.0, 0.0, 0.0])

        vo2 = run.predict_vo2(speed=speeds, grade=grades)

        assert len(vo2) == 3
        assert all(isinstance(v, (float, np.floating)) for v in vo2)

    def test_predict_speed_from_vo2(self):
        """Test speed prediction from VO2."""
        run = Run()

        # Test at VO2=35 ml/kg/min, 0% grade
        speed = run.predict_speed(vo2=35.0, grade=0.0)

        # Verify by predicting VO2 back from speed
        vo2_check = run.predict_vo2(speed=speed[0], grade=0.0)
        np.testing.assert_almost_equal(vo2_check[0], 35.0, decimal=1)

    def test_predict_grade_from_vo2_speed(self):
        """Test grade prediction from VO2 and speed."""
        run = Run()

        # Test at VO2=40 ml/kg/min, speed=10 km/h
        grade = run.predict_grade(vo2=40.0, speed=10.0)

        # Verify by predicting VO2 back
        vo2_check = run.predict_vo2(speed=10.0, grade=grade[0])
        np.testing.assert_almost_equal(vo2_check[0], 40.0, decimal=1)


class TestBikeClass:
    """Tests for Bike ACSM equations."""

    def test_predict_vo2_zero_power_male(self):
        """Test VO2 prediction with zero power for males."""
        bike = Bike()

        # Test at 0 W, 75 kg male
        vo2 = bike.predict_vo2(power=0, weight=75, gender="Male")

        # Should be 3.5 ml/kg/min (resting)
        assert vo2[0] == pytest.approx(3.5, rel=0.01)

    def test_predict_vo2_with_power_male(self):
        """Test VO2 prediction with power output for males."""
        bike = Bike()

        # Test at 100 W, 75 kg male
        vo2 = bike.predict_vo2(power=100, weight=75, gender="Male")

        # Expected: 10.7712 * 100 / 75 + 3.5 = 14.36 + 3.5 = 17.86
        expected = 10.7712 * 100 / 75 + 3.5
        assert vo2[0] == pytest.approx(expected, rel=0.01)

    def test_predict_vo2_with_power_female(self):
        """Test VO2 prediction with power output for females."""
        bike = Bike()

        # Test at 100 W, 60 kg female
        vo2 = bike.predict_vo2(power=100, weight=60, gender="Female")

        # Expected: 10.098 * 100 / 60 + 3.5 = 16.83 + 3.5 = 20.33
        expected = 10.098 * 100 / 60 + 3.5
        assert vo2[0] == pytest.approx(expected, rel=0.01)

    def test_predict_power_from_vo2_male(self):
        """Test power prediction from VO2 for males."""
        bike = Bike()

        # Test at VO2=20 ml/kg/min, 75 kg male
        power = bike.predict_power(vo2=20, weight=75, gender="Male")

        # Verify by predicting VO2 back
        vo2_check = bike.predict_vo2(power=power[0], weight=75, gender="Male")
        np.testing.assert_almost_equal(vo2_check[0], 20.0, decimal=1)
