"""
Test suite for Winter derivative functions.

Tests verify winter_derivative1 and winter_derivative2 implementations
according to Winter 2009 biomechanics reference.
"""

import numpy as np
import pytest

from labanalysis.signalprocessing import winter_derivative1, winter_derivative2


class TestWinterDerivative1:
    """Tests for first derivative using Winter's method."""

    def test_linear_signal(self):
        """
        Test first derivative of linear signal.

        Expected:
            Derivative of y = 2x + 3 should be constant 2
        """
        x = np.linspace(0, 10, 100)
        y = 2 * x + 3
        dy = winter_derivative1(y, x_signal=x)

        # All derivative values should be ~2
        assert np.allclose(dy, 2.0, atol=0.01)

    def test_quadratic_signal(self):
        """
        Test first derivative of quadratic signal.

        Expected:
            Derivative of y = x^2 should be 2x
        """
        x = np.linspace(0, 10, 100)
        y = x**2
        dy = winter_derivative1(y, x_signal=x)

        # Derivative at center points
        x_center = x[1:-1]
        expected = 2 * x_center

        assert np.allclose(dy, expected, atol=0.1)

    def test_sinusoidal_signal(self):
        """
        Test first derivative of sine wave.

        Expected:
            Derivative of sin(t) should be cos(t)
        """
        t = np.linspace(0, 2 * np.pi, 1000)
        y = np.sin(t)
        dy = winter_derivative1(y, x_signal=t)

        # Expected derivative at center points
        t_center = t[1:-1]
        expected = np.cos(t_center)

        assert np.allclose(dy, expected, atol=0.01)

    def test_with_time_diff_only(self):
        """
        Test derivative with uniform time_diff (no x_signal).

        Expected:
            Should compute derivative using time_diff as sampling interval
        """
        dt = 0.01
        t = np.arange(0, 1, dt)
        y = 3 * t**2  # dy/dt = 6t
        dy = winter_derivative1(y, time_diff=dt)

        # Expected derivative at center indices
        t_center = t[1:-1]
        expected = 6 * t_center

        assert np.allclose(dy, expected, atol=0.1)

    def test_output_length(self):
        """
        Test that output length is len(input) - 2.

        Expected:
            Winter's method loses one sample at each end
        """
        y = np.random.randn(100)
        dy = winter_derivative1(y, time_diff=1.0)

        assert len(dy) == len(y) - 2

    def test_zero_signal(self):
        """
        Test derivative of constant signal.

        Expected:
            Derivative of constant should be zero
        """
        y = np.ones(100) * 5.0
        dy = winter_derivative1(y, time_diff=1.0)

        assert np.allclose(dy, 0.0, atol=1e-10)

    def test_velocity_from_position(self):
        """
        Test computing velocity from position (biomechanics application).

        Expected:
            Velocity should match known motion profile
        """
        # Uniform acceleration: x = 0.5*a*t^2, v = a*t
        t = np.linspace(0, 2, 200)
        a = 9.81  # m/s^2
        position = 0.5 * a * t**2
        velocity = winter_derivative1(position, x_signal=t)

        # Expected velocity at center points
        t_center = t[1:-1]
        expected_velocity = a * t_center

        assert np.allclose(velocity, expected_velocity, atol=0.1)


class TestWinterDerivative2:
    """Tests for second derivative using Winter's method."""

    def test_linear_signal(self):
        """
        Test second derivative of linear signal.

        Expected:
            Second derivative of y = 2x + 3 should be zero
        """
        x = np.linspace(0, 10, 100)
        y = 2 * x + 3
        d2y = winter_derivative2(y, x_signal=x)

        assert np.allclose(d2y, 0.0, atol=0.01)

    def test_quadratic_signal(self):
        """
        Test second derivative of quadratic signal.

        Expected:
            Second derivative of y = x^2 should be constant 2
        """
        x = np.linspace(0, 10, 100)
        y = x**2
        d2y = winter_derivative2(y, x_signal=x)

        # All second derivative values should be ~2
        assert np.allclose(d2y, 2.0, atol=0.1)

    def test_sinusoidal_signal(self):
        """
        Test second derivative of sine wave.

        Expected:
            Second derivative of sin(t) should be -sin(t)
        """
        t = np.linspace(0, 2 * np.pi, 1000)
        y = np.sin(t)
        d2y = winter_derivative2(y, x_signal=t)

        # Expected second derivative at center points
        t_center = t[1:-1]
        expected = -np.sin(t_center)

        assert np.allclose(d2y, expected, atol=0.05)

    def test_cubic_signal(self):
        """
        Test second derivative of cubic signal.

        Expected:
            Second derivative of y = x^3 should be 6x
        """
        x = np.linspace(0, 5, 100)
        y = x**3
        d2y = winter_derivative2(y, x_signal=x)

        # Expected at center points
        x_center = x[1:-1]
        expected = 6 * x_center

        assert np.allclose(d2y, expected, atol=0.5)

    def test_with_time_diff_only(self):
        """
        Test second derivative with uniform time_diff.

        Expected:
            Should compute second derivative using time_diff
        """
        dt = 0.01
        t = np.arange(0, 1, dt)
        y = 0.5 * 10 * t**2  # d2y/dt2 = 10
        d2y = winter_derivative2(y, time_diff=dt)

        assert np.allclose(d2y, 10.0, atol=0.5)

    def test_output_length(self):
        """
        Test that output length is len(input) - 2.

        Expected:
            Winter's method loses one sample at each end
        """
        y = np.random.randn(100)
        d2y = winter_derivative2(y, time_diff=1.0)

        assert len(d2y) == len(y) - 2

    def test_zero_acceleration(self):
        """
        Test second derivative of linear velocity (constant acceleration = 0).

        Expected:
            Second derivative of v = constant should be zero
        """
        t = np.linspace(0, 1, 100)
        y = 5 * t  # Linear velocity
        d2y = winter_derivative2(y, x_signal=t)

        assert np.allclose(d2y, 0.0, atol=1e-5)

    def test_acceleration_from_position(self):
        """
        Test computing acceleration from position (biomechanics application).

        Expected:
            Acceleration should match known value
        """
        # Uniform acceleration: x = 0.5*a*t^2, a = constant
        t = np.linspace(0, 2, 200)
        a_true = 9.81  # m/s^2
        position = 0.5 * a_true * t**2
        acceleration = winter_derivative2(position, x_signal=t)

        assert np.allclose(acceleration, a_true, atol=0.5)

    def test_parabolic_trajectory(self):
        """
        Test second derivative on parabolic projectile motion.

        Expected:
            Vertical acceleration should be -g (gravity)
        """
        t = np.linspace(0, 1, 200)
        v0 = 10  # m/s initial velocity
        g = 9.81  # m/s^2
        y = v0 * t - 0.5 * g * t**2
        d2y = winter_derivative2(y, x_signal=t)

        # Second derivative should be -g
        assert np.allclose(d2y, -g, atol=0.5)
