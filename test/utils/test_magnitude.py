"""
Test suite for magnitude function.

Tests verify magnitude calculation with various bases and values.
"""

import pytest

from labanalysis.utils import magnitude


def test_magnitude_positive_values():
    """
    Test magnitude function with positive values.

    Expected:
        - magnitude(100, 10) should return 2 (ceil(log10(100)) = 2)
        - magnitude(1000, 10) should return 3 (ceil(log10(1000)) = 3)
        - magnitude(0.1, 10) should return -1
        - magnitude(0.01, 10) should return -2
    """
    assert magnitude(100, 10) == 2
    assert magnitude(1000, 10) == 3
    assert magnitude(0.1, 10) == -1
    assert magnitude(0.01, 10) == -2


def test_magnitude_negative_values():
    """
    Test magnitude function with negative values.

    Expected:
        Magnitude should work with absolute value of input
    """
    assert magnitude(-100, 10) == 2
    assert magnitude(-1000, 10) == 3


def test_magnitude_zero():
    """
    Test magnitude function with zero value.

    Expected:
        magnitude(0, base) should return 0
    """
    assert magnitude(0, 10) == 0
    assert magnitude(0, 2) == 0


def test_magnitude_base_2():
    """
    Test magnitude function with base 2 (binary).

    Expected:
        - magnitude(8, 2) should return 3 (ceil(log2(8)) = 3)
        - magnitude(16, 2) should return 4 (ceil(log2(16)) = 4)
    """
    assert magnitude(8, 2) == 3
    assert magnitude(16, 2) == 4


def test_magnitude_base_zero():
    """
    Test magnitude function with base zero.

    Expected:
        magnitude(value, 0) should return 0
    """
    assert magnitude(100, 0) == 0


def test_magnitude_fractional_values():
    """
    Test magnitude with fractional values between 0 and 1.

    Expected:
        Should return negative magnitudes for fractions
    """
    assert magnitude(0.5, 10) == -1
    assert magnitude(0.001, 10) == -3


def test_magnitude_exact_powers():
    """
    Test magnitude with exact powers of base.

    Expected:
        magnitude(base^n, base) should return n
    """
    assert magnitude(1, 10) == 0
    assert magnitude(10, 10) == 1
    assert magnitude(100, 10) == 2
    assert magnitude(1000, 10) == 3


def test_magnitude_base_e():
    """
    Test magnitude with base e (Euler's number).

    Expected:
        Should work with non-integer base
    """
    import math
    e = math.e
    result = magnitude(e**3, e)
    assert result == 3
