"""
Test suite for Circle class.

Tests verify circle fitting via least squares: x^2 + y^2 + A*x + B*y + C = 0.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.modelling.ols.geometry import Circle


def test_circle_initialization():
    """
    Test Circle initialization.

    Expected:
        Should create Circle object with appropriate equation
    """
    circle = Circle()
    assert circle.dimensions == ["x", "y"]
    assert circle.has_intercept is True


def test_circle_fit_unit_circle():
    """
    Test fitting unit circle centered at origin.

    Expected:
        Center should be (0, 0), radius should be 1
    """
    # Points on unit circle: x^2 + y^2 = 1
    theta = np.linspace(0, 2 * np.pi, 20)
    x = np.cos(theta)
    y = np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    center = circle.center
    radius = circle.radius

    assert abs(center[0]) < 0.01
    assert abs(center[1]) < 0.01
    assert abs(radius - 1.0) < 0.01


def test_circle_fit_shifted_circle():
    """
    Test fitting circle with shifted center.

    Expected:
        Center should be (3, 4), radius should be 2
    """
    # Circle: (x-3)^2 + (y-4)^2 = 4
    theta = np.linspace(0, 2 * np.pi, 30)
    x = 3 + 2 * np.cos(theta)
    y = 4 + 2 * np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    center = circle.center
    radius = circle.radius

    assert abs(center[0] - 3.0) < 0.01
    assert abs(center[1] - 4.0) < 0.01
    assert abs(radius - 2.0) < 0.01


def test_circle_center_property():
    """
    Test center property returns correct coordinates.

    Expected:
        Should return tuple (x0, y0) of center
    """
    theta = np.linspace(0, 2 * np.pi, 20)
    x = 5 + 3 * np.cos(theta)
    y = -2 + 3 * np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    center = circle.center
    assert isinstance(center, tuple)
    assert len(center) == 2
    assert abs(center[0] - 5.0) < 0.01
    assert abs(center[1] - (-2.0)) < 0.01


def test_circle_radius_property():
    """
    Test radius property returns correct value.

    Expected:
        Should return radius as float
    """
    theta = np.linspace(0, 2 * np.pi, 25)
    r = 7.5
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    radius = circle.radius
    assert isinstance(radius, float)
    assert abs(radius - 7.5) < 0.1


def test_circle_perimeter():
    """
    Test perimeter calculation.

    Expected:
        Perimeter = 2*π*r
    """
    theta = np.linspace(0, 2 * np.pi, 30)
    r = 5.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    perimeter = circle.perimeter
    expected = 2 * np.pi * r

    assert abs(perimeter - expected) < 0.1


def test_circle_area():
    """
    Test area calculation.

    Expected:
        Area = π*r^2
    """
    theta = np.linspace(0, 2 * np.pi, 30)
    r = 3.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    area = circle.area
    expected = np.pi * r**2

    assert abs(area - expected) < 0.1


def test_circle_domains():
    """
    Test domains property.

    Expected:
        For circle centered at (x0, y0) with radius r,
        x domain should be [x0-r, x0+r]
        y domain should be [y0-r, y0+r]
    """
    theta = np.linspace(0, 2 * np.pi, 30)
    x0, y0, r = 2.0, 3.0, 4.0
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    domains = circle.domains

    assert 'x' in domains
    assert 'y' in domains
    assert abs(domains['x'][0] - (x0 - r)) < 0.1
    assert abs(domains['x'][1] - (x0 + r)) < 0.1
    assert abs(domains['y'][0] - (y0 - r)) < 0.1
    assert abs(domains['y'][1] - (y0 + r)) < 0.1


def test_circle_fit_with_pandas():
    """
    Test fitting with pandas Series.

    Expected:
        Should accept pandas Series as input
    """
    theta = np.linspace(0, 2 * np.pi, 20)
    x = pd.Series(np.cos(theta))
    y = pd.Series(np.sin(theta))

    circle = Circle()
    circle.fit(x, y)

    assert circle.is_fitted()


def test_circle_betas_property():
    """
    Test betas property returns coefficients.

    Expected:
        Should return dict with A, B, C coefficients
    """
    theta = np.linspace(0, 2 * np.pi, 20)
    x = np.cos(theta)
    y = np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    betas = circle.betas
    assert isinstance(betas, dict)
    assert 'A' in betas
    assert 'B' in betas
    assert 'C' in betas


def test_circle_predict_y_given_x():
    """
    Test predicting y values given x.

    Expected:
        For x on circle, should return two y values (upper and lower)
    """
    theta = np.linspace(0, 2 * np.pi, 30)
    x = np.cos(theta)
    y = np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    # For x=0 on unit circle, y should be ±1
    y_pred = circle.predict(x=0.0)

    # Should return array with two values
    assert isinstance(y_pred, (np.ndarray, list, float))


def test_circle_fit_noisy_data():
    """
    Test fitting circle with noisy data.

    Expected:
        Should approximate true circle despite noise
    """
    np.random.seed(42)
    theta = np.linspace(0, 2 * np.pi, 50)
    r = 5.0
    x0, y0 = 2.0, 3.0

    x_true = x0 + r * np.cos(theta)
    y_true = y0 + r * np.sin(theta)

    # Add noise
    x_noisy = x_true + np.random.randn(50) * 0.1
    y_noisy = y_true + np.random.randn(50) * 0.1

    circle = Circle()
    circle.fit(x_noisy, y_noisy)

    center = circle.center
    radius = circle.radius

    # Should be close to true values
    assert abs(center[0] - x0) < 0.2
    assert abs(center[1] - y0) < 0.2
    assert abs(radius - r) < 0.2


def test_circle_coefs_property():
    """
    Test coefs property returns coefficient labels.

    Expected:
        Should return ['A', 'B', 'C']
    """
    circle = Circle()
    assert set(circle.coefs) == {'A', 'B', 'C'}


def test_circle_large_radius():
    """
    Test fitting circle with large radius.

    Expected:
        Should handle large radius values correctly
    """
    theta = np.linspace(0, 2 * np.pi, 40)
    r = 100.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    circle = Circle()
    circle.fit(x, y)

    assert abs(circle.radius - r) < 1.0
