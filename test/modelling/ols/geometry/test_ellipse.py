"""
Test suite for Ellipse class.

Tests verify ellipse fitting via least squares: A*x^2 + B*xy + C*y^2 + D*x + E*y + F = 0.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.modelling.ols.geometry import Ellipse


def test_ellipse_initialization():
    """
    Test Ellipse initialization.

    Expected:
        Should create Ellipse object with appropriate equation
    """
    ellipse = Ellipse()
    assert ellipse.dimensions == ["x", "y"]
    assert ellipse.has_intercept is True


def test_ellipse_fit_unit_circle():
    """
    Test fitting unit circle (special case of ellipse).

    Expected:
        Center should be (0, 0), semi-axes should be (1, 1)
    """
    theta = np.linspace(0, 2 * np.pi, 30)
    x = np.cos(theta)
    y = np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    center = ellipse.center
    semi_axes = ellipse.semi_axes

    assert abs(center[0]) < 0.05
    assert abs(center[1]) < 0.05
    assert abs(semi_axes[0] - 1.0) < 0.05
    assert abs(semi_axes[1] - 1.0) < 0.05


def test_ellipse_fit_horizontal():
    """
    Test fitting horizontal ellipse (semi-major along x).

    Expected:
        Center (0, 0), semi-axes (3, 2)
    """
    theta = np.linspace(0, 2 * np.pi, 40)
    a, b = 3.0, 2.0
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    center = ellipse.center
    semi_axes = ellipse.semi_axes

    assert abs(center[0]) < 0.1
    assert abs(center[1]) < 0.1
    # Semi-axes returned as (a, b) but order may vary
    axes_set = set([round(semi_axes[0], 1), round(semi_axes[1], 1)])
    assert axes_set == {3.0, 2.0} or axes_set == {2.0, 3.0}


def test_ellipse_fit_shifted():
    """
    Test fitting shifted ellipse.

    Expected:
        Center (4, 5), semi-axes (3, 2)
    """
    theta = np.linspace(0, 2 * np.pi, 50)
    a, b = 3.0, 2.0
    x0, y0 = 4.0, 5.0
    x = x0 + a * np.cos(theta)
    y = y0 + b * np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    center = ellipse.center

    assert abs(center[0] - x0) < 0.2
    assert abs(center[1] - y0) < 0.2


def test_ellipse_center_property():
    """
    Test center property returns correct coordinates.

    Expected:
        Should return tuple (h, k) of center
    """
    theta = np.linspace(0, 2 * np.pi, 30)
    x = 2 + 4 * np.cos(theta)
    y = 3 + 2 * np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    center = ellipse.center
    assert isinstance(center, tuple)
    assert len(center) == 2
    assert abs(center[0] - 2.0) < 0.2
    assert abs(center[1] - 3.0) < 0.2


def test_ellipse_semi_axes_property():
    """
    Test semi_axes property returns lengths.

    Expected:
        Should return tuple (a, b) of semi-axis lengths
    """
    theta = np.linspace(0, 2 * np.pi, 35)
    a, b = 5.0, 3.0
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    semi_axes = ellipse.semi_axes
    assert isinstance(semi_axes, tuple)
    assert len(semi_axes) == 2
    # Verify both values are present (order may vary)
    axes_sorted = sorted(semi_axes)
    assert abs(axes_sorted[0] - 3.0) < 0.2
    assert abs(axes_sorted[1] - 5.0) < 0.2


def test_ellipse_perimeter():
    """
    Test perimeter calculation using Ramanujan approximation.

    Expected:
        Perimeter ≈ π * (a + b) * (1 + 3h/(10 + sqrt(4-3h)))
    """
    theta = np.linspace(0, 2 * np.pi, 40)
    a, b = 4.0, 3.0
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    perimeter = ellipse.perimeter
    # Ramanujan approximation for a=4, b=3
    h = ((a - b) ** 2) / ((a + b) ** 2)
    expected = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))

    assert abs(perimeter - expected) < 1.0


def test_ellipse_area():
    """
    Test area calculation.

    Expected:
        Area = π * a * b
    """
    theta = np.linspace(0, 2 * np.pi, 40)
    a, b = 5.0, 3.0
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    area = ellipse.area
    expected = np.pi * a * b

    assert abs(area - expected) < 1.0


def test_ellipse_domains():
    """
    Test domains property.

    Expected:
        For ellipse centered at (x0, y0) with semi-axes (a, b),
        x domain should be [x0-a, x0+a]
        y domain should be [y0-b, y0+b]
    """
    theta = np.linspace(0, 2 * np.pi, 40)
    a, b = 4.0, 2.0
    x0, y0 = 1.0, 2.0
    x = x0 + a * np.cos(theta)
    y = y0 + b * np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    domains = ellipse.domains

    assert 'x' in domains
    assert 'y' in domains
    # Domain should approximately match x0 ± a
    assert abs(domains['x'][0] - (x0 - a)) < 0.5
    assert abs(domains['x'][1] - (x0 + a)) < 0.5
    assert abs(domains['y'][0] - (y0 - b)) < 0.5
    assert abs(domains['y'][1] - (y0 + b)) < 0.5


def test_ellipse_major_axis():
    """
    Test major_axis property.

    Expected:
        Should return Line2D object representing major axis
    """
    theta = np.linspace(0, 2 * np.pi, 40)
    x = 4 * np.cos(theta)
    y = 2 * np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    major_axis = ellipse.major_axis
    # Should be a Line2D object
    from labanalysis.modelling.ols.geometry import Line2D
    assert isinstance(major_axis, Line2D)


def test_ellipse_minor_axis():
    """
    Test minor_axis property.

    Expected:
        Should return Line2D object representing minor axis
    """
    theta = np.linspace(0, 2 * np.pi, 40)
    x = 5 * np.cos(theta)
    y = 3 * np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    minor_axis = ellipse.minor_axis
    # Should be a Line2D object
    from labanalysis.modelling.ols.geometry import Line2D
    assert isinstance(minor_axis, Line2D)


def test_ellipse_fit_with_pandas():
    """
    Test fitting with pandas Series.

    Expected:
        Should accept pandas Series as input
    """
    theta = np.linspace(0, 2 * np.pi, 30)
    x = pd.Series(3 * np.cos(theta))
    y = pd.Series(2 * np.sin(theta))

    ellipse = Ellipse()
    ellipse.fit(x, y)

    assert ellipse.is_fitted()


def test_ellipse_betas_property():
    """
    Test betas property returns coefficients.

    Expected:
        Should return dict with A, B, C, D, E, F coefficients
    """
    theta = np.linspace(0, 2 * np.pi, 30)
    x = np.cos(theta)
    y = np.sin(theta)

    ellipse = Ellipse()
    ellipse.fit(x, y)

    betas = ellipse.betas
    assert isinstance(betas, dict)
    assert 'A' in betas
    assert 'B' in betas
    assert 'C' in betas
    assert 'D' in betas
    assert 'E' in betas
    assert 'F' in betas


def test_ellipse_coefs_property():
    """
    Test coefs property returns coefficient labels.

    Expected:
        Should return ['A', 'B', 'C', 'D', 'E', 'F']
    """
    ellipse = Ellipse()
    assert set(ellipse.coefs) == {'A', 'B', 'C', 'D', 'E', 'F'}


def test_ellipse_fit_noisy_data():
    """
    Test fitting ellipse with noisy data.

    Expected:
        Should approximate true ellipse despite noise
    """
    np.random.seed(42)
    theta = np.linspace(0, 2 * np.pi, 60)
    a, b = 6.0, 4.0
    x0, y0 = 2.0, 3.0

    x_true = x0 + a * np.cos(theta)
    y_true = y0 + b * np.sin(theta)

    # Add noise
    x_noisy = x_true + np.random.randn(60) * 0.2
    y_noisy = y_true + np.random.randn(60) * 0.2

    ellipse = Ellipse()
    ellipse.fit(x_noisy, y_noisy)

    center = ellipse.center

    # Should be close to true values
    assert abs(center[0] - x0) < 0.5
    assert abs(center[1] - y0) < 0.5
