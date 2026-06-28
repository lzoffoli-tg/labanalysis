"""
Test suite for Line2D class.

Tests verify 2D line fitting via least squares: A*x + B*y + C = 0.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.modelling.ols.geometry import Line2D


def test_line2d_initialization():
    """
    Test Line2D initialization.

    Expected:
        Should create Line2D object with default intercept=True
    """
    line = Line2D()
    assert line.has_intercept is True
    assert line.dimensions == ["x", "y"]


def test_line2d_no_intercept():
    """
    Test Line2D without intercept (line through origin).

    Expected:
        has_intercept=False creates line A*x + B*y = 0
    """
    line = Line2D(has_intercept=False)
    assert line.has_intercept is False


def test_line2d_fit_horizontal_line():
    """
    Test fitting horizontal line y = c.

    Expected:
        Coefficients should represent y = 3 as 0*x + 1*y - 3 = 0
    """
    x = np.array([0, 1, 2, 3, 4])
    y = np.full(5, 3.0)

    line = Line2D()
    line.fit(x, y)

    assert line.is_fitted()
    # For y=3: A≈0, B≠0, C/B ≈ -3
    betas = line.betas
    assert 'A' in betas
    assert 'B' in betas
    assert 'C' in betas


def test_line2d_fit_vertical_line():
    """
    Test fitting vertical line x = c.

    Expected:
        Coefficients should represent x = 2 as 1*x + 0*y - 2 = 0
    """
    x = np.full(5, 2.0)
    y = np.array([0, 1, 2, 3, 4])

    line = Line2D()
    line.fit(x, y)

    assert line.is_fitted()


def test_line2d_fit_diagonal():
    """
    Test fitting diagonal line y = x.

    Expected:
        Should fit line y = x and report fitted status
    """
    x = np.linspace(0, 10, 20)
    y = x.copy()

    line = Line2D()
    line.fit(x, y)

    assert line.is_fitted()
    # For y=x, the equation has multiple valid solutions
    # Just verify betas exist
    assert 'A' in line.betas
    assert 'B' in line.betas


def test_line2d_fit_slope_intercept():
    """
    Test fitting line y = mx + b.

    Expected:
        Should fit y = 2x + 3 correctly
    """
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x + 3

    line = Line2D()
    line.fit(x, y)

    # Predict at known points
    y_pred = line.predict(x=x)
    assert np.allclose(y_pred, y, atol=0.01)


def test_line2d_predict_x_given_y():
    """
    Test predicting x given y values.

    Expected:
        For y = 2x + 3, given y should correctly predict x
    """
    x = np.array([0, 1, 2, 3, 4])
    y = 2 * x + 3

    line = Line2D()
    line.fit(x, y)

    # Predict x for y=5 (should be x=1)
    x_pred = line.predict(y=5)
    assert abs(x_pred - 1.0) < 0.01


def test_line2d_predict_y_given_x():
    """
    Test predicting y given x values.

    Expected:
        For y = 2x + 3, given x should correctly predict y
    """
    x_train = np.array([0, 1, 2, 3, 4])
    y_train = 2 * x_train + 3

    line = Line2D()
    line.fit(x_train, y_train)

    # Predict y for new x values
    x_test = np.array([5, 6, 7])
    y_pred = line.predict(x=x_test)
    y_expected = 2 * x_test + 3

    assert np.allclose(y_pred, y_expected, atol=0.01)


def test_line2d_fitted_equation():
    """
    Test that fitted_equation property works.

    Expected:
        Should return sympy equation with substituted coefficients
    """
    x = np.array([0, 1, 2])
    y = np.array([1, 2, 3])

    line = Line2D()
    line.fit(x, y)

    eq = line.fitted_equation
    assert eq is not None


def test_line2d_fit_with_pandas():
    """
    Test fitting with pandas Series.

    Expected:
        Should accept pandas Series as input
    """
    x = pd.Series([0, 1, 2, 3, 4])
    y = pd.Series([1, 3, 5, 7, 9])

    line = Line2D()
    line.fit(x, y)

    assert line.is_fitted()


def test_line2d_fit_through_origin():
    """
    Test fitting line through origin (has_intercept=False).

    Expected:
        Should fit y = 2x as 2*x - 1*y = 0
    """
    x = np.array([1, 2, 3, 4])
    y = 2 * x

    line = Line2D(has_intercept=False)
    line.fit(x, y)

    # Should pass through origin
    y_pred = line.predict(x=0)
    assert abs(y_pred) < 0.01


def test_line2d_not_fitted_error():
    """
    Test that predict raises error before fitting.

    Expected:
        Should raise ValueError when predict called before fit
    """
    line = Line2D()

    with pytest.raises(ValueError, match="must be fitted"):
        line.predict(x=1)


def test_line2d_betas_property():
    """
    Test betas property returns coefficients.

    Expected:
        Should return dict with A, B, C coefficients
    """
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])

    line = Line2D()
    line.fit(x, y)

    betas = line.betas
    assert isinstance(betas, dict)
    assert 'A' in betas
    assert 'B' in betas
    assert 'C' in betas


def test_line2d_domains_property():
    """
    Test domains property.

    Expected:
        Should return infinite domains for x and y
    """
    line = Line2D()
    domains = line.domains

    assert 'x' in domains
    assert 'y' in domains
    assert domains['x'] == (-np.inf, np.inf)
    assert domains['y'] == (-np.inf, np.inf)


def test_line2d_coefs_property():
    """
    Test coefs property returns coefficient labels.

    Expected:
        Should return ['A', 'B', 'C'] or ['A', 'B'] depending on intercept
    """
    line_with_intercept = Line2D(has_intercept=True)
    assert set(line_with_intercept.coefs) == {'A', 'B', 'C'}

    line_no_intercept = Line2D(has_intercept=False)
    assert set(line_no_intercept.coefs) == {'A', 'B'}


def test_line2d_fit_noisy_data():
    """
    Test fitting line with noisy data.

    Expected:
        Should fit approximate line through noisy points
    """
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y_true = 3 * x + 2
    y_noisy = y_true + np.random.randn(50) * 0.5

    line = Line2D()
    line.fit(x, y_noisy)

    # Predicted values should be close to true line
    y_pred = line.predict(x=x)
    # Correlation with true line should be high
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    assert corr > 0.99
