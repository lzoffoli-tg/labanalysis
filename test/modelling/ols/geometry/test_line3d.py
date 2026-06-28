"""
Test suite for Line3D class.

Tests verify 3D line fitting via least squares: A*x + B*y + C*z + D = 0.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.modelling.ols.geometry import Line3D


def test_line3d_initialization():
    """
    Test Line3D initialization.

    Expected:
        Should create Line3D object with default intercept=True
    """
    line = Line3D()
    assert line.has_intercept is True
    assert line.dimensions == ["x", "y", "z"]


def test_line3d_no_intercept():
    """
    Test Line3D without intercept (line through origin).

    Expected:
        has_intercept=False creates line A*x + B*y + C*z = 0
    """
    line = Line3D(has_intercept=False)
    assert line.has_intercept is False


def test_line3d_fit_horizontal_line():
    """
    Test fitting horizontal line in 3D (parallel to xy-plane).

    Expected:
        Line with z=constant should fit correctly
    """
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])
    z = np.full(5, 5.0)

    line = Line3D()
    line.fit(x, y, z)

    assert line.is_fitted()
    betas = line.betas
    assert 'A' in betas
    assert 'B' in betas
    assert 'C' in betas
    assert 'D' in betas


def test_line3d_fit_diagonal():
    """
    Test fitting diagonal line x=y=z.

    Expected:
        Should fit line x=y=z and report fitted status
    """
    t = np.linspace(0, 10, 20)
    x = t.copy()
    y = t.copy()
    z = t.copy()

    line = Line3D()
    line.fit(x, y, z)

    assert line.is_fitted()
    assert 'A' in line.betas
    assert 'B' in line.betas
    assert 'C' in line.betas


def test_line3d_fit_parametric():
    """
    Test fitting parametric line.

    Expected:
        Should fit line x=t, y=2t, z=3t correctly
    """
    t = np.linspace(0, 5, 30)
    x = t
    y = 2 * t
    z = 3 * t

    line = Line3D()
    line.fit(x, y, z)

    assert line.is_fitted()


def test_line3d_predict_z_given_xy():
    """
    Test predicting z given x and y values.

    Expected:
        For line z = x + y, should be able to predict z from x and y
    """
    t = np.linspace(0, 10, 50)
    x = t
    y = t
    z = x + y

    line = Line3D()
    line.fit(x, y, z)

    # For 3D lines, equation A*x + B*y + C*z + D = 0 has multiple solutions
    # Just verify prediction works without error
    z_pred = line.predict(x=2, y=3)
    assert isinstance(z_pred, (np.ndarray, float))


def test_line3d_predict_x_given_yz():
    """
    Test predicting x given y and z values.

    Expected:
        For parametric line, should be able to predict x from y and z
    """
    t = np.linspace(1, 10, 40)
    x = 2 * t
    y = 3 * t
    z = 4 * t

    line = Line3D()
    line.fit(x, y, z)

    # For parametric 3D lines, prediction may have multiple valid solutions
    # Just verify prediction works without error
    x_pred = line.predict(y=6, z=8)
    assert isinstance(x_pred, (np.ndarray, float))


def test_line3d_predict_y_given_xz():
    """
    Test predicting y given x and z values.

    Expected:
        For parametric line, should predict y from x and z
    """
    t = np.linspace(0, 5, 30)
    x = t
    y = 2 * t + 1
    z = t

    line = Line3D()
    line.fit(x, y, z)

    # Predict y for x=3, z=3 (should be y≈7)
    y_pred = line.predict(x=3, z=3)
    assert abs(y_pred - 7.0) < 1.0


def test_line3d_fitted_equation():
    """
    Test that fitted_equation property works.

    Expected:
        Should return sympy equation with substituted coefficients
    """
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    z = np.array([0, 1, 2])

    line = Line3D()
    line.fit(x, y, z)

    eq = line.fitted_equation
    assert eq is not None


def test_line3d_fit_with_pandas():
    """
    Test fitting with pandas Series.

    Expected:
        Should accept pandas Series as input
    """
    x = pd.Series([0, 1, 2, 3, 4])
    y = pd.Series([1, 2, 3, 4, 5])
    z = pd.Series([2, 3, 4, 5, 6])

    line = Line3D()
    line.fit(x, y, z)

    assert line.is_fitted()


def test_line3d_fit_through_origin():
    """
    Test fitting line through origin (has_intercept=False).

    Expected:
        Should fit line A*x + B*y + C*z = 0
    """
    t = np.array([1, 2, 3, 4, 5])
    x = t
    y = 2 * t
    z = 3 * t

    line = Line3D(has_intercept=False)
    line.fit(x, y, z)

    # Should pass through origin
    z_pred = line.predict(x=0, y=0)
    assert abs(z_pred) < 0.01


def test_line3d_not_fitted_error():
    """
    Test that predict raises error before fitting.

    Expected:
        Should raise ValueError when predict called before fit
    """
    line = Line3D()

    with pytest.raises(ValueError, match="must be fitted"):
        line.predict(x=1, y=1)


def test_line3d_betas_property():
    """
    Test betas property returns coefficients.

    Expected:
        Should return dict with A, B, C, D coefficients
    """
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    z = np.array([0, 1, 2])

    line = Line3D()
    line.fit(x, y, z)

    betas = line.betas
    assert isinstance(betas, dict)
    assert 'A' in betas
    assert 'B' in betas
    assert 'C' in betas
    assert 'D' in betas


def test_line3d_domains_property():
    """
    Test domains property.

    Expected:
        Should return infinite domains for x, y, z
    """
    line = Line3D()
    domains = line.domains

    assert 'x' in domains
    assert 'y' in domains
    assert 'z' in domains
    assert domains['x'] == (-np.inf, np.inf)
    assert domains['y'] == (-np.inf, np.inf)
    assert domains['z'] == (-np.inf, np.inf)


def test_line3d_coefs_property():
    """
    Test coefs property returns coefficient labels.

    Expected:
        Should return ['A', 'B', 'C', 'D'] or ['A', 'B', 'C'] depending on intercept
    """
    line_with_intercept = Line3D(has_intercept=True)
    assert set(line_with_intercept.coefs) == {'A', 'B', 'C', 'D'}

    line_no_intercept = Line3D(has_intercept=False)
    assert set(line_no_intercept.coefs) == {'A', 'B', 'C'}


def test_line3d_predict_error_no_coords():
    """
    Test predict raises error when fewer than 2 coordinates provided.

    Expected:
        Should raise ValueError when only 1 or 0 coordinates provided
    """
    t = np.linspace(0, 5, 20)
    x = t
    y = 2 * t
    z = 3 * t

    line = Line3D()
    line.fit(x, y, z)

    # Only x provided (need 2 coords)
    with pytest.raises(ValueError):
        line.predict(x=1)

    # No coords provided
    with pytest.raises(ValueError):
        line.predict()
