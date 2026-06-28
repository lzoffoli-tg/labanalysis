"""
Test suite for BaseRegression class.

Tests verify base regression functionality for OLS models.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.modelling.ols.regression import BaseRegression


def test_base_regression_initialization_default():
    """
    Test default initialization of BaseRegression.

    Expected:
        fit_intercept=True, positive=False, empty betas DataFrame
    """
    model = BaseRegression()
    assert model.fit_intercept is True
    assert callable(model.transform)
    assert model.positive is False
    assert isinstance(model.betas, pd.DataFrame)
    assert model.betas.empty


def test_base_regression_initialization_custom():
    """
    Test initialization with custom parameters.

    Expected:
        Custom transform function, fit_intercept=False, positive=True
    """
    transform_fn = lambda x: x**2
    model = BaseRegression(
        fit_intercept=False, transform=transform_fn, positive=True
    )
    assert model.fit_intercept is False
    assert model.transform == transform_fn
    assert model.positive is True


def test_base_regression_transform_property():
    """
    Test transform property getter.

    Expected:
        Should return assigned transform function
    """
    transform_fn = lambda x: np.log(x)
    model = BaseRegression(transform=transform_fn)
    assert model.transform == transform_fn


def test_base_regression_betas_property():
    """
    Test betas property getter.

    Expected:
        Should return pandas DataFrame
    """
    model = BaseRegression()
    assert isinstance(model.betas, pd.DataFrame)


def test_base_regression_get_feature_names_in():
    """
    Test get_feature_names_in method.

    Expected:
        Should return None before fitting
    """
    model = BaseRegression()
    assert model.get_feature_names_in() is None


def test_base_regression_get_feature_names_out():
    """
    Test get_feature_names_out method.

    Expected:
        Should return None before fitting
    """
    model = BaseRegression()
    assert model.get_feature_names_out() is None


def test_base_regression_simplify_ndarray_1d():
    """
    Test _simplify with 1D numpy array.

    Expected:
        Should convert to DataFrame with shape (n, 1)
    """
    model = BaseRegression()
    arr = np.array([1, 2, 3, 4, 5])
    result = model._simplify(arr, "X")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 1)
    assert result.columns[0] == "X0"


def test_base_regression_simplify_ndarray_2d():
    """
    Test _simplify with 2D numpy array.

    Expected:
        Should convert to DataFrame with appropriate column names
    """
    model = BaseRegression()
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    result = model._simplify(arr, "Y")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert list(result.columns) == ["Y0", "Y1"]


def test_base_regression_simplify_dataframe():
    """
    Test _simplify with pandas DataFrame.

    Expected:
        Should preserve DataFrame structure
    """
    model = BaseRegression()
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = model._simplify(df, "X")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    pd.testing.assert_frame_equal(result, df.astype(float))


def test_base_regression_simplify_series():
    """
    Test _simplify with pandas Series.

    Expected:
        Should convert to DataFrame with shape (n, 1)
    """
    model = BaseRegression()
    series = pd.Series([1, 2, 3, 4, 5], name="values")
    result = model._simplify(series, "Y")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 1)


def test_base_regression_simplify_list():
    """
    Test _simplify with list.

    Expected:
        Should convert to DataFrame with shape (n, 1)
    """
    model = BaseRegression()
    lst = [1, 2, 3, 4, 5]
    result = model._simplify(lst, "X")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 1)


def test_base_regression_simplify_scalar():
    """
    Test _simplify with scalar value.

    Expected:
        Should convert to DataFrame with shape (1, 1)
    """
    model = BaseRegression()
    scalar = 42
    result = model._simplify(scalar, "X")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)
    assert result.values[0, 0] == 42


def test_base_regression_copy():
    """
    Test copy method.

    Expected:
        Should create deep copy with same parameters
    """
    model = BaseRegression(fit_intercept=False, positive=True)
    model._names_in = ["X0", "X1"]
    model._names_out = ["Y0"]

    copied = model.copy()
    assert isinstance(copied, BaseRegression)
    assert copied.fit_intercept == model.fit_intercept
    assert copied.positive == model.positive
    assert copied._names_in == model._names_in
    assert copied._names_out == model._names_out
    # Ensure deep copy
    assert copied is not model


def test_base_regression_call_method():
    """
    Test __call__ method exists.

    Expected:
        Should have callable __call__ method
    """
    model = BaseRegression()
    assert callable(model)
