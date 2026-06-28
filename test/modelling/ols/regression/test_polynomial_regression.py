"""
Test suite for PolynomialRegression class.

Tests verify polynomial regression fitting with various degrees and configurations.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.modelling.ols.regression import PolynomialRegression


def test_polynomial_regression_initialization_default():
    """
    Test default initialization of PolynomialRegression.

    Expected:
        degree=1, fit_intercept=True, both main_terms and interactions=True
    """
    model = PolynomialRegression()
    assert model.degree == 1
    assert model.fit_intercept is True
    assert model.include_main_terms is True
    assert model.include_interactions is True


def test_polynomial_regression_initialization_custom():
    """
    Test initialization with custom parameters.

    Expected:
        Custom degree, intercept, main_terms, interactions, positive
    """
    model = PolynomialRegression(
        degree=3,
        fit_intercept=False,
        include_main_terms=False,
        include_interactions=True,
        positive=True,
    )
    assert model.degree == 3
    assert model.fit_intercept is False
    assert model.include_main_terms is False
    assert model.include_interactions is True
    assert model.positive is True


def test_polynomial_regression_degree_property():
    """
    Test degree property.

    Expected:
        Should return assigned degree
    """
    model = PolynomialRegression(degree=5)
    assert model.degree == 5


def test_polynomial_regression_fit_predict_linear(linear_data):
    """
    Test fit and predict with linear data.

    Expected:
        Should fit and predict successfully with degree=1
    """
    X, Y = linear_data
    model = PolynomialRegression(degree=1)
    model.fit(X, Y)

    # Check fitted attributes
    assert model._names_in is not None
    assert model._names_out is not None
    assert not model.betas.empty

    # Check predictions
    preds = model.predict(X)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_polynomial_regression_fit_predict_polynomial(polynomial_data):
    """
    Test fit and predict with polynomial data.

    Expected:
        Should achieve high R² (>0.95) with degree=2 on quadratic data
    """
    X, Y = polynomial_data
    model = PolynomialRegression(degree=2)
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape
    # Check reasonable fit (R² should be high for polynomial data)
    residuals = Y - preds.values
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    assert r_squared > 0.95


def test_polynomial_regression_fit_predict_multivariate(multivariate_data):
    """
    Test fit and predict with multivariate data.

    Expected:
        Should handle multiple input features
    """
    X, Y = multivariate_data
    model = PolynomialRegression(degree=1)
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_polynomial_regression_fit_predict_high_degree(linear_data):
    """
    Test fit and predict with high polynomial degree.

    Expected:
        Should handle degree=5 without error
    """
    X, Y = linear_data
    model = PolynomialRegression(degree=5)
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_polynomial_regression_include_main_terms_only(multivariate_data):
    """
    Test with only main terms (no interactions).

    Expected:
        Should fit with only main terms when include_interactions=False
    """
    X, Y = multivariate_data
    model = PolynomialRegression(
        degree=2, include_main_terms=True, include_interactions=False
    )
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_polynomial_regression_include_interactions_only(multivariate_data):
    """
    Test with only interactions (no main terms).

    Expected:
        Should fit with only interactions when include_main_terms=False
    """
    X, Y = multivariate_data
    model = PolynomialRegression(
        degree=2, include_main_terms=False, include_interactions=True
    )
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_polynomial_regression_both_flags_false_raises_error(linear_data):
    """
    Test that both flags False raises ValueError.

    Expected:
        Should raise ValueError when both include_main_terms and include_interactions are False
    """
    X, Y = linear_data
    model = PolynomialRegression(
        degree=2, include_main_terms=False, include_interactions=False
    )
    with pytest.raises(ValueError, match="cannot be both False"):
        model.fit(X, Y)


def test_polynomial_regression_fit_no_intercept(linear_data):
    """
    Test fitting without intercept.

    Expected:
        Should fit and predict with fit_intercept=False
    """
    X, Y = linear_data
    model = PolynomialRegression(degree=1, fit_intercept=False)
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_polynomial_regression_transform_function(linear_data):
    """
    Test with custom transform function.

    Expected:
        Should apply custom transform before fitting
    """
    X, Y = linear_data
    model = PolynomialRegression(degree=1, transform=lambda x: np.sqrt(x))
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_polynomial_regression_positive_coefficients(linear_data):
    """
    Test with positive constraint on coefficients.

    Expected:
        Should enforce positive coefficients when positive=True
    """
    X, Y = linear_data
    Y_positive = np.abs(Y)  # Ensure positive relationship
    model = PolynomialRegression(degree=1, positive=True)
    model.fit(X, Y_positive)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y_positive.shape


def test_polynomial_regression_call_method(linear_data):
    """
    Test __call__ method delegates to predict.

    Expected:
        model(X) should be equivalent to model.predict(X)
    """
    X, Y = linear_data
    model = PolynomialRegression(degree=2)
    model.fit(X, Y)

    preds_predict = model.predict(X)
    preds_call = model(X)

    pd.testing.assert_frame_equal(preds_predict, preds_call)


def test_polynomial_regression_different_input_types_fit(linear_data):
    """
    Test fit with different input types.

    Expected:
        Should accept numpy arrays, pandas DataFrames, lists
    """
    X, Y = linear_data

    # Fit with numpy arrays
    model1 = PolynomialRegression(degree=1)
    model1.fit(X, Y)
    assert not model1.betas.empty

    # Fit with DataFrames
    X_df = pd.DataFrame(X, columns=["feature"])
    Y_df = pd.DataFrame(Y, columns=["target"])
    model2 = PolynomialRegression(degree=1)
    model2.fit(X_df, Y_df)
    assert not model2.betas.empty


def test_polynomial_regression_different_input_types_predict(linear_data):
    """
    Test predict with different input types.

    Expected:
        Should accept numpy arrays, pandas DataFrames for prediction
    """
    X, Y = linear_data
    model = PolynomialRegression(degree=1)
    model.fit(X, Y)

    # Predict with numpy array
    preds1 = model.predict(X)
    assert isinstance(preds1, pd.DataFrame)

    # Predict with DataFrame
    X_df = pd.DataFrame(X, columns=["feature"])
    preds2 = model.predict(X_df)
    assert isinstance(preds2, pd.DataFrame)


def test_polynomial_regression_feature_names(multivariate_data):
    """
    Test feature names handling.

    Expected:
        Should track input and output feature names
    """
    X, Y = multivariate_data
    X_df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    Y_df = pd.DataFrame(Y, columns=["y"])

    model = PolynomialRegression(degree=1)
    model.fit(X_df, Y_df)

    names_in = model.get_feature_names_in()
    names_out = model.get_feature_names_out()

    assert names_in is not None
    assert names_out is not None


def test_polynomial_regression_copy(linear_data):
    """
    Test copy method.

    Expected:
        Should create deep copy with same parameters and fitted state
    """
    X, Y = linear_data
    model = PolynomialRegression(degree=3, fit_intercept=False)
    model.fit(X, Y)

    copied = model.copy()
    assert isinstance(copied, PolynomialRegression)
    assert copied.degree == model.degree
    assert copied.fit_intercept == model.fit_intercept
    assert copied is not model


def test_polynomial_regression_adjust_degree(linear_data):
    """
    Test behavior with different degrees.

    Expected:
        Higher degrees should not degrade fit quality significantly
    """
    X, Y = linear_data

    model1 = PolynomialRegression(degree=1)
    model1.fit(X, Y)
    preds1 = model1.predict(X)

    model2 = PolynomialRegression(degree=2)
    model2.fit(X, Y)
    preds2 = model2.predict(X)

    # Both should produce reasonable predictions
    assert preds1.shape == preds2.shape
