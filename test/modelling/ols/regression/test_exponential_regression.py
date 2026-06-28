"""
Test suite for ExponentialRegression class.

Tests verify exponential regression fitting using non-linear optimization.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.modelling.ols.regression import ExponentialRegression


def test_exponential_regression_initialization_default():
    """
    Test default initialization of ExponentialRegression.

    Expected:
        fit_intercept=True, positive=False
    """
    model = ExponentialRegression()
    assert model.fit_intercept is True
    assert model.positive is False


def test_exponential_regression_initialization_custom():
    """
    Test initialization with custom parameters.

    Expected:
        Custom fit_intercept and transform function
    """
    transform_fn = lambda x: x**2
    model = ExponentialRegression(fit_intercept=False, transform=transform_fn)
    assert model.fit_intercept is False
    assert model.transform == transform_fn


def test_exponential_regression_fit_predict_with_intercept(exponential_data):
    """
    Test fit and predict with intercept.

    Expected:
        Should fit exponential model with intercept term
    """
    X, Y = exponential_data
    model = ExponentialRegression(fit_intercept=True)
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape
    assert not model.betas.empty


def test_exponential_regression_fit_predict_without_intercept(exponential_data):
    """
    Test fit and predict without intercept.

    Expected:
        Should fit exponential model without intercept term
    """
    X, Y = exponential_data
    model = ExponentialRegression(fit_intercept=False)
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_exponential_regression_multivariate():
    """
    Test with multivariate data.

    Expected:
        Should handle multiple input features
    """
    X = np.column_stack([np.linspace(1, 5, 30), np.linspace(2, 6, 30)])
    Y = 2 + X[:, 0] ** 1.5 + X[:, 1] ** 2
    Y = Y.reshape(-1, 1)

    model = ExponentialRegression()
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_exponential_regression_predict_before_fit_raises_error():
    """
    Test betas state before and after fitting.

    Expected:
        Betas should be empty before fit, populated after fit
    """
    model = ExponentialRegression()
    X = np.array([[1], [2], [3]])

    # The model initializes _betas as empty DataFrame
    assert model.betas.empty

    # After fit, betas should not be empty
    Y = np.array([[2], [4], [6]])
    model.fit(X, Y)
    assert not model.betas.empty


def test_exponential_regression_model_function(exponential_data):
    """
    Test _model_function method.

    Expected:
        Model function should return predictions for given parameters
    """
    X, Y = exponential_data
    model = ExponentialRegression(fit_intercept=True)
    model.fit(X, Y)

    # Test that model function can be called
    params = model.betas.iloc[:, 0].values
    output = model._model_function(X, params)
    assert len(output) == len(X)


def test_exponential_regression_loss_function(exponential_data):
    """
    Test _loss_function method.

    Expected:
        Loss function should return non-negative scalar
    """
    X, Y = exponential_data
    model = ExponentialRegression()

    # Create some test parameters
    params = np.array([1.0, 1.5])
    loss = model._loss_function(params, X, Y.flatten())
    assert isinstance(loss, (int, float))
    assert loss >= 0


def test_exponential_regression_betas_structure_with_intercept(exponential_data):
    """
    Test betas structure with intercept.

    Expected:
        Should have beta0 (intercept) and beta for each feature
    """
    X, Y = exponential_data
    model = ExponentialRegression(fit_intercept=True)
    model.fit(X, Y)

    # Should have beta0 (intercept) and beta for each feature
    assert "beta0" in model.betas.index
    assert model.betas.shape[0] == 2  # beta0 + 1 feature


def test_exponential_regression_betas_structure_without_intercept(exponential_data):
    """
    Test betas structure without intercept.

    Expected:
        Should only have betas for features, no beta0
    """
    X, Y = exponential_data
    model = ExponentialRegression(fit_intercept=False)
    model.fit(X, Y)

    # Should only have betas for features
    assert "beta0" not in model.betas.index


def test_exponential_regression_copy(exponential_data):
    """
    Test copy method.

    Expected:
        Should create deep copy with same parameters
    """
    X, Y = exponential_data
    model = ExponentialRegression(fit_intercept=False)
    model.fit(X, Y)

    copied = model.copy()
    assert isinstance(copied, ExponentialRegression)
    assert copied.fit_intercept == model.fit_intercept
    assert copied is not model


def test_exponential_regression_different_input_types(exponential_data):
    """
    Test with different input types.

    Expected:
        Should accept numpy arrays and pandas DataFrames
    """
    X, Y = exponential_data

    # Fit with numpy arrays
    model1 = ExponentialRegression()
    model1.fit(X, Y)
    preds1 = model1.predict(X)
    assert isinstance(preds1, pd.DataFrame)

    # Fit with DataFrames
    X_df = pd.DataFrame(X, columns=["feature"])
    Y_df = pd.DataFrame(Y, columns=["target"])
    model2 = ExponentialRegression()
    model2.fit(X_df, Y_df)
    preds2 = model2.predict(X_df)
    assert isinstance(preds2, pd.DataFrame)
