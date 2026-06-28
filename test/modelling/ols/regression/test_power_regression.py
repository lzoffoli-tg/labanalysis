"""
Test suite for PowerRegression class.

Tests verify power law regression fitting (Y = a * X^b) with positive data.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.modelling.ols.regression import PowerRegression


def test_power_regression_initialization():
    """
    Test PowerRegression initialization.

    Expected:
        fit_intercept=True, positive=False, degree=1
    """
    model = PowerRegression()
    assert model.fit_intercept is True
    assert model.positive is False
    assert model.degree == 1


def test_power_regression_initialization_custom():
    """
    Test initialization with custom parameters.

    Expected:
        Custom transform function and positive constraint
    """
    transform_fn = lambda x: x * 2
    model = PowerRegression(transform=transform_fn, positive=True)
    assert model.transform == transform_fn
    assert model.positive is True


def test_power_regression_fit_predict_positive_data(power_data):
    """
    Test fit and predict with positive power data.

    Expected:
        Should fit and predict successfully on positive data
    """
    X, Y = power_data
    model = PowerRegression()
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape
    assert not model.betas.empty


def test_power_regression_fit_predict_multivariate_positive():
    """
    Test fit and predict with multivariate positive data.

    Expected:
        Should handle multiple positive features
    """
    X = np.column_stack([np.linspace(1, 10, 50), np.linspace(2, 20, 50)])
    Y = 2 * X[:, 0] ** 1.5 * X[:, 1] ** 0.5
    Y = Y.reshape(-1, 1)

    model = PowerRegression()
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_power_regression_negative_values_raise_error():
    """
    Test that negative X values raise ValueError.

    Expected:
        Should raise ValueError when X contains negative values
    """
    X = np.array([[-1], [2], [3]])
    Y = np.array([[1], [2], [3]])

    model = PowerRegression()
    with pytest.raises(ValueError, match="must be positive"):
        model.fit(X, Y)


def test_power_regression_zero_values_raise_error():
    """
    Test that zero X values raise ValueError.

    Expected:
        Should raise ValueError when X contains zeros
    """
    X = np.array([[0], [1], [2]])
    Y = np.array([[1], [2], [3]])

    model = PowerRegression()
    with pytest.raises(ValueError, match="must be positive"):
        model.fit(X, Y)


def test_power_regression_negative_y_raise_error():
    """
    Test that negative Y values raise ValueError.

    Expected:
        Should raise ValueError when Y contains negative values
    """
    X = np.array([[1], [2], [3]])
    Y = np.array([[-1], [2], [3]])

    model = PowerRegression()
    with pytest.raises(ValueError, match="must be positive"):
        model.fit(X, Y)


def test_power_regression_predict_negative_values_raise_error(power_data):
    """
    Test that predicting with negative values raises ValueError.

    Expected:
        Should raise ValueError when predict called with negative X
    """
    X, Y = power_data
    model = PowerRegression()
    model.fit(X, Y)

    X_neg = np.array([[-1], [2]])
    with pytest.raises(ValueError, match="must be positive"):
        model.predict(X_neg)


def test_power_regression_different_input_types(power_data):
    """
    Test with different input types.

    Expected:
        Should accept numpy arrays, pandas DataFrames
    """
    X, Y = power_data

    # Fit with numpy arrays
    model1 = PowerRegression()
    model1.fit(X, Y)
    preds1 = model1.predict(X)
    assert isinstance(preds1, pd.DataFrame)

    # Fit with DataFrames
    X_df = pd.DataFrame(X, columns=["feature"])
    Y_df = pd.DataFrame(Y, columns=["target"])
    model2 = PowerRegression()
    model2.fit(X_df, Y_df)
    preds2 = model2.predict(X_df)
    assert isinstance(preds2, pd.DataFrame)


def test_power_regression_copy(power_data):
    """
    Test copy method.

    Expected:
        Should create deep copy with same parameters
    """
    X, Y = power_data
    model = PowerRegression(positive=True)
    model.fit(X, Y)

    copied = model.copy()
    assert isinstance(copied, PowerRegression)
    assert copied.positive == model.positive
    assert copied is not model


def test_power_regression_betas_structure(power_data):
    """
    Test betas DataFrame structure.

    Expected:
        Betas should be DataFrame with coefficient information
    """
    X, Y = power_data
    model = PowerRegression()
    model.fit(X, Y)

    betas = model.betas
    assert isinstance(betas, pd.DataFrame)
    assert not betas.empty
