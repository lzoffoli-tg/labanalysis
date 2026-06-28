"""
Test suite for MultiSegmentRegression class.

Tests verify multi-segment piecewise polynomial regression fitting.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.modelling.ols.regression import MultiSegmentRegression


def test_multisegment_regression_initialization_default():
    """
    Test default initialization of MultiSegmentRegression.

    Expected:
        degree=1, n_segments=1, min_samples=degree+1
    """
    model = MultiSegmentRegression()
    assert model.degree == 1
    assert model.n_segments == 1
    assert model.min_samples == 2  # degree + 1


def test_multisegment_regression_initialization_custom():
    """
    Test initialization with custom parameters.

    Expected:
        Custom degree, n_segments, min_samples, positive
    """
    model = MultiSegmentRegression(
        degree=2, n_segments=3, min_samples=5, positive=True
    )
    assert model.degree == 2
    assert model.n_segments == 3
    assert model.min_samples == 5
    assert model.positive is True


def test_multisegment_regression_n_segments_property():
    """
    Test n_segments property.

    Expected:
        Should return assigned n_segments value
    """
    model = MultiSegmentRegression(n_segments=4)
    assert model.n_segments == 4


def test_multisegment_regression_min_samples_property():
    """
    Test min_samples property.

    Expected:
        Should return assigned min_samples value
    """
    model = MultiSegmentRegression(degree=3, min_samples=10)
    assert model.min_samples == 10


def test_multisegment_regression_min_samples_default_from_degree():
    """
    Test min_samples defaults to degree + 1.

    Expected:
        min_samples should be degree + 1 when not specified
    """
    model = MultiSegmentRegression(degree=5)
    assert model.min_samples == 6


def test_multisegment_regression_fit_predict_single_segment(linear_data):
    """
    Test fit and predict with single segment.

    Expected:
        Should behave like standard polynomial regression with n_segments=1
    """
    X, Y = linear_data
    model = MultiSegmentRegression(degree=1, n_segments=1)
    model.fit(X[:, 0], Y)
    preds = model.predict(X[:, 0])

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == Y.shape


def test_multisegment_regression_fit_predict_two_segments(segmented_data):
    """
    Test fit and predict with two segments.

    Expected:
        Should identify two distinct segments in the data
    """
    X, Y = segmented_data
    model = MultiSegmentRegression(degree=1, n_segments=2)
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == len(Y)


def test_multisegment_regression_fit_predict_multiple_segments(segmented_data):
    """
    Test fit and predict with multiple segments.

    Expected:
        Should handle 3+ segments with appropriate min_samples
    """
    X, Y = segmented_data
    model = MultiSegmentRegression(degree=1, n_segments=3, min_samples=5)
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == len(Y)


def test_multisegment_regression_2d_x_raises_error():
    """
    Test that 2D X array raises ValueError.

    Expected:
        Should raise ValueError for multivariate X (only accepts 1D)
    """
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Y = np.array([[1], [2], [3]])

    model = MultiSegmentRegression()
    with pytest.raises(ValueError, match="must be a 1D array"):
        model.fit(X, Y)


def test_multisegment_regression_predict_2d_x_raises_error(linear_data):
    """
    Test that predicting with 2D X raises ValueError.

    Expected:
        Should raise ValueError when predict called with 2D X
    """
    X, Y = linear_data
    model = MultiSegmentRegression()
    model.fit(X[:, 0], Y)

    X_2d = np.column_stack([X[:, 0], X[:, 0]])
    with pytest.raises(ValueError, match="must be a 1D array"):
        model.predict(X_2d)


def test_multisegment_regression_mismatched_shapes_raise_error():
    """
    Test that mismatched X and Y shapes raise ValueError.

    Expected:
        Should raise ValueError when X and Y have different lengths
    """
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([[1], [2], [3]])

    model = MultiSegmentRegression()
    with pytest.raises(ValueError, match="equal sample size"):
        model.fit(X, Y)


def test_multisegment_regression_high_degree_polynomial():
    """
    Test with high degree polynomial in segments.

    Expected:
        Should fit high-degree polynomials within each segment
    """
    X = np.linspace(0, 10, 100)
    Y = np.sin(X) + np.random.randn(100) * 0.1

    model = MultiSegmentRegression(degree=3, n_segments=2, min_samples=10)
    model.fit(X, Y)
    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == (100, 1)


def test_multisegment_regression_betas_structure(segmented_data):
    """
    Test betas DataFrame structure.

    Expected:
        Betas should have MultiIndex columns with FEATURE, X0, X1
    """
    X, Y = segmented_data
    model = MultiSegmentRegression(degree=1, n_segments=2)
    model.fit(X, Y)

    # Betas should have MultiIndex columns with FEATURE, X0, X1
    assert isinstance(model.betas.columns, pd.MultiIndex)
    assert model.betas.columns.names == ["FEATURE", "X0", "X1"]


def test_multisegment_regression_segment_boundaries(segmented_data):
    """
    Test that segment boundaries are properly identified.

    Expected:
        Should identify correct number of segments
    """
    X, Y = segmented_data
    model = MultiSegmentRegression(degree=1, n_segments=2)
    model.fit(X, Y)

    # Check that we have proper segment boundaries in betas
    boundaries = model.betas.columns.get_level_values("X0").unique()
    assert len(boundaries) == 2  # Two segments


def test_multisegment_regression_different_input_types():
    """
    Test with different input types.

    Expected:
        Should accept lists, numpy arrays, pandas Series
    """
    model = MultiSegmentRegression(degree=1, n_segments=1)

    # Lists
    X_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Y_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    model.fit(X_list, Y_list)
    preds = model.predict(X_list)
    assert isinstance(preds, pd.DataFrame)

    # Series
    X_series = pd.Series(X_list)
    Y_series = pd.Series(Y_list)
    model.fit(X_series, Y_series)
    preds = model.predict(X_series)
    assert isinstance(preds, pd.DataFrame)


def test_multisegment_regression_copy(segmented_data):
    """
    Test copy method.

    Expected:
        Should create deep copy with same parameters
    """
    X, Y = segmented_data
    model = MultiSegmentRegression(degree=2, n_segments=2)
    model.fit(X, Y)

    copied = model.copy()
    assert isinstance(copied, MultiSegmentRegression)
    assert copied.degree == model.degree
    assert copied.n_segments == model.n_segments
    assert copied is not model
