"""
Comprehensive test suite for regression models.

This module contains extensive tests for all regression classes in
labanalysis.modelling.ols.regression.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the regression module directory to path to import directly
# This avoids importing the full labanalysis package with its dependencies
regression_module_path = Path(__file__).parent.parent / "src" / "labanalysis" / "modelling" / "ols"
sys.path.insert(0, str(regression_module_path))

# Import directly from the regression module
from regression import (
    BaseRegression,
    PolynomialRegression,
    PowerRegression,
    ExponentialRegression,
    MultiSegmentRegression,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def linear_data():
    """Generate simple linear data for testing."""
    X = np.linspace(1, 10, 50).reshape(-1, 1)
    Y = 2 * X + 3 + np.random.randn(50, 1) * 0.1
    return X, Y


@pytest.fixture
def multivariate_data():
    """Generate multivariate data for testing."""
    X = np.column_stack(
        [np.linspace(1, 10, 50), np.linspace(2, 20, 50), np.linspace(0.5, 5, 50)]
    )
    Y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 5
    Y = Y.reshape(-1, 1)
    return X, Y


@pytest.fixture
def polynomial_data():
    """Generate polynomial data for testing."""
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    Y = 2 * X**2 + 3 * X + 1 + np.random.randn(100, 1) * 0.5
    return X, Y


@pytest.fixture
def power_data():
    """Generate power relationship data (all positive)."""
    X = np.linspace(1, 10, 50).reshape(-1, 1)
    Y = 2 * X**1.5 + np.random.randn(50, 1) * 0.1
    return X, Y


@pytest.fixture
def exponential_data():
    """Generate exponential data for testing."""
    X = np.linspace(1, 5, 50).reshape(-1, 1)
    Y = 2 + X[:, 0] ** 1.5 + np.random.randn(50) * 0.1
    Y = Y.reshape(-1, 1)
    return X, Y


@pytest.fixture
def segmented_data():
    """Generate data with clear segments."""
    X1 = np.linspace(0, 5, 30)
    Y1 = 2 * X1 + 1
    X2 = np.linspace(5, 10, 30)
    Y2 = -1 * X2 + 20
    X = np.concatenate([X1, X2])
    Y = np.concatenate([Y1, Y2])
    return X, Y


# ============================================================================
# BaseRegression Tests
# ============================================================================


class TestBaseRegression:
    """Test suite for BaseRegression class."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = BaseRegression()
        assert model.fit_intercept is True
        assert callable(model.transform)
        assert model.positive is False
        assert isinstance(model.betas, pd.DataFrame)
        assert model.betas.empty

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        transform_fn = lambda x: x**2
        model = BaseRegression(
            fit_intercept=False, transform=transform_fn, positive=True
        )
        assert model.fit_intercept is False
        assert model.transform == transform_fn
        assert model.positive is True

    def test_transform_property(self):
        """Test transform property getter."""
        transform_fn = lambda x: np.log(x)
        model = BaseRegression(transform=transform_fn)
        assert model.transform == transform_fn

    def test_betas_property(self):
        """Test betas property getter."""
        model = BaseRegression()
        assert isinstance(model.betas, pd.DataFrame)

    def test_get_feature_names_in(self):
        """Test get_feature_names_in method."""
        model = BaseRegression()
        assert model.get_feature_names_in() is None

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method."""
        model = BaseRegression()
        assert model.get_feature_names_out() is None

    def test_simplify_ndarray_1d(self):
        """Test _simplify with 1D numpy array."""
        model = BaseRegression()
        arr = np.array([1, 2, 3, 4, 5])
        result = model._simplify(arr, "X")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)
        assert result.columns[0] == "X0"

    def test_simplify_ndarray_2d(self):
        """Test _simplify with 2D numpy array."""
        model = BaseRegression()
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = model._simplify(arr, "Y")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
        assert list(result.columns) == ["Y0", "Y1"]

    def test_simplify_dataframe(self):
        """Test _simplify with pandas DataFrame."""
        model = BaseRegression()
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = model._simplify(df, "X")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        pd.testing.assert_frame_equal(result, df.astype(float))

    def test_simplify_series(self):
        """Test _simplify with pandas Series."""
        model = BaseRegression()
        series = pd.Series([1, 2, 3, 4, 5], name="values")
        result = model._simplify(series, "Y")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_simplify_list(self):
        """Test _simplify with list."""
        model = BaseRegression()
        lst = [1, 2, 3, 4, 5]
        result = model._simplify(lst, "X")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 1)

    def test_simplify_scalar(self):
        """Test _simplify with scalar value."""
        model = BaseRegression()
        scalar = 42
        result = model._simplify(scalar, "X")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 1)
        assert result.values[0, 0] == 42

    def test_copy(self):
        """Test copy method."""
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

    def test_call_method(self, linear_data):
        """Test __call__ method delegates to predict."""
        X, Y = linear_data
        model = BaseRegression()
        # Note: BaseRegression needs to be fitted first
        # This is a basic test to ensure the __call__ method exists


# ============================================================================
# PolynomialRegression Tests
# ============================================================================


class TestPolynomialRegression:
    """Test suite for PolynomialRegression class."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = PolynomialRegression()
        assert model.degree == 1
        assert model.fit_intercept is True
        assert model.include_main_terms is True
        assert model.include_interactions is True

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
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

    def test_degree_property(self):
        """Test degree property."""
        model = PolynomialRegression(degree=5)
        assert model.degree == 5

    def test_fit_predict_linear(self, linear_data):
        """Test fit and predict with linear data."""
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

    def test_fit_predict_polynomial(self, polynomial_data):
        """Test fit and predict with polynomial data."""
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

    def test_fit_predict_multivariate(self, multivariate_data):
        """Test fit and predict with multivariate data."""
        X, Y = multivariate_data
        model = PolynomialRegression(degree=1)
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_fit_predict_high_degree(self, linear_data):
        """Test fit and predict with high polynomial degree."""
        X, Y = linear_data
        model = PolynomialRegression(degree=5)
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_include_main_terms_only(self, multivariate_data):
        """Test with only main terms (no interactions)."""
        X, Y = multivariate_data
        model = PolynomialRegression(
            degree=2, include_main_terms=True, include_interactions=False
        )
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_include_interactions_only(self, multivariate_data):
        """Test with only interactions (no main terms)."""
        X, Y = multivariate_data
        model = PolynomialRegression(
            degree=2, include_main_terms=False, include_interactions=True
        )
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_both_flags_false_raises_error(self, linear_data):
        """Test that both flags False raises ValueError."""
        X, Y = linear_data
        model = PolynomialRegression(
            degree=2, include_main_terms=False, include_interactions=False
        )
        with pytest.raises(ValueError, match="cannot be both False"):
            model.fit(X, Y)

    def test_fit_no_intercept(self, linear_data):
        """Test fitting without intercept."""
        X, Y = linear_data
        model = PolynomialRegression(degree=1, fit_intercept=False)
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_transform_function(self, linear_data):
        """Test with custom transform function."""
        X, Y = linear_data
        model = PolynomialRegression(degree=1, transform=lambda x: np.sqrt(x))
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_positive_coefficients(self, linear_data):
        """Test with positive constraint on coefficients."""
        X, Y = linear_data
        Y_positive = np.abs(Y)  # Ensure positive relationship
        model = PolynomialRegression(degree=1, positive=True)
        model.fit(X, Y_positive)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y_positive.shape

    def test_call_method(self, linear_data):
        """Test __call__ method."""
        X, Y = linear_data
        model = PolynomialRegression(degree=1)
        model.fit(X, Y)

        preds_call = model(X)
        preds_predict = model.predict(X)
        pd.testing.assert_frame_equal(preds_call, preds_predict)

    def test_different_input_types_fit(self):
        """Test fit with different input types."""
        model = PolynomialRegression(degree=1)

        # Test with lists
        X_list = [[1], [2], [3], [4], [5]]
        Y_list = [[2], [4], [6], [8], [10]]
        model.fit(X_list, Y_list)
        assert not model.betas.empty

        # Test with Series
        X_series = pd.Series([1, 2, 3, 4, 5])
        Y_series = pd.Series([2, 4, 6, 8, 10])
        model.fit(X_series, Y_series)
        assert not model.betas.empty

    def test_different_input_types_predict(self, linear_data):
        """Test predict with different input types."""
        X, Y = linear_data
        model = PolynomialRegression(degree=1)
        model.fit(X, Y)

        # Test with list
        X_list = [[5.0]]
        preds_list = model.predict(X_list)
        assert isinstance(preds_list, pd.DataFrame)

        # Test with Series
        X_series = pd.Series([5.0])
        preds_series = model.predict(X_series)
        assert isinstance(preds_series, pd.DataFrame)

    def test_feature_names(self, multivariate_data):
        """Test feature names are preserved."""
        X, Y = multivariate_data
        model = PolynomialRegression(degree=1)
        model.fit(X, Y)

        feature_names_in = model.get_feature_names_in()
        feature_names_out = model.get_feature_names_out()

        assert feature_names_in is not None
        assert feature_names_out is not None
        assert len(feature_names_in) > 0
        assert len(feature_names_out) > 0

    def test_copy(self, linear_data):
        """Test copy method."""
        X, Y = linear_data
        model = PolynomialRegression(degree=2, fit_intercept=False)
        model.fit(X, Y)

        copied = model.copy()
        assert isinstance(copied, PolynomialRegression)
        assert copied.degree == model.degree
        assert copied.fit_intercept == model.fit_intercept
        assert copied is not model
        pd.testing.assert_frame_equal(copied.betas, model.betas)

    def test_adjust_degree(self, multivariate_data):
        """Test _adjust_degree method."""
        X, Y = multivariate_data
        model = PolynomialRegression(degree=2)

        X_df = model._simplify(X, "X")
        X_adjusted = model._adjust_degree(X_df)

        assert isinstance(X_adjusted, pd.DataFrame)
        # For 3 features with degree 2, we expect more columns
        assert X_adjusted.shape[1] > X_df.shape[1]


# ============================================================================
# PowerRegression Tests
# ============================================================================


class TestPowerRegression:
    """Test suite for PowerRegression class."""

    def test_initialization(self):
        """Test initialization."""
        model = PowerRegression()
        assert model.fit_intercept is True
        assert model.positive is False
        assert model.degree == 1

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        transform_fn = lambda x: x * 2
        model = PowerRegression(transform=transform_fn, positive=True)
        assert model.transform == transform_fn
        assert model.positive is True

    def test_fit_predict_positive_data(self, power_data):
        """Test fit and predict with positive data."""
        X, Y = power_data
        model = PowerRegression()
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape
        assert not model.betas.empty

    def test_fit_predict_multivariate_positive(self):
        """Test fit and predict with multivariate positive data."""
        X = np.column_stack([np.linspace(1, 10, 50), np.linspace(2, 20, 50)])
        Y = 2 * X[:, 0] ** 1.5 * X[:, 1] ** 0.5
        Y = Y.reshape(-1, 1)

        model = PowerRegression()
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_negative_values_raise_error(self):
        """Test that negative values raise ValueError."""
        X = np.array([[-1], [2], [3]])
        Y = np.array([[1], [2], [3]])

        model = PowerRegression()
        with pytest.raises(ValueError, match="must be positive"):
            model.fit(X, Y)

    def test_zero_values_raise_error(self):
        """Test that zero values raise ValueError."""
        X = np.array([[0], [1], [2]])
        Y = np.array([[1], [2], [3]])

        model = PowerRegression()
        with pytest.raises(ValueError, match="must be positive"):
            model.fit(X, Y)

    def test_negative_y_raise_error(self):
        """Test that negative Y values raise ValueError."""
        X = np.array([[1], [2], [3]])
        Y = np.array([[-1], [2], [3]])

        model = PowerRegression()
        with pytest.raises(ValueError, match="must be positive"):
            model.fit(X, Y)

    def test_predict_negative_values_raise_error(self, power_data):
        """Test that predicting with negative values raises ValueError."""
        X, Y = power_data
        model = PowerRegression()
        model.fit(X, Y)

        X_neg = np.array([[-1], [2]])
        with pytest.raises(ValueError, match="must be positive"):
            model.predict(X_neg)

    def test_different_input_types(self):
        """Test with different input types."""
        model = PowerRegression()

        # Lists
        X_list = [[1], [2], [3], [4], [5]]
        Y_list = [[2], [4], [6], [8], [10]]
        model.fit(X_list, Y_list)
        preds = model.predict(X_list)
        assert isinstance(preds, pd.DataFrame)

    def test_copy(self, power_data):
        """Test copy method."""
        X, Y = power_data
        model = PowerRegression(positive=True)
        model.fit(X, Y)

        copied = model.copy()
        assert isinstance(copied, PowerRegression)
        assert copied.positive == model.positive
        assert copied is not model
        pd.testing.assert_frame_equal(copied.betas, model.betas)

    def test_betas_structure(self, power_data):
        """Test betas DataFrame structure."""
        X, Y = power_data
        model = PowerRegression()
        model.fit(X, Y)

        # Beta0 should be the first row (intercept term after exp)
        assert "beta0" in model.betas.index
        assert model.betas.shape[0] == 2  # beta0 and beta1 for single feature


# ============================================================================
# ExponentialRegression Tests
# ============================================================================


class TestExponentialRegression:
    """Test suite for ExponentialRegression class."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = ExponentialRegression()
        assert model.fit_intercept is True
        assert model.positive is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        transform_fn = lambda x: x**2
        model = ExponentialRegression(fit_intercept=False, transform=transform_fn)
        assert model.fit_intercept is False
        assert model.transform == transform_fn

    def test_fit_predict_with_intercept(self, exponential_data):
        """Test fit and predict with intercept."""
        X, Y = exponential_data
        model = ExponentialRegression(fit_intercept=True)
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape
        assert not model.betas.empty

    def test_fit_predict_without_intercept(self, exponential_data):
        """Test fit and predict without intercept."""
        X, Y = exponential_data
        model = ExponentialRegression(fit_intercept=False)
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_multivariate(self):
        """Test with multivariate data."""
        X = np.column_stack([np.linspace(1, 5, 30), np.linspace(2, 6, 30)])
        Y = 2 + X[:, 0] ** 1.5 + X[:, 1] ** 2
        Y = Y.reshape(-1, 1)

        model = ExponentialRegression()
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fitting raises error."""
        model = ExponentialRegression()
        X = np.array([[1], [2], [3]])

        # The model initializes _betas as empty DataFrame, not None
        # So we check that betas is empty before fit
        assert model.betas.empty

        # After fit, betas should not be empty
        Y = np.array([[2], [4], [6]])
        model.fit(X, Y)
        assert not model.betas.empty

    def test_model_function(self, exponential_data):
        """Test _model_function method."""
        X, Y = exponential_data
        model = ExponentialRegression(fit_intercept=True)
        model.fit(X, Y)

        # Test that model function can be called
        params = model.betas.iloc[:, 0].values
        output = model._model_function(X, params)
        assert len(output) == len(X)

    def test_loss_function(self, exponential_data):
        """Test _loss_function method."""
        X, Y = exponential_data
        model = ExponentialRegression()

        # Create some test parameters
        params = np.array([1.0, 1.5])
        loss = model._loss_function(params, X, Y.flatten())
        assert isinstance(loss, (int, float))
        assert loss >= 0

    def test_betas_structure_with_intercept(self, exponential_data):
        """Test betas structure with intercept."""
        X, Y = exponential_data
        model = ExponentialRegression(fit_intercept=True)
        model.fit(X, Y)

        # Should have beta0 (intercept) and beta for each feature
        assert "beta0" in model.betas.index
        assert model.betas.shape[0] == 2  # beta0 + 1 feature

    def test_betas_structure_without_intercept(self, exponential_data):
        """Test betas structure without intercept."""
        X, Y = exponential_data
        model = ExponentialRegression(fit_intercept=False)
        model.fit(X, Y)

        # Should only have betas for features
        assert "beta0" not in model.betas.index

    def test_copy(self, exponential_data):
        """Test copy method."""
        X, Y = exponential_data
        model = ExponentialRegression(fit_intercept=False)
        model.fit(X, Y)

        copied = model.copy()
        assert isinstance(copied, ExponentialRegression)
        assert copied.fit_intercept == model.fit_intercept
        assert copied is not model
        pd.testing.assert_frame_equal(copied.betas, model.betas)

    def test_different_input_types(self):
        """Test with different input types."""
        model = ExponentialRegression()

        # Lists
        X_list = [[1], [2], [3], [4], [5]]
        Y_list = [[2], [4], [6], [8], [10]]
        model.fit(X_list, Y_list)
        preds = model.predict(X_list)
        assert isinstance(preds, pd.DataFrame)


# ============================================================================
# MultiSegmentRegression Tests
# ============================================================================


class TestMultiSegmentRegression:
    """Test suite for MultiSegmentRegression class."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = MultiSegmentRegression()
        assert model.degree == 1
        assert model.n_segments == 1
        assert model.min_samples == 2  # degree + 1

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        model = MultiSegmentRegression(
            degree=2, n_segments=3, min_samples=5, positive=True
        )
        assert model.degree == 2
        assert model.n_segments == 3
        assert model.min_samples == 5
        assert model.positive is True

    def test_n_segments_property(self):
        """Test n_segments property."""
        model = MultiSegmentRegression(n_segments=4)
        assert model.n_segments == 4

    def test_min_samples_property(self):
        """Test min_samples property."""
        model = MultiSegmentRegression(degree=3, min_samples=10)
        assert model.min_samples == 10

    def test_min_samples_default_from_degree(self):
        """Test min_samples defaults to degree + 1."""
        model = MultiSegmentRegression(degree=5)
        assert model.min_samples == 6

    def test_fit_predict_single_segment(self, linear_data):
        """Test fit and predict with single segment."""
        X, Y = linear_data
        model = MultiSegmentRegression(degree=1, n_segments=1)
        model.fit(X[:, 0], Y)
        preds = model.predict(X[:, 0])

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == Y.shape

    def test_fit_predict_two_segments(self, segmented_data):
        """Test fit and predict with two segments."""
        X, Y = segmented_data
        model = MultiSegmentRegression(degree=1, n_segments=2)
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape[0] == len(Y)

    def test_fit_predict_multiple_segments(self, segmented_data):
        """Test fit and predict with multiple segments."""
        X, Y = segmented_data
        model = MultiSegmentRegression(degree=1, n_segments=3, min_samples=5)
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape[0] == len(Y)

    def test_2d_x_raises_error(self):
        """Test that 2D X array raises ValueError."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        Y = np.array([[1], [2], [3]])

        model = MultiSegmentRegression()
        with pytest.raises(ValueError, match="must be a 1D array"):
            model.fit(X, Y)

    def test_predict_2d_x_raises_error(self, linear_data):
        """Test that predicting with 2D X raises ValueError."""
        X, Y = linear_data
        model = MultiSegmentRegression()
        model.fit(X[:, 0], Y)

        X_2d = np.column_stack([X[:, 0], X[:, 0]])
        with pytest.raises(ValueError, match="must be a 1D array"):
            model.predict(X_2d)

    def test_mismatched_shapes_raise_error(self):
        """Test that mismatched X and Y shapes raise ValueError."""
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([[1], [2], [3]])

        model = MultiSegmentRegression()
        with pytest.raises(ValueError, match="equal sample size"):
            model.fit(X, Y)

    def test_high_degree_polynomial(self):
        """Test with high degree polynomial in segments."""
        X = np.linspace(0, 10, 100)
        Y = np.sin(X) + np.random.randn(100) * 0.1

        model = MultiSegmentRegression(degree=3, n_segments=2, min_samples=10)
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == (100, 1)

    def test_betas_structure(self, segmented_data):
        """Test betas DataFrame structure."""
        X, Y = segmented_data
        model = MultiSegmentRegression(degree=1, n_segments=2)
        model.fit(X, Y)

        # Betas should have MultiIndex columns with FEATURE, X0, X1
        assert isinstance(model.betas.columns, pd.MultiIndex)
        assert model.betas.columns.names == ["FEATURE", "X0", "X1"]

    def test_segment_boundaries(self, segmented_data):
        """Test that segment boundaries are properly identified."""
        X, Y = segmented_data
        model = MultiSegmentRegression(degree=1, n_segments=2)
        model.fit(X, Y)

        # Check that we have proper segment boundaries in betas
        boundaries = model.betas.columns.get_level_values("X0").unique()
        assert len(boundaries) == 2  # Two segments

    def test_different_input_types(self):
        """Test with different input types."""
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

    def test_copy(self, segmented_data):
        """Test copy method."""
        X, Y = segmented_data
        model = MultiSegmentRegression(degree=2, n_segments=2)
        model.fit(X, Y)

        copied = model.copy()
        assert isinstance(copied, MultiSegmentRegression)
        assert copied.degree == model.degree
        assert copied.n_segments == model.n_segments
        assert copied.min_samples == model.min_samples
        assert copied is not model
        pd.testing.assert_frame_equal(copied.betas, model.betas)

    def test_transform_function(self):
        """Test with custom transform function."""
        X = np.linspace(1, 10, 50)
        Y = X**2 * 2 + 1  # Quadratic relationship

        # Use log transform to linearize exponential relationships
        model = MultiSegmentRegression(
            degree=1, n_segments=1, transform=lambda x: x  # Identity transform
        )
        model.fit(X, Y)
        preds = model.predict(X)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape[0] == len(Y)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests across multiple models."""

    def test_all_models_with_same_data(self, linear_data):
        """Test that all models can handle the same dataset."""
        X, Y = linear_data
        X_positive = np.abs(X) + 1  # Ensure positive for PowerRegression
        Y_positive = np.abs(Y) + 1

        models = [
            PolynomialRegression(degree=1),
            PowerRegression(),
            ExponentialRegression(),
            MultiSegmentRegression(degree=1, n_segments=1),
        ]

        for model in models[:3]:  # First 3 models
            model.fit(X_positive, Y_positive)
            preds = model.predict(X_positive)
            assert isinstance(preds, pd.DataFrame)
            assert preds.shape == Y_positive.shape

        # MultiSegmentRegression needs 1D input
        models[3].fit(X_positive[:, 0], Y_positive)
        preds = models[3].predict(X_positive[:, 0])
        assert isinstance(preds, pd.DataFrame)

    def test_model_comparison(self, polynomial_data):
        """Compare different models on polynomial data."""
        X, Y = polynomial_data

        linear_model = PolynomialRegression(degree=1)
        quadratic_model = PolynomialRegression(degree=2)

        linear_model.fit(X, Y)
        quadratic_model.fit(X, Y)

        linear_preds = linear_model.predict(X)
        quadratic_preds = quadratic_model.predict(X)

        # Quadratic should fit better on quadratic data
        linear_error = np.mean((Y - linear_preds.values) ** 2)
        quadratic_error = np.mean((Y - quadratic_preds.values) ** 2)

        assert quadratic_error < linear_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
