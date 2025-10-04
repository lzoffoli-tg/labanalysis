import pytest
import numpy as np
import pandas as pd
import sys
from os.path import dirname, abspath, join

sys.path += [join(dirname(dirname(abspath(__file__))), "src")]
from labanalysis import (
    BaseRegression,
    PolynomialRegression,
    PowerRegression,
    ExponentialRegression,
    MultiSegmentRegression,
)


@pytest.fixture
def sample_data():
    X = np.atleast_2d(np.linspace(1, 10, 10)).T
    X = np.concatenate([X, X**0.5], axis=1)
    Y = 2 * X + 3
    return X, Y


def test_base_regression(sample_data):
    X, Y = sample_data
    model = BaseRegression()
    assert isinstance(model.betas, pd.DataFrame)
    assert callable(model.transform)
    model_copy = model.copy()
    assert isinstance(model_copy, BaseRegression)


def test_polynomial_regression(sample_data):
    X, Y = sample_data
    model = PolynomialRegression(degree=2)
    model.fit(X, Y)
    preds = model.predict(X)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == len(X)
    assert isinstance(model.betas, pd.DataFrame)
    assert model.degree == 2
    model_copy = model.copy()
    assert isinstance(model_copy, PolynomialRegression)


def test_power_regression(sample_data):
    X, Y = sample_data
    model = PowerRegression()
    model.fit(X, Y)
    preds = model.predict(X)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == len(X)
    assert isinstance(model.betas, pd.DataFrame)
    model_copy = model.copy()
    assert isinstance(model_copy, PowerRegression)


def test_exponential_regression(sample_data):
    X, Y = sample_data
    model = ExponentialRegression()
    model.fit(X, Y)
    preds = model.predict(X)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == len(X)
    assert isinstance(model.betas, pd.DataFrame)
    model_copy = model.copy()
    assert isinstance(model_copy, ExponentialRegression)


def test_multisegment_regression(sample_data):
    X, Y = sample_data
    model = MultiSegmentRegression(degree=1, n_segments=2)
    model.fit(X[:, 0], Y)
    preds = model.predict(X[:, 0])
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == len(X)
    assert isinstance(model.betas, pd.DataFrame)
    assert model.n_segments == 2
    assert model.min_samples == 2
    model_copy = model.copy()
    assert isinstance(model_copy, MultiSegmentRegression)
