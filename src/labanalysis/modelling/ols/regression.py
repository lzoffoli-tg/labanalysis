"""regression module"""

#! IMPORTS

import copy
import itertools as it
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

__all__ = [
    "BaseRegression",
    "PolynomialRegression",
    "MultiSegmentRegression",
    "PowerRegression",
    "ExponentialRegression",
]


#! CLASSES


class BaseRegression(LinearRegression):
    """
    Base class for regression models with support for input transformation
    and intercept control.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.

    transform : Callable, default=lambda x: x
        A callable function applied element-wise to each input value of X.
        Useful for applying transformations such as normalization, log-scaling, etc.

    positive : bool, default=False
        If True, forces the coefficients to be non-negative.

    Attributes
    ----------
    betas : pandas.DataFrame
        Estimated model coefficients, including the intercept if applicable.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        transform: Callable = lambda x: x,
        positive: bool = False,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            positive=positive,
        )
        self._transform = transform
        self._betas = pd.DataFrame()
        self._names_in = None
        self._names_out = None

    def fit(self, xarr, yarr):
        """
        Fit the polynomial regression model to the given data.

        Parameters
        ----------
        xarr : array-like
            Input features.

        yarr : array-like
            Target values.

        Returns
        -------
        self : PolynomialRegression
            The fitted model instance.
        """
        return NotImplementedError

    def predict(self, xarr):
        """
        Predict target values using the fitted polynomial regression model.

        Parameters
        ----------
        xarr : array-like
            Input features.

        Returns
        -------
        y_pred : pandas.DataFrame
            Predicted target values.
        """
        return NotImplementedError

    def __call__(self, xarr):
        """
        Predict target values using the fitted polynomial regression model.

        Parameters
        ----------
        xarr : array-like
            Input features.

        Returns
        -------
        y_pred : pandas.DataFrame
            Predicted target values.
        """
        return self.predict(xarr)

    @property
    def transform(self):
        """
        Returns the transformation function applied to input data.

        Returns
        -------
        Callable
            The transformation function.
        """
        return self._transform

    @property
    def betas(self):
        """
        Returns the estimated coefficients of the model.

        Returns
        -------
        pandas.DataFrame
            Model coefficients (betas).
        """
        return self._betas

    def get_feature_names_in(self):
        """
        Returns the names of input features seen during fitting.

        Returns
        -------
        list or None
            Input feature names.
        """
        return self._names_in

    def get_feature_names_out(self):
        """
        Returns the names of output features seen during fitting.

        Returns
        -------
        list or None
            Output feature names.
        """
        return self._names_out

    def _simplify(self, vec, label=""):
        """
        Converts various input types into a standardized pandas DataFrame
        with labeled columns.

        Parameters
        ----------
        vec : array-like, scalar, pandas object
            Input data to be simplified.

        label : str, default=""
            Prefix used for column names.

        Returns
        -------
        pandas.DataFrame
            Simplified representation of the input.
        """

        def simplify_array(v: np.ndarray, l: str):
            if v.ndim == 1:
                d = np.atleast_2d(v).T
            elif v.ndim == 2:
                d = v
            else:
                raise ValueError(v)
            cols = [f"{l}{i}" for i in range(d.shape[1])]
            return pd.DataFrame(d.astype(float), columns=cols)

        if isinstance(vec, pd.DataFrame):
            return vec.astype(float)
        if isinstance(vec, pd.Series):
            return pd.DataFrame(vec).astype(float)
        if isinstance(vec, list):
            return simplify_array(np.array(vec), label)
        if isinstance(vec, np.ndarray):
            return simplify_array(vec, label)
        if np.isscalar(vec):
            return simplify_array(np.array([vec]), label)
        raise NotImplementedError(vec)

    def copy(self):
        """
        Creates a deep copy of the model including fitted parameters.

        Returns
        -------
        BaseRegression
            A copy of the current model instance with learned attributes.
        """
        new_model = self.__class__(
            fit_intercept=self.fit_intercept,  # type: ignore
            transform=self.transform,
            positive=self.positive,  # type: ignore
        )
        new_model._betas = copy.deepcopy(self._betas)
        new_model._names_in = copy.deepcopy(self._names_in)
        new_model._names_out = copy.deepcopy(self._names_out)
        return new_model


class PolynomialRegression(BaseRegression):
    """
    Polynomial regression model of the form:
        Y = b0 + b1 * fn(X)^1 + ... + bn * fn(X)^n + e

    Parameters
    ----------
    degree : int, default=1
        The degree of the polynomial used to expand the input features.

    fit_intercept : bool, default=True
        Whether to include an intercept term in the model.

    transform : Callable, default=lambda x: x
        A transformation function applied element-wise to the input features.

    positive : bool, default=False
        If True, forces the estimated coefficients to be non-negative.

    Attributes
    ----------
    degree : int
        The polynomial degree used for feature expansion.
    """

    def __init__(
        self,
        degree: int = 1,
        fit_intercept: bool = True,
        transform: Callable = lambda x: x,
        positive: bool = False,
    ):
        super().__init__(
            fit_intercept=fit_intercept, transform=transform, positive=positive
        )
        self._degree = degree

    @property
    def degree(self):
        """
        Returns the polynomial degree used in the model.

        Returns
        -------
        int
            The degree of the polynomial.
        """
        return self._degree

    def _adjust_degree(self, xarr: pd.DataFrame):
        """
        Expands the input features to the specified polynomial degree.

        Parameters
        ----------
        xarr : pandas.DataFrame
            Input features to be expanded.

        Returns
        -------
        pandas.DataFrame
            Transformed features including polynomial terms.
        """
        feats = []
        for i in np.arange(self.degree):
            new = xarr.copy() ** (i + 1)
            lbl = "" if i == 0 else f"^{i+1}"
            new.columns = new.columns.map(lambda x: x + lbl)
            feats += [new]
        feats = pd.concat(feats, axis=1)
        return feats

    def fit(self, xarr, yarr):
        """
        Fit the polynomial regression model to the given data.

        Parameters
        ----------
        xarr : array-like
            Input features.

        yarr : array-like
            Target values.

        Returns
        -------
        self : PolynomialRegression
            The fitted model instance.
        """
        Y = self._simplify(yarr, "Y")
        self._names_out = Y.columns.tolist()
        X = self._simplify(xarr, "X").map(self.transform)
        X = self._adjust_degree(X)
        self._names_in = X.columns.tolist()
        fitted = super().fit(X, Y)
        beta_0 = np.atleast_2d(fitted.intercept_).T
        beta_n = np.atleast_2d(fitted.coef_)
        coefs = [beta_0, fitted.coef_]
        coefs = np.concatenate([beta_0, beta_n], axis=1).T
        self._betas = pd.DataFrame(
            data=coefs,
            index=[f"beta{i}" for i in range(coefs.shape[0])],
            columns=self.get_feature_names_out(),
        )
        return self

    def predict(self, xarr):
        """
        Predict target values using the fitted polynomial regression model.

        Parameters
        ----------
        xarr : array-like
            Input features.

        Returns
        -------
        y_pred : pandas.DataFrame
            Predicted target values.
        """
        X = self._simplify(xarr).map(self.transform)
        X = self._adjust_degree(X)
        return pd.DataFrame(
            data=super().predict(X.values),
            columns=self.get_feature_names_out(),
            index=X.index,
        )

    def copy(self):
        """
        Creates a deep copy of the model including fitted parameters.

        Returns
        -------
        PolynomialRegression
            A copy of the current model instance with learned attributes.
        """
        new_model = self.__class__(
            degree=self._degree,
            fit_intercept=self.fit_intercept,  # type: ignore
            transform=self.transform,
            positive=self.positive,  # type: ignore
        )
        new_model._betas = copy.deepcopy(self._betas)
        new_model._names_in = copy.deepcopy(self._names_in)
        new_model._names_out = copy.deepcopy(self._names_out)
        return new_model


class PowerRegression(PolynomialRegression):
    """
    Power regression model of the form:
    Y = b0 * X1**b1 * X2**b2 * ... * Xn**bn + e

    This model supports multivariate input features and multi-output targets.
    It fits a log-log linear model to estimate the power coefficients.

    Parameters
    ----------
    transform : Callable, default=lambda x: x
        A callable function applied element-wise to each input value of X
        before applying the power transformation.

    positive : bool, default=False
        If True, forces the estimated coefficients to be non-negative.
        This option is only supported for dense arrays.

    Attributes
    ----------
    betas : pandas.DataFrame
        A DataFrame containing the estimated regression coefficients.
        The first row corresponds to b0 (intercept), and the remaining rows
        correspond to the power coefficients for each input feature.
    """

    def __init__(
        self,
        transform: Callable = lambda x: x,
        positive: bool = False,
    ):
        super().__init__(
            fit_intercept=True,
            transform=transform,
            positive=positive,
            degree=1,
        )

    def fit(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        X = self._simplify(xarr, "X")
        Y = self._simplify(yarr, "Y")

        K = X.map(self.transform)
        if (K <= 0).any().any() or (Y <= 0).any().any():
            raise ValueError(
                "All values in X and Y must be positive for log transformation."
            )

        Xt = K.map(np.log)
        Yt = Y.map(np.log)

        fitted = super().fit(Xt, Yt)

        b0 = np.atleast_2d(np.exp(fitted.intercept_))
        b1 = np.atleast_2d(fitted.coef_)
        if Y.shape[1] == 1:
            b1 = b1.T

        coefs = np.vstack([b0, b1])
        index = ["beta0"] + [f"beta{i+1}" for i in range(b1.shape[0])]
        self._betas = pd.DataFrame(coefs, index=index, columns=Y.columns)

        self._names_in = X.columns.tolist()
        self._names_out = Y.columns.tolist()
        return self

    def predict(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        X = self._simplify(xarr, "X")
        K = X.map(self.transform)
        K.insert(0, "beta0", 1)

        if (K <= 0).any().any():
            raise ValueError(
                "All values in X must be positive for power transformation."
            )

        y_pred = np.prod(
            K.values[:, :, np.newaxis] ** self.betas.values[np.newaxis, :, :],
            axis=1,
        )
        return pd.DataFrame(y_pred, columns=self._names_out, index=X.index)

    def copy(self):
        """
        Creates a deep copy of the model including fitted parameters.

        Returns
        -------
        PowerRegression
            A copy of the current model instance with learned attributes.
        """
        new_model = self.__class__(
            transform=self.transform,
            positive=self.positive,  # type: ignore
        )
        new_model._betas = copy.deepcopy(self._betas)
        new_model._names_in = copy.deepcopy(self._names_in)
        new_model._names_out = copy.deepcopy(self._names_out)
        return new_model


class ExponentialRegression(BaseRegression):
    """
    Exponential regression model of the form:

        Y = b0 + X1**b1 + X2**b2 + ... + Xn**bn    if fit_intercept=True
        Y = X1**b1 + X2**b2 + ... + Xn**bn         if fit_intercept=False

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term in the model.

    transform : Callable, default=lambda x: x
        A transformation function applied element-wise to the input features.

    Attributes
    ----------
    betas : pandas.DataFrame
        Estimated model coefficients for each output feature.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        transform: Callable = lambda x: x,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            transform=transform,
            positive=False,
        )

    def _model_function(self, X: np.ndarray, params: np.ndarray):
        """
        Compute the model output given input features and parameters.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        params : np.ndarray
            Model parameters (intercept and exponents).

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.fit_intercept:  # type: ignore
            b0 = params[0]
            exponents = params[1:]
            out = b0 + np.sum(X**exponents, axis=1)
        else:
            out = np.sum(X**params, axis=1)
        return np.array(out).astype(float).flatten()

    def _loss_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        Compute the loss (sum of squared errors) for given parameters.

        Parameters
        ----------
        params : np.ndarray
            Model parameters.

        X : np.ndarray
            Input features.

        y : np.ndarray
            Target values.

        Returns
        -------
        float
            Sum of squared errors.
        """
        y_pred = self._model_function(X, params)
        return np.sum((y - y_pred) ** 2)

    def fit(self, xarr, yarr):
        """
        Fit the exponential regression model to the given data.

        Parameters
        ----------
        xarr : array-like
            Input features.

        yarr : array-like
            Target values.

        Returns
        -------
        self : ExponentialRegression
            The fitted model instance.
        """
        X = self._simplify(xarr, "X").map(self.transform)
        Y = self._simplify(yarr, "Y")
        self._names_in = X.columns.tolist()
        self._names_out = Y.columns.tolist()

        beta_matrix = []
        for col in Y.columns:
            y = Y[col].to_numpy().flatten()
            n_features = X.shape[1]
            initial_params = (
                np.array([0.0] + [1.0] * n_features)
                if self.fit_intercept  # type: ignore
                else np.array([1.0] * n_features)
            )
            result = minimize(
                self._loss_function, initial_params, args=(X.values, y), method="BFGS"
            )
            beta_matrix.append(result.x)

        beta_matrix = np.array(beta_matrix).T
        index = self._names_in
        if self.fit_intercept:  # type: ignore
            index = ["beta0"] + index
        self._betas = pd.DataFrame(
            beta_matrix,
            index=index,
            columns=self._names_out,
        )
        return self

    def predict(self, xarr):
        """
        Predict target values using the fitted exponential regression model.

        Parameters
        ----------
        xarr : array-like
            Input features.

        Returns
        -------
        y_pred : pandas.DataFrame
            Predicted values.
        """
        if self._betas is None:
            raise ValueError("Model must be fitted before prediction.")

        X = self._simplify(xarr, "X").map(self.transform)
        predictions = {}
        for col in self._betas.columns:
            params = self._betas[col].to_numpy()
            predictions[col] = self._model_function(X.to_numpy(), params)

        return pd.DataFrame(predictions, index=X.index)

    def copy(self):
        """
        Creates a deep copy of the model including fitted parameters.

        Returns
        -------
        ExponentialRegression
            A copy of the current model instance with learned attributes.
        """
        new_model = self.__class__(
            fit_intercept=self.fit_intercept,  # type: ignore
            transform=self.transform,
        )
        new_model._betas = copy.deepcopy(self._betas)
        new_model._names_in = copy.deepcopy(self._names_in)
        new_model._names_out = copy.deepcopy(self._names_out)
        return new_model


class MultiSegmentRegression(PolynomialRegression):
    """
    Multi-segment polynomial regression model.

    This model fits multiple polynomial segments to the input data, each defined
    over a specific range of the x-axis. It automatically selects the best
    segmentation based on minimizing the sum of squared errors (SSE).

    Parameters
    ----------
    degree : int, default=1
        Degree of the polynomial used in each segment.

    n_segments : int, default=1
        Number of segments to divide the input space into.

    min_samples : int or None, default=None
        Minimum number of unique x-values required in each segment.
        If None, defaults to degree + 1.

    transform : Callable, default=lambda x: x
        Transformation function applied to input features.

    positive : bool, default=False
        If True, forces the estimated coefficients to be non-negative.

    Attributes
    ----------
    betas : pandas.DataFrame
        Estimated coefficients for each segment and output feature.
    """

    def __init__(
        self,
        degree: int = 1,
        n_segments: int = 1,
        min_samples: int | None = None,
        transform: Callable = lambda x: x,
        positive: bool = False,
    ):
        super().__init__(
            degree=degree, transform=transform, fit_intercept=False, positive=positive
        )
        self._n_segments = n_segments
        self._min_samples = min_samples if min_samples is not None else self.degree + 1

    @property
    def n_segments(self):
        """
        Returns the number of segments used in the model.

        Returns
        -------
        int
            Number of segments.
        """
        return self._n_segments

    @property
    def min_samples(self):
        """
        Returns the minimum number of unique x-values required per segment.

        Returns
        -------
        int
            Minimum number of samples per segment.
        """
        return self._min_samples

    def fit(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        """
        Fit the multi-segment polynomial regression model.

        Parameters
        ----------
        xarr : array-like
            Input features (must be 1D).

        yarr : array-like
            Target values.

        Returns
        -------
        self : MultiSegmentRegression
            The fitted model instance.
        """
        X = self._simplify(xarr, "X")
        Y = self._simplify(yarr, "Y")

        if X.shape[1] != 1:
            raise ValueError("xarr must be a 1D array")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("xarr and yarr must have equal sample size")

        unique_x = np.unique(X.values.flatten()).astype(float)
        n_unique = len(unique_x)
        X = X.map(self.transform)

        # Generate all valid segment combinations
        combs = []
        for i in np.arange(1, self.n_segments):
            start = self.min_samples * i
            stop = n_unique - self.min_samples * (self.n_segments - i)
            combs += [np.arange(start, stop)]
        combs = list(it.product(*combs))
        combs = [i for i in combs if np.all(np.diff(i) >= self.min_samples)]

        # Build segment boundaries
        combs = (
            np.zeros((len(combs), 1)),
            np.atleast_2d(combs),
            np.ones((len(combs), 1)) * (n_unique - 1),
        )
        combs = np.hstack(combs).astype(int)

        # Evaluate each segmentation
        betas_list = []
        sses_list = []
        for comb in combs:
            combination_sse = 0
            combination_betas_list = []
            y0 = 0
            x0 = np.atleast_1d(X.values[0]).astype(float)

            for i, (i0, i1) in enumerate(zip(comb[:-1], comb[1:])):
                unq_vals = unique_x[np.arange(i0, i1 + 1)]
                index = [np.where(X.values[:, 0] == j)[0] for j in unq_vals]
                index = np.concatenate(index)
                ymat = Y.iloc[index] - y0
                xmat = self._adjust_degree(X.iloc[index] - x0)
                if i == 0:
                    xmat.insert(0, "Intercept", np.ones((xmat.shape[0],)))

                # Estimate coefficients
                bs = (xmat @ np.linalg.inv(xmat.T @ xmat)).T @ ymat

                # Compute error
                ypred = (xmat @ bs.values + y0).values
                ytrue = ymat.values + y0
                sse = float(((ytrue - ypred) ** 2).sum())
                combination_sse += sse

                # Format coefficients
                if i != 0:
                    bs.index = pd.Index([f"beta{j + 1}" for j in range(bs.shape[0])])
                    bs.loc["beta0", bs.columns] = y0
                else:
                    bs.index = pd.Index([f"beta{j}" for j in range(bs.shape[0])])
                bs.loc["alpha0", bs.columns] = x0
                bs.sort_index(inplace=True)
                r0 = -np.inf if i0 == 0 else x0
                r1 = +np.inf if i1 == (n_unique - 1) else unq_vals[-1]
                bs.columns = pd.MultiIndex.from_product(
                    iterables=[ymat.columns.tolist(), [r0], [r1]],  # type: ignore
                    names=["FEATURE", "X0", "X1"],
                )
                combination_betas_list += [bs]

                y0 = ypred[-1]
                x0 = r1

            combination_betas = pd.concat(combination_betas_list, axis=1)
            combination_betas.sort_index(axis=1, inplace=True)
            betas_list += [combination_betas]
            sses_list += [combination_sse]

        # Select best segmentation
        index = np.argmin(sses_list)
        self._betas = betas_list[index]

        return self

    def predict(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        """
        Predict target values using the fitted multi-segment regression model.

        Parameters
        ----------
        xarr : array-like
            Input features (must be 1D).

        Returns
        -------
        y_pred : pandas.DataFrame
            Predicted values.
        """
        X = self._simplify(xarr, "X").map(self.transform)
        if X.shape[1] != 1:
            raise ValueError("xarr must be a 1D array")

        feats = self.betas.columns.to_frame().FEATURE.values.astype(str)
        feats = np.unique(feats)
        Y = pd.DataFrame(index=X.index, columns=feats)

        for feat, i0, i1 in self.betas.columns:
            idx = np.where((X.values >= i0) & (X.values <= i1))[0]
            coefs = self.betas[[(feat, i0, i1)]].values.astype(float)
            x0 = coefs[0]
            betas = coefs[1:]
            xmat = self._adjust_degree(X.iloc[idx] - x0)
            xmat.insert(0, "Intercept", np.ones((xmat.shape[0],)))
            Y.loc[X.index[idx], [feat]] = (xmat @ betas).values

        return Y

    def copy(self):
        """
        Creates a deep copy of the model including fitted parameters.

        Returns
        -------
        MultiSegmentRegression
            A copy of the current model instance with learned attributes.
        """
        new_model = self.__class__(
            degree=self._degree,
            transform=self._transform,
            n_segments=self._n_segments,
            min_samples=self._min_samples,
            positive=self.positive,  # type: ignore
        )
        new_model._betas = copy.deepcopy(self._betas)
        new_model._names_in = copy.deepcopy(self._names_in)
        new_model._names_out = copy.deepcopy(self._names_out)
        return new_model
