"""
2D and 3D space objects extrapolated by least squares approaches

Classes
---------
Line2D
    line object in a 2D space having general form:
            A * x + B * y + C = 0

Line3D
    line object in a 3D space having general form:
            A * x + B * y + C * z + D = 0
"""

#! IMPORTS


import copy
from math import pi as PI
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import sympy

__all__ = ["GeometricObject", "Line2D", "Line3D", "Circle", "Ellipse"]


#! CLASSES


@runtime_checkable
class GeometricObject(Protocol):
    """
    General geometric object class. This object is thought to be extended
    according to the geometric object of interest.

    The standard use case is to extend this class and implement at least the
    method '_expand_dimensions'". Optionally also the methods fit and predict
    might be re-implemented to make better specify the input data.
    """

    _betas: dict[str, float]
    _general_equation: sympy.Eq
    _dimensions: list[str]
    _has_intercept: bool

    @property
    def has_intercept(self):
        """return True if the model has intercept"""
        return self._has_intercept

    @property
    def betas(self):
        """return the betas coefficients"""
        return self._betas

    @property
    def coefs(self):
        """return the labels of the coefficients in the general equation"""
        symbols = [str(i) for i in self.general_equation.free_symbols]
        return sorted([i for i in symbols if i not in self.dimensions])

    @property
    def general_equation(self):
        """return the equation of the model"""
        return self._general_equation

    @property
    def fitted_equation(self):
        """return the equation with the appropriate coefficients"""
        if not self.is_fitted():
            msg = "the fit method has to be used before 'fitted_equation'"
            raise ValueError(msg)
        return self.general_equation.subs(self.betas)

    @property
    def dimensions(self):
        """get the dimensions of the shape"""
        return self._dimensions

    def _simplify(
        self,
        vec: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        label: str = "",
    ):
        """
        internal method to format the entries in the constructor and call
        methods.

        Parameters
        ----------
        vec: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None
            the data to be formatter

        label: str
            in case an array is provided, the label is used to define the
            columns of the output DataFrame.

        Returns
        -------
        dfr: pd.DataFrame
            the data formatted as DataFrame.
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
            out = vec.astype(float)
        elif isinstance(vec, pd.Series):
            out = pd.DataFrame(vec).T.astype(float)
        elif isinstance(vec, list):
            out = simplify_array(np.array(vec), label)
        elif isinstance(vec, np.ndarray):
            out = simplify_array(vec, label)
        elif np.isreal(vec):
            out = simplify_array(np.array([vec]), label)
        else:
            raise NotImplementedError(vec)
        return out.values.astype(float).flatten()

    def is_fitted(self):
        """check if the current objects has the required beta coefficients"""
        return len(self.betas) > 0

    def _check_symbols(self):
        """check the symbols provided"""
        target_symbols = list(self.general_equation.free_symbols)
        target_symbols = np.array(target_symbols).astype(str)
        available_symbols = self.dimensions + list(self.betas.keys())
        missing = [i for i in target_symbols if i not in available_symbols]
        if len(missing) > 0:
            msg = f"General Equation: {self.general_equation}\n"
            msg += f"Missing symbols: {missing}"
            raise ValueError(msg)

    def __str__(self):
        if self.is_fitted():
            return str(self.fitted_equation.args[0].__str__())
        return str(self.general_equation.args[0].__str__())

    def __repr__(self):
        return self.__str__()

    def __init__(
        self,
        general_equation: sympy.Equality,
        dimensions: list[str],
        has_intercept: bool = True,
    ):
        if not isinstance(general_equation, sympy.Equality):
            raise ValueError(f"general_equation must be {sympy.Equality}")
        self._general_equation = general_equation

        if not isinstance(dimensions, list):
            raise ValueError(f"dimensions must be {list} of {str}")
        if not all([isinstance(i, str) for i in dimensions]):
            raise ValueError(f"dimensions must be {list} of {str}")
        self._dimensions = dimensions

        if not isinstance(has_intercept, bool):
            raise ValueError("'has_intercept' must be a bool")
        self._has_intercept = has_intercept

        self._betas = {}

        # check ghe proper general equation
        """
        if len(self.coefs) != len(list(self.general_equation.args[0].args)):
            msg = "dimensions does not match with the provided number of "
            msg += "coefficients."
            raise ValueError(msg)
        """

    def _expand_dimensions(self, **dimensions: np.ndarray) -> np.ndarray: ...

    def copy(self):
        """create a copy of the current object."""
        return copy.deepcopy(self)

    def predict(
        self,
        **known_dimensions: (
            np.ndarray | list[float] | int | float | pd.Series | pd.DataFrame
        ),
    ):
        """
        predict the unknown dimensions giving all but one other dimensions.

        Parameters
        ----------
        known_dimensions: key-valued ArrayLike
            the input dimensions.

        Returns
        -------
        predicted_dimension: ArrayLike
            the value corresponding to each known dimension pair.
        """
        # check the status of the object
        if not self.is_fitted():
            raise ValueError("this object must be fitted before using 'predict'")
        self._check_symbols()

        # check the input dimensions
        known_dims = {}
        for key, val in known_dimensions.items():
            if key not in self.dimensions:
                raise ValueError(f"valid dimensions are {self.dimensions}")
            known_dims[key] = self._simplify(val)
        if np.sum(np.diff([len(i) for i in list(known_dims.values())])) != 0:
            msg = "All known dimensions must have the same number of samples."
            raise ValueError(msg)
        keys = list(known_dims.keys())
        samples = np.array(list(known_dims.values())).T

        # get the unknown dimension
        unknown_dim = [i for i in self.dimensions if i not in list(known_dims.keys())]
        if len(unknown_dim) != 1:
            raise ValueError("Only one unknown dimension is possible.")
        unknown_dim = unknown_dim[0]

        # fit the values
        out = []
        for vals in samples:
            eq = self.fitted_equation.subs({i: v for i, v in zip(keys, vals)})
            out += [sympy.solve(eq, unknown_dim)]
        return np.squeeze(out).astype(float)

    def fit(
        self,
        **dimensions_samples: (
            np.ndarray | list[float] | int | float | pd.Series | pd.DataFrame
        ),
    ):
        """
        fit the current object to the provided dimensions

        Parameters
        ----------
        dimensions_samples: key-valued ArrayLike
            the input dimensions.
        """

        # check the inputs
        keys = list(dimensions_samples.keys())
        if not all([i in keys for i in self.dimensions]):
            msg = f"The following dimensions must be provided: {self.dimensions}"
            raise ValueError(msg)
        vals = [self._simplify(j, i) for i, j in dimensions_samples.items()]
        if not all([len(i) for i in vals]):
            msg = "all provided dimensions must have the same number of samples."
            raise ValueError(msg)

        # get the design matrix
        mat = self._expand_dimensions(**{i: v for i, v in zip(keys, vals)})

        # consider the intercept
        if self.has_intercept:
            mat = np.hstack([mat, np.ones((mat.shape[0], 1))])

        # np.hstack performs a loop over all samples and creates
        # a row in J for each x,y,z sample:
        offset = -mat[:, -1]
        mat = mat[:, :-1]
        betas = (np.linalg.pinv(mat.T @ mat) @ mat.T @ offset).astype(float)
        betas = np.round(np.concatenate([betas, [1]]), 15)
        self._betas = {i: float(v) for i, v in zip(self.coefs, betas)}

        return self


class Line2D(GeometricObject):
    """
    line object in a 2D space having general form:

            A * x + B * y + C = 0

    Parameters
    ----------
    has_intercept: bool (default = True)
        if False the C coefficient is excluded.
    """

    def __init__(self, has_intercept: bool = True):
        if not isinstance(has_intercept, bool):
            raise ValueError("'has_intercept' must be True or False")

        if has_intercept:
            x, y, A, B, C = sympy.symbols("x,y,A,B,C")
            equation = sympy.Equality(A * x + B * y + C, 0)
        else:
            x, y, A, B = sympy.symbols("x,y,A,B")
            equation = sympy.Equality(A * x + B * y, 0)

        super().__init__(
            general_equation=equation,  # type: ignore
            dimensions=["x", "y"],
            has_intercept=has_intercept,
        )

    @property
    def domains(self):
        """return the value limits accepted for each dimension"""
        return {
            "x": (-np.inf, np.inf),
            "y": (-np.inf, np.inf),
        }

    def _expand_dimensions(self, **dimensions: np.ndarray):
        """prepare the design matrix"""
        out = [self._simplify(dimensions[i], i) for i in ["x", "y"]]
        return np.vstack(np.atleast_2d(*out)).T

    def fit(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int,
        y: np.ndarray | pd.Series | list[float | int] | float | int,
    ):
        """
        fit the model with the required data

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int
            the y-axis coordinates

        Returns
        -------
        fitted: Line2D
            a fitted Line2D object
        """
        return super().fit(x=x, y=y)

    def predict(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        y: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
    ):
        """
        get predictions of the not provided coordinate

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the y-axis coordinates

        Returns
        -------
        out: np.ndarray
            the coordinates of the axis not provided

        Note
        ----
        only the 'x' or 'y' argument has to be provided.
        """
        if x is None and y is not None:
            return super().predict(y=y)
        if y is None and x is not None:
            return super().predict(x=x)
        raise ValueError("Just 'x' or 'y' must be not None.")


class Line3D(GeometricObject):
    """
    line object in a 3D space having general form:

            A * x + B * y + C * z + D = 0

    Parameters
    ----------
    has_intercept: bool (default = True)
        if False the D coefficient is excluded.
    """

    def __init__(self, has_intercept: bool = True):
        if not isinstance(has_intercept, bool):
            raise ValueError("'has_intercept' must be True or False")

        if has_intercept:
            x, y, z, A, B, C, D = sympy.symbols("x,y,z,A,B,C,D")
            equation = sympy.Equality(A * x + B * y + C * z + D, 0)
        else:
            x, y, z, A, B, C = sympy.symbols("x,y,z,A,B,C")
            equation = sympy.Equality(A * x + B * y + C * z, 0)

        super().__init__(
            general_equation=equation,  # type: ignore
            dimensions=["x", "y", "z"],
            has_intercept=has_intercept,
        )

    @property
    def domains(self):
        """return the value limits accepted for each dimension"""
        return {
            "x": (-np.inf, np.inf),
            "y": (-np.inf, np.inf),
            "z": (-np.inf, np.inf),
        }

    def _expand_dimensions(self, **dimensions: np.ndarray):
        """prepare the design matrix"""
        out = [self._simplify(dimensions[i], i) for i in ["x", "y", "z"]]
        return np.vstack(np.atleast_2d(*out)).T

    def fit(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int,
        y: np.ndarray | pd.Series | list[float | int] | float | int,
        z: np.ndarray | pd.Series | list[float | int] | float | int,
    ):
        """
        fit the model with the required data

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int
            the y-axis coordinates

        z: np.ndarray | pd.Series | list[float | int] | float | int
            the z-axis coordinates

        Returns
        -------
        fitted: Line3D
            a fitted Line3D object
        """
        return super().fit(x=x, y=y, z=z)

    def predict(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        y: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        z: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
    ):
        """
        get predictions of the not provided coordinate

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the y-axis coordinates

        z: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the z-axis coordinates

        Returns
        -------
        out: np.ndarray
            the coordinates of the axis not provided

        Note
        ----
        only 2 of the three axes have to be provided.
        """
        if x is None and y is not None and z is not None:
            return super().predict(y=y, z=z)
        if y is None and x is not None and z is not None:
            return super().predict(x=x, z=z)
        if z is None and x is not None and y is not None:
            return super().predict(x=x, y=y)
        raise ValueError("Just one of 'x', 'y' or 'z' must be not None.")


class Circle(GeometricObject):
    """
    circle object in a 2D space having general form:

            x^2 + y^2 + A * x + B * y + C = 0

    Equivalently another expression of the circle is:

            (x - x0)^2 + (y - y0)^2 = r^2

    Where:
        x0 and y0 are the coordinates of the center of the circle
        r is the radius of the circle.

    Hence:
        x0 = -A/2
        y0 = -B/2
        r = (A^2 / 4 + B^2 / 4 - C) ^ 0.5
    """

    @property
    def center(self):
        """return the coordinates of the center of the circle."""
        return -self.betas["A"] / 2, -self.betas["B"] / 2

    @property
    def radius(self):
        """return the radius of the circle"""
        A = self.betas["A"]
        B = self.betas["B"]
        C = self.betas["C"]
        return float(((A**2) / 4 + (B**2) / 4 - C) ** 0.5)

    @property
    def domains(self):
        """return the value limits accepted for each dimension"""
        if not self.is_fitted():
            raise ValueError("'domains' must be called after fit.")
        x0, y0 = self.center
        r = self.radius
        return {
            "x": (x0 - r, x0 + r),
            "y": (y0 - r, y0 + r),
        }

    @property
    def perimeter(self):
        """return the perimeter of the circle"""
        return 2 * PI * self.radius

    @property
    def area(self):
        """return the area of the circle"""
        return PI * self.radius**2

    def __init__(self):
        x, y, A, B, C = sympy.symbols("x,y,A,B,C")
        equation = sympy.Equality(x**2 + y**2 + A * x + B * y + C, 0)
        super().__init__(
            general_equation=equation,  # type: ignore
            dimensions=["x", "y"],
            has_intercept=True,
        )

    def _expand_dimensions(self, **dimensions: np.ndarray):
        """prepare the design matrix"""
        x, y = [self._simplify(dimensions[i], i) for i in ["x", "y"]]
        return np.column_stack((x, y, np.ones_like(x))), -(x**2 + y**2)

    def fit(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int,
        y: np.ndarray | pd.Series | list[float | int] | float | int,
    ):
        """
        fit the model with the required data

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int
            the y-axis coordinates

        Returns
        -------
        fitted: Circle
            a fitted Circle object
        """

        # Convert to numpy arrays
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        # Construct the design matrix
        A, b = self._expand_dimensions(x=x, y=y)

        # Solve the normal equations A.T * A * coef = A.T * b
        betas = np.linalg.solve(A.T @ A, A.T @ b)

        # Extract coefficients
        self._betas = {i: float(v) for i, v in zip(self.coefs, betas)}

        return self

    def predict(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        y: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
    ):
        """
        get predictions of the not provided coordinate

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the y-axis coordinates

        Returns
        -------
        out: np.ndarray
            the coordinates of the axis not provided

        Note
        ----
        only the 'x' or 'y' argument has to be provided.
        """
        if x is None and y is not None:
            return super().predict(y=y)
        if y is None and x is not None:
            return super().predict(x=x)
        raise ValueError("Just 'x' or 'y' must be not None.")

    def is_inside(
        self,
        x: int | float,
        y: int | float,
    ):
        """
        check whether the point (x, y) is inside the Ellipse.

        Parameters
        ----------
        x: float
            the x axis coordinate

        y: float
            the y axis coordinate

        Returns
        -------
        i: bool
            True if the provided point is contained by the Ellipse.
        """
        out = self.predict(x=x)
        if out is None:
            return False
        y0, y1 = out
        return bool((y0 is not None) & (y > min(y0, y1)) & (y <= max(y0, y1)))


class Ellipse(GeometricObject):
    """
    fit an Ellipse in a 2D space having form:
        A * x**2 + B * xy + C * y**2 + D * x + E * y + F = 0

    References
    ----------
    Halir R, Flusser J. Numerically stable direct least squares fitting of
        ellipses. InProc. 6th International Conference in Central Europe on
        Computer Graphics and Visualization. WSCG 1998 (Vol. 98, pp. 125-132).
        Citeseer. https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=DF7A4B034A45C75AFCFF861DA1D7B5CD?doi=10.1.1.1.7559&rep=rep1&type=pdf
    """

    @property
    def center(self):
        """return the coordinates of the center of the ellipse."""
        A, B, C, D, E, F = list(self.betas.values())
        denom = B**2 - 4 * A * C
        h = (2 * C * D - B * E) / denom
        k = (2 * A * E - B * D) / denom
        return (float(h), float(k))

    @property
    def domains(self):
        """return the value limits accepted for each dimension"""
        if not self.is_fitted():
            raise ValueError("'domains' must be called after fit.")

        # Trattiamo l'equazione come un quadratico in y:
        # C*y^2 + (Bx + E)*y + (Ax^2 + Dx + F)
        x, y = sympy.symbols("x, y")
        A, B, C, D, E, F = list(self.betas.values())
        ax = C
        bx = B * x + E
        cx = A * x**2 + D * x + F

        # Trattiamo l'equazione come un quadratico in y:
        # A*x^2 + (By + D)*x + (Cy^2 + Ey + F)
        ay = A
        by = B * y + D
        cy = C * y**2 + E * y + F

        # Calcolo del discriminante del quadratico in y
        discriminant_x = bx**2 - 4 * ax * cx
        discriminant_y = by**2 - 4 * ay * cy

        # Risolvo l'inequazione discriminante >= 0 per trovare il dominio
        domain = sympy.solve(discriminant_x >= 0, x)
        x = (float(domain.args[0].args[0]), float(domain.args[1].args[1]))
        codomain = sympy.solve(discriminant_y >= 0, y)
        y = (float(codomain.args[0].args[0]), float(codomain.args[1].args[1]))

        return {"x": x, "y": y}

    @property
    def perimeter(self):
        """
        return the perimeter of the ellipse using the Ramanujan approximation
        """
        a, b = self.semi_axes
        h = ((a - b) ** 2) / ((a + b) ** 2)
        perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))

        return float(perimeter)

    @property
    def area(self):
        """
        Calcola l'area dell'ellisse data l'equazione generale:
        A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0
        Assumendo che rappresenti un'ellisse reale.
        """
        a, b = self.semi_axes
        return np.pi * a * b

    @property
    def major_axis(self):
        """
        Dati i coefficienti dell'equazione Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0,
        restituisce la linea che definisce l'asse maggiore.
        """
        A = self.betas["A"]
        B = self.betas["B"]
        C = self.betas["C"]

        # Matrice della forma quadratica
        Q = np.array([[A, B / 2], [B / 2, C]])

        # Calcolo autovalori e autovettori
        eigenvalues, eigenvectors = np.linalg.eigh(Q)

        # L'autovettore associato al minore autovalore è la direzione dell'asse maggiore
        min_index = np.argmin(eigenvalues)
        direction = eigenvectors[:, min_index]

        # Calcolo la retta dell'asse maggiore
        dx, dy = direction
        x0, y0 = self.center
        out = Line2D()
        out._betas = {"A": -dy, "B": dx, "C": -(-dy * x0 + dx * y0)}
        return out

    @property
    def minor_axis(self):
        """
        Dati i coefficienti dell'equazione Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0,
        restituisce la linea che definisce l'asse minore.
        """
        A = self.betas["A"]
        B = self.betas["B"]
        C = self.betas["C"]

        # Matrice della forma quadratica
        Q = np.array([[A, B / 2], [B / 2, C]])

        # Calcolo autovalori e autovettori
        eigenvalues, eigenvectors = np.linalg.eigh(Q)

        # L'autovettore associato al minore autovalore è la direzione dell'asse maggiore
        max_index = np.argmax(eigenvalues)
        direction = eigenvectors[:, max_index]

        # calcolo la retta dell'asse minore
        dx, dy = direction
        x0, y0 = self.center
        out = Line2D()
        out._betas = {"A": -dy, "B": dx, "C": -(-dy * x0 + dx * y0)}
        return out

    @property
    def semi_axes(self):
        """return the length of the semi axes"""
        A, B, C, D, E, F = list(self.betas.values())

        # Verifica se l'equazione rappresenta un'ellisse
        discriminant = B**2 - 4 * A * C
        if discriminant >= 0:
            raise ValueError("L'equazione non rappresenta un'ellisse.")

        # Matrice del termine quadratico
        M = np.array([[A, B / 2], [B / 2, C]])

        # Coordinate del centro dell'ellisse
        x0, y0 = self.center

        # Valore di F traslato al centro
        f0 = A * x0**2 + B * x0 * y0 + C * y0**2 + D * x0 + E * y0 + F

        # Autovalori della matrice M
        eigvals = np.linalg.eigvals(M)
        if f0 == 0 or np.any(eigvals == 0):
            raise ValueError("Impossibile determinare i semiassi")

        # Calcolo dei semiassi
        a, b = np.sqrt(-f0 / eigvals[:2])

        # Ordina per ottenere semiasse maggiore e minore
        return float(a), float(b)

    @property
    def rotation_angle(self):
        """return the rotation angle of the ellipse in degrees"""
        A, B, C, D, E, F = list(self.betas.values())
        theta_rad = 0.5 * np.arctan(B / (A - C))
        return float(np.degrees(theta_rad))

    @property
    def foci(self):

        # Distanza focale
        a, b = self.semi_axes
        c = np.sqrt(abs(a**2 - b**2))

        # Fuochi nel sistema ruotato
        if a > b:
            f1_rot = np.array([c, 0])
            f2_rot = np.array([-c, 0])
        else:
            f1_rot = np.array([0, c])
            f2_rot = np.array([0, -c])

        # Rotazione inversa per tornare al sistema originale
        theta = self.rotation_angle / 180 * np.pi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        center = self.center
        x1, y1 = R @ f1_rot + center
        x2, y2 = R @ f2_rot + center

        return (float(x1), float(y1)), (float(x2), float(y2))

    @property
    def eccentricity(self):
        """calculate the eccentricity of the ellipse"""
        a, b = self.semi_axes
        return float(np.sqrt(1 - (b**2 / a**2)))

    def is_inside(
        self,
        x: int | float,
        y: int | float,
    ):
        """
        check whether the point (x, y) is inside the Ellipse.

        Parameters
        ----------
        x: float
            the x axis coordinate

        y: float
            the y axis coordinate

        Returns
        -------
        i: bool
            True if the provided point is contained by the Ellipse.
        """
        domain, codomain = list(self.domains.values())
        return (
            (x >= domain[0])
            & (x <= domain[1])
            & (y >= codomain[0])
            & (y <= codomain[1])
        )

    def __init__(self):
        x, y, A, B, C, D, E, F = sympy.symbols("x,y,A,B,C,D,E,F")
        equation = sympy.Equality(
            A * x**2 + B * x * y + C * y**2 + D * x + E * y + F, 0
        )
        super().__init__(
            general_equation=equation,  # type: ignore
            dimensions=["x", "y"],
            has_intercept=True,
        )

    def _expand_dimensions(self, **dimensions: np.ndarray):
        """prepare the design matrix"""
        return None

    def fit(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int,
        y: np.ndarray | pd.Series | list[float | int] | float | int,
    ):
        """
        fit the model with the required data

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int
            the y-axis coordinates

        Returns
        -------
        fitted: Ellipse
            a fitted Ellipse object
        """
        # verifico gli input
        x = np.squeeze(self._simplify(x, "x"))
        y = np.squeeze(self._simplify(y, "y"))

        # quadratic part of he design matrix
        d_1 = np.vstack([x**2, x * y, y**2]).T.astype(float)

        # linear part of the design matrix
        d_2 = np.vstack([x, y, np.ones(len(x))]).T

        # quadratic part of the scatter matrix
        s_1 = d_1.T @ d_1

        # combined part of the scatter matrix
        s_2 = d_1.T @ d_2

        # linear part of the scatter matrix
        s_3 = d_2.T @ d_2

        # reduced scatter matrix
        cnd = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        trc = -np.linalg.inv(s_3) @ s_2.T
        mat = np.linalg.inv(cnd) @ (s_1 + s_2 @ trc)

        # solve the eigen system
        eigvec = np.linalg.eig(mat)[1]

        # evaluate the coefficients
        con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
        eiv_pos = eigvec[:, np.nonzero(con > 0)[0]]
        coefs = np.concatenate((eiv_pos, trc @ eiv_pos)).ravel()
        self._betas = {i: v for i, v in zip(self.coefs, coefs)}

        return self

    def predict(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        y: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
    ):
        """
        get predictions of the not provided coordinate

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the y-axis coordinates

        Returns
        -------
        out: np.ndarray
            the coordinates of the axis not provided

        Note
        ----
        only the 'x' or 'y' argument has to be provided.
        """
        if x is None and y is not None:
            return super().predict(y=y)
        if y is None and x is not None:
            return super().predict(x=x)
        raise ValueError("Just 'x' or 'y' must be not None.")
