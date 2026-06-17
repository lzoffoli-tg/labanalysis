"""
timeseries module
"""

# -*- coding: utf-8 -*-


#! IMPORTS

import inspect
from typing import Literal

import numpy as np
import pandas as pd
import pint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..signalprocessing import fillna as sp_fillna
from ..signalprocessing import gram_schmidt
from ..utils import FloatArray1D, FloatArray2D, TextArray1D, ureg
from .indexers import TimeseriesLocIndexer, TimeseriesILocIndexer

__all__ = ["Timeseries", "Signal1D", "Signal3D", "EMGSignal", "Point3D"]


class Timeseries:
    """
    Time-indexed multi-column data container with unit support.

    Base class for time-series data providing pandas-like indexing, arithmetic
    operations, unit conversion, and signal processing capabilities. Designed for
    biomechanical and physiological signals.

    Attributes
    ----------
    index : np.ndarray
        Time index array (1D).
    columns : np.ndarray
        Column labels array (1D).
    _data : np.ndarray
        Internal data storage (2D array: rows=time, cols=variables).
    _unit : pint.Quantity or str
        Unit of measurement for the data.

    Properties
    ----------
    ix : TimeseriesIXIndexer
        Integer-based indexer for iloc-style access.
    unit : str
        String representation of the unit of measurement.
    shape : tuple
        Shape of the data array (n_timepoints, n_columns).

    Notes
    -----
    - Supports arithmetic operations (+, -, *, /, etc.) with broadcasting
    - Integrates with pint for unit conversion and validation
    - Provides fillna() for gap filling via interpolation
    - Can be converted to pandas DataFrame or numpy array
    - Supports method chaining with inplace operations

    See Also
    --------
    Signal1D : 1D time-series specialization
    Signal3D : 3D vector time-series
    Point3D : 3D position trajectory
    EMGSignal : Electromyography signal

    Examples
    --------
    >>> import numpy as np
    >>> from labanalysis import Timeseries
    >>> data = np.random.randn(100, 3)
    >>> index = np.linspace(0, 10, 100)
    >>> ts = Timeseries(data, index, columns=['X', 'Y', 'Z'], unit='m')
    >>> ts.shape
    (100, 3)
    >>> ts.unit
    'm'
    """

    @property
    def loc(self):
        """Label-based indexer (pandas .loc analog)."""
        return TimeseriesLocIndexer(self)

    @property
    def iloc(self):
        """Position-based indexer (pandas .iloc analog)."""
        return TimeseriesILocIndexer(self)

    @property
    def unit(self):
        """
        Get the unit of measurement.

        Returns
        -------
        str
            The unit of measurement.
        """
        if isinstance(self._unit, pint.Quantity):
            return f"{self._unit.units:~}"
        else:
            return str(self._unit)

    @property
    def shape(self):
        return (len(self.index), len(self.columns))

    def set_unit(self, unit: str | pint.Quantity | pint.Unit):
        # set the unit of measurement
        msg = "unit must be a string representing a conventional unit "
        msg += "of measurement in the SI sytem or a pint.Quantity or Unit"
        if not isinstance(unit, (str, pint.Quantity, pint.Unit)):
            raise ValueError(msg)
        if isinstance(unit, str):
            try:
                unit = ureg(unit)
            except Exception as exc:
                raise ValueError(msg) from exc
        self._unit = unit

    def to_dataframe(self):
        """
        Convert to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame representation.
        """
        return pd.DataFrame(self._data, index=self.index, columns=self.columns)

    def strip(self, axis: int | None = None, inplace: bool = False):
        """
        Remove leading/trailing rows or columns that are all NaN from all
        contained Timeseries-like objects.

        Parameters
        ----------
        inplace : bool, optional
            If True, modifies in place. If False, returns a new TimeseriesRecord.

        Returns
        -------
        TimeseriesRecord or None
            Stripped TimeseriesRecord if inplace is False, otherwise None.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if axis is not None:
            if not isinstance(axis, int) or axis not in [0, 1]:
                raise ValueError("axis must be None or 0 or 1")
        out = self if inplace else self.copy()
        if axis is None or axis == 0:
            index = out.to_dataframe().dropna(how="all", axis=0).index.to_numpy()
            if len(index) > 0:
                start = float(np.min(index))
                stop = float(np.max(index))
                if inplace:
                    # For inplace, use loc accessor
                    temp = out.loc[start:stop, :]
                    out._data = temp._data
                    out.index = temp.index
                else:
                    out = out.loc[start:stop, :]
            # If index is empty (all rows are NaN), leave out unchanged
        if axis is None or axis == 1:
            cols = out.columns
            nonan_cols = out.to_dataframe().dropna(how="all", axis=1).columns.to_numpy()
            indices = [i for i, v in enumerate(out.columns) if v in nonan_cols]
            if len(indices) > 0:
                indices = np.arange(np.min(indices), np.max(indices) + 1)
                if inplace:
                    # For inplace, modify internal attributes
                    temp = out[:, cols[indices]]
                    out._data = temp._data
                    out.columns = temp.columns
                else:
                    out = out[:, cols[indices]]
            # If indices is empty (all columns are NaN), leave out unchanged
        if not inplace:
            return out

    def reset_time(self, inplace=False):
        """
        Reset the time index to start at zero.

        Parameters
        ----------
        inplace : bool, optional
            If True, modify in place. If False, return a new Timeseries.

        Returns
        -------
        Timeseries or None
            If inplace is False, returns a new Timeseries with reset time.
            If inplace is True, returns None.
        """
        min_time = np.min(self.index)
        new_index = [float(round(i - min_time, 3)) for i in self.index]
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if inplace:
            self.index = np.array(new_index)
        else:
            out = self.copy()
            out.index = np.array(new_index)
            return out


    def fillna(self, value=None, regressors=None, inplace=False):
        """
        Return a copy with NaNs replaced by the specified value or using advanced imputation for all contained objects.

        Parameters
        ----------
        value : float or int or None, optional
            Value to use for NaNs. If None, use interpolation or regression.
        regressors : np.ndarray or pd.DataFrame or pd.Series or None, optional
            Independent variables for multiple linear regression imputation.
            If provided, missing values are predicted using linear regression
            with these regressors as predictors. If None, cubic spline
            interpolation is applied to each column independently.
        inplace : bool, optional
            If True, fill in place. If False, return a new object.

        Returns
        -------
        Timeseries
            Filled object.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        vals = sp_fillna(
            self._data.copy(),
            value,
            regressors,
            False,
        )
        vals = np.asarray(vals, float)
        if inplace:
            self[:, :] = vals
        else:
            out = self.copy()
            out[:, :] = vals
            return out

    def apply(
        self,
        func,
        axis: int = 0,
        new_unit: str | None = None,
        inplace: bool = False,
        *args,
        **kwargs,
    ):
        """
        Apply a function to the underlying data.

        Parameters
        ----------
        func : callable or ProcessingPipeline
            Function, class, or method to apply to the data, or a
            ProcessingPipeline.
        axis : int, optional
            0 to apply by row, 1 to apply by column (default: 0).
        inplace : bool, optional
            If True, modifies self. If False, returns a new object.
        *args, **kwargs : additional arguments to pass to func.

        Returns
        -------
        Timeseries or result of func
            If inplace is False, returns a new LabeledArray with the function
            applied.
            If inplace is True, returns None.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")

        if hasattr(func, "apply") and callable(getattr(func, "apply", None)):
            if inplace:
                func.apply(self, inplace=True, *args, **kwargs)
            else:
                return func.apply(self, inplace=False, *args, **kwargs)
        else:
            result = np.apply_along_axis(
                func,
                axis,
                self._data,
                *args,
                **kwargs,
            )
            if all([i == j for i, j in zip(result.shape, self.shape)]):
                if inplace:
                    self[:, :] = result
                    if new_unit is not None:
                        self.set_unit(new_unit)
                else:
                    out = self.copy()
                    out[:, :] = result
                    if new_unit is not None:
                        out.set_unit(new_unit)
                    return out
            else:
                if inplace:
                    msg = "inplace must be False if the applied function "
                    msg += "changes the shape of the target object."
                    raise ValueError(msg)
                return result

    def to_plotly_figure(self):
        df = self.to_dataframe()
        fig = make_subplots(
            rows=df.shape[1],
            cols=1,
            shared_xaxes=True,
            shared_yaxes=False,
            row_titles=[i.rsplit("_", 1)[0] for i in df.columns],
        )
        for i, (column, values) in enumerate(df.items()):
            lbl, unit = str(column).rsplit("_", 1)
            fig.add_trace(
                row=i + 1,
                col=1,
                trace=go.Scatter(
                    x=df.index.to_list(),
                    y=values.to_numpy().astype(float).flatten().tolist(),
                    name=lbl,
                    mode="lines",
                ),
            )
            fig.update_yaxes(row=i + 1, col=1, title=unit)
        fig.update_layout(title=fig.__class__.__name__, template="simple_white")
        return fig

    def isna(self):
        """
        Return a boolean array indicating NaNs.

        Returns
        -------
        np.ndarray
            Boolean mask of NaNs.
        """
        return np.isnan(self._data)

    def to_numpy(self):
        return self._data

    def is_empty(self):
        """
        Check if the Timeseries is empty (all NaNs).

        Returns
        -------
        bool
            True if all data is NaN, False otherwise.
        """
        return bool(np.all(np.isnan(self._data)) or self._data.size == 0)

    def _get_object_args(self, attr_map=None):
        """
        Extracts constructor arguments and internal attributes from the current
        instance to allow dynamic instantiation of self.__class__, with support
        for nested objects and customizable attribute name mapping.

        Parameters:
            attr_map (dict): Optional mapping from constructor parameter names
                to attribute names.

        Returns:
            dict: A dictionary of constructor arguments and internal attributes.
        """
        sig = inspect.signature(self.__class__.__init__)
        args = {}

        # Extract constructor parameters
        for name, param in sig.parameters.items():
            if name == "self":
                continue

            attr_name = attr_map.get(name, name) if attr_map else name

            value = None
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
            elif hasattr(self, f"_{attr_name}"):
                value = getattr(self, f"_{attr_name}")
            elif param.default is not inspect.Parameter.empty:
                value = param.default
            else:
                raise AttributeError(f"Missing required constructor argument: '{name}'")

            args[name] = value

        # Include all class-level attributes starting with "_"
        for attr in dir(self):
            if attr.startswith("_") and not attr.startswith("__"):
                if attr not in args and hasattr(self.__class__, attr):
                    value = getattr(self, attr)
                    if not callable(value):
                        args[attr] = getattr(self, attr)

        return args

    def _check_consistency(self):
        """Internal: Check if data shape matches index/columns."""
        if self._data.shape != (len(self.index), len(self.columns)):
            raise ValueError(
                "Inconsistent shape: data shape does not match index and columns length"
            )

    def _view(
        self,
        rows: slice | list[float | bool] | np.ndarray | None = None,
        cols: list[str | bool] | None = None,
    ):
        # Default to all rows/columns
        if rows is None:
            row_mask = slice(None)
        elif isinstance(rows, slice):
            start = self.index[0] if rows.start is None else rows.start
            stop = self.index[-1] if rows.stop is None else rows.stop
            row_mask = (self.index >= start) & (self.index <= stop)
        else:
            # rows can be a list/ndarray of labels or boolean/positional indices
            # If numeric floats are provided, match using isclose to tolerate
            # floating point rounding differences in time indices.
            try:
                arr = np.asarray(rows)
            except Exception:
                arr = rows

            # boolean mask
            if isinstance(arr, (np.ndarray, list)) and all(
                [isinstance(i, (bool, np.bool_)) for i in arr]
            ):
                row_mask = np.asarray(arr, dtype=bool)
            # numeric array: use isclose for floats, isin for ints
            elif isinstance(arr, np.ndarray) and arr.dtype.kind == "f":
                idx = np.asarray(self.index, float)
                mask = np.zeros(len(idx), dtype=bool)
                for v in arr.astype(float):
                    mask |= np.isclose(idx, v, rtol=1e-6, atol=1e-8)
                row_mask = mask
            elif isinstance(arr, (list, np.ndarray)) and all(
                [isinstance(i, (int, np.integer)) for i in arr]
            ):
                # treat as positional/label integers
                row_mask = np.isin(self.index, arr)
            else:
                # fallback to isin for other label types (strings, etc.)
                row_mask = np.isin(self.index, rows)
        if cols is None:
            col_mask = slice(None)
        elif isinstance(cols, slice):
            start = cols.start
            if start is None:
                start = self.columns[0]
            stop = cols.stop
            if stop is None:
                stop = self.columns[-1]
            col_idx = [i for i, v in enumerate(self.columns) if v in [start, stop]]
            col_idx = np.arange(col_idx[0], col_idx[-1] + 1)
            col_mask = np.isin(col_idx, np.arange(len(self.columns)))
        else:
            col_mask = np.isin(self.columns, np.asarray(cols))

        # If the selection includes all columns, treat as no-column-selection
        # to preserve the original object type when returning the view.
        if not isinstance(col_mask, slice):
            try:
                if col_mask.size == len(self.columns) and np.all(col_mask):
                    cols = None
                    col_mask = slice(None)
            except Exception:
                pass

        # Extract data and prepare for view creation
        # Convert boolean masks to index arrays to ensure numpy creates a view, not a copy
        if isinstance(row_mask, np.ndarray) and row_mask.dtype == bool:
            row_indices = np.where(row_mask)[0]
        elif isinstance(row_mask, slice):
            row_indices = row_mask
        else:
            row_indices = row_mask

        if isinstance(col_mask, np.ndarray) and col_mask.dtype == bool:
            col_indices = np.where(col_mask)[0]
        elif isinstance(col_mask, slice):
            col_indices = col_mask
        else:
            col_indices = col_mask

        # Use np.ix_ for proper view creation when both are arrays
        if isinstance(row_indices, np.ndarray) and isinstance(col_indices, np.ndarray):
            view_data = self._data[np.ix_(row_indices, col_indices)]
        elif isinstance(row_indices, slice) and isinstance(col_indices, np.ndarray):
            view_data = self._data[row_indices, col_indices]
        elif isinstance(row_indices, np.ndarray) and isinstance(col_indices, slice):
            view_data = self._data[row_indices, col_indices]
        else:
            # Both are slices
            view_data = self._data[row_indices, col_indices]

        view_index = self.index[row_mask]
        view_columns = self.columns[col_mask]

        # Create a new object that shares the same data buffer and
        # handle the appropriate object type. When `cols` is provided
        # we always return a generic Timeseries to keep the object's
        # class stable regardless of the number of selected columns.
        if cols is None:
            view_obj = self.__new__(type(self))
            # Copy essential attribute (_unit) for all Timeseries objects
            try:
                view_obj._unit = self._unit
            except AttributeError:
                view_obj._unit = "dimensionless"
            # For Signal3D, copy axis attributes
            if hasattr(self, '_vertical_axis'):
                view_obj._vertical_axis = self._vertical_axis
            if hasattr(self, '_anteroposterior_axis'):
                view_obj._anteroposterior_axis = self._anteroposterior_axis
        else:
            view_obj = Timeseries.__new__(Timeseries)
            # for generic Timeseries, only copy essential attributes
            try:
                view_obj._unit = self._unit
            except AttributeError:
                view_obj._unit = "dimensionless"

        view_obj._data = view_data
        # Copy index and columns to avoid aliasing issues with numpy views
        view_obj.index = view_index.copy() if isinstance(view_index, np.ndarray) else view_index
        view_obj.columns = view_columns.copy() if isinstance(view_columns, np.ndarray) else view_columns

        # return the prepared view
        return view_obj

    def _binary_op(self, other, op):
        if isinstance(other, Timeseries):
            val = op(self._data, other._data)
        else:
            val = op(self._data, other)
        out = self.copy()
        out[:, :] = val
        return out

    def __add__(self, other):
        """Element-wise addition."""
        return self._binary_op(other, np.add)

    def __iadd__(self, other):
        """In-place element-wise addition."""
        result = self._binary_op(other, np.add)
        self._data = result._data
        return self

    def __radd__(self, other):
        """Right element-wise addition."""
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        """Element-wise subtraction."""
        return self._binary_op(other, np.subtract)

    def __isub__(self, other):
        """In-place element-wise subtraction."""
        result = self._binary_op(other, np.subtract)
        self._data = result._data
        return self

    def __rsub__(self, other):
        """Right element-wise subtraction."""
        # right subtraction: other - self
        return self._binary_op(other, lambda x, y: np.subtract(y, x))

    def __mul__(self, other):
        """Element-wise multiplication."""
        return self._binary_op(other, np.multiply)

    def __imul__(self, other):
        """In-place element-wise multiplication."""
        result = self._binary_op(other, np.multiply)
        self._data = result._data
        return self

    def __rmul__(self, other):
        """Right element-wise multiplication."""
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other):
        """Element-wise division."""
        return self._binary_op(other, np.divide)

    def __itruediv__(self, other):
        """In-place element-wise division."""
        result = self._binary_op(other, np.divide)
        self._data = result._data
        return self

    def __rtruediv__(self, other):
        """Right element-wise division."""
        # right division: other / self
        return self._binary_op(other, lambda x, y: np.divide(y, x))

    def __pow__(self, other):
        """Element-wise exponentiation."""
        return self._binary_op(other, np.power)

    def __ipow__(self, other):
        """In-place element-wise exponentiation."""
        result = self._binary_op(other, np.power)
        self._data = result._data
        return self

    def __rpow__(self, other):
        """Right element-wise exponentiation."""
        # right power: other ** self
        return self._binary_op(other, lambda x, y: np.power(y, x))

    def __eq__(self, other):
        """Element-wise equality comparison."""
        if isinstance(other, Timeseries):
            return self._data == other._data
        return self._data == other

    def __ne__(self, other):
        """Element-wise inequality comparison."""
        if isinstance(other, Timeseries):
            return self._data != other._data
        return self._data != other

    def __lt__(self, other):
        """Element-wise less-than comparison."""
        if isinstance(other, Timeseries):
            return self._data < other._data
        return self._data < other

    def __gt__(self, other):
        """Element-wise greater-than comparison."""
        if isinstance(other, Timeseries):
            return self._data > other._data
        return self._data > other

    def __le__(self, other):
        """Element-wise less-than-or-equal comparison."""
        if isinstance(other, Timeseries):
            return np.less_equal(self._data, other._data)
        return np.less_equal(self._data, other)

    def __ge__(self, other):
        """Element-wise greater-than-or-equal comparison."""
        if isinstance(other, Timeseries):
            return np.greater_equal(self._data, other._data)
        return np.greater_equal(self._data, other)

    def __repr__(self):
        df = self.to_dataframe()
        return df.__repr__() + f"\nshape = {self._data.shape}, unit = '{self.unit}'"

    def __str__(self):
        df = self.to_dataframe()
        return df.__str__() + f"\nshape = {self._data.shape}, unit = '{self.unit}'"

    def __setattr__(self, name: str, value: object):
        if name in ["index", "columns"]:
            try:
                value = np.asarray(value).flatten()
            except Exception as exc:
                raise ValueError(f"{name} must be a 1D array") from exc
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        # we return the "column" named "name"
        # Use object.__getattribute__ to safely access attributes during unpickling
        try:
            columns = object.__getattribute__(self, "columns")
            if name in columns:
                data = object.__getattribute__(self, "_data")
                loc = np.where(columns == name)[0]
                index = object.__getattribute__(self, "index")
                unit = object.__getattribute__(self, "_unit")
                return Timeseries(
                    data[:, loc],
                    index,
                    [name],
                    unit,
                )
        except AttributeError:
            # columns or other attributes don't exist yet (during unpickling)
            pass

        # if None, raise an error
        data = object.__getattribute__(self, "_data")
        attr = getattr(data, name, None)
        if attr is None:
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")

        # is a function (e.g. arr.mean())
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # Wrapping solo se il risultato è un ndarray con shape uguale
                if (
                    isinstance(result, np.ndarray)
                    and result.ndim == 2
                    and all([i == v for i, v in zip(result.shape, self._data.shape)])
                    and not name.startswith("to_")
                ):
                    return self.__class__(**self._get_object_args())

                return result

            return wrapper

        # its a property to be returned
        return attr

    def __getitem__(self, key):
        """
        Get values using label-based indexing (delegates to .loc).

        This method maintains backward compatibility while internally using .loc accessor.
        For explicit position-based indexing, use .iloc instead.

        Examples
        --------
        >>> ts["X"]                      # Get entire column 'X'
        >>> ts[['X', 'Y']]               # Get multiple columns
        >>> ts[0.5, 'X']                 # Get value at time 0.5, column 'X'
        >>> ts[1, ['X', 'Y']]            # Get multiple columns at time 1
        >>> ts[np.array([0.1, 1]), 'X']  # Get specific times for column 'X'
        >>> ts[:, 'X']                   # Get entire column via slice
        >>> ts[0:1]                      # Get time range
        """
        # Handle backward compatibility: single string or list of strings are column names
        if isinstance(key, str) or (isinstance(key, list) and all(isinstance(k, str) for k in key)):
            # Column selection: ts['X'] or ts[['X', 'Y']]
            # BUT: if it's a single string and it's not a column name, try property access
            if isinstance(key, str) and key not in self.columns:
                # Try to access as a property/attribute (backward compatibility for ts['module'], etc.)
                try:
                    return getattr(self, key)
                except AttributeError:
                    # If attribute doesn't exist, let .loc raise KeyError about column
                    pass
            return self.loc[:, key]
        # Otherwise delegate to loc
        return self.loc[key]

    def __setitem__(self, key, value):
        """
        Set values using label-based indexing (delegates to .loc).

        This method maintains backward compatibility while internally using .loc accessor.
        For explicit position-based indexing, use .iloc instead.

        Examples
        --------
        >>> ts["X"] = 0                      # Set entire column 'X'
        >>> ts[['X', 'Y']] = [[1, 2]]        # Set multiple columns
        >>> ts[0.5, 'X'] = 10                # Set value at time 0.5, column 'X'
        >>> ts[1, ['X', 'Y']] = [1, 2]       # Set multiple columns at time 1
        >>> ts[np.array([0.1, 1]), 'X'] = 5  # Set specific times for column 'X'
        >>> ts[:, 'X'] = 0                   # Set entire column via slice
        >>> ts[0:1] = [[1, 2, 3]]            # Set time range
        """
        # Handle backward compatibility: single string or list of strings are column names
        if isinstance(key, str) or (isinstance(key, list) and all(isinstance(k, str) for k in key)):
            # Column selection: ts['X'] = value or ts[['X', 'Y']] = value
            self.loc[:, key] = value
        else:
            self.loc[key] = value
        self._check_consistency()

    def __init__(
        self,
        data: FloatArray2D,
        index: FloatArray1D | list[float | int],
        columns: list[str] | TextArray1D,
        unit: str | pint.Quantity,
    ):
        try:
            self._data = np.asarray(data, dtype=float)
        except Exception as exc:
            raise ValueError("data must be a 2D array castable to float.") from exc
        if self._data.ndim != 2:
            raise ValueError("data must be a 2D array castable to float.")
        self.index = np.asarray(index, float)
        self.columns = np.asarray(columns, str)
        self._check_consistency()
        self.set_unit(unit)

    def copy(self):
        return Timeseries(
            self._data.copy(),
            self.index.copy(),
            self.columns.copy(),
            self.unit,
        )


class Signal1D(Timeseries):
    """
    A 1D signal (single column) time series.
    """

    def __init__(
        self,
        data: np.ndarray,
        index: list[float] | FloatArray1D,
        unit: str | pint.Quantity,
    ):
        """
        Initialize a Signal1D.

        Parameters
        ----------
        data : array-like
            2D data array with one column.
        index : list of float
            Time values.
        unit : str or pint.Quantity
            Unit of measurement for the data.

        Raises
        ------
        ValueError
            If data does not have exactly one column.
        """
        data_array = np.asarray(data, float)
        if data_array.ndim == 1:
            data_array = np.atleast_2d(data_array).T
        if data_array.ndim != 2 or data_array.shape[1] != 1:
            raise ValueError("Signal1D must have exactly one column")
        if not isinstance(unit, (str, pint.Quantity, pint.Unit)):
            raise ValueError("unit must be a str or a pint.Quantity")
        super().__init__(
            data=data_array,
            index=index,
            columns=["amplitude"],
            unit=unit,
        )

    def copy(self):
        return Signal1D(
            self._data.copy(),
            self.index.copy(),
            self.unit,
        )

    def to_dataframe(self):
        df = super().to_dataframe()
        df.columns = pd.Index([self.unit.replace(" ", "")])
        return df


class Signal3D(Timeseries):
    """
    A 3D signal (three columns: X, Y, Z) time series.
    """

    @property
    def vertical_axis(self):
        return self._vertical_axis

    @property
    def anteroposterior_axis(self):
        return self._anteroposterior_axis

    @property
    def lateral_axis(self):
        bounded = [self.vertical_axis, self.anteroposterior_axis]
        col = [i for i in self.columns if i not in bounded]
        if len(col) == 0:
            raise ValueError("no lateral axis could be found.")
        return str(col[0])

    @property
    def module(self):
        return Signal1D(
            data=(self._data.copy() ** 2).sum(axis=1) ** 0.5,
            index=self.index.copy(),
            unit=self.unit,
        )

    def __init__(
        self,
        data: np.ndarray,
        index: list[float] | FloatArray1D,
        unit: pint.Quantity | str,
        columns: list[str] | TextArray1D = ["X", "Y", "Z"],
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
    ):
        super().__init__(
            data,
            index,
            columns,
            unit,
        )

        # check dimensions
        if data.shape[1] != 3:
            raise ValueError("Signal3D must have exactly 3 columns.")

        # check axes
        self.set_vertical_axis(vertical_axis)
        self.set_anteroposterior_axis(anteroposterior_axis)

    def set_vertical_axis(self, axis: str):
        if axis not in self.columns:
            raise ValueError(f"vertical_axis must be any of {self.columns}")
        self._vertical_axis = str(axis)

    def set_anteroposterior_axis(self, axis: str):
        if axis not in self.columns:
            raise ValueError(f"anteroposterior_axis must be any of {self.columns}")
        self._anteroposterior_axis = str(axis)

    def change_reference_frame(
        self,
        new_x: (
            np.ndarray
            | list[int | float]
            | tuple[int | float, int | float, int | float]
        ) = [1, 0, 0],
        new_y: (
            np.ndarray
            | list[int | float]
            | tuple[int | float, int | float, int | float]
        ) = [0, 1, 0],
        new_z: (
            np.ndarray
            | list[int | float]
            | tuple[int | float, int | float, int | float]
        ) = [0, 0, 1],
        new_origin: (
            np.ndarray
            | list[int | float]
            | tuple[int | float, int | float, int | float]
        ) = [0, 0, 0],
        inplace: bool = False,
    ):
        """
        Rotate and translate each sample using the new reference frame defined by
        orthonormal versors new_x, new_y, new_z and origin new_origin.

        A point can be aligned to this reference frame by:
            new = np.einsum("nij,nj->ni", R, old - O)

        Where R is the rotation matrix (N, 3, 3) and O (N, 3) is the origin of
        the reference frame.

        Parameters
        ----------
        new_x, new_y, new_z : array-like
            Orthonormal basis vectors.
        new_origin : array-like
            New origin.

        Returns
        -------
        Signal3D
            Transformed signal.

        Raises
        ------
        ValueError
            If input vectors are not valid.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        i = np.atleast_2d(new_x)
        if i.shape[0] == 1:
            i = np.ones(self.shape) * i
        j = np.atleast_2d(new_y)
        if j.shape[0] == 1:
            j = np.ones(self.shape) * j
        k = np.atleast_2d(new_z)
        if k.shape[0] == 1:
            k = np.ones(self.shape) * k
        o = np.atleast_2d(new_origin)
        if o.shape[0] == 1:
            o = np.ones(self.shape) * o
        rmat = gram_schmidt(i, j, k)
        rmat = rmat.transpose([0, 2, 1])
        new = np.einsum("nij,nj->ni", rmat, self._data.copy() - o)
        out = self if inplace else self.copy()
        out[:, :] = new
        if not inplace:
            return out

    def copy(self):
        return Signal3D(
            self._data.copy(),
            self.index.copy(),
            self.unit,
            self.columns.copy(),
            self.vertical_axis,
            self.anteroposterior_axis,
        )


class EMGSignal(Signal1D):
    """
    A 1D EMG signal, automatically converted to microvolts (uV).
    """

    _muscle_name: str
    _side: Literal["left", "right", "bilateral"]

    def __init__(
        self,
        data,
        index,
        muscle_name: str,
        side: Literal["left", "right", "bilateral"],
        unit: str | pint.Quantity = "uV",
    ):
        """
        Initialize an EMGSignal.

        Parameters
        ----------
        data : array-like
            2D data array with one column.
        index : list of float
            Time values.
        muscle_name : str
            Name of the muscle.
        side : {'left', 'right', 'bilateral'}
            Side of the body.
        unit : str or pint.Quantity, optional
            Unit of measurement for the data, by default "uV".

        Raises
        ------
        ValueError
            If unit is not valid.
        """
        # check the unit and convert if required
        if isinstance(unit, str):
            unit = ureg(unit)
        if unit.check("V"):
            unt = pint.Quantity("uV")
            magnitude = unit.to(unt).magnitude
        elif unit == ureg("%"):
            unt = ureg("%")
            magnitude = 1
        else:
            raise ValueError("unit must represent voltage or percentages.")

        # check the side
        valid_sides = ["left", "right", "bilateral"]
        if (
            not isinstance(side, (str, Literal["left", "right", "bilateral"]))
            or side not in valid_sides
        ):
            raise ValueError(f"side must be any of: {valid_sides}")

        # check the muscle name
        if not isinstance(muscle_name, str):
            raise ValueError("muscle_name must be a str.")

        # build the object
        values = np.squeeze(data) * magnitude
        super().__init__(
            data=values,
            index=index,
            unit=unt,  # type: ignore
        )
        self.set_side(side)
        self.set_muscle_name(muscle_name)

    def set_side(self, side: Literal["left", "right", "bilateral"] | str):
        if not isinstance(side, str) or not any(
            [side == i for i in ["left", "right", "bilataral"]]
        ):
            raise ValueError("side must be 'left', 'right' or 'bilateral'.")
        self._side = side  # type: ignore

    @property
    def side(self):
        """
        Get the side of the body.

        Returns
        -------
        {'left', 'right', 'bilateral'}
            The side of the body.
        """
        return str(self._side)

    def set_muscle_name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("name must be a string.")
        self._name = name  # type: ignore

    @property
    def muscle_name(self):
        """
        Get the name of the muscle.

        Returns
        -------
        str
            The name of the muscle.
        """
        return self._name

    def copy(self):
        return EMGSignal(
            self._data.copy(),
            self.index.copy(),
            self.muscle_name,
            self.side,  # type: ignore
            self.unit,
        )


class Point3D(Signal3D):
    """
    A 3D point time series, automatically converted to meters (m).
    """

    def __init__(
        self,
        data: np.ndarray | FloatArray2D,
        index: list[float] | FloatArray1D,
        unit: str | pint.Quantity = "m",
        columns: list[str] | TextArray1D = ["X", "Y", "Z"],
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
    ):
        """
        Initialize a Point3D.

        Parameters
        ----------
        data : array-like
            2D data array with three columns.
        index : list of float
            Time values.
        unit : str or pint.Quantity, optional
            Unit of measurement for the data, by default "m".
        columns : list, optional
            Column labels, must be 'X', 'Y', 'Z', by default ["X", "Y", "Z"].

        Raises
        ------
        ValueError
            If units are not valid or not unique.
        """
        super().__init__(
            data,
            index,
            unit,
            columns,
            vertical_axis,
            anteroposterior_axis,
        )

        # check the unit
        # check the unit and convert to uV if required
        if not self._unit.check("[length]"):
            raise ValueError("unit must represent length.")
        meters = pint.Quantity("m")
        magnitude = self._unit.to(meters).magnitude
        self[:, :] = self.to_numpy() * magnitude
        self._unit = meters  # type: ignore

    def copy(self):
        return Point3D(
            self._data.copy(),
            self.index.copy(),
            self.unit,
            self.columns.copy(),
            self.vertical_axis,
            self.anteroposterior_axis,
        )
