"""
timeseries module
"""

# -*- coding: utf-8 -*-


#! IMPORTS

import inspect

import numpy as np
import pandas as pd
import pint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...signalprocessing import fillna as sp_fillna
from ...utils import FloatArray1D, FloatArray2D, TextArray1D

ureg = pint.UnitRegistry()

__all__ = ["Timeseries"]


class Timeseries:

    _unit: pint.Quantity
    _data: FloatArray2D
    index: FloatArray1D
    columns: TextArray1D

    @property
    def ix(self):
        class TimeseriesIXIndexer:
            def __init__(self, ts: Timeseries):
                self.ts = ts

            def __getitem__(self, key):
                if isinstance(key, (tuple, list)) and len(key) == 2:
                    rows, cols = key
                    rows = self.ts.index[rows]
                    cols = self.ts.columns[cols]
                    return self.ts[rows, cols]
                return self.ts[self.ts.index[key]]

            def __setitem__(self, key, value):

                def ensure_2d(val):
                    arr = np.asarray(val, dtype=float)
                    if arr.ndim == 0:
                        return arr.reshape(1, 1)
                    elif arr.ndim == 1:
                        return arr.reshape(-1, 1)
                    return arr

                vals = ensure_2d(value)

                if isinstance(key, int):  # gestione riga singola
                    self.ts._data[key, :] = vals.flatten()
                elif isinstance(key, (np.ndarray, list, slice)):
                    self.ts._data[key, :] = vals
                elif isinstance(key, tuple) and len(key) == 2:
                    row_key, col_key = key
                    self.ts._data[row_key, col_key] = vals
                else:
                    raise ValueError("Unsupported key")

        return TimeseriesIXIndexer(self)

    @property
    def unit(self):
        """
        Get the unit of measurement.

        Returns
        -------
        str
            The unit of measurement.
        """
        return f"{self._unit.units:~}"

    @property
    def shape(self):
        return (len(self.index), len(self.columns))

    def set_unit(self, unit: str | pint.Quantity):
        # set the unit of measurement
        msg = "unit must be a string representing a conventional unit "
        msg += "of measurement in the SI sytem or a pint.Quantity"
        if not isinstance(unit, (str, pint.Quantity)):
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
        out = self.copy()
        if axis is None or axis == 0:
            index = out.to_dataframe().dropna(how="all", axis=0).index.to_numpy()
            start = float(np.min(index))
            stop = float(np.max(index))
            out = out[start:stop]
        if axis is None or axis == 1:
            cols = out.to_dataframe().dropna(how="all", axis=1).columns.to_numpy()
            out = out[:cols]
        if inplace:
            self.__setitem__(
                (np.isin(self.index, out.index), np.isin(self.columns, out.columns)),
                out._data,
            )
        else:
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

    def copy(self):
        """
        Return a deep copy of the Timeseries.

        Returns
        -------
        Timeseries
            A new Timeseries object with the same data, index, columns, and unit.
        """
        return self.__class__(
            **{
                i: v.copy() if hasattr(v, "copy") else v
                for i, v in self._get_object_args().items()
            }
        )

    def fillna(self, value=None, n_regressors=None, inplace=False):
        """
        Return a copy with NaNs replaced by the specified value or using advanced imputation for all contained objects.

        Parameters
        ----------
        value : float or int or None, optional
            Value to use for NaNs. If None, use interpolation or regression.
        n_regressors : int or None, optional
            Number of regressors to use for regression-based imputation. If None, use cubic spline interpolation.
        inplace : bool, optional
            If True, fill in place. If False, return a new object.

        Returns
        -------
        LabeledArray
            Filled object.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        vals = sp_fillna(
            self._data.copy(),
            value,
            n_regressors,
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
                    y=values.values.astype(float).flatten().tolist(),
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
            row_mask = np.isin(self.index, rows)
        if cols is None:
            col_mask = slice(None)
        elif isinstance(cols, slice):
            start = cols.start
            stop = cols.stop
            col_mask = np.isin([start, stop], self.columns)
            col_idx = np.where(col_mask)[0]
            col_idx = np.arange(col_idx[0], col_idx[-1] + 1)
            col_mask = np.isin(col_idx, np.arange(len(self.columns)))
        else:
            col_mask = np.isin(self.columns, np.asarray(cols))

        # Create a new object that shares the same data buffer and
        # handle the appropriate object type
        if cols is None:
            view_obj = self.__new__(type(self))
        else:
            view_obj = Timeseries.__new__(Timeseries)

        # populate the object
        for key in dir(self):
            try:
                setattr(view_obj, key, getattr(self, key))
            except Exception as exc:
                pass
        view_obj._data = self._data[row_mask, :][:, col_mask]
        view_obj.index = self.index[row_mask]
        if hasattr(view_obj, "columns"):
            view_obj.columns = self.columns[col_mask]

        # return
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
        attr = getattr(self._data, name, None)
        if attr is None:
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")

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

        return attr

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)) and len(key) == 2:
            row_key, col_key = key
            return self._view(row_key, col_key)
        elif isinstance(key, (slice, np.ndarray)):
            return self._view(key, None)
        elif isinstance(key, str) and key in self.columns:
            return self._view(None, [key])
        elif isinstance(key, int):
            return self._view([key], None)
        else:
            raise ValueError(f"{key} type not supported as item.")

    def __setitem__(self, key, value):

        def ensure_2d(val):
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 0:
                return arr.reshape(1, 1)
            elif arr.ndim == 1:
                return arr.reshape(-1, 1)
            return arr

        def add_rows(key: list[int | float | bool]):
            if np.all([isinstance(i, bool) for i in key]):
                mask = np.asarray(key, bool)
                mask = np.isin(self.index[mask], self.index)
                mask = mask.astype(bool)
            else:
                mask = np.isin(key, self.index)  # type: ignore
                mask = mask.astype(bool)
            if not np.all(mask):
                new_data = np.full(
                    (len(mask) - np.sum(mask), self._data.shape[1]),
                    np.nan,
                )
                self._data = np.vstack([self._data, new_data])
                self.index = np.append(self.index, key[~mask])  # type: ignore
            mask = np.isin(key, self.index)  # type: ignore
            return mask.astype(bool)

        def add_cols(key: list[str]):
            mask = np.isin(key, self.columns)  # type: ignore
            mask = mask.astype(bool)
            if not np.all(mask):
                new_data = np.full(
                    (self._data.shape[0], len(mask) - np.sum(mask)),
                    np.nan,
                )
                self._data = np.hstack([self._data, new_data])
                self.columns = np.append(self.columns, key[mask])  # type: ignore
            mask = np.isin(key, self.columns)  # type: ignore
            return mask.astype(bool)

        vals = ensure_2d(value)

        if isinstance(key, str):  # gestione colonna singola
            self._data[:, add_cols([key])] = vals.flatten()
        elif isinstance(key, (int, float)):  # gestione riga singola
            self._data[add_rows([key]), :] = vals.flatten()
        elif isinstance(key, (np.ndarray, list)):
            if isinstance(key, np.ndarray):
                key = key.tolist()

            # gestione righe multiple
            if np.all([isinstance(i, (float, int, bool)) for i in key]):
                self._data[add_rows(key), :] = vals

            # gestione colonne multiple
            elif np.all([isinstance(i, str) for i in key]):
                self._data[:, add_cols(key)] = vals
            else:
                raise ValueError("key must include string only or int/float only.")
        elif isinstance(key, slice):
            start = self.index[key.start if key.start is not None else 0]
            stop = self.index[key.stop if key.stop is not None else -1]
            row_key = (self.index >= start) & (self.index <= stop)
            row_key = self.index[row_key].tolist()
            self._data[add_rows(row_key), :] = vals
        elif isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key

            # Gestione colonne
            if isinstance(col_key, str):
                col_key = [col_key]
            elif isinstance(col_key, np.ndarray):
                col_key = col_key.tolist()
            elif isinstance(col_key, slice):
                start = 0 if col_key.start is None else col_key.start
                stop = len(self.columns) if col_key.stop is None else col_key.stop
                step = col_key.step if col_key.step is None else col_key.step
                col_key = np.arange(start, stop, step)
                col_key = self.columns[col_key].tolist()
            if not isinstance(col_key, list):
                raise ValueError("Unsupported column key")
            else:
                col_mask = add_cols(col_key)

            # Gestione righe
            if isinstance(row_key, (int, float)):
                row_key = [row_key]
            elif isinstance(row_key, slice):
                start = self.index[0 if row_key.start is None else row_key.start]
                stop = self.index[-1 if row_key.stop is None else row_key.stop]
                row_key = (self.index >= start) & (self.index <= stop)
                row_key = self.index[row_key].tolist()
            if not isinstance(row_key, list):
                raise ValueError("Unsupported row key")
            else:
                row_mask = add_rows(row_key)

            # Assegna i valori
            if vals.shape != (len(row_key), len(col_key)):
                raise ValueError("Shape mismatch between value and target slice")
            self._data[np.ix_(row_mask, col_mask)] = vals

            # ordina righe
            sorting_index = np.argsort(self.index)
            self.index = self.index[sorting_index]
            self._data = self._data[sorting_index, :]

        else:
            raise ValueError("Unsupported key type for __setitem__")

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
