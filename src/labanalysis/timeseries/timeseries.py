"""
Base Timeseries class for time-indexed data.
"""

import inspect

import numpy as np
import pandas as pd
import pint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..events import Signal
from ..indexers.timeseries_iloc_indexer import TimeseriesILocIndexer
from ..indexers.timeseries_loc_indexer import TimeseriesLocIndexer
from ..signalprocessing import fillna as sp_fillna
from ..utils import FloatArray1D, FloatArray2D, TextArray1D, ureg


class Timeseries:
    """
    Time-indexed multi-column data container with unit support.

    Base class for time-series data providing pandas-like indexing, arithmetic
    operations, unit conversion, and signal processing capabilities. Designed for
    biomechanical and physiological signals.
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

        # Optimize: create DataFrame only once if needed, or use np.isnan directly
        if axis is None:
            # Need both operations, create DataFrame once
            df = out.to_dataframe()

            # Axis 0: remove all-NaN rows
            row_mask = ~df.isna().all(axis=1).to_numpy()
            if row_mask.any():
                row_indices = np.where(row_mask)[0]
                start_idx = row_indices[0]
                stop_idx = row_indices[-1] + 1

                # Direct slicing instead of using .loc
                out._data = out._data[start_idx:stop_idx, :]
                out.index = out.index[start_idx:stop_idx]

            # Axis 1: remove all-NaN columns
            col_mask = ~df.isna().all(axis=0).to_numpy()
            if col_mask.any():
                col_indices = np.where(col_mask)[0]
                start_col = col_indices[0]
                stop_col = col_indices[-1] + 1

                # Direct slicing
                out._data = out._data[:, start_col:stop_col]
                out.columns = out.columns[start_col:stop_col]

        elif axis == 0:
            # Only axis 0: use isnan directly on _data (avoid DataFrame creation)
            row_mask = ~np.isnan(out._data).all(axis=1)
            if row_mask.any():
                row_indices = np.where(row_mask)[0]
                start_idx = row_indices[0]
                stop_idx = row_indices[-1] + 1

                out._data = out._data[start_idx:stop_idx, :]
                out.index = out.index[start_idx:stop_idx]

        elif axis == 1:
            # Only axis 1: use isnan directly on _data (avoid DataFrame creation)
            col_mask = ~np.isnan(out._data).all(axis=0)
            if col_mask.any():
                col_indices = np.where(col_mask)[0]
                start_col = col_indices[0]
                stop_col = col_indices[-1] + 1

                out._data = out._data[:, start_col:stop_col]
                out.columns = out.columns[start_col:stop_col]

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

    def fillna(self, value=None, mice: bool = False, max_iter: int = 10, inplace=False):
        """
        Return a copy with NaNs replaced by the specified value or using advanced imputation for all contained objects.

        Parameters
        ----------
        value : float or int or None, optional
            Value to use for NaNs. If None, use interpolation or regression.
        mice : bool, optional
            If True, use multiple imputation by chained equations.
        max_iter : int, optional
            Maximum number of iterations for multiple imputation.
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
            mice,
            max_iter,
            None,
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

    def _view(self, rows=None, cols=None):
        """
        Return a view/subset of this Timeseries-like object while preserving
        the concrete subclass type only when the column structure is unchanged.

        If the selection changes the number of columns (e.g. 3 -> 2, 3 -> 1,
        4 -> 3), the returned object is a base Timeseries instead of the
        specialized subclass.
        """

        def _normalize_selector(selector, labels):
            if selector is None:
                return slice(None)

            if isinstance(selector, slice):
                return selector

            if isinstance(selector, (np.ndarray, list, tuple)):
                arr = np.asarray(selector)

                if arr.dtype == bool:
                    return arr.astype(bool, copy=False)

                if arr.dtype.kind in "iu":
                    return arr.astype(int, copy=False)

                labels_arr = np.asarray(labels)

                if labels_arr.dtype.kind in "fiu":
                    mask = np.zeros(len(labels_arr), dtype=bool)
                    for value in arr.astype(float, copy=False):
                        mask |= np.isclose(
                            labels_arr.astype(float),
                            value,
                            rtol=1e-6,
                            atol=1e-8,
                        )
                    return np.flatnonzero(mask)

                mask = np.zeros(len(labels_arr), dtype=bool)
                for value in arr.tolist():
                    mask |= labels_arr == value
                return np.flatnonzero(mask)

            if isinstance(selector, (bool, np.bool_)):
                return np.array([selector], dtype=bool)

            if isinstance(selector, (int, np.integer)):
                return np.array([int(selector)], dtype=int)

            labels_arr = np.asarray(labels)
            if labels_arr.dtype.kind in "fiu":
                mask = np.isclose(
                    labels_arr.astype(float),
                    float(selector),
                    rtol=1e-6,
                    atol=1e-8,
                )
                return np.flatnonzero(mask)

            mask = labels_arr == selector
            return np.flatnonzero(mask)

        row_sel = _normalize_selector(rows, self.index)
        col_sel = _normalize_selector(cols, self.columns)

        if isinstance(row_sel, np.ndarray) and row_sel.dtype == bool:
            row_sel = np.flatnonzero(row_sel)
        if isinstance(col_sel, np.ndarray) and col_sel.dtype == bool:
            col_sel = np.flatnonzero(col_sel)

        if isinstance(row_sel, slice) and isinstance(col_sel, slice):
            view_data = self._data[row_sel, col_sel]
        elif isinstance(row_sel, slice):
            view_data = self._data[row_sel, col_sel]
        elif isinstance(col_sel, slice):
            view_data = self._data[row_sel, col_sel]
        else:
            view_data = self._data[np.ix_(row_sel, col_sel)]

        view_index = self.index[row_sel]
        view_columns = self.columns[col_sel]

        view_obj = self.__new__(self._get_view_class(rows=row_sel, cols=col_sel))
        try:
            view_obj._unit = self._unit
        except AttributeError:
            view_obj._unit = "dimensionless"

        self._copy_view_attributes(view_obj)

        view_obj._data = view_data
        view_obj.index = (
            view_index.copy() if isinstance(view_index, np.ndarray) else view_index
        )
        view_obj.columns = (
            view_columns.copy()
            if isinstance(view_columns, np.ndarray)
            else view_columns
        )

        return view_obj

    def _get_view_class(self, rows=None, cols=None):
        """
        Return the class to use for a view object.

        Preserve the concrete subclass only when the selected columns keep the same
        dimensionality as the original object. If the number of selected columns
        changes, return the base Timeseries class.
        """
        if cols is None:
            return type(self)

        if isinstance(cols, slice):
            if cols == slice(None):
                selected_columns = self._data.shape[1]
            else:
                selected_columns = len(range(*cols.indices(self._data.shape[1])))
        elif isinstance(cols, np.ndarray):
            if cols.dtype == bool:
                selected_columns = int(np.count_nonzero(cols))
            else:
                selected_columns = len(cols)
        elif isinstance(cols, (list, tuple)):
            selected_columns = len(cols)
        else:
            selected_columns = 1

        original_columns = self._data.shape[1]

        if selected_columns != original_columns:
            return Timeseries

        return type(self)

    def _copy_view_attributes(self, view_obj):
        """
        Copy subclass-specific attributes to a view object.

        This method is called during slicing operations to preserve
        subclass-specific attributes. Subclasses can override this method
        to specify which attributes should be copied.

        The default implementation copies common attributes used by
        Signal3D and EMGSignal subclasses.

        Parameters
        ----------
        view_obj : Timeseries
            The new view object to copy attributes to.

        Notes
        -----
        Subclasses should call super()._copy_view_attributes(view_obj)
        to ensure parent class attributes are also copied.

        Examples
        --------
        Override in a subclass:

        >>> def _copy_view_attributes(self, view_obj):
        ...     super()._copy_view_attributes(view_obj)
        ...     if hasattr(self, '_my_custom_attr'):
        ...         view_obj._my_custom_attr = self._my_custom_attr
        """
        # Copy Signal3D/Point3D attributes
        if hasattr(self, "_vertical_axis"):
            view_obj._vertical_axis = self._vertical_axis
        if hasattr(self, "_anteroposterior_axis"):
            view_obj._anteroposterior_axis = self._anteroposterior_axis

        # Copy EMGSignal attributes
        if hasattr(self, "_name"):
            view_obj._name = self._name
        if hasattr(self, "_side"):
            view_obj._side = self._side

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
            pass

        data = object.__getattribute__(self, "_data")
        attr = getattr(data, name, None)
        if attr is None:
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")

        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
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
        if isinstance(key, str) or (
            isinstance(key, list) and all(isinstance(k, str) for k in key)
        ):
            if isinstance(key, str) and key not in self.columns:
                try:
                    return getattr(self, key)
                except AttributeError:
                    pass
            return self.loc[:, key]
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
        if isinstance(key, str) or (
            isinstance(key, list) and all(isinstance(k, str) for k in key)
        ):
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


__all__ = ["Timeseries"]
