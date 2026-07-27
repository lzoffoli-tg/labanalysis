"""Record base class module."""

from __future__ import annotations

import inspect
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..indexers.record_iloc_indexer import RecordILocIndexer
from ..indexers.record_loc_indexer import RecordLocIndexer
from ..io.read.btsbioengineering import read_tdf
from ..signalprocessing import fillna as sp_fillna
from ..timeseries import *


class Record:
    """
    A dictionary-like container for Timeseries objects,
    supporting type filtering and DataFrame conversion.

    Parameters
    ----------
    **signals : dict
        Key-value pairs of Timeseries or Record objects to include in the record.

    Attributes
    ----------
    index : ndarray
        The sorted union of all time indices across contained objects.
    shape : tuple of int
        The shape of the Record as a DataFrame (n_rows, n_columns).
    loc : RecordLocIndexer
        Label-based indexer.
    iloc : RecordILocIndexer
        Position-based indexer.

    Methods
    -------
    copy()
        Return a deep copy of the Record.
    strip(axis=None, inplace=False, independent=False)
        Remove leading/trailing rows or columns that are all NaN from all contained objects.
    reset_time(inplace=False, time_zero=None)
        Reset the time index to start at zero for all contained objects.
    fillna(value=None, mice=False, max_iter=10, inplace=False)
        Fill NaNs for all contained objects.
    to_dataframe()
        Convert the record to a pandas DataFrame with MultiIndex columns.
    to_plotly_figure()
        Convert the record to a Plotly figure with subplots.
    get(key, default=None)
        Get an item with a default fallback.
    drop(key, inplace=False)
        Remove one or more items from the Record.
    from_tdf(filename)
        Create a Record from a TDF file (class method).
    """

    @property
    def name(self):
        """
        Get the name of the class.

        Returns
        -------
        str
            The class name.
        """
        return self.__class__.__name__

    @property
    def index(self):
        """
        Get the index shared across all elements in the record.

        Returns
        -------
        1D numpy array of floats
            A sorted, unique array of all time indices.
        """
        return np.unique(np.concatenate([i.index for i in self.values()]))

    @property
    def shape(self):
        """
        Get the shape of the Record as a DataFrame.

        Returns
        -------
        tuple of int
            A (n_rows, n_columns) tuple representing the shape.
        """
        return self.to_dataframe().shape

    @property
    def loc(self):
        """
        Label-based indexer for Record items.

        Returns
        -------
        RecordLocIndexer
            An indexer object for label-based slicing.
        """
        return RecordLocIndexer(self)

    @property
    def iloc(self):
        """
        Position-based indexer for Record items.

        Returns
        -------
        RecordILocIndexer
            An indexer object for integer-position based slicing.
        """
        return RecordILocIndexer(self)

    def __len__(self):
        """
        Get the number of items in the Record.

        Returns
        -------
        int
            Number of key-value pairs.
        """
        return len(self._data)

    def _view(
        self,
        rows: slice | list[int | float | bool] | np.ndarray | None = None,
    ):
        """
        Create a view of the Record with a subset of rows.

        Parameters
        ----------
        rows : slice, list of int/float/bool, ndarray, or None
            Row selection to apply to all contained objects.

        Returns
        -------
        Record or subclass
            A view object with the same type as self.
        """
        # get a view
        view_obj = type(self).__new__(type(self))
        keys = self.__dict__
        for key in keys:
            if key != "_data":
                setattr(view_obj, key, getattr(self, key))

        # set the views
        view_obj._data = {}
        for key in self.keys():
            view_obj._data[key] = self._data[key][rows]

        # return
        return view_obj

    def __getitem__(self, key):
        """
        Get an item from the Record by key or indexing.

        Parameters
        ----------
        key : str, slice, ndarray, list, int, or float
            If str, retrieves from _data or as an attribute.
            If slice/ndarray/list/int/float, creates a view with row selection.

        Returns
        -------
        Timeseries, Record, or None
            The requested item or view.

        Raises
        ------
        ValueError
            If the key type is not supported.
        """
        # Se è una stringa, controlla sia in _data che come attributo/property
        if isinstance(key, str):
            # Prima controlla in _data
            if key in self.keys():
                return self._data[key]
            # Altrimenti prova come property/attributo
            elif hasattr(self, key):
                return getattr(self, key)
            else:
                # raise KeyError(f"'{key}' not found in _data or as attribute")
                return None
        elif key in self.keys():
            return self._data[key]
        elif isinstance(key, (slice, np.ndarray, list)):
            return self._view(key)
        elif isinstance(key, (int, float)):
            return self._view([key])
        else:
            raise ValueError(f"{key} type not supported as item.")

    def __setitem__(self, key, value):
        """
        Set an item in the Record.

        Parameters
        ----------
        key : str
            The key for the item.
        value : Timeseries or Record
            The value to set.

        Raises
        ------
        ValueError
            If key is not a string or value is not a Timeseries/Record.
        """
        if not isinstance(key, str):
            raise ValueError("key must be a str")

        if not isinstance(value, (Timeseries, Record)):
            raise ValueError("value must be a Timeseries or Record")
        # if key in self.keys() and hasattr(self, key):
        #     raise ValueError(f"{key} is a property of this Record.")
        self._data[key] = value

    def __getattr__(self, name):
        # Use object.__getattribute__ to avoid infinite recursion during unpickling
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError:
            raise AttributeError("_data is not a valid attribute of this Record")
        if name in data.keys():
            return data[name]
        raise AttributeError(f"{name} is not a valid attribute of this Record")

    def __setattr__(self, key: str, value: object):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __repr__(self):
        """
        Return a string representation of the Record.

        Returns
        -------
        str
            String representation of the internal _data dictionary.
        """
        return self._data.__repr__()

    def __init__(self, **signals: Timeseries | Record):
        """
        Initialize the Record with key-value pairs.

        Parameters
        ----------
        **signals : dict
            Key-value pairs where values are Timeseries or Record objects.
        """
        self._data: dict[str, Timeseries | Record] = {}
        for key, value in signals.items():
            self[key] = value

    def items(self):
        """
        Return a list of (key, value) pairs.

        Returns
        -------
        list of tuple
            List of (key, value) tuples from the internal dictionary.
        """
        return list(zip(self.keys(), self.values()))

    def keys(self):
        """
        Return a list of keys.

        Returns
        -------
        list of str
            List of all keys in the Record.
        """
        return list(self._data.keys())

    def values(self):
        """
        Return a list of values.

        Returns
        -------
        list of Timeseries or Record
            List of all values in the Record.
        """
        return list(self._data.values())

    def to_dataframe(self):
        """
        Convert the record to a pandas DataFrame with MultiIndex columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all the data from the Record.
        """
        if len(self._data) == 0:
            return pd.DataFrame()
        dfr_list = []
        for key, val in self.items():
            new = val.to_dataframe()
            cols = [
                " ".join([key, i]) if not i.startswith(key) else i for i in new.columns
            ]
            new.columns = pd.Index(cols)
            dfr_list += [new]
        return pd.concat(dfr_list, axis=1).sort_index(axis=0)

    def _get_constructor_args(self):
        """
        Extracts constructor arguments and internal attributes from the current
        instance to allow dynamic instantiation of self.__class__.

        Returns
        -------
        dict
            A dictionary of constructor arguments and internal attributes.
        """
        sig = inspect.signature(self.__class__.__init__)
        args = {}

        # Capture constructor parameters
        for name, param in sig.parameters.items():
            if name == "self":
                continue

            value = None
            if hasattr(self, name) and name not in self._data:
                value = getattr(self, name)
            elif hasattr(self, f"_{name}"):
                value = getattr(self, f"_{name}")
            elif param.default is not inspect.Parameter.empty:
                value = param.default
            else:
                # Don't raise, just skip - may be handled by **kwargs
                continue

            args[name] = value

        # Additionally capture internal attributes (like Timeseries._get_object_args does)
        for attr in dir(self):
            if attr.startswith("_") and not attr.startswith("__"):
                # Skip if already captured or if it's _data (handled separately)
                if attr in args or attr == "_data":
                    continue
                # Only capture non-callable attributes
                if hasattr(self.__class__, attr):
                    value = getattr(self, attr)
                    if not callable(value):
                        args[attr] = value

        return args

    def copy(self):
        """
        Return a deep copy of the Record, preserving the concrete subclass type.

        Returns
        -------
        Record or subclass
            A new instance of the same class with copied data and attributes.
        """
        # Get constructor arguments and internal attributes
        constructor_args = self._get_constructor_args()

        # Deep copy the _data dictionary
        data_copy = {key: val.copy() for key, val in self._data.items()}

        # Merge constructor args with data copy
        # Data copy takes precedence to ensure we have the latest state
        return self.__class__(**{**constructor_args, **data_copy})

    def strip(
        self,
        inplace: bool = False,
        independent: bool = False,
    ):
        """
        Remove leading/trailing rows or columns that are all NaN from all
        contained Timeseries-like objects.

        Parameters
        ----------
        inplace : bool, optional
            If True, modifies in place. If False, returns a new Record (default False).
        independent : bool, optional
            Controls whether elements are stripped independently or share a common
            timeframe (default False).
            - If True: Each element is stripped based on its own non-NaN values,
              potentially resulting in different timeframes per element (original behavior).
            - If False: All elements share a common timeframe from the first time index
              where at least one element has a non-NaN value to the last time index
              where at least one element has a non-NaN value. This ensures all elements
              span the same time period after stripping.

        Note: When axis=1 (column stripping), this parameter has no effect as
            columns are always stripped independently per element.

        Returns
        -------
        Record or None
            Stripped Record if inplace is False, otherwise None. When independent=False,
            all elements will have identical time index ranges after stripping.

        Examples
        --------
        >>> from records.records import Record
        >>> from records.timeseries import Signal1D
        >>> import numpy as np
        >>>
        >>> # Create two signals with different NaN patterns
        >>> data_a = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
        >>> data_b = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        >>> sig_a = Signal1D(data_a, index=[0, 1, 2, 3, 4], unit="m")
        >>> sig_b = Signal1D(data_b, index=[0, 1, 2, 3, 4], unit="m")
        >>> rec = Record(signal_a=sig_a, signal_b=sig_b)
        >>>
        >>> # Independent stripping (each element has own timeframe)
        >>> rec_ind = rec.strip(independent=True)
        >>> rec_ind['signal_a'].index  # [1, 2, 3]
        >>> rec_ind['signal_b'].index  # [0, 2, 4]
        >>>
        >>> # Shared timeframe stripping (all elements share timeframe)
        >>> rec_shared = rec.strip(independent=False)
        >>> rec_shared['signal_a'].index  # [0, 1, 2, 3, 4]
        >>> rec_shared['signal_b'].index  # [0, 1, 2, 3, 4]
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if not isinstance(independent, bool):
            raise ValueError("independent must be True or False")

        out = self if inplace else self.copy()
        if independent:
            for key in out.keys():
                out[key].strip(inplace=True)
        elif len(out._data) > 0:

            # find all valid indices
            def get_stripped_index(x: Timeseries | Record):
                if isinstance(x, Timeseries):
                    return x.index[~np.isnan(x._data).all(axis=1)]
                elif isinstance(x, EMGSignal):
                    return np.array([])
                else:
                    return np.concatenate(
                        [get_stripped_index(i) for i in x.values()]
                    ).flatten()

            all_valid_indices = np.unique(get_stripped_index(out))

            # resize all data in out
            def resize_data(x: Timeseries | Record, start: float, stop: float):
                if isinstance(x, Timeseries):
                    mask = (x.index >= start) & (x.index <= stop)
                    setattr(x, "_data", x._data[mask, :])
                    setattr(x, "index", x.index[mask])
                else:
                    for v in x.values():
                        resize_data(v, start, stop)

            start = float(np.min(all_valid_indices))
            stop = float(np.max(all_valid_indices))
            resize_data(out, start, stop)

        if not inplace:
            return out

    def reset_time(self, time_zero: float = 0.0, inplace: bool = False):
        """
        Reset the time index to start at zero for all contained Timeseries-like
        objects.

        Parameters
        ----------
        inplace : bool, optional
            If True, modify in place. If False, return a new Record.

        time_zero: float, optional
            the zero time of the new index

        Returns
        -------
        Record or None
            A Record with reset time if inplace is False, otherwise
            None.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if not isinstance(time_zero, float):
            raise ValueError("time_zero must be a float.")

        out = self if inplace else self.copy()

        def adjust_time(x: Timeseries | Record):
            if isinstance(x, Timeseries):
                x.index = x.index - x.index[0] + time_zero
            else:
                for v in x.values():
                    adjust_time(v)

        adjust_time(out)

        if not inplace:
            return out

    def fillna(self, value=None, mice: bool = False, max_iter: int = 50, inplace=False):
        """
        Return a copy with NaNs replaced by the specified value or using
        advanced imputation for all contained objects.

        Parameters
        ----------
        value : float or int or None, optional
            Value to use for NaNs. If None, use interpolation or regression.
        mice:bool, optional
            If True, use multiple imputation by chained equations.
        max_iter : int, optional
            Maximum number of iterations for multiple imputation.
        inplace : bool, optional
            If True, fill in place. If False, return a new object.

        Returns
        -------
        Record
            Filled record.
        """

        def fill_record(
            record: Record,
            vals: np.ndarray,
            counter: int,
        ):
            for key, value in record.items():
                if isinstance(value, Record):
                    counter = fill_record(
                        value,
                        vals,
                        counter,
                    )
                cols = record[key].shape[1]
                record[key][:, :] = vals[:, np.arange(cols) + counter]
                counter += cols

            return counter

        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        vals = sp_fillna(
            self.to_dataframe(),
            value,
            mice,
            max_iter,
            None,
            False,
        )
        vals = np.asarray(vals, float)
        if inplace:
            _ = fill_record(self, vals, 0)
        else:
            out = self.copy()
            _ = fill_record(out, vals, 0)
            return out

    def get(self, key: str, default=None):
        """
        Get an item from the Record with a default fallback.

        Parameters
        ----------
        key : str
            The key to retrieve.
        default : any, optional
            The value to return if key is not found (default None).

        Returns
        -------
        Timeseries, Record, or default
            The value associated with key, or default if not found.
        """
        return self._data.get(key, default)

    def drop(self, key: str | list[str], inplace: bool = False):
        """
        Remove one or more items from the Record.

        Parameters
        ----------
        key : str or list of str
            The key(s) to remove.
        inplace : bool, optional
            If True, modify in place. If False, return a new Record (default False).

        Returns
        -------
        Record or None
            Modified Record if inplace is False, otherwise None.

        Warnings
        --------
        Warns if a key is not found.
        """
        if isinstance(key, str):
            key = [key]
        out = self if inplace else self.copy()
        for element in key:
            if element not in out.keys():
                warn(f"{element} not found.")
            else:
                _ = out._data.pop(element)
        if not inplace:
            return out

    def to_plotly_figure(self):
        """
        Convert the Record to a Plotly figure with subplots.

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure with one row per column in the DataFrame,
            each showing a time series plot.
        """
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

    @property
    def vertical_axis(self):
        """
        Get the vertical axis from the first element that defines one.

        Returns
        -------
        str or None
            The vertical axis label, or None if not found.
        """
        for val in self.values():
            if hasattr(val, "vertical_axis"):
                axis = val.vertical_axis
                if axis is not None:
                    return str(axis)
        return None

    @property
    def anteroposterior_axis(self):
        """
        Get the anteroposterior axis from the first element that defines one.

        Returns
        -------
        str or None
            The anteroposterior axis label, or None if not found.
        """
        for val in self.values():
            if hasattr(val, "anteroposterior_axis"):
                axis = val.anteroposterior_axis
                if axis is not None:
                    return str(axis)
        return None

    @property
    def lateral_axis(self):
        """
        Get the lateral axis from the first element that defines one.

        Returns
        -------
        str or None
            The lateral axis label, or None if not found.
        """
        for val in self.values():
            if hasattr(val, "lateral_axis"):
                axis = val.lateral_axis
                if axis is not None:
                    return str(axis)
        return None

    @property
    def points3d(self):
        """
        Get all Point3D objects.

        Returns
        -------
        Record
        """
        return self._filter_by_type(Point3D)

    @property
    def signals3d(self):
        """
        Get all Signal3D objects.

        Returns
        -------
        Record
        """
        return self._filter_by_type(Signal3D)

    @property
    def signals1d(self):
        """
        Get all Signal1D objects.

        Returns
        -------
        Record
        """
        return self._filter_by_type(Signal1D)

    @property
    def emgsignals(self):
        """
        Get all EMGSignal objects.

        Returns
        -------
        Record
        """
        return self._filter_by_type(EMGSignal)

    @property
    def forceplatforms(self):
        """
        Get all ForcePlatform objects.

        Returns
        -------
        Record
        """
        from .forceplatform import ForcePlatform

        return self._filter_by_type(ForcePlatform)

    @property
    def metabolicrecords(self):
        """
        Get all MetabolicRecord objects.

        Returns
        -------
        Record
        """
        from .metabolicrecord import MetabolicRecord

        return self._filter_by_type(MetabolicRecord)

    @property
    def planes3d(self):
        """
        Get all Plane3D objects.

        Returns
        -------
        Record
        """
        return self._filter_by_type(Plane3D)

    @property
    def timeseries(self):
        """
        Get all Timeseries objects.

        Returns
        -------
        Record
        """
        return self._filter_by_type(Timeseries)

    @property
    def resultant_force(self):
        """
        Return a ForcePlatform object representing the resultant of all
        available force platforms.

        Returns
        -------
        ForcePlatform
            A combined force platform with summed forces, torques, and
            weighted-average origins.

        Raises
        ------
        ValueError
            If no force platforms are found in the Record.
        """
        platforms = list(self.forceplatforms.values())
        if len(platforms) == 0:
            raise ValueError("No forceplatforms found within the Record.")

        # Indice temporale comune a tutte le piattaforme
        common_times_array = [platform.index for platform in platforms]
        common_times_array = np.unique(np.concatenate(common_times_array))
        common_times = common_times_array.tolist()
        common_time_to_idx = {t: i for i, t in enumerate(common_times)}
        n_samples = len(common_times)

        force_accum = np.zeros((n_samples, 3), dtype=float)
        torque_accum = np.zeros((n_samples, 3), dtype=float)
        origin_weighted = np.zeros((n_samples, 3), dtype=float)
        weight_accum = np.zeros(n_samples, dtype=float)
        valid_count = np.zeros(n_samples, dtype=int)

        axes = []
        units = {}

        for platform in platforms:
            force = platform.force
            origin = platform.origin
            torque = platform.torque

            f_arr = np.asarray(force.to_numpy(), dtype=float)
            o_arr = np.asarray(origin.to_numpy(), dtype=float)
            m_arr = np.asarray(torque.to_numpy(), dtype=float)

            if f_arr.ndim != 2 or f_arr.shape[1] != 3:
                continue
            if o_arr.shape != f_arr.shape or m_arr.shape != f_arr.shape:
                continue

            platform_times = list(platform.index)
            platform_ids = np.fromiter(
                (common_time_to_idx[t] for t in platform_times),
                dtype=int,
                count=len(platform_times),
            )

            v_axis_idx = next(
                (
                    i
                    for i, col in enumerate(force.columns)
                    if col == force.vertical_axis
                ),
                None,
            )
            if v_axis_idx is None:
                weights = np.linalg.norm(f_arr, axis=1)
            else:
                weights = np.abs(f_arr[:, v_axis_idx]).reshape(-1)

            valid_mask = (
                np.isfinite(f_arr).all(axis=1)
                & np.isfinite(o_arr).all(axis=1)
                & np.isfinite(m_arr).all(axis=1)
                & np.isfinite(weights)
            )

            if not np.any(valid_mask):
                continue

            ids = platform_ids[valid_mask]
            f_valid = f_arr[valid_mask]
            o_valid = o_arr[valid_mask]
            m_valid = m_arr[valid_mask]
            w_valid = weights[valid_mask]

            moments = m_valid + np.cross(o_valid, f_valid)

            np.add.at(force_accum, ids, f_valid)
            np.add.at(torque_accum, ids, moments)
            np.add.at(origin_weighted, ids, o_valid * w_valid[:, None])
            np.add.at(weight_accum, ids, w_valid)
            np.add.at(valid_count, ids, 1)

            if not units:
                units["origin"] = origin.unit
                units["force"] = force.unit
                units["torque"] = torque.unit
                axes = list(origin.columns)

        force_out = np.full((n_samples, 3), np.nan, dtype=float)
        torque_out = np.full((n_samples, 3), np.nan, dtype=float)
        origin_out = np.full((n_samples, 3), np.nan, dtype=float)

        valid_samples = valid_count > 0
        force_out[valid_samples] = force_accum[valid_samples]
        torque_out[valid_samples] = torque_accum[valid_samples]

        with np.errstate(divide="ignore", invalid="ignore"):
            nonzero_weight = valid_samples & (weight_accum != 0)
            origin_out[nonzero_weight] = (
                origin_weighted[nonzero_weight] / weight_accum[nonzero_weight, None]
            )

        cop = Point3D(origin_out, common_times, units["origin"], axes)
        force = Signal3D(force_out, common_times, units["force"], axes)
        torque = Signal3D(torque_out, common_times, units["torque"], axes)

        from .forceplatform import ForcePlatform

        return ForcePlatform(cop, force, torque)

    def _filter_by_type(self, cls):
        """
        Internal: Filter contained items by type.

        Parameters
        ----------
        cls : type

        Returns
        -------
        Record
            A view (not a copy) of the filtered items.
            Changes to elements affect the original Record.
        """

        return Record(
            **{k: v for k, v in self.items() if type(v) == cls},
        )

    @classmethod
    def from_tdf(cls, filename: str | Path):
        """
        Create a Record from a TDF file.

        Parameters
        ----------
        filename : str
            Path to the TDF file.

        Returns
        -------
        Record
            A Record populated with the data from the TDF file.
        """
        data = read_tdf(filename)
        vals = {}

        # Handle 3D points from CAMERA TRACKED
        if data.get("CAMERA") and data["CAMERA"].get("TRACKED"):  # type: ignore
            df = data["CAMERA"]["TRACKED"]["TRACKS"]  # type: ignore
            for label in df.columns.get_level_values(0).unique():
                sub_df: pd.DataFrame = df[label]
                vals[label] = Point3D(
                    data=sub_df.values,
                    index=sub_df.index.tolist(),
                    columns=sub_df.columns.get_level_values(0).tolist(),
                    unit=sub_df.columns[0][-1],
                )

        # Handle EMG signals
        if data.get("EMG") and data["EMG"].get("TRACKS") is not None:  # type: ignore
            df = data["EMG"]["TRACKS"]  # type: ignore
            for col in df.columns:
                signal: pd.Series = df[col]
                muscle_name, side, unit = col
                vals[f"{side}_{muscle_name}".lower()] = EMGSignal(
                    data=signal.to_numpy().astype(float).flatten(),
                    index=df.index.tolist(),
                    muscle_name=muscle_name.lower(),
                    side=side.lower(),
                    unit=unit,
                )

        # Handle Force Platforms
        from .forceplatform import ForcePlatform

        if data.get("FORCE_PLATFORM") and data["FORCE_PLATFORM"].get("TRACKED"):  # type: ignore
            df = data["FORCE_PLATFORM"]["TRACKED"]["TRACKS"]  # type: ignore
            for label in df.columns.get_level_values("LABEL").unique():
                origin: pd.DataFrame = df[label]["ORIGIN"]
                force: pd.DataFrame = df[label]["FORCE"]
                torque: pd.DataFrame = df[label]["TORQUE"]
                vals[label] = ForcePlatform(
                    origin=Point3D(
                        data=origin.values,
                        index=origin.index.tolist(),
                        columns=origin.columns.get_level_values(0).tolist(),
                        unit=origin.columns[0][-1],
                    ),
                    force=Signal3D(
                        data=force.values,
                        index=force.index.tolist(),
                        columns=force.columns.get_level_values(0).tolist(),
                        unit=force.columns[0][-1],
                    ),
                    torque=Signal3D(
                        data=torque.values,
                        index=torque.index.tolist(),
                        columns=torque.columns.get_level_values(0).tolist(),
                        unit=torque.columns[0][-1],
                    ),
                )

        return cls(**vals)


__all__ = ["Record"]
