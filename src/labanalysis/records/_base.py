"""Record base class module."""

import inspect
from warnings import warn

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..signalprocessing import fillna as sp_fillna
from ..timeseries import Timeseries, Signal1D, Signal3D, EMGSignal, Point3D
from ._loc_indexer import RecordLocIndexer
from ._iloc_indexer import RecordILocIndexer


class Record:
    """
    A dictionary-like container for Timeseries,
    supporting type filtering and DataFrame conversion.

    Parameters
    ----------
    vertical_axis : str, optional
        The label for the vertical axis (default "Y").
    anteroposterior_axis : str, optional
        The label for the anteroposterior axis (default "Z").
    strip : bool, optional
        If True, remove leading/trailing rows or columns that are all NaN from all contained objects (default True).
    reset_time : bool, optional
        If True, reset the time index to start at zero for all contained objects (default True).
    **signals : dict
        Key-value pairs of Timeseries subclasses, TimeseriesRecord, or ForcePlatform to include in the record.

    Attributes
    ----------
    _vertical_axis : str
        The vertical axis label.
    _antpos_axis : str
        The anteroposterior axis label.

    Methods
    -------
    copy()
        Return a deep copy of the TimeseriesRecord.
    strip(axis=0, inplace=False, independent=False)
        Remove leading/trailing rows or columns that are all NaN from all contained objects.
        When independent=False (default), all elements share a common timeframe based on
        the union of non-NaN time points.
    reset_time(inplace=False)
        Reset the time index to start at zero for all contained objects.
    apply(func, axis=0, inplace=False, *args, **kwargs)
        Apply a function or ProcessingPipeline to all contained objects.
    fillna(value=None, regressors=None, inplace=False)
        Fill NaNs for all contained objects.
    to_dataframe()
        Convert the record to a pandas DataFrame with MultiIndex columns.
    from_tdf(filename)
        Create a TimeseriesRecord from a TDF file.
    """

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
        return self.to_dataframe().shape

    @property
    def loc(self):
        """Label-based indexer for Record items."""
        return RecordLocIndexer(self)

    @property
    def iloc(self):
        """Position-based indexer for Record items."""
        return RecordILocIndexer(self)

    def __len__(self):
        return len(self._data)

    def _view(
        self,
        rows: slice | list[int | float | bool] | np.ndarray | None = None,
    ):
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
        # Se è una stringa, controlla sia in _data che come attributo/property
        if isinstance(key, str):
            # Prima controlla in _data
            if key in self.keys():
                return self._data[key]
            # Altrimenti prova come property/attributo
            elif hasattr(self, key):
                return getattr(self, key)
            else:
                raise KeyError(f"'{key}' not found in _data or as attribute")
        elif key in self.keys():
            return self._data[key]
        elif isinstance(key, (slice, np.ndarray, list)):
            return self._view(key)
        elif isinstance(key, (int, float)):
            return self._view([key])
        else:
            raise ValueError(f"{key} type not supported as item.")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError("key must be a str")
        if not isinstance(value, (Timeseries, Signal1D, Signal3D, EMGSignal, Point3D)):
            raise ValueError("value must be a Timeseries or Record")
        if key in self.keys() and hasattr(self, key):
            raise ValueError(f"{key} is a property of this Record.")
        self._data[key] = value

    def __getattr__(self, name):
        # Use object.__getattribute__ to avoid infinite recursion during unpickling
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError:
            raise AttributeError(f"{name} is not a valid attribute of this Record")

        if name in data.keys():
            return data[name]
        raise AttributeError(f"{name} is not a valid attribute of this Record")

    def __setattr__(self, key: str, value: object):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __repr__(self):
        return self._data.__repr__()

    def __init__(
        self,
        **signals: Timeseries | Signal1D | Signal3D | EMGSignal | Point3D,
    ):
        self._data: dict[
            str,
            Timeseries | Signal1D | Signal3D | EMGSignal | Point3D,
        ] = {}
        for key, value in signals.items():
            self[key] = value

    def items(self):
        return list(zip(self.keys(), self.values()))

    def keys(self):
        return list(self._data.keys())

    def values(self):
        return list(self._data.values())

    def to_dataframe(self):
        """
        Convert the record to a pandas DataFrame with MultiIndex columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all the data from the TimeseriesRecord.
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
        Extracts constructor arguments from the current instance to allow
        dynamic instantiation of self.__class__.
        """
        sig = inspect.signature(self.__class__.__init__)
        args = {}
        for name, param in sig.parameters.items():
            try:
                if name == "self":
                    continue
                if hasattr(self, name):
                    args[name] = getattr(self, name)
                elif hasattr(self, f"_{name}"):
                    args[name] = getattr(self, f"_{name}")
                elif param.default is not inspect.Parameter.empty:
                    args[name] = param.default
                else:
                    raise AttributeError(
                        f"Missing required constructor argument: '{name}'"
                    )
            except Exception as exc:
                pass
        return args

    def copy(self):
        """
        Return a deep copy of the TimeseriesRecord.

        Returns
        -------
        TimeseriesRecord
            A new TimeseriesRecord object with the same data.
        """
        return Record(**{i: v.copy() for i, v in self._data.items()})

    def strip(self, axis: int | None = None, inplace: bool = False, independent: bool = False):
        """
        Remove leading/trailing rows or columns that are all NaN from all
        contained Timeseries-like objects.

        Parameters
        ----------
        axis : int or None, optional
            If 0, strip rows (time axis). If 1, strip columns. If None, strip both
            (default None).
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
        >>> from records.records import Record, TimeseriesRecord
from records.timeseries import Signal1D
        >>> import numpy as np
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
        if axis is not None:
            if not isinstance(axis, int) or axis not in [0, 1]:
                raise ValueError("axis must be None or 0 or 1")
        if not isinstance(independent, bool):
            raise ValueError("independent must be True or False")

        out = self if inplace else self.copy()

        # Handle column stripping (axis=1) - always independent
        if axis == 1:
            for key in out.keys():
                out[key].strip(axis=1, inplace=True)
            if not inplace:
                return out

        # Handle row/time stripping (axis=0 or axis=None)
        if independent:
            # Original behavior: each element stripped independently
            for key in out.keys():
                out[key].strip(axis=axis, inplace=True)
        else:
            # New behavior: shared timeframe across all elements
            if len(out._data) > 0:
                # Combine all elements into single DataFrame
                combined_df = out.to_dataframe()

                # Find rows where at least one element has non-NaN value
                non_empty_rows = combined_df.dropna(how="all", axis=0)

                if len(non_empty_rows) > 0:
                    # Extract shared timeframe boundaries
                    shared_index = non_empty_rows.index.to_numpy()
                    start = float(np.min(shared_index))
                    stop = float(np.max(shared_index))

                    # Apply shared timeframe to each element
                    for key in out.keys():
                        out._data[key] = out._data[key].loc[start:stop, :]

                    # Handle column stripping if axis=None
                    if axis is None:
                        for key in out.keys():
                            out[key].strip(axis=1, inplace=True)

        if not inplace:
            return out

    def reset_time(self, inplace=False, time_zero: float | int | None = None):
        """
        Reset the time index to start at zero for all contained Timeseries-like
        objects.

        Parameters
        ----------
        inplace : bool, optional
            If True, modify in place. If False, return a new TimeseriesRecord.

        Returns
        -------
        TimeseriesRecord or None
            A TimeseriesRecord with reset time if inplace is False, otherwise
            None.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if time_zero is not None:
            if not isinstance(time_zero, (float, int)):
                raise ValueError("time_zero must be int, float or None")
            t0 = time_zero
        else:
            t0 = float(self.index[0])
        if inplace:
            for v in self._data.values():
                v.index = v.index - t0
        else:
            out = self.copy()
            for v in out._data.values():
                v.index = v.index - t0
            return out

    def fillna(self, value=None, regressors=None, inplace=False):
        """
        Return a copy with NaNs replaced by the specified value or using
        advanced imputation for all contained objects.

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
        Record
            Filled record.
        """

        def fill_record(
            record: "Record",
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
            regressors,
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
        return self._data.get(key, default)

    def drop(self, key: str | list[str], inplace: bool = False):
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


__all__ = ["Record"]
