"""record module"""

import inspect
from warnings import warn
import numpy as np
import pandas as pd

from ..timeseries.emgsignal import EMGSignal
from ..timeseries.signal1d import Signal1D
from ..timeseries.signal3d import Signal3D
from ..timeseries.point3d import Point3D

from ..timeseries.timeseries import Timeseries
from ...signalprocessing import fillna as sp_fillna
from plotly.subplots import make_subplots
import plotly.graph_objects as go


__all__ = ["Record"]


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
    strip(axis=0, inplace=False)
        Remove leading/trailing rows or columns that are all NaN from all contained objects.
    reset_time(inplace=False)
        Reset the time index to start at zero for all contained objects.
    apply(func, axis=0, inplace=False, *args, **kwargs)
        Apply a function or ProcessingPipeline to all contained objects.
    fillna(value=None, n_regressors=None, inplace=False)
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
        return np.unique(self.to_dataframe().index.to_numpy())

    @property
    def shape(self):
        return self.to_dataframe().shape

    def _view(
        self,
        rows: slice | list[int | float | bool] | np.ndarray | None = None,
    ):
        # get a view
        view_obj = self.__new__(type(self))
        keys = self.__dict__
        for key in keys:
            if key != "_data":
                setattr(view_obj, key, getattr(self, key))

        # set the views
        view_obj._data = {}
        for key in self.keys():
            view_obj._data[key] = self._data[key][rows]  # type: ignore

        # return
        return view_obj

    def __getitem__(self, key):
        if key in self.keys():
            if hasattr(self, key):
                return getattr(self, key)
            else:
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
        if name in self._data.keys():
            return self._data[name]
        raise ValueError(f"{name} is not a valid attribute of this Record")

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
                "_".join([key, i]) if not i.startswith(key) else i for i in new.columns
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
        args = {
            i: v.copy() if hasattr(v, "copy") else v
            for i, v in self._get_constructor_args().items()
        }
        args.update(**{i: v.copy() for i, v in self._data.items() if i not in args})
        return self.__class__(**args)

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
        if inplace:
            out = self.copy()
            for key, val in out.items():
                val.strip(axis=axis, inplace=True)
            return out
        for key, val in self.items():
            val.strip(axis=axis, inplace=True)

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

    def fillna(self, value=None, n_regressors=None, inplace=False):
        """
        Return a copy with NaNs replaced by the specified value or using
        advanced imputation for all contained objects.

        Parameters
        ----------
        value : float or int or None, optional
            Value to use for NaNs. If None, use interpolation or regression.
        n_regressors : int or None, optional
            Number of regressors to use for regression-based imputation.
            If None, use cubic spline interpolation.
        inplace : bool, optional
            If True, fill in place. If False, return a new object.

        Returns
        -------
        TimeseriesRecord
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
            n_regressors,
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
                    y=values.values.astype(float).flatten().tolist(),
                    name=lbl,
                    mode="lines",
                ),
            )
            fig.update_yaxes(row=i + 1, col=1, title=unit)
        fig.update_layout(title=fig.__class__.__name__, template="simple_white")
        return fig
