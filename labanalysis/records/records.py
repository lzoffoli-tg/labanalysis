"""record module"""

import inspect
from warnings import warn
import numpy as np
import pandas as pd

from ..io.read.btsbioengineering import read_tdf

from .timeseries import *
from ..signalprocessing import fillna as sp_fillna
from plotly.subplots import make_subplots
import plotly.graph_objects as go


__all__ = ["Record", "ForcePlatform", "TimeseriesRecord"]


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


class ForcePlatform(Record):
    """
    Represents a force platform measurement system.

    Parameters
    ----------
    origin : Point3D
        The center of pressure (CoP) location over time.
    force : Signal3D
        The 3D ground reaction force vector over time.
    torque : Signal3D
        The 3D torque vector over time.
    vertical_axis : str, optional
        The label for the vertical axis (default "Y").
    anteroposterior_axis : str, optional
        The label for the anteroposterior axis (default "Z").
    strip : bool, optional
        If True, remove leading/trailing rows or columns that are all NaN from
        all contained objects (default True).
    reset_time : bool, optional
        If True, reset the time index to start at zero for all contained objects
        (default True).

    Methods
    -------
    copy()
        Return a deep copy of the ForcePlatform.
    """

    @property
    def vertical_axis(self):
        origin: Point3D = self["origin"]  # type: ignore
        return origin.vertical_axis

    @property
    def anteroposterior_axis(self):
        origin: Point3D = self["origin"]  # type: ignore
        return origin.anteroposterior_axis

    @property
    def lateral_axis(self):
        origin: Point3D = self["origin"]  # type: ignore
        return origin.lateral_axis

    def __init__(self, origin: Point3D, force: Signal3D, torque: Signal3D):
        """
        Initialize a ForcePlatform.

        Parameters
        ----------
        origin : Point3D
        force : Signal3D
        torque : Signal3D

        Raises
        ------
        TypeError
            If any argument is not of the correct type.
        """
        if not isinstance(origin, Point3D):
            raise TypeError("origin must be an instance of Point3D")
        if not isinstance(force, Signal3D):
            raise TypeError("force must be an instance of Signal3D")
        if not isinstance(torque, Signal3D):
            raise TypeError("torque must be an instance of Signal3D")

        # check the axes
        if (
            origin.vertical_axis != force.vertical_axis
            or origin.vertical_axis != torque.vertical_axis
        ):
            msg = "vertical axes must be the same across origin, "
            msg += "force and torque elements."
            raise ValueError(msg)
        if (
            origin.anteroposterior_axis != force.anteroposterior_axis
            or origin.anteroposterior_axis != torque.anteroposterior_axis
        ):
            msg = "anteroposterior axes must be the same across origin, "
            msg += "force and torque elements."
            raise ValueError(msg)

        super().__init__(origin=origin, force=force, torque=torque)

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__("_data", value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if not key in ["origin", "force", "torque"] or not isinstance(
            value, (Point3D, Signal3D)
        ):
            msg = "only 'origin', 'force' and 'torque' attributes can be "
            msg += " passed to ForcePlatform instances."
            raise ValueError(msg)
        if not isinstance(value, (Signal3D, Point3D)):
            raise ValueError("value must be a Timeseries or Record")
        self._data[key] = value

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

        Parameters
        ----------
        new_x, new_y, new_z : array-like
            Orthonormal basis vectors.
        new_origin : array-like
            New origin.

        Returns
        -------
        ForcePlatform
            Transformed signal.

        Raises
        ------
        ValueError
            If input vectors are not valid.

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if inplace:
            for val in self.values():
                if isinstance(val, (Point3D, Signal3D)):
                    val.change_reference_frame(
                        new_x,
                        new_y,
                        new_z,
                        new_origin,
                        True,
                    )
        else:
            out = self.copy()
            for val in out.values():
                if isinstance(val, (Point3D, Signal3D)):
                    val.change_reference_frame(
                        new_x,
                        new_y,
                        new_z,
                        new_origin,
                        True,
                    )
            return out


class TimeseriesRecord(Record):
    """
    A dictionary-like container for Timeseries, TimeseriesRecord, and ForcePlatform objects,
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

    _data: dict[
        str, Timeseries | Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
    ]

    @property
    def vertical_axis(self):
        for val in self.values():
            if hasattr(val, "vertical_axis"):
                axis = val.vertical_axis
                if axis is not None:
                    return str(axis)
        return None

    @property
    def anteroposterior_axis(self):
        for val in self.values():
            if hasattr(val, "anteroposterior_axis"):
                axis = val.anteroposterior_axis
                if axis is not None:
                    return str(axis)
        return None

    @property
    def lateral_axis(self):
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
        TimeseriesRecord
        """
        return self._filter_by_type(Point3D)

    @property
    def signals3d(self):
        """
        Get all Signal3D objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(Signal3D)

    @property
    def signals1d(self):
        """
        Get all Signal1D objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(Signal1D)

    @property
    def emgsignals(self):
        """
        Get all EMGSignal objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(EMGSignal)

    @property
    def forceplatforms(self):
        """
        Get all ForcePlatform objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(ForcePlatform)

    @property
    def resultant_force(self):
        forces = self.forceplatforms
        rows, cols = forces.shape
        if cols == 0:
            return None
        i_total = np.asarray(forces.index, float)
        f_total = np.zeros((rows, 3))
        m_total = np.zeros_like(f_total)
        axes = []
        units = {}
        for obj in forces.values():
            f_arr = obj["force"].to_numpy()
            r_arr = obj["origin"].to_numpy()
            m_arr = obj["torque"].to_numpy()
            i_arr = np.asarray(obj.index, float)
            mask = np.where(np.isin(i_total, i_arr))[0]
            f_total[mask] = f_total[mask] + f_arr
            m_total[mask] = m_total[mask] + m_arr + np.cross(r_arr, f_arr)
            if len(axes) == 0:
                axes = obj["origin"].columns
                units = {i: obj[i].unit for i in ["origin", "force", "torque"]}

        num = np.cross(f_total, m_total)
        den = np.sum(f_total**2, axis=1)[:, np.newaxis]

        # generate the force platform
        cop = Point3D(
            num / den,
            forces.index.tolist(),
            units["origin"],
            axes,
        )
        force = Signal3D(
            f_total,
            forces.index.tolist(),
            units["force"],
            axes,
        )
        torque = Signal3D(
            m_total,
            forces.index.tolist(),
            units["torque"],
            axes,
        )

        return ForcePlatform(cop, force, torque)

    def _filter_by_type(self, cls):
        """
        Internal: Filter contained items by type.

        Parameters
        ----------
        cls : type

        Returns
        -------
        TimeseriesRecord
            A view (not a copy) of the filtered items.
            Changes to elements affect the original TimeseriesRecord.
        """
        return TimeseriesRecord(
            **{k: v for k, v in self.items() if type(v) == cls},
        )

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

        Parameters
        ----------
        new_x, new_y, new_z : array-like
            Orthonormal basis vectors.
        new_origin : array-like
            New origin.

        Returns
        -------
        TimeseriesRecord
            Transformed signal.

        Raises
        ------
        ValueError
            If input vectors are not valid.

        Notes
        -----
        rotations are applied only to 3D objects like Signal3D, Point3D and
        ForcePlatform(s)
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if inplace:
            for key, value in self.items():
                if isinstance(value, (Point3D, Signal3D, ForcePlatform)):
                    value.change_reference_frame(
                        new_x,
                        new_y,
                        new_z,
                        new_origin,
                        True,
                    )
        else:
            out = self.copy()
            for key, value in out.values():
                if isinstance(value, (Point3D, Signal3D, ForcePlatform)):
                    value.change_reference_frame(
                        new_x,
                        new_y,
                        new_z,
                        new_origin,
                        True,
                    )
            return out

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError("key must be a str")
        types = (Timeseries, Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform)
        if not isinstance(value, types):
            raise ValueError(f"value must be any of {types}")
        self._data[key] = value

    @classmethod
    def from_tdf(cls, filename: str):
        """
        Create a TimeseriesRecord from a TDF file.

        Parameters
        ----------
        filename : str
            Path to the TDF file.

        Returns
        -------
        TimeseriesRecord
            A TimeseriesRecord populated with the data from the TDF file.
        """
        data = read_tdf(filename)
        record = cls()

        # Handle 3D points from CAMERA TRACKED
        if data.get("CAMERA") and data["CAMERA"].get("TRACKED"):  # type: ignore
            df = data["CAMERA"]["TRACKED"]["TRACKS"]  # type: ignore
            for label in df.columns.get_level_values(0).unique():
                sub_df: pd.DataFrame = df[label]
                record[label] = Point3D(
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
                record[f"{side}_{muscle_name}".lower()] = EMGSignal(
                    data=signal.values.astype(float).flatten(),
                    index=df.index.tolist(),
                    muscle_name=muscle_name.lower(),
                    side=side.lower(),
                    unit=unit,
                )

        # Handle Force Platforms
        if data.get("FORCE_PLATFORM") and data["FORCE_PLATFORM"].get("TRACKED"):  # type: ignore
            df = data["FORCE_PLATFORM"]["TRACKED"]["TRACKS"]  # type: ignore
            for label in df.columns.get_level_values("LABEL").unique():
                origin: pd.DataFrame = df[label]["ORIGIN"]
                force: pd.DataFrame = df[label]["FORCE"]
                torque: pd.DataFrame = df[label]["TORQUE"]
                record[label] = ForcePlatform(
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

        return record

    def __init__(
        self,
        **signals: (
            Timeseries | Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        ),
    ):
        """
        Initialize a TimeseriesRecord.

        Parameters
        ----------
        **signals : dict
            Key-value pairs of Timeseries subclasses, TimeseriesRecord, or ForcePlatform.
        """
        super().__init__()
        for key, value in signals.items():
            self[key] = value

        # check the axes
        vt = None
        ap = None
        for value in self.values():
            if isinstance(value, (ForcePlatform, Point3D, Signal3D)):
                vtaxis = value.vertical_axis
                apaxis = value.anteroposterior_axis
                if vt is None:
                    vt = vtaxis
                elif vt != vtaxis:
                    raise ValueError("vertical axes are not aligned.")
                if ap is None:
                    ap = apaxis
                elif ap != apaxis:
                    raise ValueError("anteroposterior axes are not aligned.")
