"""Timeseries record container module."""

import numpy as np
import pandas as pd

from ..io.read.btsbioengineering import read_tdf
from ..timeseries import Timeseries, Signal1D, Signal3D, EMGSignal, Point3D, Plane3D
from .record import Record
from .forceplatform import ForcePlatform
from .metabolicrecord import MetabolicRecord


class TimeseriesRecord(Record):
    """
    A dictionary-like container for Timeseries, TimeseriesRecord,
    and ForcePlatform objects, supporting type filtering and DataFrame conversion.

    Parameters
    ----------
    vertical_axis : str, optional
        The label for the vertical axis (default "Y").
    anteroposterior_axis : str, optional
        The label for the anteroposterior axis (default "Z").
    strip : bool, optional
        If True, remove leading/trailing rows or columns that are all NaN from
        all contained objects (default True).
    reset_time : bool, optional
        If True, reset the time index to start at zero for all contained
        objects (default True).
    **signals : dict
        Key-value pairs of Timeseries subclasses, TimeseriesRecord, or
        ForcePlatform to include in the record.

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
        Remove leading/trailing rows or columns that are all NaN from all
        contained objects. When independent=False (default), all elements share
        a common timeframe based on the union of non-NaN time points.
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

    _data: dict[
        str,
        Timeseries
        | Signal1D
        | Signal3D
        | EMGSignal
        | Point3D
        | Plane3D
        | ForcePlatform
        | MetabolicRecord,

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
    def metabolicrecords(self):
        """
        Get all MetabolicRecord objects.

        Returns
        -------
        TimeseriesRecord
        """
        return self._filter_by_type(MetabolicRecord)

    @property
    def resultant_force(self):
        """
        return a forceplatform object representing the resultant of all
        available forceplatforms
        """
        platforms = list(self.forceplatforms.values())
        if len(platforms) == 0:
            raise ValueError("No forceplatforms found within the TimeseriesRecord.")

        # Indice temporale comune a tutte le piattaforme
        common_times = list(
            dict.fromkeys([t for platform in platforms for t in platform.index])
        )
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

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError("key must be a str")
        types = (Timeseries, Signal1D, Signal3D, EMGSignal, Point3D, Plane3D, ForcePlatform, Record)
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

    def __init__(
        self,
        **signals: (
            Timeseries | Signal1D | Signal3D | EMGSignal | Point3D | Plane3D | ForcePlatform
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

    def copy(self):
        """
        Create a deep copy of the TimeseriesRecord object.

        Returns
        -------
        TimeseriesRecord
            A new instance of the same class with copies of all signals.

        Notes
        -----
        Uses self.__class__ to preserve subclass type. All internal
        Timeseries objects are deep copied.
        """
        return self.__class__(**{i: v.copy() for i, v in self._data.items()})  # type: ignore


__all__ = ["TimeseriesRecord"]
