"""TestResults Protocol for lab test results management."""

from os import makedirs
from os.path import dirname, exists, join
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..messages import askyesnocancel
from ..records import TimeseriesRecord
from ..timeseries import EMGSignal


@runtime_checkable
class TestResults(Protocol):

    _summary: pd.DataFrame | dict[str, pd.DataFrame] | None
    _analytics: pd.DataFrame | None
    _figures: dict[str, go.Figure | dict[str, go.Figure]] | None
    _include_emg: bool

    @property
    def summary(self):
        return self._summary

    @property
    def analytics(self):
        return self._analytics

    @property
    def figures(self):
        return self._figures

    @property
    def include_emg(self):
        return self._include_emg

    def save_all(self, path: str, force_overwrite: bool = False):

        # check the inputs
        if not isinstance(path, str):
            msg = "path must be a string defining the root folder where to "
            msg += "save the results of the test."
            raise ValueError(msg)
        if not isinstance(force_overwrite, bool):
            msg = "force_overwrite must be True or False"
            raise ValueError(msg)

        # generate a recursive function that automatically save
        # the data in the proper folder
        def save(filename: str, obj: pd.DataFrame | go.Figure | dict):
            def make_filename(file: str):
                if exists(file) and not force_overwrite:
                    if not askyesnocancel(
                        "File already exits.", f"Overwrite {filename}?"
                    ):
                        name, ext = filename.rsplit(".", 1)  # type: ignore
                        file = name + "(1)" + "." + ext
                makedirs(dirname(file), exist_ok=True)
                return file

            if isinstance(obj, pd.DataFrame):
                obj.to_csv(make_filename(filename + ".csv"))
            elif isinstance(obj, go.Figure):
                obj.write_image(make_filename(filename + ".svg"))
            elif isinstance(obj, dict):
                for key, val in obj.items():
                    save(join(filename, key), val)
            else:
                msg = "saving procedure is not supported for objects of type "
                msg += f"{type(obj)}."
                raise ValueError(msg)

        if self.summary is not None:
            save(join(path, "summary"), self.summary)
        if self.analytics is not None:
            save(join(path, "analytics"), self.analytics)
        if self.figures is not None:
            save(join(path, "figures"), self.figures)

    def __init__(self, test: Any, include_emg: bool):
        if not isinstance(include_emg, bool):
            raise ValueError("include_emg must be True or False.")
        self._include_emg = include_emg
        self._summary = pd.DataFrame()
        self._analytics = pd.DataFrame()
        self._figures = {}
        self._generate_results(test)

    def _get_symmetry(self, left: np.ndarray, right: np.ndarray):
        line = {"left_%": np.mean(left), "right_%": np.mean(right)}
        line = pd.DataFrame(pd.Series(line)).T
        norm = line.sum(axis=1).values.astype(float)
        line.loc[line.index, line.columns] = line.values.astype(float) / norm * 100
        return line

    def _get_muscle_symmetry(self, test: TimeseriesRecord):
        """
        Returns coordination and balance metrics from EMG signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with coordination and balance metrics, or empty if not available.
        """

        # get the muscle activations
        # (if there are no emg data return and empty dataframe)
        emgs = test.emgsignals
        if emgs.shape[1] == 0:
            return pd.DataFrame()

        # check the presence of left and right muscles
        muscles = {}
        for emg in emgs.values():
            if isinstance(emg, EMGSignal):
                name = emg.muscle_name
                side = emg.side
                if side not in ["left", "right"]:
                    continue
                if name not in list(muscles.keys()):
                    muscles[name] = {}

                # get the area under the curve of the muscle activation
                muscles[name][side] = emg.to_numpy().flatten()

        # remove those muscles not having both sides
        muscles = {i: v for i, v in muscles.items() if len(v) == 2}

        # calculate coordination and imbalance between left and right side
        out = {}
        for muscle, sides in muscles.items():
            params = self._get_symmetry(**sides)
            out.update(**{f"{muscle}_{i}": v[0] for i, v in params.items()})

        return pd.DataFrame(pd.Series(out)).T

    def _generate_results(self, test: Any):
        from .test_protocol import TestProtocol

        if not isinstance(test, TestProtocol):
            raise ValueError("test must be a TestProtocol instance.")
        self._summary = self._get_summary(test)
        self._analytics = self._get_analytics(test)
        self._figures = self._get_figures(test)

    #! MANDATORY FIELDS TO BE IMPLEMENTED

    def _get_summary(self, test: Any): ...

    def _get_analytics(self, test: Any): ...

    def _get_figures(
        self, test: Any
    ) -> dict[str, go.Figure | dict[str, go.Figure]]: ...

    def copy(self):
        """
        Create a deep copy of the TestResults instance.

        Returns
        -------
        TestResults
            A new instance of the same class with copied DataFrames and figures.

        Notes
        -----
        Creates deep copies of summary and analytics DataFrames.
        Figures dictionary is copied (figure objects themselves are immutable).
        """
        import copy as copy_module

        new_instance = object.__new__(self.__class__)
        for i in ["_summary", "_analytics", "_figures", "_include_emg"]:
            if self.__dict__.get(i) is None:
                new_instance.__dict__[i] = None
            else:
                new_instance.__dict__[i] = copy_module.deepcopy(self.__dict__[i])

        return new_instance


__all__ = ["TestResults"]
