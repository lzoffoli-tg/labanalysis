"""TestProtocol Protocol for lab test implementations."""

import pickle
from os import makedirs
from os.path import dirname, exists
from typing import Callable, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..messages import askyesnocancel
from ..records import TimeseriesRecord
from ..timeseries import EMGSignal
from .participant import Participant


@runtime_checkable
class TestProtocol(Protocol):
    """
    Protocol for lab test classes.

    Defines the required interface for test protocol implementations, including participant data,
    normative data, and methods for saving/loading and summarizing results.

    Parameters
    ----------
    participant : Participant
        The participant associated with the test.
    normative_data : pandas DataFrame, optional
        a dataframe containing normative data.

    Methods
    -------
    save(file_path: str)
        Save the test object to a file.
    load(file_path: str)
        Load a test object from a file.
    result_tables() -> dict[str, pd.DataFrame]
        Abstract method. Return a summary of the test results as a dictionary of pandas DataFrames.
    processing_pipeline
        Return the default processing pipeline for this test.
    raw_data_table()
        Return a table containing the raw data (optional, may raise NotImplementedError).
    raw_data_figure()
        Return a figure displaying the raw data (optional, may raise NotImplementedError).
    """

    _normative_data: pd.DataFrame
    _participant: Participant
    _emg_normalization_references: TimeseriesRecord
    _emg_activation_references: TimeseriesRecord
    _emg_activation_threshold: float
    _emg_normalization_function: Callable
    _relevant_muscle_map: list[str] | None

    def __init__(
        self,
        participant: Participant,
        normative_data: pd.DataFrame,
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        self.set_participant(participant)
        self.set_normative_data(normative_data)
        self.set_emg_normalization_references(emg_normalization_references)
        self.set_emg_normalization_function(emg_normalization_function)
        self.set_emg_activation_references(emg_activation_references)
        self.set_emg_activation_threshold(emg_activation_threshold)
        self.set_relevant_muscle_map(relevant_muscle_map)

    def __setstate__(self, state):
        """
        Restore object state from pickle and ensure all required attributes are initialized.
        This handles cases where older pickled objects might be missing some attributes.
        """
        self.__dict__.update(state)
        # Ensure all required attributes exist with default values if missing
        if not hasattr(self, "_emg_activation_references"):
            self._emg_activation_references = TimeseriesRecord()
        if not hasattr(self, "_emg_normalization_references"):
            self._emg_normalization_references = TimeseriesRecord()
        if not hasattr(self, "_emg_activation_threshold"):
            self._emg_activation_threshold = 3
        if not hasattr(self, "_emg_normalization_function"):
            self._emg_normalization_function = np.mean
        if not hasattr(self, "_relevant_muscle_map"):
            self._relevant_muscle_map = None

    def set_relevant_muscle_map(self, muscle_map: list[str] | None):
        if muscle_map is None or (
            isinstance(muscle_map, list)
            and all([isinstance(i, str) for i in muscle_map])
        ):
            self._relevant_muscle_map = muscle_map
        else:
            raise ValueError("muscle_map must be None or a list of muscle names.")

    @property
    def relevant_muscle_map(self):
        return self._relevant_muscle_map

    def set_emg_normalization_function(self, func: Callable):
        if not callable(func):
            raise ValueError("emg_normalization_function must be a callable.")
        self._emg_normalization_function = func

    @property
    def emg_normalization_function(self):
        return self._emg_normalization_function

    def set_emg_normalization_references(
        self, ref: TimeseriesRecord | str | Literal["self"]
    ):
        if isinstance(ref, str):
            if ref == "self":
                if isinstance(self, TimeseriesRecord):
                    self._emg_normalization_references = self.emgsignals.copy()
                else:
                    msg = "'self' cannot be used as emg_normalization_reference "
                    msg += "as it is not a TimeseriesRecord subclass."
                    raise ValueError(msg)
        elif isinstance(ref, TimeseriesRecord):
            self._emg_normalization_references = ref
        else:
            msg = "emg_normalization_references must be: 1) a TimeseriesRecord "
            msg += " instance with EMGSignal objects contained inside. 2) 'self'."
            raise ValueError(msg)

    @property
    def emg_normalization_references(self):
        return self._emg_normalization_references

    def set_emg_activation_references(
        self, ref: TimeseriesRecord | str | Literal["self"]
    ):
        if isinstance(ref, str):
            if ref == "self":
                if isinstance(self, TimeseriesRecord):
                    self._emg_activation_references = self.emgsignals.copy()
                else:
                    msg = "'self' cannot be used as emg_activation_reference "
                    msg += "as it is not a TimeseriesRecord subclass."
                    raise ValueError(msg)
        elif isinstance(ref, TimeseriesRecord):
            self._emg_activation_references = ref
        else:
            msg = "emg_activation_references must be: 1) a TimeseriesRecord "
            msg += " instance with EMGSignal objects contained inside. 2) 'self'."
            raise ValueError(msg)

    @property
    def emg_activation_references(self):
        return self._emg_activation_references

    def set_emg_activation_threshold(self, ref: float | int):
        if (not isinstance(ref, (float, int))) or ref <= 0:
            msg = "emg_activation_threshold must be a float > 0."
            raise ValueError(msg)
        self._emg_activation_threshold = ref

    @property
    def emg_activation_threshold(self):
        return self._emg_activation_threshold

    def set_normative_data(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "'normative_data' is not valid."
                + " If provided, it should be a "
                + " pandas DataFrame containing normative data."
                + " Not having valid normative references might affect"
                + " the implementation of specific test reports."
            )
        self._normative_data = data

    @property
    def normative_data(self):
        """
        Returns the normative data.
        """
        return self._normative_data

    def set_participant(self, participant: Participant):
        if not isinstance(participant, Participant):
            raise ValueError("participant must be a Participant instance.")
        self._participant = participant

    @property
    def participant(self):
        return self._participant

    @property
    def name(self):
        """
        Returns the test name.

        Returns
        -------
        str
            The name of the test.
        """
        return type(self).__name__

    @property
    def emg_normalization_values(self):
        # apply the pipeline to normalization emg data and extract mean values
        norm = self.processing_pipeline(
            self.emg_normalization_references,
            inplace=False,
        )
        if not isinstance(norm, TimeseriesRecord):
            msg = "Something went wrong during data processing."
            raise ValueError(msg)
        norms: dict[tuple[str, str], float] = {}
        for i in norm.emgsignals.values():
            if isinstance(i, EMGSignal):
                norms[(i.muscle_name, i.side)] = float(
                    self.emg_normalization_function(i.to_numpy())
                )

        return norms

    @property
    def emg_activation_thresholds(self):
        # get processed activation signals
        thresh = self.processing_pipeline(
            self.emg_activation_references,
            inplace=False,
        )
        if not isinstance(thresh, TimeseriesRecord):
            msg = "Something went wrong during data processing."
            raise ValueError(msg)
        thresh_vals = {
            (i.muscle_name, i.side): i.to_numpy().flatten()
            for i in thresh.emgsignals.values()
        }

        # get thresholds
        thresholds: dict[tuple[str, str], float] = {}
        for (tname, tside), val in thresh_vals.items():
            avg = val.mean()
            std = val.std()
            thr = float(avg + self.emg_activation_threshold * std)
            thresholds[(str(tname), str(tside))] = thr

        return thresholds

    def save(self, file_path: str, force_overwrite: bool = False):
        """
        Save the test object to a file.

        Parameters
        ----------
        file_path : str
            Path where to save the file. The file extension should match the
            test name. If not, the appropriate extension is appended.
        """
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        if not isinstance(force_overwrite, bool):
            raise ValueError("force_overwrite must be True or False.")
        extension = "." + self.__class__.__name__.lower()
        if not file_path.endswith(extension):
            file_path += extension
        if exists(file_path) and not force_overwrite:
            overwrite = askyesnocancel(
                title="File already exists",
                message="the provided file_path already exist. Overwrite?",
            )
            if not overwrite:
                file_path = file_path[: len(extension)] + "(1)" + extension
        makedirs(dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as buf:
            pickle.dump(self, buf)

    @classmethod
    def load(cls, file_path: str):
        """
        Load a test object from a file.

        Parameters
        ----------
        file_path : str
            Path to the file to load. The file extension must match the test name.

        Returns
        -------
        TestProtocol
            The loaded test object.

        Raises
        ------
        ValueError
            If file_path is not a string or does not have the correct extension.
        RuntimeError
            If loading fails.
        """
        if not isinstance(file_path, str):
            raise ValueError("'file_path' must be a str instance.")
        extension = "." + cls.__name__.lower()
        if not file_path.endswith(extension):
            raise ValueError(f"'file_path' must have {extension}.")
        try:
            with open(file_path, "rb") as buf:
                return pickle.load(buf)
        except Exception:
            raise RuntimeError(f"an error occurred importing {file_path}.")

    #! MANDATORY METHODS TO BE IMPLEMENTED

    def get_results(self, include_emg: bool = True): ...

    @property
    def processing_pipeline(self):
        """
        exercise data processing pipeline
        """
        ...

    @property
    def processed_data(self): ...


__all__ = ["TestProtocol"]
