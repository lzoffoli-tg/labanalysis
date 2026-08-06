"""TestProtocol Protocol for lab test implementations."""

import inspect
import pickle
from os import makedirs
from os.path import dirname, exists
from pathlib import Path
from typing import Callable, Protocol, Self, runtime_checkable
import copy as copy_module
import numpy as np
import pandas as pd

from ..messages import askyesnocancel
from ..pipelines.base import ProcessingPipeline
from ..records import Record
from ..timeseries import EMGSignal
from .participant import Participant
from .test_results import TestResults


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
    _emg_normalization_references: Record
    _emg_activation_references: Record
    _emg_activation_threshold: float
    _emg_normalization_function: Callable
    _relevant_muscle_map: list[str]
    _results: TestResults | None

    def __init__(
        self,
        participant: Participant,
        normative_data: pd.DataFrame,
        emg_normalization_references: Record = Record(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: Record = Record(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
        keep_all_data: bool = True,
    ):
        self.set_participant(participant)
        self.set_normative_data(normative_data)
        self.set_relevant_muscle_map(relevant_muscle_map)
        self.set_emg_activation_threshold(emg_activation_threshold)
        self.set_emg_activation_references(emg_activation_references)
        self.set_emg_normalization_function(emg_normalization_function)
        self.set_emg_normalization_references(emg_normalization_references)
        self.set_keep_all_data(keep_all_data)
        self._results = None

    def __setstate__(self, state):
        """
        Restore object state from pickle and ensure all required attributes are initialized.
        This handles cases where older pickled objects might be missing some attributes.
        """
        self.__dict__.update(state)
        # Ensure all required attributes exist with default values if missing
        if not hasattr(self, "_emg_activation_references"):
            self._emg_activation_references = Record()
        if not hasattr(self, "_emg_normalization_references"):
            self._emg_normalization_references = Record()
        if not hasattr(self, "_emg_activation_threshold"):
            self._emg_activation_threshold = 3
        if not hasattr(self, "_emg_normalization_function"):
            self._emg_normalization_function = np.mean
        if not hasattr(self, "_relevant_muscle_map"):
            self._relevant_muscle_map = []

    def set_relevant_muscle_map(self, muscle_map: list[str] | None):
        if muscle_map is None:
            self._relevant_muscle_map = []
        elif isinstance(muscle_map, list) and all(
            [isinstance(i, str) for i in muscle_map]
        ):
            self._relevant_muscle_map = muscle_map
        else:
            raise ValueError("muscle_map must be None or a list of muscle names.")

    def set_keep_all_data(self, value: bool):
        """
        set the keep all data attribute

        Parameters
        ----------
        value: bool
            the attribute to be set.
        """
        if not isinstance(value, bool):
            raise ValueError("keep all data must be True or False")
        self._keep_all_data = value

    @property
    def keep_all_data(self):
        """return the keep_all_data attribute"""
        return self._keep_all_data

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

    def set_emg_normalization_references(self, ref: Record):
        """set the Record containing EMG data to be used as normalization references"""
        if not isinstance(ref, Record):
            raise ValueError("emg normalization references must be a Record instance.")
        if len(self.relevant_muscle_map) == 0:
            muscle_map = ref.emgsignals.keys()
        else:
            muscle_map = self.relevant_muscle_map
        self._emg_normalization_references = Record(
            **{i: v for i, v in ref.emgsignals.items() if i in muscle_map}
        )

    @property
    def emg_normalization_references(self):
        return self._emg_normalization_references

    def set_emg_activation_references(self, ref: Record):
        """set the Record containing EMG data to be used as activation references"""
        if not isinstance(ref, Record):
            raise ValueError("emg activation references must be a Record instance.")
        if len(self.relevant_muscle_map) == 0:
            muscle_map = ref.emgsignals.keys()
        else:
            muscle_map = self.relevant_muscle_map
        self._emg_activation_references = Record(
            **{i: v for i, v in ref.emgsignals.items() if i in muscle_map}
        )

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
        pipeline = self.processing_pipeline
        if pipeline is not None:
            norm = pipeline(self.emg_normalization_references, inplace=False)
            if not isinstance(norm, Record):
                msg = "Something went wrong during data processing."
                raise ValueError(msg)
        else:
            norm = self.emg_normalization_references
        norms: dict[tuple[str, str], float] = {}
        for i in norm.emgsignals.values():
            if isinstance(i, EMGSignal):
                norms[(i.muscle_name, i.side)] = float(
                    self.emg_normalization_function(i.to_numpy().flatten())
                )

        return norms

    @property
    def emg_activation_thresholds(self):
        pipeline = self.processing_pipeline
        if pipeline is not None:

            # get processed activation signals
            thresh = pipeline(
                self.emg_activation_references,
                inplace=False,
            )
            if not isinstance(thresh, Record):
                msg = "Something went wrong during data processing."
                raise ValueError(msg)
        else:
            thresh = self.emg_activation_references
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

    def _get_constructor_args(self):
        """
        Extract constructor arguments and internal attributes for dynamic instantiation.

        Returns
        -------
        dict
            Dictionary of constructor arguments and internal attributes.
        """
        sig = inspect.signature(self.__class__.__init__)
        args = {}

        # Capture constructor parameters
        for name, param in sig.parameters.items():
            if name == "self":
                continue

            value = None
            if hasattr(self, name):
                value = getattr(self, name)
            elif hasattr(self, f"_{name}"):
                value = getattr(self, f"_{name}")
            elif param.default is not inspect.Parameter.empty:
                value = param.default
            else:
                # Skip - may be handled by **kwargs
                continue
            args[name] = value

        # Additionally capture internal attributes (like Timeseries._get_object_args does)
        for attr in dir(self):
            if attr.startswith("_") and not attr.startswith("__"):
                # Skip if already captured
                if attr in args:
                    continue
                # Only capture non-callable attributes
                if hasattr(self.__class__, attr):
                    value = getattr(self, attr)
                    if not callable(value):
                        args[attr] = value

        # remove unnecessary args
        to_omit = ["_abc_impl", "_is_protocol", "_is_runtime_protocol"]
        for key in to_omit:
            if key in list(args.keys()):
                args.pop(key)

        return args

    def get_results(self, *args, **kwargs):
        """
        return the results of the test

        Parameters
        ----------
        all parameters required by update_results

        Returns
        -------
        TestResults: the results of the test.
        """
        if self._results is None:
            self.update_results(*args, **kwargs)
        return self._results

    def copy(self):
        """
        Create a deep copy preserving the concrete subclass type.

        Returns
        -------
        TestProtocol or subclass
            A new instance of the same class with copied attributes.

        Notes
        -----
        - Participant objects are deep copied using copy.deepcopy()
        - DataFrames are copied using df.copy()
        - Records are copied using their .copy() method
        - Callable attributes (like emg_normalization_function) are referenced, not copied
        - Subclass-specific attributes are automatically preserved via introspection
        """
        # Get all constructor arguments and internal attributes
        args = self._get_constructor_args()

        # Deep copy Participant object
        if "participant" in args or "_participant" in args:
            participant = args.get("participant", args.get("_participant"))
            if participant is not None:
                args["participant"] = copy_module.deepcopy(participant)
                if "_participant" in args:
                    args.pop("_participant")

        # Deep copy DataFrame
        if "normative_data" in args or "_normative_data" in args:
            norm_data = args.get("normative_data", args.get("_normative_data"))
            if norm_data is not None and hasattr(norm_data, "copy"):
                args["normative_data"] = norm_data.copy()
                if "_normative_data" in args:
                    args.pop("_normative_data")

        # Copy Record objects using their .copy() method
        for key in list(args.keys()):
            if key in [
                "emg_normalization_references",
                "emg_activation_references",
                "_emg_normalization_references",
                "_emg_activation_references",
            ]:
                ref = args[key]
                if ref is not None and hasattr(ref, "copy") and callable(ref.copy):
                    param_name = key.replace("_", "", 1) if key.startswith("_") else key
                    args[param_name] = ref.copy()
                    if key.startswith("_") and param_name in args:
                        args.pop(key)

        # Remove internal attribute versions if public parameter exists
        # (avoid passing both _attr and attr to constructor)
        for key in list(args.keys()):
            if key.startswith("_"):
                public_key = key[1:]
                if public_key in args:
                    args.pop(key)

        # Callable attributes (emg_normalization_function) and simple types
        # (float, str, list) are passed as-is

        # Create new instance of the concrete class
        return self.__class__(**args)

    def save(self, file_path: str | Path, force_overwrite: bool = False):
        """
        Save the test object to a file.

        Parameters
        ----------
        file_path : str
            Path where to save the file. The file extension should match the
            test name. If not, the appropriate extension is appended.
        """
        if not isinstance(file_path, (str, Path)):
            raise ValueError("'file_path' must be a str or Path instance.")
        if not isinstance(force_overwrite, bool):
            raise ValueError("force_overwrite must be True or False.")
        if isinstance(file_path, str):
            file_path = Path(file_path)
        extension = "." + self.name.lower()
        if file_path.suffix.lower() != extension:
            file_path = file_path.with_suffix(extension)
        if exists(file_path) and not force_overwrite:
            overwrite = askyesnocancel(
                title="File already exists",
                message="the provided file_path already exist. Overwrite?",
            )
            if not overwrite:
                file_path = file_path.with_name(file_path.name + "(1)")
        makedirs(dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as buf:
            pickle.dump(self, buf)

    @classmethod
    def load(cls, file_path: str | Path):
        """
        Load a test object from a file.

        Parameters
        ----------
        file_path : str | Path
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
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not isinstance(file_path, Path):
            raise ValueError("'file_path' must be a str or Path instance.")

        extension = "." + cls.__name__.lower()
        if not file_path.suffix.endswith(extension):
            raise ValueError(f"'file_path' must have {extension}.")
        try:
            with open(file_path, "rb") as buf:
                return pickle.load(buf)
        except Exception:
            raise RuntimeError(f"an error occurred importing {file_path}.")

    #! MANDATORY METHODS TO BE IMPLEMENTED

    def update_results(self, include_emg: bool = True): ...

    @property
    def processing_pipeline(self) -> ProcessingPipeline:
        """
        exercise data processing pipeline
        """
        ...

    @property
    def processed_data(self) -> "Self": ...


__all__ = ["TestProtocol"]
