"""
base test module containing classes and functions used to perform lab tests.
"""

#! IMPORTS

import pickle
from datetime import date, datetime
from os import makedirs
from os.path import dirname, exists, join
from typing import Any, Callable, Literal, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..messages import askyesnocancel
from ..records.pipelines import ProcessingPipeline
from ..records.records import ForcePlatform, TimeseriesRecord
from ..records.timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from ..signalprocessing import butterworth_filt, rms_filt

__all__ = ["TestProtocol", "Participant", "TestResults"]


#! CLASSES


class Participant:
    """
    Represents a participant in a lab test.

    Stores participant data such as name, gender, height, weight, age, birthdate, and test recording date.
    Provides methods and properties to access and compute participant metrics.

    Parameters
    ----------
    surname : str, optional
        The participant's surname.
    name : str, optional
        The participant's name.
    gender : str, optional
        The participant's gender.
    height : int or float, optional
        The participant's height in centimeters. Will be converted to meters.
    weight : int or float, optional
        The participant's weight in kilograms.
    age : int or float, optional
        The participant's age in years.
    birthdate : datetime.date, optional
        The participant's birth date.
    recordingdate : datetime.date, optional
        The date when the test was recorded. Defaults to the current date.

    Attributes
    ----------
    units : dict
        Units of measurement for each attribute.

    Methods
    -------
    set_recordingdate(recordingdate)
        Set the test recording date.
    set_surname(surname)
        Set the participant's surname.
    set_name(name)
        Set the participant's name.
    set_gender(gender)
        Set the participant's gender.
    set_height(height)
        Set the participant's height in meters.
    set_weight(weight)
        Set the participant's weight in kilograms.
    set_age(age)
        Set the participant's age in years.
    set_birthdate(birthdate)
        Set the participant's birth date.
    copy()
        Return a copy of the object.

    Properties
    ----------
    surname : str or None
        The participant's surname.
    name : str or None
        The participant's name.
    gender : str or None
        The participant's gender.
    height : float or None
        The participant's height in meters.
    weight : float or None
        The participant's weight in kilograms.
    birthdate : datetime.date or None
        The participant's birth date.
    recordingdate : datetime.date or None
        The test recording date.
    bmi : float or None
        The participant's BMI in kg/m^2.
    fullname : str
        The participant's full name.
    age : int or None
        The participant's age in years.
    hrmax : float or None
        The maximum theoretical heart rate according to Gellish.
    units : dict
        Units of measurement for each attribute.
    dict : dict
        Dictionary representation of the participant's data.
    series : pandas.Series
        pandas.Series representation of the participant's data.
    dataframe : pandas.DataFrame
        pandas.DataFrame representation of the participant's data.
    """

    # class variables
    _name = None
    _surname = None
    _gender = None
    _height = None
    _weight = None
    _birthdate = None
    _recordingdate = date  # type:ignore
    _units = {
        "fullname": "",
        "surname": "",
        "name": "",
        "gender": "",
        "height": "m",
        "weight": "kg",
        "bmi": "kg/m^2",
        "birthdate": "",
        "age": "years",
        "hrmax": "bpm",
        "recordingdate": "",
    }

    def __init__(
        self,
        surname: str | None = None,
        name: str | None = None,
        gender: str | None = None,
        height: int | float | None = None,
        weight: int | float | None = None,
        age: int | float | None = None,
        birthdate: date | None = None,
        recordingdate: date = datetime.now().date,  # type: ignore
    ):
        """
        Initializes a Participant object.
        """
        self.set_surname(surname)
        self.set_name(name)
        self.set_gender(gender)
        self.set_height((height / 100 if height is not None else height))
        self.set_weight(weight)
        self.set_age(age)
        self.set_birthdate(birthdate)
        self.set_recordingdate(recordingdate)

    def set_recordingdate(
        self,
        recordingdate: date | None,
    ):
        """
        Sets the test recording date.

        Parameters
        ----------
        recordingdate : datetime.date or datetime.datetime or None
            The test recording date.

        Returns
        -------
        None
        """
        if recordingdate is not None:
            txt = "'recordingdate' must be a datetime.date or datetime.datetime."
            assert isinstance(recordingdate, (datetime, date)), txt
            if isinstance(recordingdate, datetime):
                self._recordingdate = recordingdate.date()
            else:
                self._recordingdate = recordingdate
        else:
            self._recordingdate = recordingdate

    def set_surname(
        self,
        surname: str | None,
    ):
        """
        Sets the participant's surname.

        Parameters
        ----------
        surname : str or None
            The surname of the participant.

        Returns
        -------
        None
        """
        if surname is not None:
            assert isinstance(surname, str), "'surname' must be a string."
        self._surname = surname

    def set_name(
        self,
        name: str | None,
    ):
        """
        Sets the participant's name.

        Parameters
        ----------
        name : str or None
            The name of the participant.

        Returns
        -------
        None
        """
        if name is not None:
            assert isinstance(name, str), "'name' must be a string."
        self._name = name

    def set_gender(
        self,
        gender: str | None,
    ):
        """
        Sets the participant's gender.

        Parameters
        ----------
        gender : str or None
            The gender of the participant.

        Returns
        -------
        None
        """
        if gender is not None:
            assert isinstance(gender, str), "'gender' must be a string."
        self._gender = gender

    def set_height(
        self,
        height: int | float | None,
    ):
        """
        Sets the participant's height in meters.

        Parameters
        ----------
        height : int or float or None
            The height of the participant.

        Returns
        -------
        None
        """
        if height is not None:
            txt = "'height' must be a float or int."
            assert isinstance(height, (int, float)), txt
        self._height = height

    def set_weight(
        self,
        weight: int | float | None,
    ):
        """
        Sets the participant's weight in kg.

        Parameters
        ----------
        weight : int or float or None
            The weight of the participant.

        Returns
        -------
        None
        """
        if weight is not None:
            txt = "'weight' must be a float or int."
            assert isinstance(weight, (int, float)), txt
        self._weight = weight

    def set_age(
        self,
        age: int | float | None,
    ):
        """
        Sets the participant's age in years.

        Parameters
        ----------
        age : int or float or None
            The age of the participant.

        Returns
        -------
        None
        """
        if age is not None:
            txt = "'age' must be a float or int."
            assert isinstance(age, (int, float)), txt
        self._age = age

    def set_birthdate(
        self,
        birthdate: date | None,
    ):
        """
        Sets the participant's birth date.

        Parameters
        ----------
        birthdate : datetime.date or None
            The birth date of the participant.

        Returns
        -------
        None
        """
        if birthdate is not None:
            txt = "'birth_date' must be a datetime.date or datetime.datetime."
            assert isinstance(birthdate, (datetime, date)), txt
            if isinstance(birthdate, datetime):
                self._birthdate = birthdate.date()
            else:
                self._birthdate = birthdate
        else:
            self._birthdate = birthdate

    @property
    def surname(self):
        """
        Gets the participant surname.

        Returns
        -------
        str or None
            The participant's surname.
        """
        return self._surname

    @property
    def name(self):
        """
        Gets the participant name.

        Returns
        -------
        str or None
            The participant's name.
        """
        return self._name

    @property
    def gender(self):
        """
        Gets the participant gender.

        Returns
        -------
        str or None
            The participant's gender.
        """
        return self._gender

    @property
    def height(self):
        """
        Gets the participant height in meters.

        Returns
        -------
        float or None
            The participant's height in meters.
        """
        return self._height

    @property
    def weight(self):
        """
        Gets the participant weight in kg.

        Returns
        -------
        float or None
            The participant's weight in kg.
        """
        return self._weight

    @property
    def birthdate(self):
        """
        Gets the participant birth date.

        Returns
        -------
        datetime.date or None
            The participant's birth date.
        """
        return self._birthdate

    @property
    def recordingdate(self):
        """
        Gets the test recording date.

        Returns
        -------
        datetime.date or None
            The test recording date.
        """
        return self._recordingdate

    @property
    def bmi(self):
        """
        Gets the participant BMI in kg/m^2.

        Returns
        -------
        float or None
            The participant's BMI in kg/m^2. Returns None if height or weight is None.
        """
        if self.height is None or self.weight is None:
            return None
        return self.weight / (self.height**2)

    @property
    def fullname(self):
        """
        Gets the participant full name.

        Returns
        -------
        str
            The participant's full name.
        """
        return f"{self.surname} {self.name}"

    @property
    def age(self):
        """
        Gets the age of the participant in years.

        Returns
        -------
        int or None
            The age of the participant in years. Returns None if age or birthdate is None.
        """
        if self._age is not None:
            return self._age
        if self._birthdate is not None:
            return int((self._recordingdate - self._birthdate).days // 365)  # type: ignore
        return None

    @property
    def hrmax(self):
        """
        Gets the maximum theoretical heart rate according to Gellish.

        Returns
        -------
        float or None
            The maximum theoretical heart rate. Returns None if age is None.

        References
        ----------
        Gellish RL, Goslin BR, Olson RE, McDonald A, Russi GD, Moudgil VK.
            Longitudinal modeling of the relationship between age and maximal
            heart rate.
            Med Sci Sports Exerc. 2007;39(5):822-9.
            doi: 10.1097/mss.0b013e31803349c6.
        """
        if self.age is None:
            return None
        return 207 - 0.7 * self.age

    @property
    def units(self):
        """
        Returns the unit of measurement of the stored data.

        Returns
        -------
        dict
            A dictionary containing the units of measurement for each attribute.
        """
        return self._units

    def copy(self):
        """
        Returns a copy of the object.

        Returns
        -------
        Participant
            A copy of the Participant object.
        """
        return Participant(
            **{
                i: getattr(self, i)
                for i in self.units.keys()
                if i not in ["fullname", "bmi", "hrmax"]
            }
        )

    @property
    def dict(self):
        """
        Returns a dict representation of self

        Returns
        -------
        dict
            A dictionary representation of the Participant object.
        """
        out = {}
        for i, v in self.units.items():
            out[i + ((" [" + v + "]") if v != "" else "")] = getattr(self, i)
        return out

    @property
    def series(self):
        """
        Returns a pandas.Series representation of self

        Returns
        -------
        pandas.Series
            A pandas.Series representation of the Participant object.
        """
        vals = [(i, v) for i, v in self.units.items()]
        vals = pd.MultiIndex.from_tuples(vals)
        return pd.Series(list(self.dict.values()), index=vals)

    @property
    def dataframe(self):
        """
        Returns a pandas.DataFrame representation of self

        Returns
        -------
        pandas.DataFrame
            A pandas.DataFrame representation of the Participant object.
        """
        return pd.DataFrame(self.series).T

    def __repr__(self):
        """
        Returns a string representation of the Participant object.
        """
        return self.__str__()

    def __str__(self):
        """
        Returns a string representation of the Participant object.
        """
        return self.dataframe.__str__()


@runtime_checkable
class TestResults(Protocol):

    _summary: pd.DataFrame
    _analytics: pd.DataFrame
    _figures: dict[str, go.Figure | dict[str, go.Figure]]
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

    def save_all(self, path: str, force_overwrite: bool = True):

        # check the inputs
        if not isinstance(path, str):
            msg = "path must be a string defining the root folder where to "
            msg += "save the results of the test."
            raise ValueError(msg)

        if not isinstance(force_overwrite, bool):
            msg = "force_overwrite must be True or False."
            raise ValueError(msg)

        # generate a recursive function that automatically save
        # the data in the proper folder
        def save(filename: str, obj: pd.DataFrame | go.Figure | dict):
            if isinstance(obj, dict):
                for key, val in obj.items():
                    save(join(filename, key), val)

            # check overwrite
            if exists(filename):
                if not askyesnocancel(
                    "Overwrite check",
                    f"{filename} exists. Overwrite?",
                ):
                    filename += "(1)"

            makedirs(dirname(filename), exist_ok=True)
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(filename + ".csv")
            elif isinstance(obj, go.Figure):
                obj.write_image(filename + ".svg")
            else:
                msg = "saving procedure is not supported for objects of type "
                msg += "{type(obj)}."
                raise ValueError(msg)

        save(join(path, "summary"), self.summary)
        save(join(path, "analytics"), self.analytics)
        save(join(path, "figures"), self.figures)

    def __init__(self, test: Any, include_emg):
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

    def _emgsignals_processing_func(self, channel: EMGSignal):
        channel[:, :] = channel - channel.mean()  # type: ignore
        fsamp = 1 / np.mean(np.diff(channel.index))
        channel.apply(
            butterworth_filt,
            fcut=[20, 450],
            fsamp=fsamp,
            order=4,
            ftype="bandpass",
            phase_corrected=True,
            inplace=True,
        )
        channel.apply(
            rms_filt,
            order=int(0.1 * fsamp),
            pad_style="reflect",
            offset=1,
            inplace=True,
        )
        return channel

    def _signal1d_processing_func(self, signal: Signal1D):
        signal.fillna(inplace=True)
        fsamp = float(1 / np.mean(np.diff(signal.index)))
        signal[:, :] = np.apply_along_axis(
            func1d=lambda x: butterworth_filt(
                x,
                fcut=6,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
            ),
            axis=0,
            arr=signal.to_numpy(),
        )
        return signal

    def _signal3d_processing_func(self, signal: Signal3D):
        signal.fillna(inplace=True, value=0)
        fsamp = float(1 / np.mean(np.diff(signal.index)))
        signal[:, :] = np.apply_along_axis(
            func1d=lambda x: butterworth_filt(
                x,
                fcut=[10, 200],
                fsamp=fsamp,
                order=4,
                ftype="bandstop",
                phase_corrected=True,
            ),
            axis=0,
            arr=signal.to_numpy(),
        )
        return signal

    def _point3d_processing_func(self, point: Point3D):
        point.fillna(inplace=True)
        fsamp = float(1 / np.mean(np.diff(point.index)))
        point[:, :] = np.apply_along_axis(
            func1d=lambda x: butterworth_filt(
                x,
                fcut=[10, 200],
                fsamp=fsamp,
                order=4,
                ftype="bandstop",
                phase_corrected=True,
            ),
            axis=0,
            arr=point.to_numpy(),
        )
        return point

    def _forceplatforms_processing_func(self, fp: ForcePlatform):
        fp["origin"] = self._point3d_processing_func(fp["origin"])  # type: ignore
        fp["force"] = self._signal3d_processing_func(fp["force"])  # type: ignore
        fp["torque"] = self._signal3d_processing_func(fp["torque"])  # type: ignore
        return fp

    def _generate_results(self, raw_test: Any):
        if not isinstance(raw_test, TestProtocol):
            raise ValueError("test must be a TestProtocol instance.")
        self._summary = self._get_summary(raw_test)
        self._analytics = self._get_analytics(raw_test)
        self._figures = self._get_figures(raw_test)

    #! MANDATORY FIELDS TO BE IMPLEMENTED

    def _get_summary(self, test: Any) -> pd.DataFrame: ...

    def _get_analytics(self, test: Any) -> pd.DataFrame: ...

    def _get_figures(
        self, test: Any
    ) -> dict[str, go.Figure | dict[str, go.Figure]]: ...


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

    def save(self, file_path: str):
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
        extension = "." + self.__class__.__name__.lower()
        if not file_path.endswith(extension):
            file_path += extension
        overwrite = False
        while exists(file_path) and not overwrite:
            overwrite = askyesnocancel(
                title="File already exists",
                message="the provided file_path already exist. Overwrite?",
            )
            if not overwrite:
                file_path = file_path[: len(extension)] + "_" + extension
        if not exists(file_path) or overwrite:
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

    def get_results(self, include_emg: bool = True) -> TestResults: ...

    @property
    def processing_pipeline(self) -> ProcessingPipeline:
        """
        exercise data processing pipeline
        """
        ...

    @property
    def processed_data(self) -> Self: ...
