"""Participant class for lab test participant data management."""

from datetime import date, datetime

import numpy as np
import pandas as pd
from os.path import exists


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

    @classmethod
    def from_cosmed_file(cls, filename: str):
        # check the filename
        msg = "filename must be the path to a .csv or .xlsx file containing"
        msg += " relevant test data."
        if not isinstance(filename, str) or not exists(filename):
            raise ValueError(msg)
        assert filename.endswith(".xlsx") or filename.endswith(".csv"), (
            filename + ' must be a ".xlsx" or a ".csv" file.'
        )

        # get the raw data
        if filename.endswith(".xlsx"):
            raw = pd.read_excel(filename)
        else:
            raw = pd.read_csv(filename, sep=";")

        # handle the participant generation if required
        try:
            name = str(raw.iloc[np.where(raw.iloc[:, 0] == "First Name")[0][0], 1])
        except Exception:
            name = None
        try:
            surname = str(raw.iloc[np.where(raw.iloc[:, 0] == "Last Name")[0][0], 1])
        except Exception:
            surname = None
        try:
            gender = str(raw.iloc[np.where(raw.iloc[:, 0] == "Gender")[0][0], 1])
        except Exception:
            gender = None
        try:
            height = int(raw.iloc[np.where(raw.iloc[:, 0] == "Height (cm)")[0][0], 1])  # type: ignore
        except Exception:
            height = None
        try:
            weight = float(raw.iloc[np.where(raw.iloc[:, 0] == "Weight (kg)")[0][0], 1])  # type: ignore
        except Exception:
            weight = None
        try:
            dob = str(raw.iloc[np.where(raw.iloc[:, 0] == "D.O.B.")[0][0], 1])
            dd, mm, aaaa = dob.split("/")
            dob = date(int(aaaa), int(mm), int(dd))
        except Exception:
            try:
                dob = str(
                    raw.iloc[
                        np.where(raw.iloc[:, 0] == "Birthdate (MMDDYYYY)")[0][0], 1
                    ]
                )
                mm, dd, aaaa = dob.split("/")
                dob = date(int(aaaa), int(mm), int(dd))
            except Exception:
                dob = None
        try:
            test_date = str(raw.iloc[np.where(raw.iloc[:, 3] == "Test date")[0][0], 4])
            dd, mm, aaaa = test_date.split("/")
            test_date = date(int(aaaa), int(mm), int(dd))
        except Exception:
            try:
                test_date = str(
                    raw.columns[
                        np.where(raw.columns == "Test date (MMDDYYYY)")[0][0] + 1
                    ]
                )
                mm, dd, aaaa = test_date.split("/")
                test_date = date(int(aaaa), int(mm), int(dd))
            except Exception:
                try:
                    test_date = str(
                        raw.columns[np.where(raw.columns == "Test date")[0][0] + 1]
                    )
                    dd, mm, aaaa = test_date.split("/")
                    test_date = date(int(aaaa), int(mm), int(dd))
                except Exception:
                    now = datetime.now()
                    test_date = date(now.year, now.month, now.day)
        return Participant(
            name=name,
            surname=surname,
            gender=gender,
            height=height,
            weight=weight,
            birthdate=dob,
            recordingdate=test_date,
        )


__all__ = ["Participant"]
