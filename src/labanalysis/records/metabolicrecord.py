"""Metabolic data record module."""

import datetime
from os.path import exists

import numpy as np
import pandas as pd

from ..utils import ureg
from ..timeseries import Signal1D
from .record import Record


class MetabolicRecord(Record):

    def __setattr__(self, key, value):
        if key in ["_data", "_breath_by_breath"]:
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if not key in ["vo2", "vco2", "ve", "hr", "rf"]:
            msg = "only 'vo2', 'vco2', 've', 'rf' and 'hr' attributes can be "
            msg += " passed to MetabolicRecord instances."
            raise ValueError(msg)
        if not isinstance(value, Signal1D):
            raise ValueError("value must be a Signal1D.")
        self._data[key] = value

    def __init__(
        self,
        vo2: Signal1D,
        hr: Signal1D,
        vco2: Signal1D,
        ve: Signal1D,
        rf: Signal1D,
        breath_by_breath: bool = False,
    ):
        super().__init__()
        self.set_vo2(vo2)
        self.set_hr(hr)
        self.set_vco2(vco2)
        self.set_ve(ve)
        self.set_breath_by_breath(breath_by_breath)
        self.set_rf(rf)

    def set_rf(self, signal: Signal1D):
        if not isinstance(signal, Signal1D):
            raise ValueError("signal must be a Signal1D instance.")
        self["rf"] = signal

    def set_vo2(self, signal: Signal1D):
        if not isinstance(signal, Signal1D):
            raise ValueError("signal must be a Signal1D instance.")
        self["vo2"] = signal

    def set_vco2(self, signal: Signal1D):
        if not isinstance(signal, Signal1D):
            raise ValueError("signal must be a Signal1D instance.")
        self["vco2"] = signal

    def set_hr(self, signal: Signal1D):
        if not isinstance(signal, Signal1D):
            raise ValueError("signal must be a Signal1D instance.")
        self["hr"] = signal

    def set_ve(self, signal: Signal1D):
        if not isinstance(signal, Signal1D):
            raise ValueError("signal must be a Signal1D instance.")
        self["ve"] = signal

    def set_breath_by_breath(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("value must be True or False.")
        self._breath_by_breath = value

    @property
    def breath_by_breath(self):
        """return the breath-by-breath setting"""
        return self._breath_by_breath

    @property
    def rq(self):
        """return the RQ values"""
        return Signal1D(
            self.vco2.to_numpy() / self.vo2.to_numpy(),
            self.index,
            1 * ureg.dimensionless,
        )

    @property
    def fat_oxidation(self):
        """return the fat oxydation status"""
        vco2 = self.vco2.to_numpy()
        vo2 = self.vo2.to_numpy()
        # from: Frayn KN. "Calculation of substrate oxidation rates in vivo from gaseous exchange." Nutrition (1983).
        fox = 1.695 * vo2 / 1000 - 1.701 * vco2 / 1000
        fox[fox < 0] = 0
        return Signal1D(fox, self.index, "g/kg/min")

    @property
    def vo2(self):
        """return the VO2 signal"""
        out: Signal1D = self["vo2"]
        return out

    @property
    def vco2(self):
        """return the VCO2 signal"""
        out: Signal1D = self["vco2"]
        return out

    @property
    def ve(self):
        """return the VE signal"""
        out: Signal1D = self["ve"]
        return out

    @property
    def hr(self):
        """return the HR signal"""
        out: Signal1D = self["hr"]
        return out

    @property
    def rf(self):
        """return the Respiratory Frequency signal"""
        out: Signal1D = self["rf"]
        return out

    def copy(self):
        """return a copy of the object"""
        return MetabolicRecord(
            vo2=self.vo2,
            vco2=self.vco2,
            hr=self.hr,
            ve=self.ve,
            rf=self.rf,
            breath_by_breath=self.breath_by_breath,
        )

    def to_dataframe(self):
        """return a dataframe view of the object"""
        out = super().to_dataframe()
        out.loc[out.index, "RQ"] = self.rq.to_numpy()
        out.loc[out.index, "Fat Oxidation g/kg/min"] = self.fat_oxidation.to_numpy()
        return out

    @classmethod
    def from_file(
        cls,
        filename: str,
        breath_by_breath: bool = False,
    ):
        """
        generate a MetabolicRecord from file

        Parameters
        ----------
        filename: str
            the path to a csv or xlsx file containing the data

        breath_by_breath: bool (default=False)
            if True the data are assumed to be sampled breath-by-breath. If
            False, the data are sampled at constant sample rate.
        """

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

        # get the weight
        wgt = float(raw.iloc[np.where(raw.iloc[:, 0] == "Weight (kg)")[0][0], 1])  # type: ignore

        # get the data values
        try:
            time_col = np.where(raw.columns == "Time")[0][0]
        except Exception:
            time_col = np.where(raw.columns == "t")[0][0]
        i0 = 0 if isinstance(raw.columns, pd.MultiIndex) else 2
        data = raw.iloc[i0:, (time_col + 1) :]

        # update the column headers
        labels = data.columns.to_numpy().astype(str)
        if isinstance(raw.columns, pd.MultiIndex):
            units = [str(i[1]) for i in data.columns]
        else:
            units = [
                i.replace("---", "")
                for i in raw.iloc[0, (time_col + 1) :].to_numpy().astype(str)
            ]
        cols = pd.MultiIndex.from_tuples([(i, j) for i, j in zip(labels, units)])
        data.columns = cols

        # update the indices
        try:
            test_date = str(data.iloc[np.where(raw.iloc[:, 3] == "Test date")[0][0], 4])
            dd, mm, aaaa = test_date.split("/")
            test_date = datetime.date(int(aaaa), int(mm), int(dd))
        except Exception:
            try:
                test_date = str(
                    raw.columns[
                        np.where(raw.columns == "Test date (MMDDYYYY)")[0][0] + 1
                    ]
                )
                mm, dd, aaaa = test_date.split("/")
                test_date = datetime.date(int(aaaa), int(mm), int(dd))
            except Exception:
                try:
                    test_date = str(
                        raw.columns[np.where(raw.columns == "Test date")[0][0] + 1]
                    )
                    dd, mm, aaaa = test_date.split("/")
                    test_date = datetime.date(int(aaaa), int(mm), int(dd))
                except Exception:
                    now = datetime.datetime.now()
                    test_date = datetime.date(now.year, now.month, now.day)
        col = []
        max_row = 0
        for i, time in enumerate(raw.iloc[i0:, time_col].to_numpy()):  # type: ignore
            try:
                hour, minute, second = [int(i) for i in str(time).split(":")]
                max_row = i
                col.append(hour * 3600 + minute * 60 + second)
            except Exception:
                break
        data = data.iloc[: max_row + 1]

        # get the signals
        time = np.array(col)
        vo2 = Signal1D(
            data[("VO2", "mL/min")].to_numpy().astype(float) / wgt,
            index=time,
            unit="mL/kg/min",
        )
        vco2 = Signal1D(
            data[("VCO2", "mL/min")].to_numpy().astype(float) / wgt,
            index=time,
            unit="mL/kg/min",
        )
        try:
            ve = Signal1D(
                data[("Ve", "L/min")].to_numpy().astype(float),
                index=time,
                unit="L/min",
            )
        except Exception:
            ve = Signal1D(
                data[("VE", "L/min")].to_numpy().astype(float),
                index=time,
                unit="L/min",
            )
        hr_data = data[("HR", "bpm")].to_numpy()
        if np.any(np.array([isinstance(x, str) and x == "-" for x in hr_data])):
            hr = Signal1D(
                np.array([], dtype=float),
                index=np.array([], dtype=float),
                unit="1/min",
            )
        else:
            hr = Signal1D(
                hr_data.astype(float),
                index=time,
                unit="1/min",
            )
        rf = Signal1D(
            data[("Rf", "1/min")].to_numpy().astype(float),
            index=time,
            unit="1/min",
        )
        return cls(
            vo2=vo2,
            vco2=vco2,
            hr=hr,
            ve=ve,
            rf=rf,
            breath_by_breath=breath_by_breath,
        )


__all__ = ["MetabolicRecord"]
