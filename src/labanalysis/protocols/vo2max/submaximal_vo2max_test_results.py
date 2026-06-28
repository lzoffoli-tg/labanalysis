"""Submaximal VO2max test results implementation."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.colors as p_colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...signalprocessing import mean_filt
from ..test_results import TestResults

if TYPE_CHECKING:
    from .submaximal_vo2max_test import SubmaximalVO2MaxTest


class SubmaximalVO2MaxTestResults(TestResults):

    def __init__(self, test: "SubmaximalVO2MaxTest"):
        self._summary = pd.DataFrame()
        self._analytics = pd.DataFrame()
        self._figures = {}
        self._generate_results(test)

    def _get_hrmax(self, test: "SubmaximalVO2MaxTest"):
        age = test.participant.age
        if age is None:
            raise ValueError("user's age or date of birth must be provided.")
        return 207 - 0.7 * age

    def _get_vo2max(self, test: "SubmaximalVO2MaxTest"):
        rq = test.metabolic_record.rq.to_numpy().flatten().astype(float)
        hr = test.metabolic_record.hr.to_numpy().flatten().astype(float)
        vo2 = test.metabolic_record.vo2.to_numpy().flatten().astype(float)
        hrmax = self._get_hrmax(test)

        # from Beck O N, Kipp S K, Byrnes W C, Kram R.
        # Use aerobic energy expenditure instead of oxygen uptake to quantify
        # exercise intensity and predict endurance performance.
        # J Appl Physiol 125: 672–674, 2018.
        # https://www.doi.org/10.1152/japplphysiol.00940.2017.
        idx = np.where(rq>0.832)[0]
        vo2_perc = (2 * rq[idx] - 1.663999) ** 0.5 + 0.301
        from_rq = max(vo2[idx] / vo2_perc)

        if test.metabolic_record.hr.is_empty():
            return float(from_rq)
        else:
            idx = np.where(rq > 0.95)[0]
            from_hr = np.polyval(
                np.polyfit(hr[idx], vo2[idx], 1),
                hrmax,
            )
            return float(min(from_hr, from_rq))

    def _get_fatmax(self, test: "SubmaximalVO2MaxTest"):
        wgt = test.participant.weight
        if wgt is None:
            raise ValueError("participant's weight must be provided.")
        return float(test.metabolic_record.fat_oxidation.max() * wgt)  # type: ignore

    def _get_vt2_params(self, test: "SubmaximalVO2MaxTest"):

        # get the data
        vo2 = test.metabolic_record.vo2.to_numpy().flatten().astype(float)
        vco2 = test.metabolic_record.vco2.to_numpy().flatten().astype(float)
        hr = test.metabolic_record.hr.to_numpy().flatten().astype(float)
        wgt = test.participant.weight
        if wgt is None:
            raise ValueError("participant's weight must be provided.")
        gender = test.participant.gender
        if gender is None or gender not in ["Male", "Female"]:
            msg = "participant's gender must be provided as Male or Female."
            raise ValueError(msg)

        # use sympy to estimate the crossing point between the identity line
        # and a third order polynomial fitted on the vco2/vo2 curve
        o2 = sympy.Symbol("VO2", real=True)
        b3, b2, b1, b0 = np.polyfit(vo2, vco2, 3)
        solutions = sympy.nroots(o2**3 * b3 + o2**2 * b2 + o2 * b1 + b0 - o2)
        real_solutions = [
            i
            for i in solutions
            if complex(i).imag == 0 and i >= np.min(vo2) and i <= np.max(vo2)
        ]
        vt_vo2 = float(real_solutions[0])
        vt_idx = np.where(vo2 >= vt_vo2)[0][0]

        # calculate the additional parameters
        vt_vo2p = vt_vo2 / self._get_vo2max(test) * 100
        if test.metabolic_record.hr.is_empty():
            vt_hr = np.nan
            vt_hrp = np.nan
        else:   
            vt_hr = float(hr[vt_idx])
            vt_hrp = vt_hr / self._get_hrmax(test) * 100
        running_speed = float(Run().predict_speed(vo2=vt_vo2, grade=0)[0])
        cycling_power = Bike().predict_power(
            vo2=vt_vo2,
            weight=wgt,
            gender=gender,  # type: ignore
        )
        cycling_power = float(cycling_power[0])
        return {
            ("VO2", "ml/kg/min"): round(vt_vo2, 1),
            ("VO2", "%VO2max"): round(vt_vo2p, 1),
            ("HR", "bpm"): '-' if hr.size == 0 else round(vt_hr, 1),
            ("HR", "%HRmax"): '-' if hr.size == 0 else round(vt_hrp, 1),
            ("Running Speed", "km/h"): round(running_speed, 1),
            ("Cycling Power", "W"): round(cycling_power, 1),
        }

    def _get_fatmax_params(self, test: "SubmaximalVO2MaxTest"):
        wgt = test.participant.weight
        if wgt is None:
            raise ValueError("participant's weight must be provided.")
        gender = test.participant.gender
        if gender is None or gender not in ["Male", "Female"]:
            msg = "participant's gender must be provided as Male or Female."
            raise ValueError(msg)
        vo2 = test.metabolic_record.vo2.to_numpy().flatten().astype(float)
        hr = test.metabolic_record.hr.to_numpy().flatten().astype(float)
        fox = test.metabolic_record.fat_oxidation.to_numpy().flatten().astype(float)
        idx = np.argmax(fox)
        vo2p = vo2 / self._get_vo2max(test) * 100
        hrp = hr / self._get_hrmax(test) * 100
        running_speed = float(Run().predict_speed(vo2=vo2[idx], grade=0)[0])
        cycling_power = Bike().predict_power(
            vo2=vo2[idx],
            weight=wgt,
            gender=gender,  # type: ignore
        )
        cycling_power = float(cycling_power[0])
        return {
            ("VO2", "ml/kg/min"): round(float(vo2[idx]), 1),
            ("VO2", "%VO2max"): round(float(vo2p[idx]), 1),
            ("HR", "bpm"): '-' if hr.size == 0 else round(float(hr[idx]), 1),
            ("HR", "%HRmax"): '-' if hr.size == 0 else round(float(hrp[idx]), 1),
            ("Running Speed", "km/h"): round(running_speed, 1),
            ("Cycling Power", "W"): round(cycling_power, 1),
        }

    def _get_summary(self, test: "SubmaximalVO2MaxTest"):
        fox = pd.DataFrame(
            data=[self._get_fatmax_params(test)],
            index=["FatMax Oxidation"],
        )
        vt2 = pd.DataFrame(
            data=[self._get_vt2_params(test)],
            index=["Anaerobic Threshold"],
        )
        out = pd.concat([fox, vt2])
        out.insert(out.shape[1], ("VO2Max", "ml/kg/min"), None)
        out.insert(out.shape[1], ("FatMax", "g/min"), None)
        out.loc["Estimated", [("VO2Max", "ml/kg/min")]] = round(
            self._get_vo2max(test),
            1,
        )
        out.loc["Estimated", [("FatMax", "g/min")]] = round(
            self._get_fatmax(test),
            2,
        )
        out.columns = pd.MultiIndex.from_tuples(
            out.columns.tolist(),  # type: ignore
        )

        return out

    def _get_analytics(self, test: "SubmaximalVO2MaxTest"):
        out = test.metabolic_record.to_dataframe()
        out.loc[out.index, "VO2 %VO2max"] = (
            test.metabolic_record.vo2.to_numpy() / self._get_vo2max(test) * 100
        )
        if test.metabolic_record.hr.is_empty():
            out.loc[out.index, "HR %HRmax"] = None
        else:
            out.loc[out.index, "HR %HRmax"] = (
                test.metabolic_record.hr.to_numpy() / self._get_hrmax(test) * 100
            )
        cols = []
        for c in out.columns:
            splits = c.rsplit(" ", 1)
            if len(splits) == 1:
                splits.append("")
            splits = [splits[0].upper(), splits[1]]
            cols.append(splits)
        out.columns = pd.MultiIndex.from_tuples(cols)

        return out.sort_index(axis=1)

    def _get_figures(self, test: "SubmaximalVO2MaxTest"):

        # get the data
        tracks = self.analytics
        time_val = (tracks.index.to_numpy() / 60).round(1)
        time_lbl = "Time (min)"
        vo2_val = tracks[("VO2", "ml/kg/min")].to_numpy().flatten()
        vo2_lbl = "VO<sub>2</sub> (ml/kg/min)"
        vco2_val = tracks[("VCO2", "ml/kg/min")].to_numpy().flatten()
        vco2_lbl = "VCO<sub>2</sub> (ml/kg/min)"
        rq_val = tracks[("RQ", "")].to_numpy().flatten()
        rq_lbl = "RQ"
        hr_val = tracks[("HR", "1/min")].to_numpy().flatten()
        hr_lbl = "HR (bpm)"
        rf_val = tracks[("RF", "1/min")].to_numpy().flatten()
        ve_val = tracks[("VE", "l/min")].to_numpy().flatten()
        ve_rf_val = ve_val / rf_val
        ve_rf_lbl = "Ventilation by Breath (l)"
        wgt = test.participant.weight
        if wgt is None:
            raise ValueError("Participant's Weight must be provided.")
        ve_vco2 = ve_val / (vco2_val / 1000 * wgt)
        ve_vo2 = ve_val / (vo2_val / 1000 * wgt)
        summary = self.summary
        fmo_vo2, vt2_vo2 = summary[[("VO2", "ml/kg/min")]].to_numpy().flatten()[:2]
        vt2_time = time_val[vo2_val <= vt2_vo2][-1]
        fmo_time = time_val[vo2_val <= fmo_vo2][-1]

        # prepare the figure
        titles = ["Oxygen Consumption and Heart Rate", "Ventilatory Equivalents"]
        titles += ["Ventilatory Pattern", "VCO<sub>2</sub> vs. VO<sub>2</sub>"]
        specs = [[{"secondary_y": True}, {}], [{"secondary_y": True}, {}]]
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=titles,
            specs=specs,
            horizontal_spacing=0.25,
            vertical_spacing=0.25,
        )
        color_map = p_colors.qualitative.Plotly

        def plot_trace(
            x: np.ndarray,
            y: np.ndarray,
            color: str,
            name: str,
            row: int,
            col: int,
            secondary_y: bool,
            legend: str,
        ):
            fig.add_trace(
                row=row,
                col=col,
                secondary_y=secondary_y,
                trace=go.Scatter(
                    x=x,
                    y=y,
                    name=name,
                    mode="markers",
                    opacity=0.6,
                    legendgroup=legend,
                    legendgrouptitle_text=legend,
                    marker_color=color,
                    showlegend=True,
                ),
            )
            fig.add_trace(
                row=row,
                col=col,
                secondary_y=secondary_y,
                trace=go.Scatter(
                    x=x,
                    y=mean_filt(y, 5),
                    name="Interpolated " + name,
                    mode="lines",
                    opacity=0.6,
                    legendgroup=legend,
                    legendgrouptitle_text=legend,
                    marker_color=color,
                    showlegend=True,
                ),
            )


__all__ = ["SubmaximalVO2MaxTestResults"]
