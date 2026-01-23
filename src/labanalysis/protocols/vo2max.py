"""isokinetic test module"""

#! IMPORTS

import numpy as np
import pandas as pd
import sympy
import plotly.graph_objects as go
import plotly.colors as p_colors
from plotly.subplots import make_subplots

from ..signalprocessing import mean_filt

from ..equations.cardio import Bike, Run
from ..records.pipelines import get_default_processing_pipeline
from ..records.records import MetabolicRecord
from .normativedata import vo2max_normative_values
from .protocols import Participant, TestProtocol, TestResults

#! CONSTANTS


__all__ = ["SubmaximalVO2MaxTest", "SubmaximalVO2MaxTestResults"]


class SubmaximalVO2MaxTest(TestProtocol):

    def __init__(
        self,
        participant: Participant,
        metabolic_record: MetabolicRecord,
        normative_data: pd.DataFrame = vo2max_normative_values,
    ):
        super().__init__(
            participant,
            normative_data,
        )
        self.set_metabolic_record(metabolic_record)

    def set_metabolic_record(self, record: MetabolicRecord):
        if not isinstance(record, MetabolicRecord):
            raise ValueError("record must be a MetabolicRecord instance.")
        self._metabolic_record = record

    @property
    def metabolic_record(self):
        return self._metabolic_record

    def copy(self):
        return SubmaximalVO2MaxTest(
            participant=self.participant.copy(),
            normative_data=self.normative_data,
            metabolic_record=self.metabolic_record,
        )

    @classmethod
    def from_files(
        cls,
        filename: str,
        participant: Participant,
        normative_data: pd.DataFrame = vo2max_normative_values,
        breath_by_breath: bool = False,
    ):
        return cls(
            participant=participant,
            normative_data=normative_data,
            metabolic_record=MetabolicRecord.from_file(
                filename=filename,
                breath_by_breath=breath_by_breath,
            ),
        )

    def get_results(self):
        return SubmaximalVO2MaxTestResults(self.processed_data)

    @property
    def processed_data(self):
        out = self.copy()
        self.processing_pipeline(out.metabolic_record, inplace=True)
        return out

    @property
    def processing_pipeline(self):
        return get_default_processing_pipeline()


class SubmaximalVO2MaxTestResults(TestResults):

    def __init__(self, test: SubmaximalVO2MaxTest):
        self._summary = pd.DataFrame()
        self._analytics = pd.DataFrame()
        self._figures = {}
        self._generate_results(test)

    def _get_hrmax(self, test: SubmaximalVO2MaxTest):
        age = test.participant.age
        if age is None:
            raise ValueError("user's age or date of birth must be provided.")
        return 207 - 0.7 * age

    def _get_vo2max(self, test: SubmaximalVO2MaxTest):
        rq = test.metabolic_record.rq.to_numpy().flatten().astype(float)
        hr = test.metabolic_record.hr.to_numpy().flatten().astype(float)
        vo2 = test.metabolic_record.vo2.to_numpy().flatten().astype(float)
        idx = np.where(rq > 0.95)[0]
        hrmax = self._get_hrmax(test)
        from_hr = np.polyval(
            np.polyfit(hr[idx], vo2[idx], 1),
            hrmax,
        )
        # from Beck O N, Kipp S K, Byrnes W C, Kram R.
        # Use aerobic energy expenditure instead of oxygen uptake to quantify
        # exercise intensity and predict endurance performance.
        # J Appl Physiol 125: 672–674, 2018.
        # https://www.doi.org/10.1152/japplphysiol.00940.2017.
        idx = np.where(rq>0.832)
        vo2_perc = (2 * rq[idx] - 1.663999) ** 0.5 + 0.301
        from_rq = max(vo2[idx] / vo2_perc)
        return float(min(from_hr, from_rq))

    def _get_fatmax(self, test: SubmaximalVO2MaxTest):
        wgt = test.participant.weight
        if wgt is None:
            raise ValueError("participant's weight must be provided.")
        return float(test.metabolic_record.fat_oxidation.max() * wgt)  # type: ignore

    def _get_vt2_params(self, test: SubmaximalVO2MaxTest):

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
            ("HR", "bpm"): round(vt_hr, 1),
            ("HR", "%HRmax"): round(vt_hrp, 1),
            ("Running Speed", "km/h"): round(running_speed, 1),
            ("Cycling Power", "W"): round(cycling_power, 1),
        }

    def _get_fatmax_params(self, test: SubmaximalVO2MaxTest):
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
            ("HR", "bpm"): round(float(hr[idx]), 1),
            ("HR", "%HRmax"): round(float(hrp[idx]), 1),
            ("Running Speed", "km/h"): round(running_speed, 1),
            ("Cycling Power", "W"): round(cycling_power, 1),
        }

    def _get_summary(self, test: SubmaximalVO2MaxTest):
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
            1,
        )
        out.columns = pd.MultiIndex.from_tuples(
            out.columns.tolist(),  # type: ignore
        )

        return out

    def _get_analytics(self, test: SubmaximalVO2MaxTest):
        out = test.metabolic_record.to_dataframe()
        out.loc[out.index, "VO2 %VO2max"] = (
            test.metabolic_record.vo2.to_numpy() / self._get_vo2max(test) * 100
        )
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

    def _get_figures(self, test: SubmaximalVO2MaxTest):

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

        # top-left plot
        plot_trace(
            time_val,
            vo2_val,
            color_map[0],
            vo2_lbl,
            1,
            1,
            False,
            titles[0],
        )
        plot_trace(
            time_val,
            hr_val,
            color_map[1],
            hr_lbl,
            1,
            1,
            True,
            titles[0],
        )

        # top-right plot
        plot_trace(
            time_val,
            ve_vo2,
            color_map[2],
            "VE/VO<sub>2</sub>",
            1,
            2,
            False,
            titles[1],
        )
        plot_trace(
            time_val,
            ve_vco2,
            color_map[3],
            "VE/VCO<sub>2</sub>",
            1,
            2,
            False,
            titles[1],
        )

        # bottom-left plot
        plot_trace(
            time_val,
            ve_rf_val,
            color_map[4],
            "Ventilation by Breath",
            2,
            1,
            False,
            titles[2],
        )
        plot_trace(
            time_val,
            rq_val,
            color_map[5],
            rq_lbl,
            2,
            1,
            True,
            titles[2],
        )

        # bottom-right plot
        fit = np.polyfit(vo2_val, vco2_val, 3)
        sorted_vo2 = np.sort(vo2_val)
        fitted = np.polyval(fit, sorted_vo2)
        fig.add_trace(
            row=2,
            col=2,
            trace=go.Scatter(
                x=vo2_val,
                y=vco2_val,
                name="VCO<sub>2</sub> vs. VO<sub>2</sub>",
                mode="markers",
                opacity=0.6,
                legendgroup=titles[3],
                legendgrouptitle_text=titles[3],
                marker_color=color_map[6],
                showlegend=True,
            ),
        )
        fig.add_trace(
            row=2,
            col=2,
            trace=go.Scatter(
                x=sorted_vo2,
                y=fitted,
                name="Interpolated VCO<sub>2</sub> vs. VO<sub>2</sub>",
                mode="lines",
                opacity=0.6,
                legendgroup=titles[3],
                legendgrouptitle_text=titles[3],
                marker_color=color_map[6],
                showlegend=True,
            ),
        )

        # add the identity line to the VCO2 vs VO2 plot
        rval = [min(vo2_val.min(), vco2_val.min())]
        rval += [max(vo2_val.max(), vco2_val.max())]
        fig.add_trace(
            row=2,
            col=2,
            trace=go.Scatter(
                x=rval,
                y=rval,
                showlegend=True,
                mode="lines",
                opacity=0.6,
                line_dash="dashdot",
                legendgroup=titles[3],
                legendgrouptitle_text=titles[3],
                line_color=color_map[7],
                name="Identity line",
            ),
        )

        # add the vt2 and fatmax lines
        fig.add_vline(
            row=1,  # type: ignore
            col=1,  # type: ignore
            line_dash="dash",
            line_color="black",
            opacity=0.6,
            name="Anaerobic Threshold",
            legendgroup="Thresholds",
            legendgrouptitle_text="Thresholds",
            x=vt2_time,
            showlegend=True,
        )
        fig.add_vline(
            row=1,  # type: ignore
            col=1,  # type: ignore
            line_dash="dot",
            name="FatMax",
            line_color="grey",
            opacity=0.6,
            legendgroup="Thresholds",
            legendgrouptitle_text="Thresholds",
            x=fmo_time,
            showlegend=True,
        )
        fig.add_vline(
            row=1,  # type: ignore
            col=2,  # type: ignore
            line_dash="dash",
            line_color="black",
            opacity=0.6,
            name="Anaerobic Threshold",
            legendgroup="Thresholds",
            legendgrouptitle_text="Thresholds",
            x=vt2_time,
        )
        fig.add_vline(
            row=1,  # type: ignore
            col=2,  # type: ignore
            line_dash="dot",
            name="FatMax",
            line_color="grey",
            opacity=0.6,
            legendgroup="Thresholds",
            legendgrouptitle_text="Thresholds",
            x=fmo_time,
        )
        fig.add_vline(
            row=2,  # type: ignore
            col=1,  # type: ignore
            line_dash="dash",
            line_color="black",
            opacity=0.6,
            name="Anaerobic Threshold",
            legendgroup="Thresholds",
            legendgrouptitle_text="Thresholds",
            x=vt2_time,
        )
        fig.add_vline(
            row=2,  # type: ignore
            col=1,  # type: ignore
            line_dash="dot",
            name="FatMax",
            line_color="grey",
            opacity=0.6,
            legendgroup="Thresholds",
            legendgrouptitle_text="Thresholds",
            x=fmo_time,
        )
        fig.add_vline(
            row=2,  # type: ignore
            col=2,  # type: ignore
            line_dash="dash",
            line_color="black",
            opacity=0.6,
            name="Anaerobic Threshold",
            legendgroup="Thresholds",
            legendgrouptitle_text="Thresholds",
            x=vt2_vo2,
        )
        fig.add_vline(
            row=2,  # type: ignore
            col=2,  # type: ignore
            line_dash="dot",
            name="FatMax",
            line_color="grey",
            opacity=0.6,
            legendgroup="Thresholds",
            legendgrouptitle_text="Thresholds",
            x=fmo_vo2,
        )

        # update the layout
        time_range = [0, np.max(time_val)]
        fig.update_layout(
            template="plotly_white",
            height=800,
            width=1000,
            legend=dict(tracegroupgap=40),
        )
        fig.update_xaxes(
            showgrid=False,
            showline=True,
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=False,
            showline=True,
            zeroline=False,
        )
        fig.update_xaxes(
            row=1,
            title=time_lbl,
            range=time_range,
        )
        fig.update_yaxes(
            row=1,
            col=1,
            secondary_y=False,
            title=vo2_lbl,
        )
        fig.update_yaxes(
            row=1,
            col=1,
            secondary_y=True,
            title=hr_lbl,
        )
        fig.update_yaxes(
            row=1,
            col=2,
            title="Metabolic Equivalents",
        )
        fig.update_xaxes(
            row=2,
            col=1,
            title=time_lbl,
            range=time_range,
        )
        fig.update_yaxes(
            row=2,
            col=1,
            secondary_y=False,
            title=ve_rf_lbl,
        )
        fig.update_yaxes(
            row=2,
            col=1,
            secondary_y=True,
            title="RQ",
        )
        fig.update_xaxes(
            row=2,
            col=2,
            title=vo2_lbl,
            range=rval,
            constrain="domain",
        )
        fig.update_yaxes(
            row=2,
            col=2,
            title=vco2_lbl,
            range=rval,
            scaleanchor="x4",
            scaleratio=1,
        )

        return {"metabolic chart": fig}
