"""balance test module"""

__all__ = ["UprightBalanceTest", "PlankBalanceTest"]


from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..constants import RANK_4COLORS, G
from ..modelling import Ellipse
from ..records.pipelines import get_default_processing_pipeline
from ..records.posture import PronePosture, UprightPosture
from ..records.records import ForcePlatform, TimeseriesRecord
from ..records.timeseries import EMGSignal, Point3D
from ..utils import FloatArray1D
from .normativedata import (
    plankbalance_normative_values,
    uprightbalance_normative_values,
)
from .protocols import Participant, TestProtocol, TestResults


def _get_sway_figure(
    cop_x: FloatArray1D,
    cop_y: FloatArray1D,
    normative_data: pd.DataFrame = pd.DataFrame(),
    emgsignals: TimeseriesRecord = TimeseriesRecord(),
):

    def balance_string(left: str, right: str, sep: str = " | "):
        width = max(len(left), len(right))
        nbsp = "\u00a0"  # non-breaking
        ljust = left.rjust(width).replace(" ", nbsp)
        rjust = right.ljust(width).replace(" ", nbsp)
        return sep.join([ljust, rjust])

    # collect emg signals
    emg_signals = {}
    for chn in emgsignals.values():
        if chn.muscle_name not in emg_signals:
            emg_signals[chn.muscle_name] = {}
        emg_signals[chn.muscle_name][chn.side] = chn.to_dataframe()

    # generate the figure and setup the layout
    if len(emg_signals) == 0 and normative_data.empty:
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=["Sway"],
        )
        ncols = 1
    elif len(emg_signals) == 0 and not normative_data.empty:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Sway", "Performance Overview"],
        )
        ncols = 2
    elif len(emg_signals) > 0 and normative_data.empty:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Sway", "Muscle imbalance"],
        )
        ncols = 2
    else:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Sway", "Performance Overview", "Muscle imbalance"],
        )
        ncols = 3

    # update overall layout
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Peformance<br>Levels",
        height=500,
        width=500 * ncols,
    )

    # plot normative data
    if not normative_data.empty:

        # get normative data
        norms_idx = normative_data.parameter == "area_of_stability_mm2"
        norms = normative_data.loc[norms_idx, ["mean", "std"]]
        avg, std = norms.values.astype(float).flatten()
        areas = np.array([avg, avg + 1 * std, avg + 2 * std])
        ranks = list(RANK_4COLORS.keys())[:-1][::-1]
        rank_colors = list(RANK_4COLORS.values())[:-1][::-1]

        # plot the background on the sway plot
        fig.add_shape(
            type="rect",
            xref="x domain",
            yref="y domain",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            fillcolor=RANK_4COLORS["Poor"],
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )

        # plot normative ellipses
        def build_ellipse(a, b, t, x0, y0):

            # Generate ellipse points
            theta = np.linspace(0, 2 * np.pi, 100)
            xy = np.column_stack([a * np.cos(theta), b * np.sin(theta)])

            # Apply rotation
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            xy_rot = xy @ R.T

            # translate to the center
            return (xy_rot + np.array([x0, y0])).T

        def is_within_ellipse(
            x: np.ndarray,
            y: np.ndarray,
            x0: float,
            y0: float,
            a: float,
            b: float,
            t: float,
        ):
            p1 = (((x - x0) * np.cos(t) + (y - y0) * np.sin(t)) ** 2) / a**2
            p2 = ((-(x - x0) * np.sin(t) + (y - y0) * np.cos(t)) ** 2) / b**2
            return np.asarray((p1 + p2) <= 1, dtype=bool)

        # get the sway ellipse
        ellipse = Ellipse().fit(cop_x, cop_y)

        # get the ellipse properties
        cop_x0, cop_y0 = ellipse.center
        semiaxis_a, semiaxis_b = ellipse.semi_axes
        cop_angle_rad = ellipse.rotation_angle / 180 * np.pi
        cop_area = ellipse.area

        samples_within = {}
        for area, color, label in zip(areas[::-1], rank_colors, ranks):

            # scale the axes according to the ratio between the ellipses area
            ratio = area / cop_area

            # add the ellipse
            x_ell, y_ell = build_ellipse(
                semiaxis_a * ratio,
                semiaxis_b * ratio,
                cop_angle_rad,
                cop_x0,
                cop_y0,
            )
            fig.add_trace(
                trace=go.Scatter(
                    x=x_ell,
                    y=y_ell,
                    fill="toself",
                    fillcolor=color,
                    line_width=0,
                    mode="lines",
                    name=label,
                    showlegend=False,
                    legendgroup=label,
                ),
                row=1,
                col=1,
            )

            # check the count of cop samples within the current norm
            within = is_within_ellipse(
                cop_x,
                cop_y,
                cop_x0,
                cop_y0,
                semiaxis_a * ratio,
                semiaxis_b * ratio,
                cop_angle_rad,
            )
            samples_within[label] = np.sum(within) / len(within) * 100

        # get the time spent within each norm interval
        ranks = ranks[::-1]
        for i in range(len(ranks) - 1):
            ranki = ranks[i]
            for j in range(i + 1, len(ranks)):
                rankj = ranks[j]
                samples_within[rankj] -= samples_within[ranki]
        samples_within["Poor"] = 100 - sum(samples_within.values())

        # plot the cumulative time spent at each level of norm
        ranks = list(RANK_4COLORS.keys())[::-1]
        colors = list(RANK_4COLORS.values())[::-1]
        for rank, color in zip(ranks, colors):
            value = samples_within[rank]
            fig.add_trace(
                trace=go.Bar(
                    x=[float(value)],
                    y=[rank],
                    text=[f"{value:0.2f}%"],
                    marker_color=color,
                    name=rank,
                    textposition="outside",
                    textangle=0,
                    orientation="h",
                    legendgroup=rank,
                ),
                row=1,
                col=2,
            )

        # update axes
        vrange = [0, max(20, np.max(list(samples_within.values())) * 1.25)]
        fig.update_yaxes(
            row=1,
            col=2,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor="black",
            showgrid=False,
        )
        fig.update_xaxes(
            row=1,
            col=2,
            title="Time lapsed (%)",
            zeroline=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            range=vrange,
        )

    # plot the emg signals
    if len(emg_signals) > 0:
        col = 2 if normative_data.empty else 3

        # plot the muscle balance
        cscales = [
            [i / (len(RANK_4COLORS) - 1), col]
            for i, col in enumerate(RANK_4COLORS.values())
        ]
        vals = []
        for i, (muscle, dct) in enumerate(emg_signals.items()):
            lt, rt = [float(m.mean()) for m in dct.values()]
            symm = (lt - rt) / (rt + lt) * 100
            val = max(-50, min(50, symm))
            vals.append(val)
            lbl = f"{abs(symm):0.1f}%" if -50 <= symm <= 50 else ">50.0%"
            fig.add_trace(
                trace=go.Bar(
                    x=[val],
                    y=[muscle.replace(" ", "<br>")],
                    orientation="h",
                    text=[lbl],
                    textangle=0,
                    textposition="outside",
                    name="Muscle imbalance",
                    legendgroup="Muscle imbalance",
                    showlegend=False,
                    marker=dict(
                        color=[abs(symm)],
                        colorscale=cscales,
                        cmin=0,  # range colore
                        cmax=50,
                        colorbar=dict(
                            title=dict(
                                text="Muscle<br>Imbalance<br>Levels",
                            ),
                            len=0.50,
                            y=0.5,
                            tickvals=[10, 20, 30, 40],
                            ticktext=["Minimal", "Low", "Moderate", "High"],
                        ),
                        showscale=False,
                    ),
                ),
                row=1,
                col=col,
            )

        # adjust the muscle balance axes
        vrange = max(50, np.max(abs(np.array(vals))) * 1.5)
        vrange = [-vrange, vrange]
        fig.update_xaxes(
            row=1,
            col=col,
            title=f"{balance_string("Left", "Right", "|")}<br>(%)",
            zeroline=False,
            showline=False,
            showgrid=False,
            showticklabels=False,
            range=vrange,
        )
        fig.update_yaxes(
            row=1,
            col=col,
            title="",
            showgrid=False,
            zeroline=True,
            showline=False,
            zerolinecolor="black",
            zerolinewidth=2,
        )
        fig.add_vline(
            x=0,
            row=1,  # type: ignore
            col=col,  # type: ignore
            showlegend=False,
            line_color="black",
            line_width=2,
            line_dash="solid",
        )

    # adjust the sway plot axes
    fig.update_xaxes(
        row=1,
        col=1,
        title=f"{balance_string("Left", "Right", "|")}<br>(mm)",
        scaleanchor="y",
        scaleratio=1,
        showgrid=False,
        zeroline=False,
        showline=False,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        title=f"{balance_string("Backward", "Forward", "|")}<br>(mm)",
        showgrid=False,
        zeroline=False,
        showline=False,
    )

    # add the sway
    fig.add_trace(
        trace=go.Scatter(
            x=cop_x,
            y=cop_y,
            mode="lines",
            opacity=0.5,
            line_width=1,
            line_color="black",
            name="Sway",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # add vertical and horizontal axes to the sway plot
    fig.add_hline(
        y=0,
        line_color="black",
        line_dash="dash",
        line_width=2,
        showlegend=False,
        row=1,  # type: ignore
        col=1,  # type: ignore
    )
    fig.add_vline(
        x=0,
        line_color="black",
        line_dash="dash",
        line_width=2,
        showlegend=False,
        row=1,  # type: ignore
        col=1,  # type: ignore
    )

    return fig


class UprightBalanceTest(TestProtocol):

    @property
    def eyes(self):
        return self._eyes

    @property
    def side(self):
        """
        Returns which side(s) have force data.

        Returns
        -------
        str
            "bilateral", "left", or "right".
        """
        return self.exercise.side

    def set_eyes(self, eyes: Literal["open", "closed"]):
        if eyes not in ["open", "closed"]:
            raise ValueError("eyes must be 'open' or 'closed'.")
        self._eyes = eyes

    def __init__(
        self,
        participant: Participant,
        exercise: UprightPosture,
        eyes: Literal["open", "closed"],
        normative_data: pd.DataFrame = uprightbalance_normative_values,
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float | int = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        super().__init__(
            participant=participant,
            normative_data=normative_data,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            relevant_muscle_map=relevant_muscle_map,
        )
        self.set_eyes(eyes)
        self.set_exercise(exercise)

    @classmethod
    def from_files(
        cls,
        filename: str,
        participant: Participant,
        eyes: Literal["open", "closed"],
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        normative_data: pd.DataFrame = uprightbalance_normative_values,
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float | int = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        return cls(
            participant=participant,
            normative_data=normative_data,
            eyes=eyes,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            relevant_muscle_map=relevant_muscle_map,
            emg_normalization_function=emg_normalization_function,
            exercise=UprightPosture.from_tdf(
                file=filename,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            ),
        )

    def set_exercise(self, exercise: UprightPosture):
        if not isinstance(exercise, UprightPosture):
            raise ValueError("exercise must be an UprightPosture instance.")
        self._exercise = exercise

    @property
    def exercise(self):
        return self._exercise

    def copy(self):
        return UprightBalanceTest(
            participant=self.participant,
            exercise=self.exercise,
            eyes=self.eyes,  # type: ignore
            normative_data=self.normative_data,
            emg_normalization_references=self.emg_normalization_references,
            emg_normalization_function=self.emg_normalization_function,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
            relevant_muscle_map=self.relevant_muscle_map,
        )

    @property
    def processed_data(self):

        # apply the pipeline to the test data
        exe = self.processing_pipeline(self.exercise, inplace=False)
        if not isinstance(exe, TimeseriesRecord):
            raise ValueError("Something went wrong during data processing.")

        # normalize emg data and remove non-relevant muscles
        norms = self.emg_normalization_values
        to_remove: list[str] = []
        for k, m in exe.emgsignals.items():

            # remove if non relevant
            if self.relevant_muscle_map is not None:
                if not any([i.lower() in k.lower() for i in self.relevant_muscle_map]):
                    to_remove.append(k)
                    continue

            # normalize
            if isinstance(m, EMGSignal):
                for (name, side), val in norms.items():
                    if m.muscle_name == name and m.side == side:
                        exe[k] = m / val * 100
                        exe[k].set_unit("%")  # type: ignore
                        break
        if len(to_remove) > 0:
            exe.drop(to_remove, True)

        # align the reference frame
        if self.side not in ["right", "left"]:

            def extract_cop(force: Any):
                if not isinstance(force, ForcePlatform):
                    raise ValueError("force must be a ForcePlatform instance.")
                cop = force.origin
                if not isinstance(cop, Point3D):
                    raise ValueError("force must be a ForcePlatform instance.")
                cop = cop.copy()
                return cop.to_numpy().astype(float).mean(axis=0)

            # on bilateral test, we rotate the system of forces to a
            rt = extract_cop(exe.right_foot_ground_reaction_force)
            lt = extract_cop(exe.left_foot_ground_reaction_force)

            def norm(arr):
                return arr / np.sum(arr**2) ** 0.5

            ml = norm(lt - rt)
            vt = np.array([0, 1, 0])
            ap = np.cross(ml, vt)
            origin = (rt + lt) / 2
            exe.change_reference_frame(
                ml,
                vt,
                ap,
                origin,
                inplace=True,
            )
            if exe is None:
                raise ValueError("reference frame alignment returned None")

        # return processed data
        out = self.copy()
        if not isinstance(exe, UprightPosture):
            raise ValueError("Something went wrong during data processing.")
        out.set_exercise(exe)
        return out

    @property
    def processing_pipeline(self):
        return get_default_processing_pipeline()

    def get_results(self, include_emg: bool = True):
        return UprightBalanceTestResults(
            self.processed_data,
            include_emg,
        )


class UprightBalanceTestResults(TestResults):

    def __init__(self, test: UprightBalanceTest, include_emg: bool):
        if not isinstance(test, UprightBalanceTest):
            raise ValueError("'test' must be an UprightBalanceTest instance.")
        super().__init__(test, include_emg)

    def _get_bodymass_kg(self, exe: TimeseriesRecord):
        """
        Returns the subject's body mass in kilograms.

        Returns
        -------
        float
            Body mass in kg.
        """
        return float(exe.resultant_force.force[exe.vertical_axis].to_numpy().mean() / G)

    def _get_force_symmetry(self, exe: TimeseriesRecord):
        """
        Returns coordination and balance metrics from force signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with force coordination and balance metrics,
            or empty if not available.
        """

        # get the forces from each foot and hand
        left_foot = exe.get("left_foot_ground_reaction_force")
        right_foot = exe.get("right_foot_ground_reaction_force")
        if (
            left_foot is None
            or right_foot is None
            or not isinstance(left_foot, ForcePlatform)
            or not isinstance(right_foot, ForcePlatform)
        ):
            return pd.DataFrame()
        vt = exe.vertical_axis
        left_vt = left_foot.copy().force[vt].to_numpy().flatten()  # type: ignore
        right_vt = right_foot.copy().force[vt].to_numpy().flatten()  # type: ignore

        # get the pairs to be tested
        pairs = {
            "lower_limbs": {"left_foot": left_vt, "right_foot": right_vt},
        }

        # calculate balance and coordination
        out = []
        unit = exe.resultant_force
        if unit is None:
            return pd.DataFrame()
        unit = unit.force.unit
        for region, pair in pairs.items():
            left, right = list(pair.values())
            fit = self._get_symmetry(left, right)
            line = {f"force_{i}": float(v) for i, v in fit.items()}  # type: ignore
            line = pd.DataFrame(pd.Series(line)).T
            line.insert(0, "region", region)
            out += [line]

        return pd.concat(out, ignore_index=True)

    def _get_area_of_stability_mm2(self, exe: TimeseriesRecord):
        x, y = self._get_cop_mm(exe).to_numpy().astype(float).T
        return Ellipse().fit(x, y).area

    def _get_cop_mm(self, exe: TimeseriesRecord):
        def extract_cop(force: ForcePlatform):
            cop = force.origin.copy() * 1000
            cop_x = cop.copy()[cop.lateral_axis]
            cop_x = cop_x.to_numpy().astype(float).flatten()
            cop_y = cop.copy()[cop.anteroposterior_axis]
            cop_y = cop_y.to_numpy().astype(float).flatten()
            return cop_x, cop_y

        grf = exe.resultant_force
        cop_x, cop_y = extract_cop(grf)
        lt_grf = exe.get("left_foot_ground_reaction_force")
        rt_grf = exe.get("right_foot_ground_reaction_force")
        if (
            lt_grf is not None
            and rt_grf is not None
            and isinstance(lt_grf, ForcePlatform)
            and isinstance(rt_grf, ForcePlatform)
        ):
            lt_x, lt_y = extract_cop(lt_grf)
            rt_x, rt_y = extract_cop(rt_grf)
            cop_x0 = np.mean((lt_x + rt_x) / 2)
            cop_y0 = np.mean((lt_y + rt_y) / 2)
        else:
            cop_x0 = np.mean(cop_x)
            cop_y0 = np.mean(cop_y)
        cop_x -= cop_x0
        cop_y -= cop_y0

        return pd.DataFrame(
            {"cop_x_mm": cop_x, "cop_y_mm": cop_y},
            index=grf.index,
        )

    def _get_summary(self, test: UprightBalanceTest):
        summary = {
            "type": test.name,
            "eyes": test.eyes,
            "side": test.side,
            "bodymass_kg": self._get_bodymass_kg(test.exercise),
            "area_of_stability_mm2": self._get_area_of_stability_mm2(test.exercise),
        }
        summary = [pd.DataFrame(pd.Series(summary)).T]
        summary.append(self._get_force_symmetry(test.exercise))
        if test.side == "bilateral" and self.include_emg:
            summary.append(self._get_muscle_symmetry(test.exercise))
        return pd.concat(summary, axis=1)

    def _get_analytics(self, test: UprightBalanceTest):
        out = self._get_cop_mm(test.exercise)
        if self.include_emg:
            emgs = test.exercise.emgsignals.to_dataframe()
            out = pd.concat([out, emgs], axis=1)
        return out.dropna()

    def _get_figures(self, test: UprightBalanceTest):

        # get the cop coordinates in mm
        cop_x, cop_y = self._get_cop_mm(test.exercise).to_numpy().T

        # get the emgsignals
        if self.include_emg and test.side == "bilateral":
            emgs = test.exercise.emgsignals
        else:
            emgs = TimeseriesRecord()

        # get the normative data
        norms = test.normative_data
        norms_idx = (norms.side == test.side) & (norms.eyes == test.eyes)
        norms = norms.loc[norms_idx]

        # generate the sway figure
        out: dict[str, go.Figure] = {}
        out["sway"] = _get_sway_figure(
            cop_x,
            cop_y,
            norms,
            emgs,
        )

        return out


class PlankBalanceTest(TestProtocol):

    @property
    def eyes(self):
        return self._eyes

    def set_eyes(self, eyes: Literal["open", "closed"]):
        if eyes not in ["open", "closed"]:
            raise ValueError("eyes must be 'open' or 'closed'.")
        self._eyes = eyes

    def __init__(
        self,
        participant: Participant,
        exercise: PronePosture,
        eyes: Literal["open", "closed"],
        normative_data: pd.DataFrame = plankbalance_normative_values,
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float | int = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        super().__init__(
            participant=participant,
            normative_data=normative_data,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            relevant_muscle_map=relevant_muscle_map,
        )
        self.set_eyes(eyes)
        self.set_exercise(exercise)

    def copy(self):
        return PlankBalanceTest(
            participant=self.participant,
            exercise=self.exercise,
            eyes=self.eyes,  # type: ignore
            normative_data=self.normative_data,
            emg_normalization_references=self.emg_normalization_references,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
            emg_normalization_function=self.emg_normalization_function,
            relevant_muscle_map=self.relevant_muscle_map,
        )

    def set_exercise(self, exercise: PronePosture):
        if not isinstance(exercise, PronePosture):
            raise ValueError("exercise must be a PronePosture instance.")
        self._exercise = exercise

    @property
    def exercise(self):
        return self._exercise

    @classmethod
    def from_files(
        cls,
        filename: str,
        participant: Participant,
        eyes: Literal["open", "closed"],
        left_foot_ground_reaction_force: str = "left_foot",
        right_foot_ground_reaction_force: str = "right_foot",
        left_hand_ground_reaction_force: str = "left_hand",
        right_hand_ground_reaction_force: str = "right_hand",
        normative_data: pd.DataFrame = uprightbalance_normative_values,
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float | int = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        return cls(
            participant=participant,
            normative_data=normative_data,
            eyes=eyes,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            relevant_muscle_map=relevant_muscle_map,
            exercise=PronePosture.from_tdf(
                file=filename,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                left_hand_ground_reaction_force=left_hand_ground_reaction_force,
                right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            ),
        )

    @property
    def processed_data(self):

        # apply the pipeline to the test data
        exe = self.processing_pipeline(self.exercise, inplace=False)
        if not isinstance(exe, PronePosture):
            raise ValueError("Something went wrong during data processing.")

        # normalize emg data and remove non-relevant muscles
        norms = self.emg_normalization_values
        to_remove: list[str] = []
        for k, m in exe.emgsignals.items():

            # remove if non relevant
            if self.relevant_muscle_map is not None:
                if not any([i.lower() in k.lower() for i in self.relevant_muscle_map]):
                    to_remove.append(k)
                    continue

            # normalize
            if isinstance(m, EMGSignal):
                for (name, side), val in norms.items():
                    if m.muscle_name == name and m.side == side:
                        exe[k] = m / val * 100
                        exe[k].set_unit("%")  # type: ignore
                        break
        if len(to_remove) > 0:
            exe.drop(to_remove, True)

        # align the reference frame
        def extract_cop(force: Any):
            if not isinstance(force, ForcePlatform):
                raise ValueError("force must be a ForcePlatform instance.")
            cop = force.origin
            if not isinstance(cop, Point3D):
                raise ValueError("force must be a ForcePlatform instance.")
            cop = cop.copy()
            return cop.to_numpy().astype(float).mean(axis=0)

        # on bilateral test, we rotate the system of forces to a
        rf = extract_cop(exe.right_foot_ground_reaction_force)
        lf = extract_cop(exe.left_foot_ground_reaction_force)
        rh = extract_cop(exe.right_hand_ground_reaction_force)
        lh = extract_cop(exe.left_hand_ground_reaction_force)

        def norm(arr):
            return arr / np.sum(arr**2) ** 0.5

        ml = norm((lf + lh) / 2 - (rf + rh) / 2)
        vt = np.array([0, 1, 0])
        ap = np.cross(ml, vt)
        origin = (rf + lf + rh + lh) / 4
        exe = exe.change_reference_frame(
            ml,
            vt,
            ap,
            origin,
            inplace=False,
        )
        if exe is None:
            raise ValueError("reference frame alignment returned None")

        # return processed data
        out = self.copy()
        out.set_exercise(exe)  # type: ignore
        return out

    @property
    def processing_pipeline(self):
        return get_default_processing_pipeline()

    def get_results(self, include_emg: bool = True):
        return PlankBalanceTestResults(
            self.processed_data,
            include_emg,
        )


class PlankBalanceTestResults(TestResults):

    def __init__(self, test: PlankBalanceTest, include_emg: bool):
        if not isinstance(test, PlankBalanceTest):
            raise ValueError("'test' must be a PlankBalanceTest instance.")
        super().__init__(test, include_emg)

    def _get_force_symmetry(self, exe: TimeseriesRecord):
        """
        Returns coordination and balance metrics from force signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with force coordination and balance metrics,
            or empty if not available.
        """

        # get the forces from each foot and hand
        left_foot = exe.get("left_foot_ground_reaction_force")
        right_foot = exe.get("right_foot_ground_reaction_force")
        left_hand = exe.get("left_hand_ground_reaction_force")
        right_hand = exe.get("right_hand_ground_reaction_force")
        if (
            left_foot is None
            or right_foot is None
            or left_hand is None
            or right_hand is None
        ):
            return pd.DataFrame()
        vt = exe.vertical_axis
        left_foot = left_foot.copy()["force"][vt].to_numpy()
        right_foot = right_foot.copy()["force"][vt].to_numpy()
        left_hand = left_hand.copy()["force"][vt].to_numpy()
        right_hand = right_hand.copy()["force"][vt].to_numpy()

        # get the pairs to be tested
        pairs = {
            "upper/lower": {
                "upper": left_hand + right_hand,
                "lower": right_foot + left_foot,
            },
            "left/right": {
                "left": left_hand + left_foot,
                "right": right_hand + right_foot,
            },
        }

        # calculate balance and coordination
        out = []
        unit = exe.resultant_force
        if unit is None:
            return pd.DataFrame()
        unit = unit.force.unit
        for region, pair in pairs.items():
            left, right = list(pair.values())
            fit = self._get_symmetry(left, right)
            line = {f"force_{i}": float(v.values[0]) for i, v in fit.items()}  # type: ignore
            line = pd.DataFrame(pd.Series(line)).T
            line.insert(0, "region", region)
            out += [line]

        return pd.concat(out, ignore_index=True)

    def _get_bodymass_kg(self, exe: TimeseriesRecord):
        """
        Returns the subject's body mass in kilograms.

        Returns
        -------
        float
            Body mass in kg.
        """
        return float(exe.resultant_force.force[exe.vertical_axis].to_numpy().mean() / G)

    def _get_area_of_stability_mm2(self, exe: TimeseriesRecord):
        x, y = self._get_cop_mm(exe).to_numpy().astype(float).T
        return Ellipse().fit(x, y).area

    def _get_cop_mm(self, exe: TimeseriesRecord):
        def extract_cop(force: ForcePlatform):
            cop = force.origin.copy() * 1000
            cop_x = cop.copy()[:, cop.lateral_axis].to_numpy().astype(float).flatten()  # type: ignore
            cop_y = cop.copy()[:, cop.anteroposterior_axis].to_numpy().astype(float).flatten()  # type: ignore
            return cop_x, cop_y

        grf = exe.resultant_force
        cop_x, cop_y = extract_cop(grf)
        lf_x, lf_y = extract_cop(exe.left_foot_ground_reaction_force)  # type: ignore
        rf_x, rf_y = extract_cop(exe.right_foot_ground_reaction_force)  # type: ignore
        lh_x, lh_y = extract_cop(exe.left_hand_ground_reaction_force)  # type: ignore
        rh_x, rh_y = extract_cop(exe.right_hand_ground_reaction_force)  # type: ignore
        cop_x0 = np.mean((lf_x + rf_x + lh_x + rh_x) / 4)
        cop_y0 = np.mean(cop_y)
        cop_x -= cop_x0
        cop_y -= cop_y0

        return pd.DataFrame(
            {"cop_x_mm": cop_x, "cop_y_mm": cop_y},
            index=grf.index,
        )

    def _get_summary(self, test: PlankBalanceTest):
        summary = {
            "type": test.__class__.__name__,
            "eyes": test.eyes,
            "bodymass_kg": self._get_bodymass_kg(test.exercise),
            "area_of_stability_mm2": self._get_area_of_stability_mm2(test.exercise),
        }
        summary = [pd.DataFrame(pd.Series(summary)).T]
        summary.append(self._get_force_symmetry(test.exercise))
        if self.include_emg:
            summary.append(self._get_muscle_symmetry(test.exercise))
        return pd.concat(summary, axis=1)

    def _get_analytics(self, test: PlankBalanceTest):
        out = self._get_cop_mm(test.exercise)
        if self.include_emg:
            emgs = test.exercise.emgsignals.to_dataframe()
            out = pd.concat([out, emgs], axis=1)
        return out.dropna()

    def _get_figures(self, test: PlankBalanceTest):

        # get the cop coordinates in mm
        cop_x, cop_y = self._get_cop_mm(test.exercise).to_numpy().T

        # get the emgsignals
        emgs = test.exercise.emgsignals if self.include_emg else TimeseriesRecord()

        # get the normative data
        norms = test.normative_data
        norms_idx = norms.eyes == test.eyes
        norms = norms.loc[norms_idx]

        # generate the sway figure
        out: dict[str, go.Figure] = {}
        out["sway"] = _get_sway_figure(
            cop_x,
            cop_y,
            norms,
            emgs,
        )

        return out
