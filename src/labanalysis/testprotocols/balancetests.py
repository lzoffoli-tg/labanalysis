"""balance test module"""

__all__ = ["UprightBalanceTest", "PlankBalanceTest"]


from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..constants import G
from ..modelling import Ellipse
from ..records.bodies import WholeBody
from ..records.pipelines import ProcessingPipeline
from ..records.records import ForcePlatform
from ..records.timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from .normativedata import (
    plankbalance_normative_values,
    uprightbalance_normative_values,
)
from .protocols import Participant, TestProtocol, TestResults


class UprightBalanceTest(WholeBody, TestProtocol):

    _eyes: Literal["open", "closed"]

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
        left_foot = self.get("left_foot_ground_reaction_force")
        right_foot = self.get("right_foot_ground_reaction_force")
        if left_foot is not None and right_foot is not None:
            return "bilateral"
        if left_foot is not None:
            return "left"
        if right_foot is not None:
            return "right"
        raise ValueError("both left_foot and right_foot are None")

    def set_eyes(self, eyes: Literal["open", "closed"]):
        self._eyes = eyes

    def __init__(
        self,
        participant: Participant,
        eyes: Literal["open", "closed"],
        normative_data: pd.DataFrame = uprightbalance_normative_values,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):

        super().__init__(
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            **signals,  # type: ignore
        )
        self.set_participant(participant)
        self.set_normative_data(normative_data)
        self.set_eyes(eyes)

    def copy(self):
        return UprightBalanceTest(
            participant=self.participant.copy(),
            eyes=self.eyes,
            normative_data=self.normative_data,
            **{i: v.copy() for i, v in self.items()},  # type: ignore
        )

    @property
    def results(self):
        return UprightBalanceTestResults(self)

    @classmethod
    def from_tdf(
        cls,
        filename: str,
        participant: Participant,
        eyes: Literal["open", "closed"],
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        normative_data: pd.DataFrame = uprightbalance_normative_values,
    ):
        tdf = WholeBody.from_tdf(
            filename=filename,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
        )
        mandatory = [
            "left_foot_ground_reaction_force",
            "right_foot_ground_reaction_force",
        ]
        mandatory = {i: tdf.get(i) for i in mandatory}
        signals = {i: v for i, v in tdf.items() if i not in mandatory}
        return cls(
            participant=participant,
            normative_data=normative_data,
            eyes=eyes,
            **mandatory,  # type: ignore
            **signals,  # type: ignore
        )


class UprightBalanceTestResults(TestResults):

    def __init__(self, test: UprightBalanceTest):
        if not isinstance(test, UprightBalanceTest):
            raise ValueError("'test' must be an UprightBalanceTest instance.")
        super().__init__(test)

    def _get_bodymass_kg(self, test: UprightBalanceTest):
        """
        Returns the subject's body mass in kilograms.

        Returns
        -------
        float
            Body mass in kg.
        """
        grf = test.resultant_force
        if grf is None:
            return np.nan
        return float(grf.force[test.vertical_axis].to_numpy().mean() / G)

    def _get_force_symmetry(self, test: UprightBalanceTest):
        """
        Returns coordination and balance metrics from force signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with force coordination and balance metrics, or empty if not available.
        """

        # get the forces from each foot and hand
        left_foot = test.get("left_foot_ground_reaction_force")
        right_foot = test.get("right_foot_ground_reaction_force")
        if (
            left_foot is None
            or right_foot is None
            or not isinstance(left_foot, ForcePlatform)
            or not isinstance(right_foot, ForcePlatform)
        ):
            return pd.DataFrame()
        left_vt = left_foot.copy().force[test.vertical_axis].to_numpy().flatten()  # type: ignore
        right_vt = right_foot.copy().force[test.vertical_axis].to_numpy().flatten()  # type: ignore

        # get the pairs to be tested
        pairs = {
            "lower_limbs": {"left_foot": left_vt, "right_foot": right_vt},
        }

        # calculate balance and coordination
        out = []
        unit = test.resultant_force
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

    def _get_area_of_stability_mm2(self, test: UprightBalanceTest):
        cop = test.resultant_force
        if cop is None:
            return np.nan
        cop = cop.origin.to_dataframe().dropna()
        horizontal_axes = [test.anteroposterior_axis, test.lateral_axis]
        ap, ml = cop[horizontal_axes].values.astype(float).T * 1000
        ellipse = Ellipse().fit(ap.astype(float), ml.astype(float))
        return ellipse.area

    def _align_referenceframe_func(self, obj: UprightBalanceTest):

        # in case of single-leg tests, no rotation is required
        if obj.side in ["right", "left"]:
            return obj

        # on bilateral test, we rotate the system of forces to a
        rt = obj.right_foot_ground_reaction_force.origin.to_numpy().mean(axis=0)  # type: ignore
        lt = obj.left_foot_ground_reaction_force.origin.to_numpy().mean(axis=0)  # type: ignore

        def norm(arr):
            return arr / np.sum(arr**2) ** 0.5

        ml = norm(lt - rt)
        vt = np.array([0, 1, 0])
        ap = np.cross(ml, vt)
        origin = (rt + vt) / 2
        out = obj.change_reference_frame(
            ml,
            vt,
            ap,
            origin,
            inplace=False,
        )
        if out is None:
            raise ValueError("reference frame alignment returned None")

        return out

    def _get_sway_figure(self, test: UprightBalanceTest):

        # fit an ellipse on the cop
        cop = test.resultant_force.origin.copy() * 1000
        cop_x = cop[test.lateral_axis].values.astype(float).flatten()  # type: ignore
        cop_y = cop[test.anteroposterior_axis].values.astype(float).flatten()  # type: ignore
        ellipse = Ellipse().fit(cop_x, cop_y)

        # get the ellipse properties
        cop_x0, cop_y0 = ellipse.center
        semiaxis_a, semiaxis_b = ellipse.semi_axes
        cop_angle_rad = ellipse.rotation_angle / 180 * np.pi
        cop_area = ellipse.area

        # collect emg signals
        emg_signals = {}
        for chn in test.emgsignals.values():
            if chn.muscle_name not in emg_signals:
                emg_signals[chn.muscle_name] = {}
            emg_signals[chn.muscle_name][chn.side] = chn.to_dataframe()

        # generate the figure and setup the layout
        if len(emg_signals) == 0:
            fig = make_subplots(
                rows=3,
                cols=1,
                specs=[[{"rowspan": 2}], [None], [{}]],
            )
            fig.update_layout(title="Sway")
        else:
            specs = [[{"rowspan": len(emg_signals) - 1, "colspan": 2}, None, {}]]
            specs += [[None, None, {}] for _ in range(len(emg_signals) - 2)]
            specs += [[{"colspan": 2}, None, {}]]
            fig = make_subplots(
                rows=len(emg_signals),
                cols=3,
                subplot_titles=["Sway"] + list(emg_signals.keys()),
                shared_yaxes=False,
                shared_xaxes=False,
                specs=specs,
                row_titles=list(emg_signals.keys()),
            )
        fig.update_xaxes(
            row=1,
            col=1,
            title="Left | Right<br>(mm)",
            scaleanchor="x",
            scaleratio=1,
        )
        fig.update_yaxes(
            row=1,
            col=1,
            title="Backward | Forward<br>(mm)",
            scaleanchor="y",
            scaleratio=1,
        )
        fig.add_hline(
            y=0,
            line_color="black",
            line_dash="dash",
            opacity=1,
            line_width=2,
            showlegend=False,
            row=1,  # type: ignore
            col=1,  # type: ignore
        )
        fig.add_vline(
            0,
            line_color="black",
            line_dash="dash",
            opacity=1,
            line_width=2,
            showlegend=False,
            row=1,  # type: ignore
            col=1,  # type: ignore
        )

        # get normative data
        norms = test.normative_data
        norms_idx = norms.parameter == "area_of_stability_mm2"
        if "side" in norms.columns:
            norms_idx &= norms.side == test.side
        norms_idx &= norms.eyes == test.eyes
        norms = norms.loc[norms_idx, ["mean", "std"]]
        avg, std = norms.values.astype(float).flatten()
        areas = np.array([avg, avg + 1 * std, avg + 2 * std])
        colors = ["#00D338", "#CAFF4D", "#EEFF00", "#D01C00"]
        ranks = ["good", "normal", "fair"]

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
            p1 = ((x - x0) * np.cos(t) + (y - y0 * np.sin(t)) ** 2) / a**2
            p2 = ((-(x - x0) * np.sin(t) + (y - y0) * np.cos(t)) ** 2) / b**2
            return np.asarray((p1 + p2) <= 1, dtype=bool)

        samples_within = {}
        for area, color, label in zip(areas[::-1], colors[::-1], ranks[::-1]):

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
        for i in range(1, len(ranks)):
            rank = ranks[i]
            samples_within[rank] -= samples_within[ranks[i - 1]]
        samples_within["poor"] = 100 - sum(samples_within.values())

        # add the sway
        fig.add_trace(
            trace=go.Scatter(
                x=cop_x.tolist(),
                y=cop_y.tolist(),
                mode="lines",
                opacity=0.5,
                line_width=2,
                line_color="#000000",
                name="centro di pressione",
            ),
            row=1,
            col=1,
        )

        # plot the cumulative time spent at each level of norm
        for i, (rank, value) in enumerate(samples_within.items()):
            fig.add_trace(
                trace=go.Bar(
                    x=value,
                    y=0,
                    text=f"{value:0.1f}%",
                    marker_color=colors[i],
                    name=rank,
                    orientation="h",
                ),
                row=3,
                col=1,
            )

        # plot the emg signals
        sides = ["left", "right"]
        colors = ["#00A9D3", "#FF5F4D"]
        for i, (muscle, dct) in enumerate(emg_signals.items()):
            symm = self._get_symmetry(dct["left"], dct["right"])
            values = symm.values.astype(float).flatten()
            for side, color, value in zip(sides, colors, values):
                fig.add_trace(
                    trace=go.Bar(
                        x=side,
                        y=value,
                        marker_color=color,
                        text=f"{value:0.1f}%",
                        name=side,
                        legendgroup=side,
                        showlegend=bool(i == 0),
                    ),
                    row=i + 1,
                    col=1,
                )

        return fig

    def _get_figures(self, test: UprightBalanceTest):

        return {"sway": self._get_sway_figure(test)}

    def _get_summary(self, test: UprightBalanceTest):
        summary = {
            "type": test.__class__.__name__,
            "eyes": test.eyes,
            "side": test.side,
            "bodymass_kg": self._get_bodymass_kg(test),
            "area_of_stability_mm2": self._get_area_of_stability_mm2(test),
        }
        summary = pd.DataFrame(pd.Series(summary)).T
        summary = [
            summary,
            self._get_force_symmetry(test),
            self._get_muscle_symmetry(test),
        ]
        return pd.concat(summary, axis=1)

    def _get_analytics(self, test: UprightBalanceTest):
        cop = test.resultant_force.copy()["origin"].to_dataframe().dropna()
        emgs = test.emgsignals.to_dataframe().dropna()
        return pd.concat([cop, emgs], axis=1)

    def _get_processed_data(self, test: UprightBalanceTest):
        processing_pipeline = ProcessingPipeline(
            {
                "ForcePlatform": [self._forceplatforms_processing_func],
                "EMGSignal": [self._emgsignals_processing_func],
            }
        )
        out = processing_pipeline(test, inplace=False)
        if out is None:
            raise ValueError("Something went wrong during data processing.")
        out = self._align_referenceframe_func(out)  # type: ignore
        if out is None:
            raise ValueError("Something went wrong during data processing.")

        return out


class PlankBalanceTest(UprightBalanceTest):

    def __init__(
        self,
        participant: Participant,
        left_foot_ground_reaction_force: ForcePlatform,
        right_foot_ground_reaction_force: ForcePlatform,
        left_hand_ground_reaction_force: ForcePlatform,
        right_hand_ground_reaction_force: ForcePlatform,
        eyes: Literal["open", "closed"],
        normative_data: pd.DataFrame = plankbalance_normative_values,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):

        super().__init__(
            participant=participant,
            normative_data=normative_data,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            eyes=eyes,
            **signals,  # type: ignore
        )

    def copy(self):
        return PlankBalanceTest(
            participant=self.participant.copy(),
            eyes=self.eyes,
            normative_data=self.normative_data,
            **{i: v.copy() for i, v in self.items()},  # type: ignore
        )

    @classmethod
    def from_tdf(
        cls,
        filename: str,
        participant: Participant,
        eyes: Literal["open", "closed"],
        left_foot_ground_reaction_force: str,
        right_foot_ground_reaction_force: str,
        left_hand_ground_reaction_force: str,
        right_hand_ground_reaction_force: str,
        normative_data: pd.DataFrame = plankbalance_normative_values,
    ):
        tdf = WholeBody.from_tdf(
            filename=filename,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
        )
        mandatory = [
            "left_foot_ground_reaction_force",
            "right_foot_ground_reaction_force",
            "left_hand_ground_reaction_force",
            "right_hand_ground_reaction_force",
        ]
        mandatory_dict = {i: tdf.get(i) for i in mandatory}
        if any([i is None for i in list(mandatory_dict.values())]):
            raise ValueError(
                "all the following ForcePlatform elements must be included "
                + "into the provided file"
            )
        signals = {i: v for i, v in tdf.items() if i not in mandatory}
        return cls(
            participant=participant,
            normative_data=normative_data,
            eyes=eyes,
            **mandatory_dict,  # type: ignore
            **signals,  # type: ignore
        )

    @property
    def results(self):
        return PlankBalanceTestResults(self)


class PlankBalanceTestResults(UprightBalanceTestResults):

    def __init__(self, test: PlankBalanceTest):
        if not isinstance(test, PlankBalanceTest):
            raise ValueError("'test' must be a PlankBalanceTest instance.")
        super().__init__(test)

    def _get_force_symmetry(self, test):
        """
        Returns coordination and balance metrics from force signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with force coordination and balance metrics,
            or empty if not available.
        """

        # get the forces from each foot and hand
        left_foot = test.get("left_foot_ground_reaction_force")
        right_foot = test.get("right_foot_ground_reaction_force")
        left_hand = test.get("left_hand_ground_reaction_force")
        right_hand = test.get("right_hand_ground_reaction_force")
        if (
            left_foot is None
            or right_foot is None
            or left_hand is None
            or right_hand is None
        ):
            return pd.DataFrame()
        left_foot = left_foot.copy()["force"][test.vertical_axis].to_numpy()
        right_foot = right_foot.copy()["force"][test.vertical_axis].to_numpy()
        left_hand = left_hand.copy()["force"][test.vertical_axis].to_numpy()
        right_hand = right_hand.copy()["force"][test.vertical_axis].to_numpy()

        # get the pairs to be tested
        pairs = {
            "lower_limbs": {"left_foot": left_foot, "right_foot": right_foot},
            "upper_limbs": {"left_hand": left_hand, "right_hand": right_hand},
            "whole_body": {
                "left": left_foot + left_hand,
                "right": right_foot + right_hand,
            },
        }

        # calculate balance and coordination
        out = []
        unit = test.resultant_force
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
