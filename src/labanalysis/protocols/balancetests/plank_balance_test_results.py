"""Plank balance test results implementation."""

from ...records import TimeseriesRecord
from ...records import ForcePlatform
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from ..test_results import TestResults
from ._plotting import _get_sway_figure

if TYPE_CHECKING:
    from .plank_balance_test import PlankBalanceTest


class PlankBalanceTestResults(TestResults):

    def __init__(self, test: "PlankBalanceTest", include_emg: bool):
        from .plank_balance_test import PlankBalanceTest
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

    def _get_summary(self, test: "PlankBalanceTest"):
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

    def _get_analytics(self, test: "PlankBalanceTest"):
        out = self._get_cop_mm(test.exercise)
        if self.include_emg:
            emgs = test.exercise.emgsignals.to_dataframe()
            out = pd.concat([out, emgs], axis=1)
        return out.dropna()

    def _get_figures(self, test: "PlankBalanceTest"):

        # get the cop coordinates in mm
        cop_x, cop_y = self._get_cop_mm(test.exercise).to_numpy().T

        # get the emgsignals
        emgs = test.exercise.emgsignals if self.include_emg else TimeseriesRecord()

        # get the normative data
        norms = test.normative_data
        norms_idx = (norms.side == "bilateral") & (norms.eyes == test.eyes)
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


__all__ = ["PlankBalanceTestResults"]
