"""Upright balance test results implementation."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...modelling.ols.geometry import Ellipse

from ...constants import G
from ...records import ForcePlatform, TimeseriesRecord
from ..test_results import TestResults
from ._plotting import _get_sway_figure

if TYPE_CHECKING:
    from .upright_balance_test import UprightBalanceTest


class UprightBalanceTestResults(TestResults):

    def __init__(self, test: "UprightBalanceTest", include_emg: bool):
        from .upright_balance_test import UprightBalanceTest

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
            line = {f"force_{i}": float(v.iloc[0]) for i, v in fit.items()}  # type: ignore
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

    def _get_summary(self, test: "UprightBalanceTest"):
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

    def _get_analytics(self, test: "UprightBalanceTest"):
        out = self._get_cop_mm(test.exercise)
        if self.include_emg:
            emgs = test.exercise.emgsignals.to_dataframe()
            out = pd.concat([out, emgs], axis=1)
        return out.dropna()

    def _get_figures(self, test: "UprightBalanceTest"):

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


__all__ = ["UprightBalanceTestResults"]
