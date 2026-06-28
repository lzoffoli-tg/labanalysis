"""Isokinetic 1RM test results implementation."""

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from ...records.strength import IsokineticExercise
from ..test_results import TestResults
from ._plotting import _get_force_figure

if TYPE_CHECKING:
    from .isokinetic_1rm_test import Isokinetic1RMTest


class Isokinetic1RMTestResults(TestResults):

    @property
    def estimate_1rm(self):
        return self._estimate_1rm

    def set_estimate_1rm(self, estimate: bool):
        if not isinstance(estimate, bool):
            raise ValueError("estimate must be a bool instance.")
        self._estimate_1rm = estimate

    @property
    def include_emg(self):
        return self._include_emg

    def set_include_emg(self, estimate: bool):
        if not isinstance(estimate, bool):
            raise ValueError("estimate must be a bool instance.")
        self._include_emg = estimate

    @property
    def include_force_balance(self):
        return self._include_force_balance

    def set_include_force_balance(self, estimate: bool):
        if not isinstance(estimate, bool):
            raise ValueError("estimate must be a bool instance.")
        self._include_force_balance = estimate

    def __init__(
        self,
        test: "Isokinetic1RMTest",
        include_emg: bool = True,
        estimate_1rm: bool = False,
        include_force_balance: bool = True,
    ):
        from .isokinetic_1rm_test import Isokinetic1RMTest
        self._summary = pd.DataFrame()
        self._analytics = pd.DataFrame()
        self._figures = {}
        self.set_include_emg(include_emg)
        self.set_include_force_balance(include_force_balance)
        self.set_estimate_1rm(estimate_1rm)
        self._generate_results(test)

    def _get_estimated_1rm(
        self,
        test: IsokineticExercise,
        coefs: dict[str, float],
    ):
        return self._get_peak_force(test) / G * coefs["beta1"] + coefs["beta0"]

    def _get_peak_force(self, test: IsokineticExercise):
        return float(test.force.to_numpy().max())

    def _get_summary(self, test: "Isokinetic1RMTest"):
        trials = [test.left, test.right, test.bilateral]
        sides = ["left", "right", "bilateral"]
        metrics = pd.DataFrame()
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            emg_norms = test.emg_normalization_values
            for rep in trial.repetitions:
                new = pd.DataFrame()
                new.loc["rom (m)", side] = rep.rom_m
                if self.include_emg:
                    for m in rep.emgsignals.values():
                        ename = str(m.muscle_name)
                        eside = str(m.side)
                        keys = emg_norms.keys()
                        check = [i[0] == ename and i[1] == eside for i in keys]
                        ename += " (%)" if any(check) else " (uV)"
                        new.loc[ename, eside] = m.to_numpy().mean()
                metrics = pd.concat([metrics, new])
            if self.estimate_1rm:
                metrics.loc["estimated 1RM (kg)", side] = self._get_estimated_1rm(
                    trial,
                    test.rm1_coefs,
                )
            metrics.loc["peak force (N)", side] = self._get_peak_force(trial)
        metrics.insert(0, "parameter", metrics.index)
        metrics.reset_index(inplace=True, drop=True)
        summary = metrics.groupby("parameter", as_index=False).mean()

        # add left/right symmetries
        for row in range(summary.shape[0]):
            line = summary.iloc[row]
            if "left" in line.index and "right" in line.index:
                lt, rt = line[["left", "right"]].to_numpy().flatten()
                lt = float(lt)
                rt = float(rt)
                symm = 100 * (rt - lt) / ((rt + lt))
                summary.loc[summary.index[row], "symmetry (%)"] = symm

        return summary

    def _get_analytics(self, test: "Isokinetic1RMTest"):
        analytics = []
        trials = [test.left, test.right, test.bilateral]
        sides = ["left", "right", "bilateral"]
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            for i, rep in enumerate(trial.repetitions):
                cycle = rep.copy()
                if not self.include_emg:
                    cycle.drop(cycle.emgsignals.keys(), inplace=True)
                cycle = cycle.to_dataframe()
                time = cycle.index - cycle.index.min()
                cycle.insert(0, "time_s", time)
                cycle.insert(0, "repetition", i + 1)
                cycle.insert(0, "side", side)
                analytics.append(cycle)
        return pd.concat(analytics, ignore_index=True)

    def _get_figures(self, test: "Isokinetic1RMTest"):

        # force data
        tracks = self.analytics
        if self.include_emg:
            cols = tracks.columns
        else:
            cols = [
                "side",
                "repetition",
                "time_s",
                "force_amplitude",
                "position_amplitude",
            ]
        force_data = tracks[cols].copy()
        dfs = []
        for (side, rep), dfr in force_data.groupby(by=["side", "repetition"]):
            df = dfr.drop(
                ["side", "repetition", "time_s", "position_amplitude"],
                axis=1,
            ).copy()
            arr = df.to_numpy()
            interp = np.apply_along_axis(
                cubicspline_interp,
                arr=arr,
                axis=0,
                nsamp=201,
            )
            df = pd.DataFrame(dict(zip(df.columns, interp.T)))
            df.insert(
                0,
                "time_%",
                np.linspace(0, 100, df.shape[0]),
            )
            df = df.melt(
                id_vars=["time_%"],
                var_name="parameter",
                value_name="value",
            )
            df.insert(
                0,
                "side",
                side,
            )
            df.insert(
                0,
                "limb",
                "bilateral" if side == "bilateral" else "unilateral",
            )
            df.insert(
                0,
                "repetition",
                rep,
            )
            dfs.append(df)
        dfr = pd.concat(dfs, ignore_index=True)

        out: dict[str, go.Figure] = {}
        for i, v in dfr.groupby("limb"):
            out[i] = _get_force_figure(
                v,
                self.summary,
                self.include_emg,
            )

        return out


__all__ = ["Isokinetic1RMTestResults"]
