"""Isokinetic 1RM test results implementation."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...constants import G
from ...exercises.strength import IsokineticExercise
from ...signalprocessing import cubicspline_interp
from ..test_results import TestResults
from ._plotting import _get_force_figure

if TYPE_CHECKING:
    from .isokinetic_1rm_test import Isokinetic1RMTest


class Isokinetic1RMTestResults(TestResults):
    """
    Results container for isokinetic 1RM prediction test analysis.

    Isokinetic1RMTestResults processes Isokinetic1RMTest data to predict
    maximal strength (1RM) from constant-velocity contractions. The class
    generates comprehensive performance summaries including peak force,
    predicted 1RM, force balance analysis, and muscle activation patterns.

    Parameters
    ----------
    test : Isokinetic1RMTest
        Processed isokinetic 1RM test data to analyze.
    include_emg : bool, optional
        Whether to include EMG analysis in results (mean amplitude per muscle).
        Default is True.
    estimate_1rm : bool, optional
        Whether to compute predicted 1RM from load-velocity relationship.
        Default is False.
    include_force_balance : bool, optional
        Whether to include force balance metrics (push/pull phase analysis).
        Default is True.

    Attributes
    ----------
    summary : pd.DataFrame
        Comprehensive table of isokinetic metrics including:
        - Peak force (N)
        - Predicted 1RM (kg) if estimate_1rm is True
        - Force balance metrics if include_force_balance is True
        - EMG mean amplitude (% or µV) per muscle if include_emg is True
        - Left/right symmetry (%)
    analytics : pd.DataFrame
        Time-series data for all trials in long format.
    figures : dict of str -> go.Figure
        Dictionary of interactive Plotly figures:
        - 'force_traces': Force-time curves for all trials
    estimate_1rm : bool
        Whether 1RM prediction is enabled.
    include_emg : bool
        Whether EMG analysis is enabled.
    include_force_balance : bool
        Whether force balance analysis is enabled.

    Methods
    -------
    set_estimate_1rm(estimate)
        Enable or disable 1RM prediction.
    set_include_emg(estimate)
        Enable or disable EMG analysis.
    set_include_force_balance(estimate)
        Enable or disable force balance analysis.

    Notes
    -----
    1RM Prediction:
    Uses linear regression from load-velocity relationship:
        predicted_1RM = (peak_force / g) * beta1 + beta0
    where beta0 and beta1 are exercise-specific coefficients from test.rm1_coefs.

    Force Balance:
    Analyzes symmetry between push and pull phases in bilateral exercises,
    or between concentric and eccentric phases in unilateral exercises.

    EMG Processing:
    - Mean amplitude computed over entire contraction phase
    - Values expressed as % of reference if normalization applied
    - Values in µV if no normalization reference provided

    Examples
    --------
    >>> from labanalysis.protocols import Isokinetic1RMTest, Participant
    >>>
    >>> # Create test from BioStrength files
    >>> participant = Participant(surname='Athlete', weight=75)
    >>> test = Isokinetic1RMTest.from_files(
    ...     participant=participant,
    ...     product='LEG PRESS',
    ...     bilateral_biostrength_filename='leg_press.txt'
    ... )
    >>>
    >>> # Get results with 1RM prediction
    >>> results = test.get_results(include_emg=True)
    >>> results.set_estimate_1rm(True)
    >>>
    >>> # View summary with predicted 1RM
    >>> print(results.summary)
    >>> print(f"Predicted 1RM: {results.summary['predicted_1rm_kg'].iloc[0]:.1f} kg")
    >>>
    >>> # Display force curves
    >>> results.figures['force_traces'].show()

    See Also
    --------
    Isokinetic1RMTest : Test protocol for isokinetic 1RM prediction.
    TestResults : Parent class for test results.
    """

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

        # Find actual force and position column names
        available_cols = tracks.columns.tolist()
        force_col = None
        position_col = None
        for col in available_cols:
            if 'force' in col.lower():
                force_col = col
            if 'position' in col.lower():
                position_col = col

        if force_col is None or position_col is None:
            return {}

        if self.include_emg:
            cols = tracks.columns
        else:
            cols = [
                "side",
                "repetition",
                "time_s",
                force_col,
                position_col,
            ]
        force_data = tracks[cols].copy()
        dfs = []
        for (side, rep), dfr in force_data.groupby(by=["side", "repetition"]):
            df = dfr.drop(
                ["side", "repetition", "time_s", position_col],
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
            # Map column names to standard names for plotting
            df['parameter'] = df['parameter'].replace({
                force_col: 'force_amplitude'
            })
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
