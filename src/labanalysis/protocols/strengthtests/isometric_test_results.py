"""Isometric test results implementation."""

from ...exercises.strength import IsometricExercise
from typing import TYPE_CHECKING

import pandas as pd

from ..test_results import TestResults
from ._plotting import _get_force_figure

if TYPE_CHECKING:
    from .isometric_test import IsometricTest


class IsometricTestResults(TestResults):
    """
    Results container for isometric strength test analysis.

    IsometricTestResults processes IsometricTest data to generate comprehensive
    performance summaries including peak force, rate of force development (RFD),
    time to peak force, and muscle activation patterns. The class provides
    automated reporting with EMG analysis and bilateral symmetry calculations.

    Parameters
    ----------
    test : IsometricTest
        Processed isometric test data to analyze.
    include_emg : bool
        Whether to include EMG analysis in results (mean amplitude per muscle).

    Attributes
    ----------
    summary : pd.DataFrame
        Comprehensive table of isometric metrics including:
        - Peak force (N)
        - Rate of force development (kN/s)
        - Time to peak force (ms)
        - EMG mean amplitude (% or µV) per muscle
        - Left/right symmetry (%)
    analytics : pd.DataFrame
        Time-series data for all trials in long format.
    figures : dict of str -> go.Figure
        Dictionary of interactive Plotly figures:
        - 'force_traces': Force-time curves for all trials

    Notes
    -----
    Metric Calculations:
    - Peak Force: Maximum force value during MVIC
    - RFD: (Peak Force - Baseline) / (Time to Peak - Baseline Time), in kN/s
    - Time to Peak: Time from contraction onset to peak force, in ms
    - Symmetry: 100 * (Right - Left) / mean(Right, Left)

    EMG Processing:
    - Mean amplitude computed over entire contraction phase
    - Values expressed as % of reference if normalization applied
    - Values in µV if no normalization reference provided

    The class automatically identifies valid repetitions and computes metrics
    for left, right, and bilateral trials when available.

    Examples
    --------
    >>> from labanalysis.protocols import IsometricTest, Participant
    >>> from labanalysis.exercises.strength import IsometricExercise
    >>>
    >>> # Create and process test
    >>> participant = Participant(surname='Athlete', weight=75)
    >>> left_ex = IsometricExercise.from_biostrength("left.txt")
    >>> test = IsometricTest(left=left_ex, right=None, bilateral=None,
    ...                       participant=participant)
    >>> results = test.get_results(include_emg=True)
    >>>
    >>> # View summary metrics
    >>> print(results.summary)
    >>>
    >>> # Display force-time curve
    >>> results.figures['force_traces'].show()

    See Also
    --------
    IsometricTest : Test protocol for isometric strength assessment.
    TestResults : Parent class for test results.
    """

    def __init__(self, test: "IsometricTest", include_emg: bool):
        from .isometric_test import IsometricTest
        if not isinstance(test, IsometricTest):
            raise ValueError("'test' must be an IsometricTest instance.")
        super().__init__(test, include_emg)

    def _get_peak_force(self, test: IsometricExercise):
        return float(test.force.to_numpy().max())

    def _get_rate_of_force_development_kns(self, exe: IsometricExercise):
        force = exe.repetitions[0].force.to_numpy().flatten()
        time = np.array(exe.index)
        peaks = find_peaks(force, height=float(np.max(force) * 0.8))
        peak = np.argmax(force) if len(peaks) == 0 else peaks[0]
        return (force[peak] - force[0]) / (time[peak] - time[0]) / 1000

    def _get_time_to_peak_force_ms(self, exe: IsometricExercise):
        force = exe.repetitions[0].force.to_numpy().flatten()
        time = np.array(exe.index)
        peak = np.argmax(force)
        return (time[peak] - time[0]) * 1000

    def _get_summary(self, test: "IsometricTest"):
        trials = [test.left, test.right, test.bilateral]
        sides = ["left", "right", "bilateral"]
        metrics = pd.DataFrame()
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            emg_norms = test.emg_normalization_values
            if self.include_emg:
                for rep in trial.repetitions:
                    new = pd.DataFrame()
                    for m in rep.emgsignals.values():
                        ename = str(m.muscle_name)
                        eside = str(m.side)
                        keys = emg_norms.keys()
                        check = [i[0] == ename and i[1] == eside for i in keys]
                        ename += " (%)" if any(check) else " (uV)"
                        new.loc[ename, eside] = m.to_numpy().mean()
                    metrics = pd.concat([metrics, new])
            metrics.loc["rate of force development (kN/s)", side] = (
                self._get_rate_of_force_development_kns(trial)
            )
            metrics.loc["time to peak force (ms)", side] = (
                self._get_time_to_peak_force_ms(trial)
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

    def _get_analytics(self, test: "IsometricTest"):
        processed = test.processed_data
        analytics = []
        trials = [processed.left, processed.right, processed.bilateral]
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

    def _get_figures(self, test: "IsometricTest"):

        # force data
        tracks = self.analytics
        sides = np.unique(tracks.side).tolist()
        force_data = [
            "side",
            "repetition",
            "time_s",
            "force_amplitude",
            "position_amplitude",
        ]
        force_data = tracks[force_data].copy()
        y_arr = {side: [] for side in sides}
        for (side, rep), dfr in force_data.groupby(by=["side", "repetition"]):
            force = dfr.force_amplitude.to_numpy().flatten()
            fint = cubicspline_interp(force, 201)
            y_arr[side].append(np.atleast_2d(fint))
        tracks = []
        for i, side in enumerate(sides):
            y = np.vstack(y_arr[side]).max(axis=0)
            x = np.linspace(0, 5, len(y))
            df = pd.DataFrame({"Time (s)": x, "y": y})
            df.insert(0, "side", side)
            tracks.append(df)
        tracks = pd.concat(tracks, ignore_index=True)

        return {
            "force_profiles_with_muscle_balance": _get_force_figure(
                tracks,
                self.summary,
            )
        }


# TODO CREATE THE POWER-VELOCITY / FORCE-VELOCITY TEST


__all__ = ["IsometricTestResults"]
