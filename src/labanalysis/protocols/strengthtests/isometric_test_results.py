"""Isometric test results implementation."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...exercises.strength import IsometricExercise

from ...signalprocessing import cubicspline_interp, find_peaks
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
        - Force at 100 ms (N)
        - Force at 200 ms (N)
        - Force at 500 ms (N)
        - Force at 1000 ms (N)
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

    def _get_peak_force(self, exe: IsometricExercise, rep_index: int):
        """Get peak force for a specific repetition."""
        force = exe.repetitions[rep_index].force.to_numpy().flatten()
        return float(np.max(force))

    def _get_rate_of_force_development_kns(self, exe: IsometricExercise, rep_index: int):
        """Get RFD for a specific repetition."""
        rep = exe.repetitions[rep_index]
        force = rep.force.to_numpy().flatten()
        time = np.array(rep.index)
        peaks = find_peaks(force, height=float(np.max(force) * 0.8))
        peak = np.argmax(force) if len(peaks) == 0 else peaks[0]
        return (force[peak] - force[0]) / (time[peak] - time[0]) / 1000

    def _get_time_to_peak_force_ms(self, exe: IsometricExercise, rep_index: int):
        """Get time to peak force for a specific repetition."""
        rep = exe.repetitions[rep_index]
        force = rep.force.to_numpy().flatten()
        time = np.array(rep.index)
        peak = np.argmax(force)
        return (time[peak] - time[0]) * 1000

    def _get_force_at_time_ms(self, exe: IsometricExercise, rep_index: int, time_ms: float):
        """Get force level at specific time point (ms) from contraction onset for a specific repetition."""
        rep = exe.repetitions[rep_index]
        force = rep.force.to_numpy().flatten()
        time = (np.array(rep.index) - rep.index[0]) * 1000  # Convert to ms from onset

        # If the requested time is beyond the data, return the last force value
        if time_ms >= time[-1]:
            return float(force[-1])

        # Interpolate force at requested time
        return float(np.interp(time_ms, time, force))

    def _get_rfd_at_interval_ms(self, exe: IsometricExercise, rep_index: int, time_ms: float):
        """Get RFD (N/s) from onset to specific time point for a specific repetition."""
        force_at_time = self._get_force_at_time_ms(exe, rep_index, time_ms)

        # Get baseline force (at onset)
        rep = exe.repetitions[rep_index]
        baseline_force = float(rep.force.to_numpy().flatten()[0])

        # RFD = (F_time - F_baseline) / (time_ms / 1000)
        # Result in N/s
        delta_force = force_at_time - baseline_force
        delta_time_s = time_ms / 1000.0

        if delta_time_s == 0:
            return 0.0

        return float(delta_force / delta_time_s)

    def _get_summary(self, test: "IsometricTest"):
        trials = [test.left, test.right, test.bilateral]
        sides = ["left", "right", "bilateral"]

        # Initialize metrics with all required columns
        active_sides = [s for s, t in zip(sides, trials) if t is not None]
        metrics = pd.DataFrame(columns=active_sides)

        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            emg_norms = test.emg_normalization_values

            # Iterate over all repetitions to get metrics for each
            for i, rep in enumerate(trial.repetitions):
                new = pd.DataFrame(columns=active_sides)

                # EMG data for this repetition
                if self.include_emg:
                    for m in rep.emgsignals.values():
                        ename = str(m.muscle_name)
                        eside = str(m.side)
                        keys = emg_norms.keys()
                        check = [i[0] == ename and i[1] == eside for i in keys]
                        ename += " (%)" if any(check) else " (uV)"
                        new.loc[ename, eside] = m.to_numpy().mean()

                # Force metrics for this repetition
                new.loc["rate of force development (kN/s)", side] = (
                    self._get_rate_of_force_development_kns(trial, i)
                )
                new.loc["time to peak force (ms)", side] = (
                    self._get_time_to_peak_force_ms(trial, i)
                )
                # Peak force in kN
                new.loc["peak force (kN)", side] = self._get_peak_force(trial, i) / 1000.0

                # Get time points from exercise
                time_points = trial.time_points

                # Calculate force and RFD at each time point
                for tp in time_points:
                    # Force in kN (divide by 1000)
                    new.loc[f"force at {tp} ms (kN)", side] = (
                        self._get_force_at_time_ms(trial, i, tp) / 1000.0
                    )
                    # RFD in kN/s (divide by 1000)
                    new.loc[f"RFD 0-{tp} ms (kN/s)", side] = (
                        self._get_rfd_at_interval_ms(trial, i, tp) / 1000.0
                    )

                metrics = pd.concat([metrics, new])
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
        analytics = self.analytics
        sides = np.unique(analytics.side).tolist()

        # Find actual force and position column names
        available_cols = analytics.columns.tolist()

        # Find force column (e.g., "force N")
        force_col = None
        for col in available_cols:
            if 'force' in col.lower():
                force_col = col
                break

        # Find position column (e.g., "position m")
        position_col = None
        for col in available_cols:
            if 'position' in col.lower():
                position_col = col
                break

        if force_col is None or position_col is None:
            # If we can't find these columns, skip figure generation
            return {}

        # Determine time limit and time points from exercises
        max_time_s = None
        time_points = [100, 200, 500, 1000]  # default
        for exe in [test.left, test.right, test.bilateral]:
            if exe is not None:
                max_time_s = exe.max_time_s
                time_points = exe.time_points
                break
        # Default to 2000 ms if max_time_s is not set
        max_time_ms = max_time_s * 1000 if max_time_s is not None else 2000

        # Process force data for each side and repetition
        # Build tracks with absolute time (ms)
        tracks_data = []
        for side, rep_group in analytics.groupby('side'):
            # Get the first repetition for this side
            first_rep = rep_group[rep_group['repetition'] == 1]
            if first_rep.empty:
                continue

            # Get time in ms and force
            time_ms = (first_rep['time_s'].to_numpy() * 1000).flatten()
            force = first_rep[force_col].to_numpy().flatten()

            # Limit to max_time_ms
            mask = time_ms <= max_time_ms
            time_ms = time_ms[mask]
            force = force[mask]

            # Store in tracks_data
            for t, val in zip(time_ms, force):
                tracks_data.append({
                    "parameter": "force_amplitude",
                    "side": side,
                    "limb": side,
                    "time_ms": t,
                    "value": val
                })

        tracks = pd.DataFrame(tracks_data)

        return {
            "force_profiles_with_muscle_balance": _get_force_figure(
                tracks,
                self.summary,
                include_emg=self.include_emg,
                time_mode='absolute',  # Use absolute time mode
                time_points=time_points,  # Pass time points to figure
                max_time_ms=max_time_ms,  # Pass max time for X-axis limit
            )
        }


# TODO CREATE THE POWER-VELOCITY / FORCE-VELOCITY TEST


__all__ = ["IsometricTestResults"]
