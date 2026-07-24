"""Running test results implementation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from ...exercises.gait import RunningStep
from ...signalprocessing import cubicspline_interp
from ...timeseries.signal1d import Signal1D
from ...timeseries.timeseries import Timeseries
from ..test_results import TestResults
from .running_test import RunningTest


class RunningTestResults(TestResults):
    """
    Results container for RunningTest protocol.

    Provides comprehensive analysis of running gait including per-step metrics,
    aggregate statistics, time-series analytics, and interactive force profile
    visualizations.

    Parameters
    ----------
    test : RunningTest
        The running test instance containing detected cycles.
    include_emg : bool, optional
        Whether to include EMG metrics in results. Default is False.

    Attributes
    ----------
    summary : dict
        Dictionary with two DataFrames:
        - 'per_step': Per-step metrics for each detected cycle
        - 'aggregate': Aggregated statistics (mean, std, CV%, asymmetry)
    analytics : pd.DataFrame
        Time-series data in long format with normalized contact phases.
    figures : dict
        Dictionary of plotly figures including 'force_profiles'.

    Notes
    -----
    Per-Step Metrics:
    - contact_time_s: Duration of foot-ground contact (s)
    - propulsion_time_s: Duration of push-off phase (s)
    - flight_time_s: Duration of aerial phase (s)
    - cadence_steps_per_min: Step frequency (steps/min)
    - peak_vertical_force_N: Maximum vertical ground reaction force (N)
    - peak_braking_force_N: Maximum braking force during loading (N)
    - peak_propulsion_force_N: Maximum propulsion force during push-off (N)
    - vertical_oscillation_mm: Vertical displacement of pelvis (mm)
    - peak_trunk_lateral_flexion_deg: Peak trunk lateral flexion angle (deg)
    - peak_pelvis_lateral_tilt_deg: Peak pelvis lateral tilt angle (deg)
    - peak_trunk_rotation_deg: Peak trunk rotation angle (deg)

    Aggregate Metrics (per side):
    - mean: Average across all steps
    - std: Standard deviation
    - cv%: Coefficient of variation (%)
    - diff_%: Left-right asymmetry (%)

    Force Profiles Figure:
    - 2×2 subplot grid (vertical/AP × left/right)
    - Mean force curves normalized to 0-100% contact phase
    - Shaded area representing ±1 standard deviation
    - Distinct colors: blue for vertical, red for anteroposterior
    """

    def __init__(
        self,
        test,
        include_emg: bool = False,
        limit_steps: int | None = None,
    ):
        """
        Initialize RunningTestResults.

        Parameters
        ----------
        test : RunningTest
            The running test instance.
        include_emg : bool, optional
            Include EMG metrics. Default is False.
        limit_steps : int | None, optional
            If provided, limits the number of steps included in the results.
        """
        if not isinstance(test, RunningTest):
            raise ValueError("test must be a RunningTest instance.")
        self._test = test

        if not isinstance(include_emg, bool):
            raise ValueError("include_emg must be a boolean.")
        self._include_emg = include_emg

        if limit_steps is not None and not isinstance(limit_steps, int):
            raise ValueError("limit_steps must be an integer or None.")
        self._limit_steps = limit_steps

        self._summary = None
        self._analytics = None
        self._figures = None
        self._generate_results()

    @property
    def limit_steps(self):
        """limit the number of steps for each exercise of the test"""
        return self._limit_steps

    @property
    def include_emg(self):
        """return whether EMG data is included in the results"""
        return self._include_emg

    @property
    def test(self):
        """return the running test instance"""
        return self._test

    def _generate_results(self):
        """Generate all results components."""
        self._summary = self._get_summary()
        self._analytics = self._get_analytics()
        self._figures = self._get_figures()

    def _get_summary(self):
        """
        Generate summary statistics.

        Returns
        -------
        dict
            Dictionary with 'per_step' and 'aggregate' DataFrames.
        """
        # get steps data
        steps_data = []
        for exe in self._test.exercises:
            count = 0
            for cycle in tqdm(
                exe.steps,
                f"SUMMARY - speed: {exe.speed} - grade: {exe.grade}",
            ):
                if not isinstance(cycle, RunningStep):
                    continue
                if self.limit_steps is None or count < self.limit_steps:
                    metrics = cycle.get_output_metrics(include_emg=self._include_emg)
                    count += 1
                    metrics[("step", "#")] = count
                    steps_data.append(metrics)
        steps = pd.DataFrame(steps_data)
        steps.columns = pd.MultiIndex.from_tuples(steps.columns)  # type: ignore

        # get aggregated data
        agg_data = pd.DataFrame(
            steps.drop([("step", "#")], axis=1)
            .groupby(
                [
                    ("speed", "km/h"),
                    ("pace", "min/km"),
                    ("grade", "%"),
                    ("side", "left/right"),
                ]
            )
            .agg(["mean", "std"])
            .stack([0, 1])
            .unstack(3)
            .stack(0)
            .unstack(-1)
        )

        return {"per_step": steps, "aggregate": agg_data}

    def _get_analytics(self):
        """
        Generate time-series analytics.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with time-series data for each cycle.
        """
        # get steps data
        steps_data = []
        for exe in self.test.exercises:
            count = 1
            for cycle in tqdm(
                exe.steps,
                f"ANALYTICS - speed: {exe.speed} - grade: {exe.grade}",
            ):
                if not isinstance(cycle, RunningStep):
                    continue
                if self.limit_steps is None or count < self.limit_steps:
                    signals = cycle.get_output_signals(include_emg=self._include_emg)
                    signals = signals.to_dataframe()
                    cols = [tuple(i.rsplit(" ", 1)) for i in signals.columns]
                    cols = pd.MultiIndex.from_tuples(cols)
                    signals.columns = cols
                    signals.insert(0, ("time", "s"), signals.index.to_numpy())
                    signals.insert(0, ("step", "#"), count)
                    signals.insert(0, ("side", "left/right"), cycle.side)
                    signals.insert(0, ("speed", "km/h"), cycle.speed)
                    signals.insert(0, ("pace", "min/km"), cycle.pace)
                    signals.insert(0, ("grade", "%"), cycle.grade)
                    signals.reset_index(drop=True, inplace=True)
                steps_data.append(signals)
        out = pd.DataFrame(pd.concat(steps_data, ignore_index=True))
        return out

    def _get_figures(self):
        """
        Generate interactive figures.

        Returns
        -------
        dict
            Dictionary with 'force_profiles' figure.
        """
        fig = self._get_force_profile_figure()
        return {"force_profiles": fig}

    def _get_force_profile_figure(self):
        """
        Create force profile figure with mean and std.

        Returns
        -------
        go.Figure
            Plotly figure with 2×2 subplots showing vertical and AP forces
            for left and right sides.
        """
        # Collect and normalize contact phase forces
        left_vertical, left_ap = [], []
        right_vertical, right_ap = [], []

        for cycle in test.cycles:
            contact = cycle.contact_phase
            if contact is None:
                continue

            res = contact.resultant_force
            if res is None:
                continue

            v_force = res.force[test.vertical_axis].to_numpy().flatten()
            ap_force = res.force[test.anteroposterior_axis].to_numpy().flatten()

            # Normalize to 101 points (0-100%)
            v_norm = self._normalize_to_101_points(v_force)
            ap_norm = self._normalize_to_101_points(ap_force)

            if cycle.side == "left":
                left_vertical.append(v_norm)
                left_ap.append(ap_norm)
            else:
                right_vertical.append(v_norm)
                right_ap.append(ap_norm)

        # Create subplot grid: 2 rows (vertical/AP) × 2 cols (left/right)
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Left Vertical Force",
                "Right Vertical Force",
                "Left Anteroposterior Force",
                "Right Anteroposterior Force",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
            x_title="Contact Phase (%)",
            y_title="Force (N)",
        )

        x_norm = np.linspace(0, 100, 101)

        # Plot vertical forces (blue)
        if left_vertical:
            mean_v = np.nanmean(left_vertical, axis=0)
            std_v = np.nanstd(left_vertical, axis=0)
            self._add_mean_std_trace(
                fig,
                x_norm,
                mean_v,
                std_v,
                row=1,
                col=1,
                color="rgb(0, 0, 255)",
                name="Left Vertical",
            )

        if right_vertical:
            mean_v = np.nanmean(right_vertical, axis=0)
            std_v = np.nanstd(right_vertical, axis=0)
            self._add_mean_std_trace(
                fig,
                x_norm,
                mean_v,
                std_v,
                row=1,
                col=2,
                color="rgb(0, 0, 255)",
                name="Right Vertical",
            )

        # Plot anteroposterior forces (red)
        if left_ap:
            mean_ap = np.nanmean(left_ap, axis=0)
            std_ap = np.nanstd(left_ap, axis=0)
            self._add_mean_std_trace(
                fig,
                x_norm,
                mean_ap,
                std_ap,
                row=2,
                col=1,
                color="rgb(255, 0, 0)",
                name="Left AP",
            )

        if right_ap:
            mean_ap = np.nanmean(right_ap, axis=0)
            std_ap = np.nanstd(right_ap, axis=0)
            self._add_mean_std_trace(
                fig,
                x_norm,
                mean_ap,
                std_ap,
                row=2,
                col=2,
                color="rgb(255, 0, 0)",
                name="Right AP",
            )

        # Update layout
        fig.update_xaxes(title_text="Contact Phase (%)", row=2)
        fig.update_yaxes(title_text="Force (N)")
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Ground Reaction Forces - Mean ± SD",
            title_x=0.5,
        )

        return fig

    def _add_mean_std_trace(self, fig, x, mean, std, row, col, color, name):
        """
        Add mean line and std area to subplot.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to add traces to.
        x : array-like
            X-axis values (0-100%).
        mean : array-like
            Mean values.
        std : array-like
            Standard deviation values.
        row : int
            Subplot row.
        col : int
            Subplot column.
        color : str
            RGB color string.
        name : str
            Trace name for legend.
        """
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=x, y=mean, mode="lines", line=dict(width=2.5, color=color), name=name
            ),
            row=row,
            col=col,
        )

        # Extract RGB values from color string
        rgb_values = color.replace("rgb(", "").replace(")", "").split(",")
        r, g, b = [int(v.strip()) for v in rgb_values]

        # Std area (shaded region)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mean + std, (mean - std)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({r}, {g}, {b}, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    def _normalize_to_101_points(self, signal):
        """
        Normalize signal to 101 points (0-100%) using cubic spline interpolation.

        Parameters
        ----------
        signal : array-like
            Input signal to normalize.

        Returns
        -------
        np.ndarray
            Signal normalized to 101 points.
        """
        if signal is None or len(signal) == 0:
            return np.full(101, np.nan)

        data = np.array(signal).flatten()

        if len(data) < 4:
            # Use linear interpolation for short signals
            old_x = np.linspace(0, 100, len(data))
            new_x = np.linspace(0, 100, 101)
            return np.interp(new_x, old_x, data)

        # Use cubic spline for smooth interpolation
        return cubicspline_interp(data, nsamp=101)


__all__ = ["RunningTestResults"]
