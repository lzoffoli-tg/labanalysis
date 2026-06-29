"""Running test results implementation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...signalprocessing import cubicspline_interp
from ..test_results import TestResults


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

    def __init__(self, test, include_emg: bool = False):
        """
        Initialize RunningTestResults.

        Parameters
        ----------
        test : RunningTest
            The running test instance.
        include_emg : bool, optional
            Include EMG metrics. Default is False.
        """
        self._test = test
        self._include_emg = include_emg
        self._summary = None
        self._analytics = None
        self._figures = None
        self._generate_results(test)

    def _generate_results(self, test):
        """Generate all results components."""
        self._summary = self._get_summary(test)
        self._analytics = self._get_analytics(test)
        self._figures = self._get_figures(test)

    def _get_summary(self, test):
        """
        Generate summary statistics.

        Returns
        -------
        dict
            Dictionary with 'per_step' and 'aggregate' DataFrames.
        """
        steps_data = []

        for i, cycle in enumerate(test.cycles, start=1):
            row = {
                'cycle': i,
                'side': cycle.side,
                'contact_time_s': cycle.contact_time_s,
                'propulsion_time_s': cycle.propulsion_time_s,
                'flight_time_s': cycle.flight_time_s,
                'cadence_steps_per_min': 60.0 / cycle.cycle_time_s,
            }

            # Peak vertical force
            if cycle.peak_force is not None and not np.isnan(cycle.peak_force):
                row['peak_vertical_force_N'] = float(cycle.peak_force)

            # Peak braking force
            braking = cycle.peak_braking_force
            if braking is not None:
                braking_val = braking.to_numpy().flatten()
                if len(braking_val) > 0:
                    row['peak_braking_force_N'] = float(braking_val[0])

            # Peak propulsion force
            propulsion = cycle.peak_propulsion_force
            if propulsion is not None:
                propulsion_val = propulsion.to_numpy().flatten()
                if len(propulsion_val) > 0:
                    row['peak_propulsion_force_N'] = float(propulsion_val[0])

            # Vertical oscillation
            oscillation = cycle.vertical_oscillation
            if oscillation is not None:
                osc_array = oscillation.to_numpy().flatten()
                if len(osc_array) > 0:
                    osc_value = float(osc_array[0])
                    unit = oscillation.unit

                    # Convert to mm if in meters
                    if unit == 'm':
                        osc_value *= 1000
                    elif unit == 'cm':
                        osc_value *= 10

                    row['vertical_oscillation_mm'] = osc_value

            # Peak trunk lateral flexion
            trunk_lat = cycle.peak_trunk_lateral_flexion
            if trunk_lat is not None:
                trunk_lat_array = trunk_lat.to_numpy().flatten()
                if len(trunk_lat_array) > 0:
                    row['peak_trunk_lateral_flexion_deg'] = float(trunk_lat_array[0])

            # Peak pelvis lateral tilt
            pelvis_tilt = cycle.peak_pelvis_lateral_tilt
            if pelvis_tilt is not None:
                pelvis_tilt_array = pelvis_tilt.to_numpy().flatten()
                if len(pelvis_tilt_array) > 0:
                    row['peak_pelvis_lateral_tilt_deg'] = float(pelvis_tilt_array[0])

            # Peak trunk rotation
            trunk_rot = cycle.peak_trunk_rotation
            if trunk_rot is not None:
                trunk_rot_array = trunk_rot.to_numpy().flatten()
                if len(trunk_rot_array) > 0:
                    row['peak_trunk_rotation_deg'] = float(trunk_rot_array[0])

            steps_data.append(row)

        summary_steps = pd.DataFrame(steps_data)

        # Aggregate statistics by side
        metrics = [
            'contact_time_s', 'propulsion_time_s', 'flight_time_s',
            'cadence_steps_per_min', 'peak_vertical_force_N',
            'peak_braking_force_N', 'peak_propulsion_force_N',
            'vertical_oscillation_mm',
            'peak_trunk_lateral_flexion_deg',
            'peak_pelvis_lateral_tilt_deg',
            'peak_trunk_rotation_deg'
        ]

        agg_data = []
        for metric in metrics:
            if metric not in summary_steps.columns:
                continue

            left_vals = summary_steps[summary_steps['side'] == 'left'][metric].dropna()
            right_vals = summary_steps[summary_steps['side'] == 'right'][metric].dropna()

            row = {'metric': metric}

            # Left side statistics
            if len(left_vals) > 0:
                row['left_mean'] = left_vals.mean()
                row['left_std'] = left_vals.std()
                if left_vals.mean() != 0:
                    row['left_cv%'] = (left_vals.std() / left_vals.mean()) * 100
                else:
                    row['left_cv%'] = np.nan

            # Right side statistics
            if len(right_vals) > 0:
                row['right_mean'] = right_vals.mean()
                row['right_std'] = right_vals.std()
                if right_vals.mean() != 0:
                    row['right_cv%'] = (right_vals.std() / right_vals.mean()) * 100
                else:
                    row['right_cv%'] = np.nan

            # Left-right asymmetry
            if len(left_vals) > 0 and len(right_vals) > 0:
                l_mean = left_vals.mean()
                r_mean = right_vals.mean()
                avg = (l_mean + r_mean) / 2
                if avg != 0:
                    row['diff_%'] = 100 * (r_mean - l_mean) / avg
                else:
                    row['diff_%'] = np.nan

            agg_data.append(row)

        summary_aggregate = pd.DataFrame(agg_data)

        return {
            'per_step': summary_steps,
            'aggregate': summary_aggregate
        }

    def _get_analytics(self, test):
        """
        Generate time-series analytics.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with time-series data for each cycle.
        """
        analytics_parts = []

        for i, cycle in enumerate(test.cycles, start=1):
            contact = cycle.contact_phase
            if contact is None:
                continue

            res = contact.resultant_force
            if res is None:
                continue

            # Extract vertical and anteroposterior forces
            v_force = res.force[test.vertical_axis].to_numpy().flatten()
            ap_force = res.force[test.anteroposterior_axis].to_numpy().flatten()
            time = res.index

            # Create DataFrame
            df = pd.DataFrame({
                'time_s': time - time[0],  # Relative time from contact start
                'vertical_force_N': v_force,
                'anteroposterior_force_N': ap_force,
            })

            # Add metadata
            df.insert(0, 'side', cycle.side)
            df.insert(0, 'cycle', i)

            analytics_parts.append(df)

        if len(analytics_parts) == 0:
            return pd.DataFrame()

        return pd.concat(analytics_parts, ignore_index=True)

    def _get_figures(self, test):
        """
        Generate interactive figures.

        Returns
        -------
        dict
            Dictionary with 'force_profiles' figure.
        """
        fig = self._get_force_profile_figure(test)
        return {'force_profiles': fig}

    def _get_force_profile_figure(self, test):
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

            if cycle.side == 'left':
                left_vertical.append(v_norm)
                left_ap.append(ap_norm)
            else:
                right_vertical.append(v_norm)
                right_ap.append(ap_norm)

        # Create subplot grid: 2 rows (vertical/AP) × 2 cols (left/right)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Left Vertical Force',
                'Right Vertical Force',
                'Left Anteroposterior Force',
                'Right Anteroposterior Force'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
            x_title='Contact Phase (%)',
            y_title='Force (N)'
        )

        x_norm = np.linspace(0, 100, 101)

        # Plot vertical forces (blue)
        if left_vertical:
            mean_v = np.nanmean(left_vertical, axis=0)
            std_v = np.nanstd(left_vertical, axis=0)
            self._add_mean_std_trace(
                fig, x_norm, mean_v, std_v,
                row=1, col=1, color='rgb(0, 0, 255)', name='Left Vertical'
            )

        if right_vertical:
            mean_v = np.nanmean(right_vertical, axis=0)
            std_v = np.nanstd(right_vertical, axis=0)
            self._add_mean_std_trace(
                fig, x_norm, mean_v, std_v,
                row=1, col=2, color='rgb(0, 0, 255)', name='Right Vertical'
            )

        # Plot anteroposterior forces (red)
        if left_ap:
            mean_ap = np.nanmean(left_ap, axis=0)
            std_ap = np.nanstd(left_ap, axis=0)
            self._add_mean_std_trace(
                fig, x_norm, mean_ap, std_ap,
                row=2, col=1, color='rgb(255, 0, 0)', name='Left AP'
            )

        if right_ap:
            mean_ap = np.nanmean(right_ap, axis=0)
            std_ap = np.nanstd(right_ap, axis=0)
            self._add_mean_std_trace(
                fig, x_norm, mean_ap, std_ap,
                row=2, col=2, color='rgb(255, 0, 0)', name='Right AP'
            )

        # Update layout
        fig.update_xaxes(title_text="Contact Phase (%)", row=2)
        fig.update_yaxes(title_text="Force (N)")
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text='Ground Reaction Forces - Mean ± SD',
            title_x=0.5
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
                x=x,
                y=mean,
                mode='lines',
                line=dict(width=2.5, color=color),
                name=name
            ),
            row=row, col=col
        )

        # Extract RGB values from color string
        rgb_values = color.replace('rgb(', '').replace(')', '').split(',')
        r, g, b = [int(v.strip()) for v in rgb_values]

        # Std area (shaded region)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mean + std, (mean - std)[::-1]]),
                fill='toself',
                fillcolor=f'rgba({r}, {g}, {b}, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
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
