"""Jump test results implementation."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...constants import G
from ...modelling.ols import Ellipse
from ...plotting import bars_with_normative_bands
from ...signalprocessing import continuous_batches, cubicspline_interp, fillna
from ...timeseries import EMGSignal
from ...exercises import DropJump, SingleJump
from ..test_results import TestResults

if TYPE_CHECKING:
    from .jump_test import JumpTest


class JumpTestResults(TestResults):
    """
    Results container for jump test analysis with automated reporting.

    JumpTestResults processes JumpTest data to generate comprehensive
    performance summaries, interactive visualizations, and normative
    comparisons. The class automatically computes all relevant jump metrics,
    generates publication-ready figures, and provides structured data export.

    Parameters
    ----------
    test : JumpTest
        Processed jump test data to analyze.
    include_emg : bool
        Whether to include EMG analysis in results (activation timing,
        amplitude, and pre-activation ratios).

    Attributes
    ----------
    summary : pd.DataFrame
        Comprehensive table of all jump metrics including elevation, contact
        time, flight time, RSI, force symmetry, and EMG metrics.
    analytics : pd.DataFrame
        Time-series data for all jumps in long format for detailed analysis.
    figures : dict of str -> go.Figure or dict
        Dictionary of interactive Plotly figures:
        - 'ground_reaction_forces': Force-time curves for all jumps
        - 'elevation': Jump height with normative bands and symmetry
        - 'contact_time': Contact time with normative ranking (DJ/repeated only)
        - 'rsi': Reactive strength index (DJ/repeated only)
        - 'muscle_activation_ratio': EMG pre-activation (DJ only, if include_emg)
        - 'muscle_activation_time': EMG onset timing (DJ only, if include_emg)

    Notes
    -----
    Metric Calculations:
    - Elevation: min(flight_time_method, impulse_method) for conservative estimate
    - Flight time method: h = (t_flight^2 * g) / 8
    - Impulse method: h = v_takeoff^2 / (2*g), where v from force integral
    - RSI: elevation / (contact_time / 1000)
    - Force symmetry: 100 * (R - L) / mean(R, L)

    EMG Metrics (Drop Jumps only):
    - Activation time: Time from landing to sustained EMG > threshold
    - Pre-activation ratio: mean(EMG_pre) / max(EMG_loading) * 100
    - Pre-window: 25ms before landing
    - Loading window: Landing to bodyweight crossing

    Figure Organization:
    Each figure type may contain multiple subplots organized by:
    - Jump type (SJ, CMJ, DJ, Repeated)
    - Side (bilateral, left, right)
    - Box height (for drop jumps)

    Normative data (if available) is displayed as colored bands overlaying
    the performance bars, with ranks typically defined as:
    - 5-level: Elite, Above Average, Average, Below Average, Poor
    - 3-level: Good, Average, Poor

    See Also
    --------
    JumpTest : Test protocol for jump assessment.
    TestResults : Parent class for test results.
    """

    def __init__(self, test: "JumpTest", include_emg: bool):
        from .jump_test import JumpTest
        if not isinstance(test, JumpTest):
            raise ValueError("'test' must be an JumpTest instance.")
        super().__init__(test, include_emg)

    def _get_jump_contact_time_ms(self, jump: SingleJump | DropJump):
        contact = jump.contact_phase
        if contact is None:
            return 0
        try:
            time = contact.index
            if len(time) == 0:
                return 0
            return int(round((time[-1] - time[0]) * 1000))
        except (ValueError, AttributeError):
            return 0

    def _get_jump_flight_time_ms(self, jump: SingleJump | DropJump):
        flight = jump.flight_phase
        if flight is None:
            return 0
        try:
            time = flight.index
            if len(time) == 0:
                return 0
            return int(round((time[-1] - time[0]) * 1000))
        except (ValueError, AttributeError):
            return 0

    def _get_takeoff_velocity_ms(
        self,
        jump: SingleJump | DropJump,
        bodyweight: float,
    ):

        # get the ground reaction force during the concentric phase
        con = jump.contact_phase.resultant_force
        if con is None:
            return np.nan
        grf = con.copy().force[jump.vertical_axis].to_numpy().flatten()
        grfy = fillna(arr=grf, value=0).flatten()  # type: ignore
        grft = con.index

        # get the output velocity
        net_grf = grfy - bodyweight * G
        return float(np.trapezoid(net_grf / bodyweight, grft))

    def _get_elevation_cm(
        self,
        jump: SingleJump | DropJump,
        bodyweight: float,
    ):

        # from flight time
        flight_time = jump.flight_phase.index
        flight_time = flight_time[-1] - flight_time[0]
        elevation_from_time = (flight_time**2) * G / 8 * 100

        # from force impulse
        elevation_from_velocity = (
            (self._get_takeoff_velocity_ms(jump, bodyweight) ** 2) / (2 * G) * 100
        )

        # return the lower of the two
        return float(min(elevation_from_time, elevation_from_velocity))

    def _get_muscle_activation_ratio(
        self,
        muscle: EMGSignal,
        landing_instant_s: float,
        t_bodyweight_s: float,
        pre_window_s: float = 0.025,
    ):

        # get the data
        time = muscle.index.copy()
        envelope = muscle.to_numpy().flatten()

        # get the pre-condition
        pre_idx = np.where(
            (time >= landing_instant_s - pre_window_s) & (time < landing_instant_s)
        )[0]
        lr_idx = np.where((time >= landing_instant_s) & (time <= t_bodyweight_s))[0]
        if len(pre_idx) == 0 or len(lr_idx) == 0:
            raise RuntimeError("time window not possible.")

        # get the mean amplitude in the pre-window the the highest activation
        # in the lr window
        pre_emg = float(envelope[pre_idx].mean())
        lr_emg = float(envelope[lr_idx].max())

        # return the ratio
        return pre_emg / lr_emg

    def _get_muscle_activation_time_ms(
        self,
        muscle: EMGSignal,
        threshold: float,
        landing_instant_s: float,
    ):

        # get the data
        time = muscle.index.copy()
        envelope = muscle.to_numpy().flatten()

        # offset time to contact_time_instant_s
        time -= landing_instant_s

        # get the steady activation time
        batches = continuous_batches(envelope >= threshold)
        fsamp = float(1 / np.mean(np.diff(time)))
        samples = int(round(fsamp * 0.025))
        batches = [i for i in batches if len(i) >= samples]
        if len(batches) == 0:
            return time[-1] * 1000
        return time[np.array(batches[0])][0] * 1000

    def _get_summary(self, test: "JumpTest"):
        muscle_map = test.relevant_muscle_map
        if test.relevant_muscle_map is None:
            muscle_map = []
        else:
            muscle_map = test.relevant_muscle_map.copy()

        def _get_jumps_summary_table(
            jumps: list[SingleJump | DropJump],
            jump_name: str,
        ):
            sides_counter = {}
            sides_df = {}
            for jump in jumps:
                if jump.side not in sides_counter:
                    sides_counter[jump.side] = 1
                    sides_df[jump.side] = pd.DataFrame()
                else:
                    sides_counter[jump.side] += 1
                out = pd.DataFrame()
                contact = jump.contact_phase

                # Skip if contact phase not found
                if contact is None:
                    continue

                jump_side = "bilateral" if jump.side == "bilateral" else "unilateral"

                # remove unnecessary EMG data
                if self.include_emg:
                    to_remove = []
                    for k in contact.emgsignals.keys():
                        if all(i.lower() not in k.lower() for i in muscle_map):
                            to_remove.append(k)
                else:
                    to_remove = contact.emgsignals.keys()
                contact.drop(to_remove, inplace=True)
                jump.drop(to_remove, inplace=True)

                # get muscle emg amplitude
                for emg in contact.emgsignals.values():
                    if emg.side == jump.side or jump.side == "bilateral":
                        name = emg.muscle_name.replace("_", " ").lower()  # type: ignore
                        name += f" amplitude ({emg.unit})"  # type: ignore
                        val = emg.to_numpy().mean()  # type: ignore
                        out.loc[name, f"{emg.side}"] = float(val)  # type: ignore

                # add activation time and ratios
                refs_keys = test.emg_activation_references.keys()
                if isinstance(jump, DropJump):

                    # get the ground contact time
                    t_gc = contact.index[0]

                    # get the time corresponding to the end of the
                    # loading response
                    grf = jump.resultant_force.force[jump.vertical_axis].copy()
                    time = grf.index.copy()
                    grf = grf.to_numpy().flatten()
                    wgt = test.participant.weight
                    if wgt is None:
                        raise ValueError("participant's weight must be provided.")
                    batches = continuous_batches((grf >= wgt * G) & (time > t_gc))
                    fsamp = float(1 / np.mean(np.diff(time)))
                    min_samples = fsamp * 0.025
                    batches = [i for i in batches if len(i) >= min_samples]
                    if len(batches) == 0:
                        raise RuntimeError("No loading response was detected.")
                    batch = batches[0]
                    t_bw = time[batch][0]

                    # extract the EMG-related metrics
                    for emg in jump.emgsignals.values():
                        if not isinstance(emg, EMGSignal):
                            continue
                        if emg.side == jump.side or jump.side == "bilateral":
                            lbl = "_".join([emg.side, emg.muscle_name])
                            muscle_name = emg.muscle_name.replace("_", " ")
                            muscle_name = muscle_name.lower()

                            # get muscle activation time
                            if lbl in refs_keys:
                                key = (emg.muscle_name, emg.side)
                                threshold = test.emg_activation_thresholds.get(key)
                                if threshold is not None:
                                    val = self._get_muscle_activation_time_ms(
                                        emg,
                                        threshold,
                                        t_gc,
                                    )
                                    name = f"{muscle_name} activation time (ms)"
                                    out.loc[name, emg.side] = val

                            # get muscle activation ratio
                            val = self._get_muscle_activation_ratio(
                                emg,
                                t_gc,
                                t_bw,
                            )
                            name = f"{muscle_name} activation ratio"
                            out.loc[name, emg.side] = val * 100

                # get force
                sides = ["left", "right"] if jump_side == "bilateral" else [jump.side]
                for side in sides:
                    frz = contact.get(f"{side}_foot_ground_reaction_force")
                    if frz is None:
                        continue
                    val = frz["force"][frz.vertical_axis].to_numpy().mean()
                    name = "vertical force (N)"
                    out.loc[name, side] = float(val)

                # add jump parameters
                wgt = test.participant.weight
                if wgt is None:
                    raise ValueError("participant's weight cannot be None.")
                ctime = self._get_jump_contact_time_ms(jump)
                ftime = self._get_jump_flight_time_ms(jump)
                tov = self._get_takeoff_velocity_ms(jump, wgt)
                elevation = self._get_elevation_cm(jump, wgt)
                for side in sides:
                    out.loc["takeoff velocity (m/s)", side] = tov
                    out.loc["elevation (cm)", side] = elevation
                    out.loc["flight time (ms)", side] = ftime
                    out.loc["contact time (ms)", side] = ctime
                    out.loc["flight-to-contact ratio", side] = float(
                        round(ftime / ctime, 2)
                    )

                # Calculate RSI for drop jumps
                if isinstance(jump, DropJump):
                    rsi = (elevation / (ctime / 1000)) if ctime > 0 else 0.0
                    for side in sides:
                        out.loc["reactive strength index", side] = float(rsi)

                # Add jump number and side
                jump_num = sides_counter[jump.side]
                out.insert(0, "jump", jump_num)
                out.insert(0, "side", jump.side)

                # Append to the summary for this side
                sides_df[jump.side] = pd.concat(
                    [sides_df[jump.side], out], ignore_index=True
                )

            # Concatenate all sides and add jump type
            result = pd.concat(list(sides_df.values()), ignore_index=True)
            result.insert(0, "type", jump_name)
            return result

        # Process each jump type
        summary_parts = []

        if test.squat_jumps:
            sj_summary = _get_jumps_summary_table(test.squat_jumps, "squat jump")
            summary_parts.append(sj_summary)

        if test.counter_movement_jumps:
            cmj_summary = _get_jumps_summary_table(
                test.counter_movement_jumps, "counter movement jump"
            )
            summary_parts.append(cmj_summary)

        if test.drop_jumps:
            dj_summary = _get_jumps_summary_table(test.drop_jumps, "drop jump")
            summary_parts.append(dj_summary)

        if test.repeated_jumps:
            rj_summary = _get_jumps_summary_table(test.repeated_jumps, "repeated jump")
            summary_parts.append(rj_summary)

        # Concatenate all summaries
        if summary_parts:
            return pd.concat(summary_parts, ignore_index=True)
        else:
            return pd.DataFrame()

    def _get_analytics(self, test: "JumpTest"):
        """
        Generate detailed analytics with time-series data for all jumps.

        Returns a DataFrame with columns:
        - type: Jump type (squat jump, CMJ, drop jump, repeated jump)
        - jump: Jump number
        - side: bilateral/left/right
        - time_s: Time relative to contact phase start
        - force columns: Force platform data
        - emg columns: EMG signals (if include_emg)
        """
        analytics_parts = []

        jump_types = [
            (test.squat_jumps, "squat jump"),
            (test.counter_movement_jumps, "counter movement jump"),
            (test.drop_jumps, "drop jump"),
            (test.repeated_jumps, "repeated jump"),
        ]

        for jumps, jump_name in jump_types:
            for jump_idx, jump in enumerate(jumps, start=1):
                # Get contact phase data
                contact = jump.contact_phase

                # Skip if no contact phase
                if contact is None:
                    continue

                contact = contact.copy()

                # Remove EMG if not requested
                if not self.include_emg:
                    emg_keys = list(contact.emgsignals.keys())
                    if emg_keys:
                        contact.drop(emg_keys, inplace=True)

                # Convert to DataFrame
                df = contact.to_dataframe()

                # Add time column relative to start
                time = df.index - df.index.min()
                df.insert(0, "time_s", time)

                # Add metadata columns
                df.insert(0, "side", jump.side)
                df.insert(0, "jump", jump_idx)
                df.insert(0, "type", jump_name)

                analytics_parts.append(df)

        if analytics_parts:
            return pd.concat(analytics_parts, ignore_index=True)
        else:
            return pd.DataFrame()

    def _get_figures(
        self, test: "JumpTest"
    ) -> dict[str, go.Figure | dict[str, go.Figure]]:
        """
        Generate interactive Plotly figures for jump test visualization.

        Returns a dictionary with keys:
        - 'elevation': Bar chart of jump heights with normative bands
        - 'contact_time': Bar chart of contact times (drop jumps only)
        - 'rsi': Reactive strength index chart (drop jumps only)
        - 'ground_reaction_forces': Force-time curves for all jumps
        """
        figures: dict[str, go.Figure | dict[str, go.Figure]] = {}

        # Create elevation figure if we have any jumps
        all_jumps = (
            test.squat_jumps
            + test.counter_movement_jumps
            + test.drop_jumps
            + test.repeated_jumps
        )

        if not all_jumps:
            return figures

        # Extract elevation data from summary
        summary = self.summary
        if summary is not None and not summary.empty:
            # Create a simple bar chart for elevation
            fig = go.Figure()

            # Group by jump type
            for jump_type in summary['type'].unique():
                type_data = summary[summary['type'] == jump_type]
                if 'elevation (cm)' in type_data.columns:
                    elevations = type_data['elevation (cm)'].dropna()
                    if len(elevations) > 0:
                        fig.add_trace(go.Bar(
                            name=jump_type,
                            x=[f"{jump_type} {i+1}" for i in range(len(elevations))],
                            y=elevations.values,
                        ))

            fig.update_layout(
                title="Jump Height (Elevation)",
                xaxis_title="Jump",
                yaxis_title="Elevation (cm)",
                showlegend=True,
            )
            figures['elevation'] = fig

        # Create force-time figure
        analytics = self.analytics
        if analytics is not None and not analytics.empty:
            # Find force columns
            force_cols = [c for c in analytics.columns if 'force' in c.lower()]

            if force_cols and 'time_s' in analytics.columns:
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=["Ground Reaction Forces"]
                )

                # Plot each jump
                for _, group in analytics.groupby(['type', 'jump', 'side']):
                    jump_type = group['type'].iloc[0]
                    jump_num = group['jump'].iloc[0]
                    side = group['side'].iloc[0]

                    # Use first force column found
                    force_col = force_cols[0]

                    fig.add_trace(
                        go.Scatter(
                            x=group['time_s'],
                            y=group[force_col],
                            mode='lines',
                            name=f"{jump_type} #{jump_num} ({side})",
                        ),
                        row=1, col=1
                    )

                fig.update_xaxes(title_text="Time (s)", row=1, col=1)
                fig.update_yaxes(title_text="Force (N)", row=1, col=1)
                fig.update_layout(height=500, showlegend=True)

                figures['ground_reaction_forces'] = fig

        # Create kinematics figure (angles + force background)
        kinematics_fig = self._get_kinematics_figure(all_jumps)
        if kinematics_fig is not None:
            figures['kinematics'] = kinematics_fig

        return figures

    def _normalize_to_101_points(self, signal):
        """
        Normalize a timeseries signal to 101 points (0-100%) using cubic spline interpolation.

        Parameters
        ----------
        signal : Signal1D or array-like
            Input signal to normalize.

        Returns
        -------
        np.ndarray
            Normalized signal with 101 points.
        """
        if signal is None:
            return np.full(101, np.nan)

        data = signal.to_numpy().flatten() if hasattr(signal, 'to_numpy') else np.array(signal).flatten()

        if len(data) == 0:
            return np.full(101, np.nan)

        if len(data) < 4:
            # Not enough points for cubic spline, use linear interpolation
            old_x = np.linspace(0, 100, len(data))
            new_x = np.linspace(0, 100, 101)
            normalized = np.interp(new_x, old_x, data)
            return normalized

        # Use cubic spline interpolation for smoother results
        # cubicspline_interp with nsamp generates evenly spaced points
        normalized = cubicspline_interp(data, nsamp=101)

        return normalized

    def _get_kinematics_figure(self, jumps: list) -> go.Figure | None:
        """
        Generate kinematics figure with joint angles and force background.

        Creates a 3x2 subplot grid:
        - Left column: left side angles and force
        - Right column: right side angles and force
        - Row 1: Hip flexion/extension
        - Row 2: Knee flexion/extension
        - Row 3: Ankle flexion/extension

        Force is shown on secondary y-axis with low opacity (background).
        Angles are shown on primary y-axis with normal visibility.

        Parameters
        ----------
        jumps : list
            List of SingleJump or DropJump objects.

        Returns
        -------
        go.Figure or None
            Plotly figure with subplots, or None if insufficient data.
        """
        if not jumps:
            return None

        # Collect normalized data for left and right sides
        left_forces = []
        right_forces = []
        left_hip_angles = []
        right_hip_angles = []
        left_knee_angles = []
        right_knee_angles = []
        left_ankle_angles = []
        right_ankle_angles = []

        for jump in jumps:
            contact = jump.contact_phase
            if contact is None or len(contact) == 0:
                continue

            # Get vertical force for left and right
            left_fp = contact.get('left_foot_ground_reaction_force')
            right_fp = contact.get('right_foot_ground_reaction_force')

            if left_fp is not None:
                try:
                    left_force = left_fp.force[contact.vertical_axis]
                    left_forces.append(self._normalize_to_101_points(left_force))
                except (AttributeError, KeyError):
                    pass

            if right_fp is not None:
                try:
                    right_force = right_fp.force[contact.vertical_axis]
                    right_forces.append(self._normalize_to_101_points(right_force))
                except (AttributeError, KeyError):
                    pass

            # Get angles
            try:
                left_hip = contact.left_hip_flexionextension
                if left_hip is not None:
                    left_hip_angles.append(self._normalize_to_101_points(left_hip))
            except AttributeError:
                pass

            try:
                right_hip = contact.right_hip_flexionextension
                if right_hip is not None:
                    right_hip_angles.append(self._normalize_to_101_points(right_hip))
            except AttributeError:
                pass

            try:
                left_knee = contact.left_knee_flexionextension
                if left_knee is not None:
                    left_knee_angles.append(self._normalize_to_101_points(left_knee))
            except AttributeError:
                pass

            try:
                right_knee = contact.right_knee_flexionextension
                if right_knee is not None:
                    right_knee_angles.append(self._normalize_to_101_points(right_knee))
            except AttributeError:
                pass

            try:
                left_ankle = contact.left_ankle_flexionextension
                if left_ankle is not None:
                    left_ankle_angles.append(self._normalize_to_101_points(left_ankle))
            except AttributeError:
                pass

            try:
                right_ankle = contact.right_ankle_flexionextension
                if right_ankle is not None:
                    right_ankle_angles.append(self._normalize_to_101_points(right_ankle))
            except AttributeError:
                pass

        # Check if we have any data
        if not any([left_forces, right_forces, left_hip_angles, right_hip_angles,
                    left_knee_angles, right_knee_angles, left_ankle_angles, right_ankle_angles]):
            return None

        # Create subplots: 3 rows (hip, knee, ankle), 2 columns (left, right)
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Left Hip Flexion/Extension', 'Right Hip Flexion/Extension',
                'Left Knee Flexion/Extension', 'Right Knee Flexion/Extension',
                'Left Ankle Flexion/Extension', 'Right Ankle Flexion/Extension'
            ],
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}],
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.08
        )

        x_norm = np.linspace(0, 100, 101)

        # Helper to add traces
        def add_angle_traces(angles_list, row, col, name):
            if angles_list:
                mean_angle = np.nanmean(angles_list, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=x_norm,
                        y=mean_angle,
                        mode='lines',
                        name=f'{name} Angle',
                        line=dict(width=2.5, color='blue'),
                        showlegend=False,
                    ),
                    row=row, col=col, secondary_y=False
                )

        def add_force_traces(forces_list, row, col, name):
            if forces_list:
                mean_force = np.nanmean(forces_list, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=x_norm,
                        y=mean_force,
                        mode='lines',
                        name=f'{name} Force',
                        line=dict(width=1.5, color='gray'),
                        opacity=0.3,
                        showlegend=False,
                    ),
                    row=row, col=col, secondary_y=True
                )

        # Row 1: Hip
        add_angle_traces(left_hip_angles, 1, 1, 'Left Hip')
        add_force_traces(left_forces, 1, 1, 'Left')
        add_angle_traces(right_hip_angles, 1, 2, 'Right Hip')
        add_force_traces(right_forces, 1, 2, 'Right')

        # Row 2: Knee
        add_angle_traces(left_knee_angles, 2, 1, 'Left Knee')
        add_force_traces(left_forces, 2, 1, 'Left')
        add_angle_traces(right_knee_angles, 2, 2, 'Right Knee')
        add_force_traces(right_forces, 2, 2, 'Right')

        # Row 3: Ankle
        add_angle_traces(left_ankle_angles, 3, 1, 'Left Ankle')
        add_force_traces(left_forces, 3, 1, 'Left')
        add_angle_traces(right_ankle_angles, 3, 2, 'Right Ankle')
        add_force_traces(right_forces, 3, 2, 'Right')

        # Update axes labels
        for row in range(1, 4):
            for col in range(1, 3):
                # Primary y-axis (angles)
                fig.update_yaxes(title_text="Angle (°)", row=row, col=col, secondary_y=False)
                # Secondary y-axis (force)
                fig.update_yaxes(title_text="Force (N)", row=row, col=col, secondary_y=True)
                # X-axis
                if row == 3:  # Only bottom row
                    fig.update_xaxes(title_text="Contact Phase (%)", row=row, col=col)
                else:
                    fig.update_xaxes(row=row, col=col)

        fig.update_layout(
            height=900,
            title_text="Joint Kinematics During Contact Phase (Mean)",
            showlegend=False,
        )

        return fig


__all__ = ["JumpTestResults"]
