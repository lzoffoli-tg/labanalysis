"""Jump test results implementation."""

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from ...utils import hex_to_rgba

from ..test_protocol import TestProtocol

from ...constants import G, RANK_3COLORS, RANK_5COLORS
from ...signalprocessing import continuous_batches, cubicspline_interp
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

    def __init__(self, test: TestProtocol, include_emg: bool):
        from .jump_test import JumpTest

        if not isinstance(test, JumpTest):
            raise ValueError("'test' must be an JumpTest instance.")
        super().__init__(test, include_emg)

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
                ctime = jump.contact_time
                if ctime is not None:
                    ctime = int(round(1000 * ctime))
                ftime = jump.flight_time
                if ftime is not None:
                    ftime = int(round(1000 * ftime))
                tov = jump.takeoff_velocity
                if tov is not None:
                    tov = int(round(tov * 100, 0))
                elevation = jump.jump_height
                if elevation is not None:
                    elevation = int(round(elevation * 100, 0))
                for side in sides:
                    out.loc["takeoff velocity (cm/s)", side] = tov
                    out.loc["elevation (cm)", side] = elevation
                    out.loc["flight time (ms)", side] = ftime
                    out.loc["contact time (ms)", side] = ctime
                    if ftime is not None and ctime is not None:
                        out.loc["flight-to-contact ratio", side] = float(
                            round(ftime / ctime, 2)
                        )

                # Calculate RSI for drop jumps
                if isinstance(jump, DropJump):
                    rsi = jump.reactive_strength_index
                    if rsi:
                        rsi = float(round(rsi, 1))
                    for side in sides:
                        out.loc["reactive strength index", side] = rsi

                # convert index in column
                out.insert(0, "parameter", out.index)
                out.reset_index(inplace=True, drop=True)

                # add jump conditions
                out.insert(0, "free hands", jump.free_hands)
                if isinstance(jump, DropJump):
                    out.insert(0, "box height (cm)", jump.box_height)

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

            # calculate symmetry
            def get_symmetry(x: pd.Series):
                if x.left and x.right:
                    return float(100 * (1 - abs(x.right - x.left) / (x.left + x.right)))
                else:
                    return None

            result.insert(
                result.shape[1], "symmetry (%)", result.apply(get_symmetry, axis=1)
            )

            return result

        # Process each jump type
        summary_parts = []

        if test.squat_jumps:
            sj_summary = _get_jumps_summary_table(
                test.squat_jumps,
                "squat jump",
            )
            summary_parts.append(sj_summary)

        if test.counter_movement_jumps:
            cmj_summary = _get_jumps_summary_table(
                test.counter_movement_jumps,
                "counter movement jump",
            )
            summary_parts.append(cmj_summary)

        if test.drop_jumps:
            dj_summary = _get_jumps_summary_table(
                test.drop_jumps,  # type: ignore
                "drop jump",
            )
            summary_parts.append(dj_summary)

        if test.repeated_jumps:
            rj_summary = _get_jumps_summary_table(
                test.repeated_jumps,
                "repeated jump",
            )
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

                # Get jump dataframe
                cf = jump.contact_phase
                ff = jump.flight_phase
                if cf is not None and ff is not None:
                    if not self.include_emg:
                        emg_keys = list(cf.emgsignals.keys())
                        if emg_keys:
                            cf.drop(emg_keys, inplace=True)
                            ff.drop(emg_keys, inplace=True)
                    cf = cf.to_dataframe()
                    ff = ff.to_dataframe()
                    cf.insert(0, "phase", "contact")
                    ff.insert(0, "phase", "flight")
                    df = pd.concat([cf, ff], axis=0)
                    df.insert(1, "time_s", df.index)

                    # Add metadata columns
                    df.insert(0, "side", jump.side)
                    df.insert(0, "free hands", jump.free_hands)
                    if isinstance(jump, DropJump):
                        df.insert(0, "box height (cm)", jump.box_height_cm)
                    df.insert(0, "jump", jump_idx)
                    df.insert(0, "type", jump_name)

                    # reset index
                    df.reset_index(inplace=True, drop=True)

                    # append
                    analytics_parts.append(df)

        if analytics_parts:
            return pd.concat(analytics_parts, ignore_index=True)
        else:
            return pd.DataFrame()

    def _interpolate_to_101_points(self, signal):
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
        data = (
            signal.to_numpy().flatten()
            if hasattr(signal, "to_numpy")
            else np.array(signal).flatten()
        )

        # Use cubic spline interpolation for smoother results
        # cubicspline_interp with nsamp generates evenly spaced points
        normalized = cubicspline_interp(data, nsamp=101)

        return normalized

    def _get_kinematics_figure(self, jumps: list[SingleJump | DropJump]):
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
            left_fp = contact.left_foot_ground_reaction_force
            right_fp = contact.right_foot_ground_reaction_force

            if left_fp is not None:
                left_force = left_fp.force[contact.vertical_axis]
                left_forces.append(self._interpolate_to_101_points(left_force))

            if right_fp is not None:
                right_force = right_fp.force[contact.vertical_axis]
                right_forces.append(self._interpolate_to_101_points(right_force))

            # Get angles
            left_hip = contact.left_hip
            if left_hip is not None:
                left_hip_angles.append(
                    self._interpolate_to_101_points(left_hip.flexionextension)
                )

            right_hip = contact.right_hip
            if right_hip is not None:
                right_hip_angles.append(
                    self._interpolate_to_101_points(right_hip.flexionextension)
                )

            left_knee = contact.left_knee
            if left_knee is not None:
                left_knee_angles.append(
                    self._interpolate_to_101_points(left_knee.flexionextension)
                )

            right_knee = contact.right_knee
            if right_knee is not None:
                right_knee_angles.append(
                    self._interpolate_to_101_points(right_knee.flexionextension)
                )

            left_ankle = contact.left_ankle
            if left_ankle is not None:
                left_ankle_angles.append(
                    self._interpolate_to_101_points(left_ankle.flexionextension)
                )

            right_ankle = contact.right_ankle
            if right_ankle is not None:
                right_ankle_angles.append(
                    self._interpolate_to_101_points(right_ankle.flexionextension)
                )

        # Check if we have any data
        if not any(
            [
                left_forces,
                right_forces,
                left_hip_angles,
                right_hip_angles,
                left_knee_angles,
                right_knee_angles,
                left_ankle_angles,
                right_ankle_angles,
            ]
        ):
            return None

        # Create subplots: 3 rows (hip, knee, ankle), 2 columns (left, right)
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Left Hip Flexion/Extension",
                "Right Hip Flexion/Extension",
                "Left Knee Flexion/Extension",
                "Right Knee Flexion/Extension",
                "Left Ankle Flexion/Extension",
                "Right Ankle Flexion/Extension",
            ],
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}],
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.08,
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
                        mode="lines",
                        name=f"{name} Angle",
                        line=dict(width=2.5, color="blue"),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                    secondary_y=False,
                )

        def add_force_traces(forces_list, row, col, name):
            if forces_list:
                mean_force = np.nanmean(forces_list, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=x_norm,
                        y=mean_force,
                        mode="lines",
                        name=f"{name} Force",
                        line=dict(width=1.5, color="gray"),
                        opacity=0.3,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )

        # Row 1: Hip
        add_angle_traces(left_hip_angles, 1, 1, "Left Hip")
        add_force_traces(left_forces, 1, 1, "Left")
        add_angle_traces(right_hip_angles, 1, 2, "Right Hip")
        add_force_traces(right_forces, 1, 2, "Right")

        # Row 2: Knee
        add_angle_traces(left_knee_angles, 2, 1, "Left Knee")
        add_force_traces(left_forces, 2, 1, "Left")
        add_angle_traces(right_knee_angles, 2, 2, "Right Knee")
        add_force_traces(right_forces, 2, 2, "Right")

        # Row 3: Ankle
        add_angle_traces(left_ankle_angles, 3, 1, "Left Ankle")
        add_force_traces(left_forces, 3, 1, "Left")
        add_angle_traces(right_ankle_angles, 3, 2, "Right Ankle")
        add_force_traces(right_forces, 3, 2, "Right")

        # Update axes labels
        for row in range(1, 4):
            for col in range(1, 3):
                # Primary y-axis (angles)
                fig.update_yaxes(
                    title_text="Angle (°)", row=row, col=col, secondary_y=False
                )
                # Secondary y-axis (force)
                fig.update_yaxes(
                    title_text="Force (N)", row=row, col=col, secondary_y=True
                )
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

    def _get_grf_figure(self, test: "JumpTest"):

        def get_data(jump: SingleJump | DropJump, n: int, typed: str):
            grf = jump.copy().resultant_force.copy()
            grf = grf.force.to_dataframe()[[jump.vertical_axis]]  # type: ignore
            grf.columns = pd.Index(["grf"])
            start = jump.flight_phase
            if start is None:
                return None
            start = start.index[0]
            grf.insert(0, "time", grf.index - start)
            grf.insert(0, "jump", n)
            grf.insert(0, "side", jump.side)
            type_name = [typed]
            if isinstance(jump, DropJump):
                type_name[0] += f" ({jump.box_height_cm:0.0f}cm)"
            if jump.free_hands:
                type_name.append("free hands")
            if jump.straight_legs:
                type_name.append("straight legs")
            grf.insert(0, "type", "-".join(type_name))
            grf = grf.loc[(grf.time > -1) & (grf.time < 2)]
            return grf

        data = []
        for i, jump in enumerate(test.squat_jumps):
            data.append(get_data(jump, i + 1, "Squat Jump"))
        for i, jump in enumerate(test.counter_movement_jumps):
            data.append(get_data(jump, i + 1, "Counter Movement Jump"))
        for i, jump in enumerate(test.drop_jumps):
            data.append(get_data(jump, i + 1, "Drop Jump"))
        for i, jump in enumerate(test.repeated_jumps):
            data.append(get_data(jump, i + 1, "Single Leg Jump"))
        df = pd.concat(data, ignore_index=True)

        fig = px.line(
            data_frame=df,
            x="time",
            y="grf",
            color="jump",
            facet_row="type",
            facet_col="side",
            facet_col_spacing=0.05,
            facet_row_spacing=0.05,
            template="plotly_white",
        )
        fig.update_traces(opacity=0.5)
        fig.update_yaxes(
            matches=None,
            showticklabels=True,
            title="Ground Reaction Force (N)",
        )
        fig.update_xaxes(
            matches=None,
            showticklabels=True,
            title="Time (s)",
        )
        fig.add_hline(
            y=test.participant.weight * G,
            line_color="red",
            line_dash="dash",
            line_width=1,
            name="Weight (N)",
            showlegend=True,
        )
        fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
        return fig

    def _get_data_and_norms(
        self,
        metric: str,
        test: "JumpTest",
        bilateral_is_unique: bool = True,
        ranks: dict[str, str] = RANK_5COLORS,
        symmetric_ranks: bool = False,
        reversed_ranks: bool = False,
    ):

        # retrieve the data of the required metric from summary
        metric_df = self.summary
        if not isinstance(metric_df, pd.DataFrame):
            raise ValueError(f"summary was expected to be a pandas.DataFrame. {type(metric_df)} was found.") 
        params: list[str] = metric_df.parameter.to_list()

        idx = [i for i, v in enumerate(params) if v.endswith(metric)]
        metric_df = metric_df.iloc[idx]
        # metric_df = metric_df.loc[metric_df["free hands"] == free_hands]
        metric_df.drop(["symmetry (%)"], axis=1, inplace=True)
        metric_df = metric_df.melt(
            id_vars=["type", "side", "parameter", "free hands", "jump"],
            var_name="limb",
            value_name="value",
        )
        if bilateral_is_unique:
            idx = (metric_df.side != "bilateral") | (metric_df.limb == "left")
            metric_df = metric_df.loc[idx]
            new_limbs = metric_df[["side", "limb"]].apply(
                lambda x: x.side if x.side == "bilateral" else x.limb,
                axis=1,
            )
            metric_df.loc[metric_df.index, "limb"] = new_limbs
        metric_df.reset_index(drop=True, inplace=True)

        # get the data sorted according to the subplots to be rendered
        data: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
        for (t, s, f), dfr in metric_df.groupby(["type", "side", "free hands"]):
            key = (f"{t} - free hands" if f else str(t), str(s))
            val: dict[str, dict[str, list[float]]] = {}
            dfr = dfr.loc[dfr['free hands'] == f]
            for param, dfp in dfr.groupby("parameter"):
                dct: dict[str, list[float]] = {}
                for side, dfs in dfp.groupby("limb"):
                    k = str(side)
                    v = dfs.sort_values("jump").value.to_numpy().flatten().tolist()
                    dct[k] = v
                val[str(param)] = dct
            data[key] = val

        # get the normative data sorted according to the subplots to be rendered
        norms: dict[
            tuple[str, str], tuple[list[float], list[float], list[str], list[str]]
        ] = {}
        combs = metric_df[["type", "side", "free hands"]].drop_duplicates().values.tolist()
        if not test.normative_data.empty:
            gender = test.participant.gender
            if gender is None:
                raise ValueError("Normative Data require gender being specified.")
            gender = gender.lower()[0]
            norm = test.normative_data.copy()
            params: list[str] = norm.parameter.to_list()
            idx = [i for i, v in enumerate(params) if v.endswith(metric)]
            norm = norm.iloc[idx]
            types = norm["type"].str.lower().tolist()
            types = [t.lower().rsplit(" (", 1)[0] for t in types]
            sides = norm["side"].str.lower().tolist()
            genders = [i.lower()[0] for i in norm["gender"]]
            for t, s, f in combs:
                k = t.lower().rsplit(" (", 1)[0]
                if f:
                    k += " - free hands"
                types_idx = [k == v for v in types]
                types_idx = np.array(types_idx)
                sides_idx = np.array([s in v for v in sides])
                gender_idx = np.array([gender == v for v in genders])
                mask = types_idx & sides_idx & gender_idx
                tnorm = norm.loc[mask]
                if tnorm.shape[0] > 1:
                    msg = "Multiple normative values found for jump elevation."
                    raise ValueError(msg)
                if not tnorm.empty:
                    avg = float(tnorm["mean"].to_numpy()[0])
                    std = float(tnorm["std"].to_numpy()[0])
                    rank_clrs = list(ranks.values())
                    rank_lbls = list(ranks.keys())
                    if reversed_ranks:
                        rank_clrs = rank_clrs[::-1]
                        rank_lbls = rank_lbls[::-1]
                    n_vals = len(ranks)
                    if symmetric_ranks:
                        rank_clrs = rank_clrs[::-1] + rank_clrs
                        rank_lbls = rank_lbls[::-1] + rank_lbls
                        rank_vals = np.arange(n_vals + 1)
                    else:
                        if n_vals % 2 == 1:
                            rank_vals = np.arange((n_vals + 1) // 2) + 1
                        else:
                            rank_vals = np.arange((n_vals + 2) // 2)
                    rank_vals = np.concatenate([rank_vals, -rank_vals]) * std + avg
                    rank_vals = np.unique(rank_vals)[::-1]
                    rank_lows = rank_vals[1:].copy().tolist()
                    rank_tops = rank_vals[:-1].copy().tolist()
                    norms[(k, s)] = (rank_lows, rank_tops, rank_lbls, rank_clrs)

        return data, norms

    def _get_performance_figure(
        self,
        performance_data: dict[str, list[float]],
        performance_norms: tuple[list[float], list[float], list[str], list[str]],
        performance_unit: str,
        performance_metric: str,
        balance_data: list[float] | None = None,
        balance_norms: (
            tuple[list[float], list[float], list[str], list[str]] | None
        ) = None,
    ):

        # generate the figure
        subplot_titles = [performance_metric.capitalize()]
        if balance_data is not None:
            subplot_titles.append("Left/Right Imbalance")
        fig = make_subplots(
            rows=1,
            cols=1 if balance_data is None else 2,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.01,
        )
        fig.update_layout(
            template="plotly_white",
            legend=dict(title_text="Legend"),
            width=1000,
            height=400,
            bargroupgap=0.25,
            #margin = dict(t = 100, r = 100, b=50, l=100), 
        )
        fig.update_xaxes(
            showgrid=False,
            showline=False,
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
        )

        # get the normative data if available
        if performance_norms is not None:
            rank_lows, rank_tops, rank_lbls, rank_clrs = performance_norms
            rank_lows = np.array(rank_lows)
            rank_tops = np.array(rank_tops)
        else:
            rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

        # plot the bars representing the performance value
        yvals = []
        colors_plotted = []
        values = []
        for k, (side, performances) in enumerate(performance_data.items()):
            for j, y in enumerate(performances):
                value = round(y, 1)
                values.append(value)

                # if normative data are available get the main bar color as
                # the color of the rank achieved by the actual value.
                # Otherwise, use the color of the side with which the jump
                # has been performed.
                if len(rank_tops) > 0:
                    idx = np.where(rank_tops >= value)[0]
                    idx = idx[-1] if len(idx) > 0 else 0  # (len(rank_clrs) - 1)
                    color = rank_clrs[idx]
                else:
                    color = SIDE_COLORS[side]  # type: ignore

                # update the y-axis range values
                yvals += rank_lows.tolist() + rank_tops.tolist() + [value]

                # plot the bar
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Bar(
                        x=[k + 1],
                        y=[value],
                        text=[f"Jump {j+1}<br>{value} {performance_unit}"],
                        textposition="outside",
                        textangle=0,
                        showlegend=(j == 0)
                        and performance_norms is None
                        and len(performance_data) > 1,
                        marker_color=[color],
                        marker_line_color=["black"],
                        name=side.capitalize(),
                        legendgroup="Limb",
                        legendgrouptitle_text="Limb",
                        offsetgroup=str(j + 1),
                    ),
                )

        # update the yaxes
        yrange = [np.min(yvals) * 0.9, np.max(yvals) * 1.2]
        fig.update_yaxes(row=1, col=1, range=yrange)

        # update the xaxes
        fig.update_xaxes(
            col=1,
            row=1,
            range=[0, len(performance_data) + 1],
            showticklabels=False,
        )
        if len(performance_data) > 1:
            fig.update_xaxes(
                col=1,
                row=1,
                showticklabels=True,
                tickvals=np.arange(len(performance_data)) + 1,
                tickmode="array",
                ticktext=[str(i).capitalize() for i in list(performance_data.keys())],
            )
        
        # plot average line
        avg = round(np.mean(values), 1)
        fig.add_hline(
            y=avg, 
            col = 1, #type: ignore
            line_dash = "dash", 
            line_color="red", 
            line_width = 1.5, 
            opacity = 0.7,
        )
        fig.add_annotation(
            col = 1, 
            row = 1,  
            x = 0, 
            y = avg, 
            text = f"media<br>{avg} cm", 
            font = dict(color = "red"), 
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            xref="x",
            yref="y",
        )
        
        # plot the norms as colored boxes behind the bars
        zipped = zip(rank_lows, rank_tops, rank_lbls, rank_clrs)
        for rlow, rtop, rlbl, rclr in zipped:
            if rlow == np.min(rank_lows) and rlow > yrange[0]:
                rlow = yrange[0]
            if rtop == np.max(rank_tops) and rtop < yrange[1]:
                rtop = yrange[1]
            fig.add_shape(
                type="rect",
                x0=0,
                x1=len(performance_data) + 1,
                y0=rlow,
                y1=rtop,
                line_width=0,
                fillcolor=hex_to_rgba(rclr, 0.25),
                layer="below",
                name=rlbl.capitalize(),
                legendgroup="Rank",
                legendgrouptitle_text="Rank",
                showlegend=rclr not in colors_plotted,
                col=1,
                row=1,
            )
            if rtop < np.max(rank_tops):
                fig.add_annotation(
                    x=len(performance_data) + 1,
                    y=rtop,
                    text=f"{rtop:0.1f} {performance_unit}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(color=rclr),
                    valign="top",
                    yshift=0,
                    name=rlbl,
                    col=1,  # type: ignore
                    row=1,  # type: ignore
                )

            # ensure that the legend is plotted once
            colors_plotted.append(rclr)

        # plot balance
        if balance_data is not None:

            # get the normative data if available
            if balance_norms is not None:
                rank_lows, rank_tops, rank_lbls, rank_clrs = balance_norms
                rank_lows = np.asarray(rank_lows)
                rank_tops = np.asarray(rank_tops)
            else:
                rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

            # plot the balance of each single jump
            for j, val in enumerate(balance_data):

                # get the bar color as the color of the rank achieved by the
                # jump height. Otherwise, use the color of the side with which the
                # jump has been performed.
                idx = np.where(rank_tops >= abs(val))[0]
                idx = idx[0] if len(idx) > 0 else (len(rank_clrs) - 1)
                color = rank_clrs[idx]

                # get the value and label
                value = max(-50, min(50, val))
                lbl = f"{abs(val):0.1f}%" if -50 <= val <= 50 else ">50.0%"
                lbl = f"Jump {j+1} ({lbl})"

                # plot the bar
                fig.add_trace(
                    col=2,
                    row=1,
                    trace=go.Bar(
                        y=[len(balance_data) - 1 - j],
                        x=[value],
                        text=[lbl],
                        textposition="outside",
                        textangle=0,
                        showlegend=False,
                        marker_color=[color],
                        marker_line_color=["black"],
                        name=f"Jump {j+1}",
                        legendgroup="Jump",
                        legendgrouptitle_text="Jump",
                        orientation="h",
                    ),
                )

            # update rank extremes
            rank_tops[-1] = 120

            # plot the norms as colored boxes behind the bars
            zipped = zip(rank_lows, rank_tops, rank_lbls, rank_clrs)
            for rlow, rtop, rlbl, rclr in zipped:
                fig.add_shape(
                    type="rect",
                    y0=-1,
                    y1=len(balance_data),
                    x0=rlow,
                    x1=rtop,
                    line_width=0,
                    fillcolor=hex_to_rgba(rclr, 0.25),
                    layer="below",
                    name=rlbl.capitalize(),
                    legendgroup="Rank",
                    legendgrouptitle_text="Rank",
                    showlegend=color not in colors_plotted,
                    col=2,
                    row=1,
                )
                fig.add_shape(
                    type="rect",
                    y0=-1,
                    y1=len(balance_data),
                    x0=-rlow,
                    x1=-rtop,
                    line_width=0,
                    fillcolor=hex_to_rgba(rclr, 0.25),
                    layer="below",
                    name=rlbl.capitalize(),
                    legendgroup="Rank",
                    legendgrouptitle_text="Rank",
                    showlegend=False,
                    col=2,
                    row=1,
                )

                # ensure that the legend is plotted once
                colors_plotted.append(rclr)

            # plot the zero line
            fig.add_vline(
                col=2,  # type: ignore
                row=1,  # type: ignore
                x=0,
                line_width=2,
                line_dash="solid",
                showlegend=False,
            )

            # update the xaxes
            xrange = [-np.max(rank_tops), np.max(rank_tops)]
            fig.update_xaxes(
                col=2,
                row=1,
                range=xrange,
                tickmode="array",
                tickvals=[xrange[0] * 0.9, 0, xrange[1] * 0.9],
                ticktext=["Left", "Perfect<br>Balance", "Right"],
                ticklen=0,
            )

            # update the yaxes
            fig.update_yaxes(
                col=2,
                row=1,
                range=[-1, len(balance_data)],
            )

        # check
        return fig

    def _get_muscle_activation_figure(
        self,
        data: dict[str, dict[str, list[float]]],
        norms: tuple[list[float], list[float], list[str], list[str]] | None,
        unit: str,
    ):

        # prepare the figure
        muscles = np.unique(list(data.keys())).tolist()
        sides = np.unique(
            [s.capitalize() for m in data.values() for s in m.keys()]
        ).tolist()
        fig = make_subplots(
            cols=len(sides),
            rows=1,
            horizontal_spacing=0.1,
            subplot_titles=sides,
        )
        fig.update_layout(
            template="plotly_white",
            height=200 * len(muscles),
            width=1200,
            legend=dict(title=dict(text="Legend")),
            bargroupgap=0.1,
        )

        # get the normative data if available
        if norms is not None:
            rank_lows, rank_tops, rank_lbls, rank_clrs = norms
            rank_lows = np.array(rank_lows)
            rank_tops = np.array(rank_tops)
        else:
            rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

        # plot the data
        color_plotted = []
        xvals = {}
        for row, muscle in enumerate(muscles):
            side_dct = data[muscle]
            for col, (side, jump_values) in enumerate(side_dct.items()):

                # plot the jumps
                for n, x in enumerate(jump_values):

                    # update the xrange values
                    if side not in xvals:
                        xvals[side] = []
                    xvals[side].append(x)

                    # if normative data are available get the main bar color as
                    # the color of the rank achieved by the actual value.
                    # Otherwise, use the color of the side with which the jump
                    # has been performed.
                    value = round(x, 1)
                    if len(rank_tops) > 0:
                        idx = np.where(rank_tops >= value)[0]
                        idx = idx[-1] if len(idx) > 0 else 0  # (len(rank_clrs) - 1)
                        color = rank_clrs[idx]
                    else:
                        color = SIDE_COLORS[side]  # type: ignore

                    # get the label
                    lbl = f"{x:0.1f}{unit}"
                    lbl = f"Jump {n+1} ({lbl})"

                    # plot the bar
                    fig.add_trace(
                        row=1,
                        col=col + 1,
                        trace=go.Bar(
                            y=[row],
                            x=[x],
                            text=[lbl],
                            textposition="outside",
                            textangle=0,
                            showlegend=color not in color_plotted and norms is None,
                            marker_color=[color],
                            marker_line_color=["black"],
                            orientation="h",
                            offsetgroup=str(n),
                            name=side,
                            legendgroup="Side",
                            legendgrouptitle_text="Side",
                        ),
                    )

                    # prevent the same color to be plotted again
                    if norms is None:
                        color_plotted.append(color)

        # plot the norms (if available)
        for col, (side, xv) in enumerate(xvals.items()):
            if norms is not None:
                r_lows = rank_lows.copy()
                r_tops = rank_tops.copy()
                r_lows[-1] = min(r_lows[-1], np.min(xv))
                r_lows[-1] *= 1.1 if r_lows[-1] < 0 else 0.9
                r_lows[-1] = min(0, r_lows[-1])
                r_tops[0] = max(r_tops[0], np.max(xv) * 2)
                zipped = zip(r_lows, r_tops, rank_lbls, rank_clrs)
                for rlow, rtop, rlbl, rclr in zipped:
                    fig.add_shape(
                        type="rect",
                        y0=-1,
                        y1=len(muscles),
                        x0=rlow,
                        x1=rtop,
                        line_width=0,
                        fillcolor=hex_to_rgba(rclr, 0.25),
                        layer="below",
                        name=rlbl,
                        legendgroup="Rank",
                        legendgrouptitle_text="Rank",
                        showlegend=rlbl not in color_plotted,
                        row=1,
                        col=col + 1,
                    )

                    # ensure that each rank level is plotted once
                    color_plotted.append(rlbl)

                # update the xrange
                xrange = [min(0, np.min(r_lows)), np.max(r_tops)]

            else:
                xrange = [
                    min(0, np.min(xv) * (1.1 if np.min(xv) < 0 else 0.9)),
                    np.max(xv) * 2,
                ]

            # update x-axis
            tickvals = r_lows[:-1]
            ticktext = [f"{i:.0f}{unit}" for i in tickvals]
            fig.update_xaxes(
                col=col + 1,
                range=xrange,
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=True,
                tickmode="array",
                ticktext=ticktext,
                tickvals=tickvals,
                tickangle=0,
            )

            # update the y-axis
            fig.update_yaxes(
                col=col + 1,
                tickvals=np.arange(len(muscles)).tolist(),
                tickangle=0,
                tickmode="array",
                ticktext=[m.replace(" ", "<br>") for m in muscles],
                showticklabels=True,
                range=[-1, len(muscles)],
            )

            # plot the zero lines
            fig.add_vline(
                x=0,
                line_width=2,
                line_dash="solid",
                showlegend=False,
            )

        return fig

    def _get_elevation_figure(self, test: "JumpTest"):

        # retrieve the jump height data
        performance_data, performance_norms = self._get_data_and_norms(
            "elevation (cm)",
            test
        )

        # since we have just one parameter (elevation), we remove the layer
        # defining the parameters for each key
        performance_data = {i: list(v.values())[0] for i, v in performance_data.items()}

        # retrieve the force balance data
        balance_df:pd.DataFrame = self.summary.copy()  # type: ignore
        balance_df.loc[balance_df.index, "type"] = balance_df.apply(lambda x: f"{x["type"]} - free hands" if x['free hands'] else x["type"], axis=1)
        balance_df = balance_df.loc[balance_df.parameter == "vertical force (N)"]
        balance_data: dict[tuple[str, str], list[float]] = {}
        for t, s in performance_data.keys():
            if s == "bilateral":
                balance = balance_df.loc[balance_df["type"] == t].copy()
                balance = balance.loc[balance["side"] == s]
                balance = balance.loc[balance["free hands"] == ("free hands" in t)]
                balance.sort_values("jump", inplace=True)
                balance = 100 * (balance["right"] / (balance["right"]+balance["left"])).to_numpy().flatten() - 50
                balance_data[(t, s)] = balance.tolist()

        # prepare the balance norms
        vals = np.array([0, 10, 20, 30, 40, 100])
        lows = vals[:-1].copy().tolist()
        tops = vals[1:].copy().tolist()
        clrs = list(RANK_5COLORS.values())
        lbls = list(RANK_5COLORS.keys())
        balance_norms = {(t, s): (lows, tops, lbls, clrs) for (t, s) in balance_data}

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in performance_data.keys():
            p_data = performance_data.get((t, s))
            p_norms = performance_norms.get((t, s))
            b_data = balance_data.get((t, s))
            b_norms = balance_norms.get((t, s))
            titolo = f"{t}-{s}".capitalize()
            if p_data is not None:
                fig = self._get_performance_figure(
                    p_data,
                    p_norms,  # type: ignore
                    "cm",
                    "Elevation",
                    b_data,
                    b_norms,
                )
                fig.update_layout(title=titolo)
                figures[titolo] = fig

        return figures

    def _get_contact_time_figure(self, test: "JumpTest"):

        # retrieve the contact time data
        performance_data, performance_norms = self._get_data_and_norms(
            "contact time (ms)",
            test,
            reversed_ranks=True,
        )

        # since we have just one parameter (contact time), we remove the layer
        # defining the parameters for each key
        performance_data = {i: list(v.values())[0] for i, v in performance_data.items()}

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in performance_data.keys():
            p_data = performance_data.get((t, s))
            p_norms = performance_norms.get((t, s))
            if p_data is not None:
                fig = self._get_performance_figure(
                    p_data,
                    p_norms,  # type: ignore
                    "ms",
                    "Contact Time",
                )
                title = f"{t}-{s}".capitalize()
                fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_rsi_figure(self, test: "JumpTest"):

        # retrieve the rsi data
        performance_data, performance_norms = self._get_data_and_norms(
            "rsi (cm/s)",
            test
        )

        # since we have just one parameter (rsi), we remove the layer
        # defining the parameters for each key
        performance_data = {i: list(v.values())[0] for i, v in performance_data.items()}

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in performance_data.keys():
            p_data = performance_data.get((t, s))
            p_norms = performance_norms.get((t, s))
            if p_data is not None:
                fig = self._get_performance_figure(
                    p_data,
                    p_norms,  # type: ignore
                    "cm/s",
                    "Reactive Strength Index (RSI)",
                )
                title = f"{t}-{s}".capitalize()
                fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_muscle_activation_ratio_figure(self, test: "JumpTest"):

        # retrieve the activation ratio data
        data_raw, norms = self._get_data_and_norms(
            "activation ratio",
            test,
            False,
            RANK_3COLORS,
            True
        )

        # we turn the name of the parameters layer into the muscle names
        data: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
        for i, v in data_raw.items():
            vals = {}
            for j, k in v.items():
                muscle = j.replace(" activation ratio", "").split(" ")
                muscle = " ".join([l.capitalize() for l in muscle])
                vals[muscle] = k
            data[i] = vals

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in data.keys():
            p_data = data.get((t, s))
            p_norms = norms.get((t, s))
            if p_data is not None:
                fig = self._get_muscle_activation_figure(
                    p_data,
                    p_norms,
                    unit="%",
                )

                # update the x-axis
                fig.update_xaxes(
                    title="Pre-Activation (%)",
                    row=len(fig._grid_ref),  # type: ignore
                )

                # update the title
                title = f"{t}-{s}".capitalize()
                fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_muscle_activation_time_figure(self, test: "JumpTest"):

        # retrieve the activation ratio data
        data_raw, norms = self._get_data_and_norms(
            "activation time (ms)",
            test,
            False,
            RANK_3COLORS,
            True,
        )

        # we turn the name of the parameters layer into the muscle names
        data: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
        for i, v in data_raw.items():
            vals = {}
            for j, k in v.items():
                muscle = j.replace(" activation time (ms)", "").split(" ")
                muscle = " ".join([l.capitalize() for l in muscle])
                vals[muscle] = k
            data[i] = vals

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in data.keys():
            p_data = data.get((t, s))
            p_norms = norms.get((t, s))
            if p_data is not None:
                fig = self._get_muscle_activation_figure(
                    p_data,
                    p_norms,
                    unit="ms",
                )

                # update the x-axis
                fig.update_xaxes(
                    title="Activation time (ms)",
                    row=len(fig._grid_ref),  # type: ignore
                )
                fig.update_xaxes(
                    tickvals=[-200, 200],
                    tickangle=0,
                    tickmode="array",
                    ticktext=["Before<br>contact", "After<br>contact"],
                    showticklabels=True,
                )

                # update the title
                title = f"{t}-{s}".capitalize()
                fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_figures(self, test: "JumpTest"):
        out: dict[str, go.Figure] = {}

        out["ground_reaction_forces"] = self._get_grf_figure(test) 

        out["elevation"] = self._get_elevation_figure(test)

        if len(test.drop_jumps) > 0 or len(test.repeated_jumps) > 0:

            out["contact_time"] = self._get_contact_time_figure(test, False)
            out["contact_time free hands"] = self._get_contact_time_figure(test, True)

            out["rsi"] = self._get_rsi_figure(test, False)
            out["rsi free hands"] = self._get_rsi_figure(test, True)

        if len(test.drop_jumps) > 0 and self.include_emg:

            macr = self._get_muscle_activation_ratio_figure(test, False)
            out["muscle_activation_ratio"] = macr
            macr = self._get_muscle_activation_ratio_figure(test, True)
            out["muscle_activation_ratio free hands"] = macr

            mact = self._get_muscle_activation_time_figure(test, False)
            orut["muscle_activation_time"] = mact
            mact = self._get_muscle_activation_time_figure(test, True)
            out["muscle_activation_time free hands"] = mact
       
        return out


__all__ = ["JumpTestResults"]
