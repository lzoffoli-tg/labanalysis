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
        time = jump.contact_phase.index
        return int(round((time[-1] - time[0]) * 1000))

    def _get_jump_flight_time_ms(self, jump: SingleJump | DropJump):
        time = jump.flight_phase.index
        return int(round((time[-1] - time[0]) * 1000))

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




__all__ = ["JumpTestResults"]
