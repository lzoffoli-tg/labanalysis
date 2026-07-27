"""
Drop jump exercise analysis module.

This module provides the DropJump class for analyzing plyometric drop jumps
from elevated surfaces. Emphasizes reactive strength index (RSI) and fast
stretch-shortening cycle performance evaluation.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ...constants import MINIMUM_CONTACT_FORCE_N
from ...signalprocessing import continuous_batches
from ...records.body import WholeBody
from ...records import ForcePlatform
from ...timeseries import Signal1D, Signal3D, EMGSignal, Point3D
from .counter_movement_jump import CounterMovementJump


class DropJump(CounterMovementJump):
    """
    Drop jump exercise for plyometric assessment and reactive strength analysis.

    DropJump extends CounterMovementJump to model drop jumps from elevated surfaces,
    adding landing phase detection and specific metrics for reactive strength
    index (RSI) and fast stretch-shortening cycle performance. The class
    automatically identifies the box drop landing, subsequent ground contact,
    and explosive propulsion phases.

    Drop jumps assess the neuromuscular system's ability to rapidly switch
    from eccentric (landing) to concentric (propulsion) muscle actions,
    measuring reactive strength and elastic energy utilization.

    Parameters
    ----------
    bodymass_kg : float
        Participant's body mass in kilograms.
    box_height_cm : float
        Height of the drop box in centimeters. Used for protocol documentation
        and performance interpretation.
    left_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for left foot. Default is None.
    right_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for right foot. Default is None.
    vertical_axis : str, optional
        Name of vertical axis in force data. Default is "Y".
    anteroposterior_axis : str, optional
        Name of anteroposterior axis in force data. Default is "X".
    **signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional biomechanical signals (markers, EMG, etc.).

    Attributes
    ----------
    box_height_cm : float
        Drop box height in centimeters.
    landing_phase : Record
        Data segment from box drop landing to end of initial ground contact.
    contact_phase : Record
        Data segment from landing to takeoff (full ground contact).
    flight_phase : Record
        Data segment during aerial phase after propulsion.
    reactive_strength_index : float
        RSI = jump_height / contact_time (unitless performance metric).

    Properties (Inherited from CounterMovementJump)
    --------------------------------------
    bodymass_kg : float
        Participant's body mass.
    side : str
        Jump execution side ("bilateral", "left", or "right").
    contact_time : float
        Ground contact duration in seconds.
    flight_time : float
        Aerial phase duration in seconds.
    jump_height : float
        Vertical jump height in centimeters.
    takeoff_velocity : float
        Vertical takeoff velocity in m/s.

    Methods
    -------
    copy()
        Return independent copy of the drop jump.
    from_tdf(file, bodymass_kg, box_height_cm, ...)
        Load drop jump from BTS TDF file.

    Notes
    -----
    Phase Detection:
    - Landing phase: Identified as force > 30N occurring before main contact phase
    - Contact phase: Continuous ground contact from landing to takeoff
    - Flight phase: Period with force < 30N after takeoff

    Performance Metrics:
    - RSI (Reactive Strength Index): Primary metric for drop jump performance,
      calculated as jump_height_cm / contact_time_s. Higher RSI indicates
      better reactive strength and elastic energy utilization.
    - Optimal box height: Typically 20-40cm for most athletes; height where
      RSI is maximized represents optimal drop height for individual.

    Applications:
    - Plyometric training assessment
    - Return-to-sport testing after lower limb injury
    - Explosive strength monitoring in power athletes
    - Stretch-shortening cycle function evaluation

    Examples
    --------
    >>> import labanalysis as laban
    >>>
    >>> # Load drop jump from 40cm box
    >>> dj = laban.DropJump.from_tdf(
    ...     file="dj_40cm.tdf",
    ...     bodymass_kg=75.0,
    ...     box_height_cm=40.0,
    ...     left_foot_ground_reaction_force="left_fp",
    ...     right_foot_ground_reaction_force="right_fp"
    ... )
    >>>
    >>> # Key performance metrics
    >>> print(f"Box height: {dj.box_height_cm} cm")
    >>> print(f"Contact time: {dj.contact_time*1000:.0f} ms")
    >>> print(f"Jump height: {dj.jump_height:.1f} cm")
    >>> print(f"RSI: {dj.reactive_strength_index:.2f}")
    >>>
    >>> # Analyze landing phase
    >>> landing_duration_ms = dj.landing_phase.duration * 1000
    >>> print(f"Landing phase: {landing_duration_ms:.0f} ms")

    See Also
    --------
    SingleJump : Base class for single jump analysis.
    RepeatedJumps : Continuous jumping for fatigue analysis.
    JumpTest : Complete jump testing protocol.
    """

    def __init__(
        self,
        box_height_cm: float,
        bodymass_kg: float | None,
        side: Literal["bilateral", "left", "right"] | None = None,
        free_hands: bool = False,
        left_hand_ground_reaction_force: ForcePlatform | None = None,
        right_hand_ground_reaction_force: ForcePlatform | None = None,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        left_heel: Point3D | None = None,
        right_heel: Point3D | None = None,
        left_toe: Point3D | None = None,
        right_toe: Point3D | None = None,
        left_first_metatarsal_head: Point3D | None = None,
        left_fifth_metatarsal_head: Point3D | None = None,
        right_first_metatarsal_head: Point3D | None = None,
        right_fifth_metatarsal_head: Point3D | None = None,
        left_ankle_medial: Point3D | None = None,
        left_ankle_lateral: Point3D | None = None,
        right_ankle_medial: Point3D | None = None,
        right_ankle_lateral: Point3D | None = None,
        left_knee_medial: Point3D | None = None,
        left_knee_lateral: Point3D | None = None,
        right_knee_medial: Point3D | None = None,
        right_knee_lateral: Point3D | None = None,
        right_trochanter: Point3D | None = None,
        left_trochanter: Point3D | None = None,
        left_asis: Point3D | None = None,
        right_asis: Point3D | None = None,
        left_psis: Point3D | None = None,
        right_psis: Point3D | None = None,
        left_shoulder_anterior: Point3D | None = None,
        left_shoulder_posterior: Point3D | None = None,
        left_acromion: Point3D | None = None,
        right_shoulder_anterior: Point3D | None = None,
        right_shoulder_posterior: Point3D | None = None,
        right_acromion: Point3D | None = None,
        left_elbow_medial: Point3D | None = None,
        left_elbow_lateral: Point3D | None = None,
        right_elbow_medial: Point3D | None = None,
        right_elbow_lateral: Point3D | None = None,
        left_wrist_medial: Point3D | None = None,
        left_wrist_lateral: Point3D | None = None,
        right_wrist_medial: Point3D | None = None,
        right_wrist_lateral: Point3D | None = None,
        s2: Point3D | None = None,
        l2: Point3D | None = None,
        c7: Point3D | None = None,
        t5: Point3D | None = None,
        sc: Point3D | None = None,
        head_anterior: Point3D | None = None,
        head_posterior: Point3D | None = None,
        head_left: Point3D | None = None,
        head_right: Point3D | None = None,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a DropJump object.

        See class docstring for detailed parameter descriptions.
        """
        super().__init__(
            bodymass_kg=bodymass_kg,
            side=side,
            free_hands=free_hands,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            left_heel=left_heel,
            right_heel=right_heel,
            left_toe=left_toe,
            right_toe=right_toe,
            left_first_metatarsal_head=left_first_metatarsal_head,
            left_fifth_metatarsal_head=left_fifth_metatarsal_head,
            right_first_metatarsal_head=right_first_metatarsal_head,
            right_fifth_metatarsal_head=right_fifth_metatarsal_head,
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral,
            right_ankle_medial=right_ankle_medial,
            right_ankle_lateral=right_ankle_lateral,
            left_knee_medial=left_knee_medial,
            left_knee_lateral=left_knee_lateral,
            right_knee_medial=right_knee_medial,
            right_knee_lateral=right_knee_lateral,
            left_trochanter=left_trochanter,
            right_trochanter=right_trochanter,
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
            left_shoulder_anterior=left_shoulder_anterior,
            left_shoulder_posterior=left_shoulder_posterior,
            left_acromion=left_acromion,
            right_shoulder_anterior=right_shoulder_anterior,
            right_shoulder_posterior=right_shoulder_posterior,
            right_acromion=right_acromion,
            left_elbow_medial=left_elbow_medial,
            left_elbow_lateral=left_elbow_lateral,
            right_elbow_medial=right_elbow_medial,
            right_elbow_lateral=right_elbow_lateral,
            left_wrist_medial=left_wrist_medial,
            left_wrist_lateral=left_wrist_lateral,
            right_wrist_medial=right_wrist_medial,
            right_wrist_lateral=right_wrist_lateral,
            s2=s2,
            l2=l2,
            c7=c7,
            t5=t5,
            sc=sc,
            head_anterior=head_anterior,
            head_posterior=head_posterior,
            head_left=head_left,
            head_right=head_right,
            **signals,
        )
        self.set_box_height_cm(box_height_cm)

    def set_box_height_cm(self, box_height_cm: float):
        """
        Set the box height in centimeters.

        Parameters
        ----------
        box_height_cm : float or int
            Height of the drop box in centimeters. Must be positive.

        Raises
        ------
        ValueError
            If box_height_cm is not a float or int.
        """
        # check box height
        if not isinstance(box_height_cm, (float, int)):
            raise ValueError("box_height_cm must be a float or int")
        self._box_height_cm = float(box_height_cm)

    @property
    def box_height_cm(self):
        """
        Get the drop box height.

        Returns
        -------
        float
            Height of the drop box in centimeters.

        Notes
        -----
        Box height is a critical parameter for drop jump interpretation:
        - Higher boxes increase impact forces and SSC demands
        - Optimal height maximizes RSI (reactive strength index)
        - Heights above individual optimum reduce performance

        See Also
        --------
        set_box_height_cm : Set the box height value.
        reactive_strength_index : Performance metric influenced by box height.
        """
        return self._box_height_cm

    @property
    def landing_phase(self):
        """
        Get the landing phase of the drop jump.

        Returns
        -------
        WholeBody or None
            Data segment from box drop landing to end of initial ground contact,
            or None if counter-movement phase cannot be determined.

        Notes
        -----
        The landing phase occurs before the main contact phase and represents
        the initial impact absorption after dropping from the box. This phase
        is critical for understanding:
        - Impact forces during box drop landing
        - Eccentric loading prior to main propulsion
        - Landing technique and shock absorption strategies

        Detection algorithm:
        1. Identify counter-movement phase start (main contact start)
        2. Landing phase = from initial data to counter-movement start

        Raises
        ------
        RuntimeError
            If no landing phase is found (counter-movement starts at data start).

        See Also
        --------
        counter_movement_phase : Main eccentric phase after landing.
        contact_phase : Full ground contact from landing to takeoff.
        """
        cmp = self.counter_movement_phase
        if cmp is None:
            return None
        t1 = cmp.index[0]
        t0 = self.index[0]
        if t0 >= t1:
            raise RuntimeError("no landing phase was found")

        return WholeBody(
            **{
                k: v.copy().loc[(v.index >= t0) & (v.index < t1)]
                for k, v in self.items()
            }
        )

    @property
    def reactive_strength_index(self):
        """
        Calculate reactive strength index (RSI).

        Returns
        -------
        float or None
            RSI in cm/s (jump height in cm / contact time in seconds),
            or None if jump height or contact time cannot be determined.

        Notes
        -----
        RSI is the primary performance metric for drop jumps, representing
        the ability to rapidly produce force during fast SSC actions:

            RSI = jump_height (cm) / contact_time (s)

        Higher RSI indicates:
        - Superior reactive strength
        - Better elastic energy utilization
        - More effective fast SSC performance

        RSI is sensitive to both jump height (performance) and contact time
        (speed), making it more comprehensive than jump height alone for
        plyometric assessment.

        Examples
        --------
        >>> dj = DropJump.from_tdf("trial.tdf", bodymass_kg=75, box_height_cm=40)
        >>> rsi = dj.reactive_strength_index
        >>> print(f"RSI: {rsi:.2f} cm/s")
        RSI: 2.35 cm/s

        See Also
        --------
        jump_height : Numerator of RSI calculation.
        contact_time : Denominator of RSI calculation.
        box_height_cm : Drop height affecting RSI.
        """
        return self.elevation / self.contact_time * 100

    @property
    def output_metrics(self):
        new = pd.DataFrame(
            [
                {
                    "type": self.name,
                    "free hands": self.free_hands,
                    "side": self.side,
                    "metric": "contact time",
                    "unit": "ms",
                    "value": self.contact_time * 1000,
                },
                {
                    "type": self.name,
                    "free hands": self.free_hands,
                    "side": self.side,
                    "metric": "reactive strength index",
                    "unit": "cm/s",
                    "value": self.reactive_strength_index,
                },
            ]
        )
        out = pd.concat([super().output_metrics, new], ignore_index=True)
        out.insert(1, "box height", self.box_height_cm)
        out.loc[out.index, "free hands"] = self.free_hands
        out.loc[out.index, "side"] = self.side
        out.loc[out.index, "type"] = self.name
        return out.sort_values(["metric", "side", "free hands", "box height"])

    @classmethod
    def from_tdf(
        cls,
        filename: str | Path,
        box_height_cm: float,
        bodymass_kg: float | int | None,
        free_hands: bool = False,
        side: Literal["bilateral", "left", "right"] | None = None,
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_heel: str | None = None,
        right_heel: str | None = None,
        left_toe: str | None = None,
        right_toe: str | None = None,
        left_first_metatarsal_head: str | None = None,
        left_fifth_metatarsal_head: str | None = None,
        right_first_metatarsal_head: str | None = None,
        right_fifth_metatarsal_head: str | None = None,
        left_ankle_medial: str | None = None,
        left_ankle_lateral: str | None = None,
        right_ankle_medial: str | None = None,
        right_ankle_lateral: str | None = None,
        left_knee_medial: str | None = None,
        left_knee_lateral: str | None = None,
        right_knee_medial: str | None = None,
        right_knee_lateral: str | None = None,
        right_trochanter: str | None = None,
        left_trochanter: str | None = None,
        left_asis: str | None = None,
        right_asis: str | None = None,
        left_psis: str | None = None,
        right_psis: str | None = None,
        left_shoulder_anterior: str | None = None,
        left_shoulder_posterior: str | None = None,
        left_acromion: str | None = None,
        right_shoulder_anterior: str | None = None,
        right_shoulder_posterior: str | None = None,
        right_acromion: str | None = None,
        left_elbow_medial: str | None = None,
        left_elbow_lateral: str | None = None,
        right_elbow_medial: str | None = None,
        right_elbow_lateral: str | None = None,
        left_wrist_medial: str | None = None,
        left_wrist_lateral: str | None = None,
        right_wrist_medial: str | None = None,
        right_wrist_lateral: str | None = None,
        s2: str | None = None,
        l2: str | None = None,
        c7: str | None = None,
        t5: str | None = None,
        sc: str | None = None,
        head_anterior: str | None = None,
        head_posterior: str | None = None,
        head_left: str | None = None,
        head_right: str | None = None,
    ):
        """
        Create DropJump instance from BTS Bioengineering TDF file.

        Parameters
        ----------
        filename : str or Path
            Path to the TDF file containing drop jump trial data.
        box_height_cm : float
            Height of the drop box in centimeters (required parameter).
        bodymass_kg : float, int, or None
            Participant's body mass in kilograms. Required for kinetic analysis.
        left_foot_ground_reaction_force : str or None
            Label for left foot force platform in TDF file.
        right_foot_ground_reaction_force : str or None
            Label for right foot force platform in TDF file.
        free_hands : bool, optional
            Whether hands were free during the jump. Default is False.
        left_hand_ground_reaction_force : str or None, optional
            Label for left hand force platform in TDF file. Default is None.
        right_hand_ground_reaction_force : str or None, optional
            Label for right hand force platform in TDF file. Default is None.
        **marker_labels : str or None, optional
            Labels for anatomical markers in TDF file (same as SquatJump).

        Returns
        -------
        DropJump
            Initialized DropJump instance with data loaded from TDF file.

        Notes
        -----
        Drop jump protocol considerations:
        - Box height should be recorded accurately for RSI interpretation
        - Multiple trials at different heights help identify optimal drop height
        - Participant should step off box (not jump up first)
        - Landing technique affects impact forces and performance

        Standard protocol variations:
        - Depth jump: Emphasis on maximum height
        - Drop jump: Emphasis on minimum contact time
        - Reactive drop jump: Emphasis on RSI (balanced height/time)

        See Also
        --------
        CounterMovementJump.from_tdf : Parent class TDF loading.
        reactive_strength_index : Primary performance metric.

        Examples
        --------
        >>> # Single drop height
        >>> dj = DropJump.from_tdf(
        ...     "dj_30cm.tdf",
        ...     box_height_cm=30,
        ...     bodymass_kg=75,
        ...     left_foot_ground_reaction_force="FP1",
        ...     right_foot_ground_reaction_force="FP2",
        ...     s2="S2"
        ... )
        >>> print(f"RSI at 30cm: {dj.reactive_strength_index:.2f}")

        >>> # Multiple heights for optimal height determination
        >>> heights = [20, 30, 40, 50]
        >>> for h in heights:
        ...     dj = DropJump.from_tdf(f"dj_{h}cm.tdf", box_height_cm=h, ...)
        ...     print(f"{h}cm: RSI = {dj.reactive_strength_index:.2f}")
        """
        record = WholeBody.from_tdf(
            filename,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            left_heel=left_heel,
            right_heel=right_heel,
            left_toe=left_toe,
            right_toe=right_toe,
            left_first_metatarsal_head=left_first_metatarsal_head,
            left_fifth_metatarsal_head=left_fifth_metatarsal_head,
            right_first_metatarsal_head=right_first_metatarsal_head,
            right_fifth_metatarsal_head=right_fifth_metatarsal_head,
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral,
            right_ankle_medial=right_ankle_medial,
            right_ankle_lateral=right_ankle_lateral,
            left_knee_medial=left_knee_medial,
            left_knee_lateral=left_knee_lateral,
            right_knee_medial=right_knee_medial,
            right_knee_lateral=right_knee_lateral,
            left_trochanter=left_trochanter,
            right_trochanter=right_trochanter,
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
            left_shoulder_anterior=left_shoulder_anterior,
            left_shoulder_posterior=left_shoulder_posterior,
            left_acromion=left_acromion,
            right_shoulder_anterior=right_shoulder_anterior,
            right_shoulder_posterior=right_shoulder_posterior,
            right_acromion=right_acromion,
            left_elbow_medial=left_elbow_medial,
            left_elbow_lateral=left_elbow_lateral,
            right_elbow_medial=right_elbow_medial,
            right_elbow_lateral=right_elbow_lateral,
            left_wrist_medial=left_wrist_medial,
            left_wrist_lateral=left_wrist_lateral,
            right_wrist_medial=right_wrist_medial,
            right_wrist_lateral=right_wrist_lateral,
            s2=s2,
            l2=l2,
            c7=c7,
            t5=t5,
            sc=sc,
            head_anterior=head_anterior,
            head_posterior=head_posterior,
            head_left=head_left,
            head_right=head_right,
        )
        return cls(
            box_height_cm=box_height_cm,
            bodymass_kg=bodymass_kg,
            free_hands=free_hands,
            side=side,
            **{i: v for i, v in record.items()},  # type: ignore
        )


__all__ = ["DropJump"]
