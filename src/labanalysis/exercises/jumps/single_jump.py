"""
Single vertical jump analysis module.

This module provides the SingleJump class for analyzing individual vertical
jump performance from force platform and motion capture data. SingleJump
extends CounterMovementJump to provide comprehensive jump analysis capabilities
including phase detection, kinetic/kinematic metrics, and reactive strength
index calculation.

SingleJump is primarily used as the base class for individual jumps within
repeated jump protocols, providing full analysis capabilities for each jump
in a continuous sequence.
"""

from pathlib import Path
from typing import Literal

import pandas as pd

from ...records import ForcePlatform
from ...records.body import WholeBody
from ...timeseries import Point3D
from .counter_movement_jump import CounterMovementJump


class SingleJump(CounterMovementJump):
    """
    Single vertical jump analysis from force platform and motion capture data.

    Analyzes individual jump performance by detecting contact and flight phases,
    computing kinetic and kinematic metrics, and estimating jump height using
    multiple methods (flight time, takeoff velocity, marker trajectory).

    SingleJump extends CounterMovementJump, inheriting all countermovement
    analysis capabilities (loading response and propulsion phases, SSC metrics)
    while adding reactive strength index (RSI) calculation. This class is
    primarily used for individual jumps within repeated jump protocols.

    Parameters
    ----------
    bodymass_kg : float or None, optional
        Participant's body mass in kilograms. Required for kinetic metrics
        (takeoff velocity, forces relative to body weight) (default: None).
    side : {"left", "right", "bilateral"} or None, optional
        Jump execution side. If None, automatically determined from available
        force platforms (default: None).
    straight_leg : bool, optional
        Whether the jump uses straight leg technique (ankle-dominant)
        (default: False).
    free_hands : bool, optional
        Whether the jump allows free arm swing (default: True).
    left_foot_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for left foot. Default is None.
    right_foot_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for right foot. Default is None.
    left_hand_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for left hand (if applicable). Default is None.
    right_hand_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for right hand (if applicable). Default is None.
    **markers : Point3D or None, optional
        3D marker trajectories for kinematic analysis. Common markers include:
        s2 (sacrum), left_heel, right_heel, left_toe, right_toe, left_ankle_*,
        right_ankle_*, left_knee_*, right_knee_*, left_trochanter,
        right_trochanter, pelvis landmarks (asis, psis), spine (c7, t5, l2),
        and head markers.
    **kwargs
        Additional keyword arguments passed to WholeBody base class.

    Attributes
    ----------
    bodymass_kg : float or None
        Participant's body mass in kg.
    side : str
        Jump execution side ("left", "right", or "bilateral").
    straight_leg : bool
        Whether straight leg technique is used.
    free_hands : bool
        Whether free arm swing is allowed.
    contact_phase : WholeBody or None
        Data segment from ground contact to takeoff.
    flight_phase : WholeBody or None
        Data segment during aerial phase.
    loading_response_phase : WholeBody or None
        Eccentric phase of countermovement (inherited from CounterMovementJump).
    propulsion_phase : WholeBody or None
        Concentric push-off phase (inherited from CounterMovementJump).
    contact_time : float or None
        Ground contact duration in seconds.
    flight_time : float or None
        Flight phase duration in seconds.
    jump_height : float or None
        Best estimate of jump height in meters.
    jump_height_from_s2 : float or None
        Jump height from S2 marker trajectory (meters).
    jump_height_from_ft : float or None
        Jump height from flight time (meters).
    jump_height_from_tov : float or None
        Jump height from takeoff velocity (meters).
    takeoff_velocity : float or None
        Vertical velocity at takeoff (m/s).
    peak_vertical_force : float or None
        Maximum vertical GRF during contact (N).
    reactive_strength_index : float or None
        RSI in cm/s (jump height / contact time).

    Raises
    ------
    ValueError
        If neither left nor right foot force platform is provided.

    Notes
    -----
    **Phase Detection:**

    - Contact phase: Force >= 30N threshold from first contact to takeoff
    - Flight phase: Force < 30N for minimum duration (typically 0.1s)
    - Longest valid flight phase is selected if multiple are detected
    - Loading response: Eccentric phase (inherited from CounterMovementJump)
    - Propulsion: Concentric push-off (inherited from CounterMovementJump)

    **Jump Height Methods:**

    1. S2 marker: Direct measurement of center of mass elevation
    2. Flight time: h = (g × t²) / 8 (tends to overestimate)
    3. Takeoff velocity: h = v² / (2g) (from force integration)

    The `jump_height` property returns the S2 method if available,
    otherwise the minimum of flight time and takeoff velocity methods
    for a conservative estimate.

    **Reactive Strength Index (RSI):**

    RSI = jump_height (cm) / contact_time (s)

    This metric is particularly relevant for repeated jump protocols to
    assess fatigue-induced changes in reactive strength.

    Examples
    --------
    Create a SingleJump from TDF file:

    >>> jump = SingleJump.from_tdf(
    ...     "trial.tdf",
    ...     bodymass_kg=75,
    ...     side="bilateral",
    ...     left_foot_ground_reaction_force="FP1",
    ...     right_foot_ground_reaction_force="FP2",
    ...     s2="S2"
    ... )
    >>> print(f"Jump height: {jump.jump_height:.3f} m")
    >>> print(f"Contact time: {jump.contact_time:.3f} s")
    >>> print(f"RSI: {jump.reactive_strength_index:.2f} cm/s")

    Access phase data:

    >>> contact = jump.contact_phase
    >>> flight = jump.flight_phase
    >>> loading = jump.loading_response_phase  # eccentric
    >>> propulsion = jump.propulsion_phase  # concentric

    See Also
    --------
    CounterMovementJump : Parent class with SSC analysis.
    SquatJump : Concentric-only jump from static position.
    DropJump : Plyometric jump from elevated surface.
    RepeatedJumps : Continuous jumping sequence using SingleJump.
    WholeBody : Base class for whole-body biomechanical data.
    ForcePlatform : Force platform data structure.
    """

    def __init__(
        self,
        bodymass_kg: float | None = None,
        side: Literal["left", "right", "bilateral"] | None = None,
        straight_leg: bool = False,
        free_hands: bool = True,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        left_hand_ground_reaction_force: ForcePlatform | None = None,
        right_hand_ground_reaction_force: ForcePlatform | None = None,
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
        **kwargs,
    ):
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
            right_trochanter=right_trochanter,
            left_trochanter=left_trochanter,
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
            **kwargs,
        )
        self.set_straight_legs(straight_leg)

    def set_straight_legs(self, value: bool):
        """
        Set whether legs were kept straight during countermovement.

        Parameters
        ----------
        value : bool
            True if legs were kept relatively straight (emphasizing ankle
            plantarflexion), False if normal knee/hip flexion allowed.

        Raises
        ------
        ValueError
            If value is not a boolean.

        Notes
        -----
        Straight-leg jumps reduce contribution from knee extensors and
        emphasize ankle plantarflexors. This variation is sometimes used
        to assess calf muscle power or to modify SSC demands.
        """
        if not isinstance(value, bool):
            raise ValueError("straight_legs must be True or False")
        self._straight_legs = value

    @property
    def straight_legs(self):
        """
        Get whether legs were kept straight during countermovement.

        Returns
        -------
        bool
            True if legs were kept relatively straight, False otherwise.

        See Also
        --------
        set_straight_legs : Set the straight legs status.
        """
        return self._straight_legs

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

        Typical RSI values:
        - <1.0: Untrained/rehabilitation
        - 1.0-2.0: Recreational athletes
        - 2.0-3.0: Trained athletes
        - >3.0: Elite power athletes

        RSI is maximized at an individual-specific optimal drop height.
        Testing across multiple heights (e.g., 20, 30, 40, 50cm) identifies
        this optimum.

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
                    "side": self.side,
                    "metric": "reactive strength index",
                    "unit": "cm/s",
                    "value": self.reactive_strength_index,
                },
            ]
        )
        out = pd.concat([super().output_metrics, new], ignore_index=True)
        out.insert(1, "straight legs", self.straight_legs)
        out.loc[out.index, "free hands"] = self.free_hands
        out.loc[out.index, "side"] = self.side
        out.loc[out.index, "type"] = self.name
        return out.sort_values(["metric", "side", "free hands", "straight legs"])

    @classmethod
    def from_tdf(
        cls,
        filename: str | Path,
        bodymass_kg: float | int,
        side: Literal["left", "right", "bilateral"] | None = None,
        straight_leg: bool = False,
        free_hands: bool = True,
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
        Create SingleJump instance from BTS Bioengineering TDF file.

        Parameters
        ----------
        filename : str or Path
            Path to the TDF file containing jump trial data.
        bodymass_kg : float or int
            Participant's body mass in kilograms (required for kinetic analysis).
        side : {"left", "right", "bilateral"} or None, optional
            Jump execution side. If None, automatically determined from
            available force platform data (default: None).
        straight_leg : bool, optional
            Whether the jump uses straight leg technique (default: False).
        free_hands : bool, optional
            Whether the jump allows free arm swing (default: True).
        left_foot_ground_reaction_force : str or None, optional
            Label for left foot force platform in TDF file. Default is None.
        right_foot_ground_reaction_force : str or None, optional
            Label for right foot force platform in TDF file. Default is None.
        left_hand_ground_reaction_force : str or None, optional
            Label for left hand force platform in TDF file (if applicable).
            Default is None.
        right_hand_ground_reaction_force : str or None, optional
            Label for right hand force platform in TDF file (if applicable).
            Default is None.
        **marker_labels : str or None, optional
            Labels for anatomical markers in TDF file (e.g., s2, left_heel,
            right_knee_medial). See Parameters section for complete list.

        Returns
        -------
        SingleJump
            Initialized SingleJump instance with data loaded from TDF file.

        Notes
        -----
        TDF files are binary files from BTS Bioengineering systems containing
        synchronized force platform and 3D motion capture data.

        At least one force platform (left or right foot) must be specified
        for jump analysis.

        Marker labels should match the marker names used in the TDF file.
        Common markers include:
        - Pelvis: s2, l2, left_asis, right_asis, left_psis, right_psis
        - Lower limb: left_heel, left_toe, left_ankle_*, left_knee_*, left_trochanter
        - Spine: c7, t5, sc
        - Head: head_anterior, head_posterior, head_left, head_right

        See Also
        --------
        WholeBody.from_tdf : Base class TDF loading method.

        Examples
        --------
        >>> jump = SingleJump.from_tdf(
        ...     "trial01.tdf",
        ...     bodymass_kg=75,
        ...     side="bilateral",
        ...     straight_leg=False,
        ...     free_hands=True,
        ...     left_foot_ground_reaction_force="FP1",
        ...     right_foot_ground_reaction_force="FP2",
        ...     s2="S2"
        ... )
        >>> print(f"RSI: {jump.reactive_strength_index:.2f} cm/s")
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
            bodymass_kg=bodymass_kg,
            side=side,
            straight_leg=straight_leg,
            free_hands=free_hands,
            **{i: v for i, v in record.items()},  # type: ignore
        )


__all__ = ["SingleJump"]
