"""Drop jump exercise module."""

import numpy as np

from ..constants import MINIMUM_CONTACT_FORCE_N
from ..signalprocessing import continuous_batches
from ..records.body import WholeBody
from ..records import ForcePlatform
from ..timeseries import Signal1D, Signal3D, EMGSignal, Point3D
from .single_jump import SingleJump


class DropJump(SingleJump):
    """
    Drop jump exercise for plyometric assessment and reactive strength analysis.

    DropJump extends SingleJump to model drop jumps from elevated surfaces,
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
    landing_phase : TimeseriesRecord
        Data segment from box drop landing to end of initial ground contact.
    contact_phase : TimeseriesRecord
        Data segment from landing to takeoff (full ground contact).
    flight_phase : TimeseriesRecord
        Data segment during aerial phase after propulsion.
    reactive_strength_index : float
        RSI = jump_height / contact_time (unitless performance metric).

    Properties (Inherited from SingleJump)
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

    @property
    def landing_phase(self):
        """
        Returns the landing phase of the drop jump.

        Procedure
        ---------
            1. get the batch with grf lower than 30N occurring before the contact phase.
        """

        # # get the time samples corresponding to the start and end of each
        # batch
        start = float(round(self.index[0], 3))
        contact_time_start = float(round(self.contact_phase.index[0], 3))
        stop = float(
            np.round(self.index[np.where(self.index < contact_time_start)[0][-1]], 3)
        )

        # return the landing phase
        signals = {k: v.copy().loc[start:stop, :] for k, v in self.items()}
        return WholeBody(**signals)

    @property
    def flight_phase(self):
        """
        Returns the flight phase of the jump.

        Returns
        -------
        TimeseriesRecord
            Data for the flight phase.

        Procedure
        ---------
            1. get the longest batch with grf lower than 30N.
            2. define 'flight_start' as the first local minima occurring after
            the start of the detected batch.
            3. define 'flight_end' as the last local minima occurring before the
            end of the detected batch.
        """

        # get vertical force
        vgrf = self.resultant_force.copy()
        grfy = vgrf.force.copy()[self.vertical_axis].to_numpy().flatten()
        grft = self.index

        # get contact phases
        contact_batches = continuous_batches(grfy > MINIMUM_CONTACT_FORCE_N)

        # if the jump starts with a contact phase (i.e. the box is on the
        # force platforms) ignore the first contact
        if contact_batches[0][0] == 0:
            contact_batches = contact_batches[1:]

        # ensure that a minimum of 2 contact phases
        if len(contact_batches) < 2:
            raise RuntimeError("No flight phase found.")

        # get the second flight phase
        start = float(round(grft[contact_batches[0][-1] + 1], 3))
        stop = float(round(grft[contact_batches[1][0] - 1], 3))

        # return the landing phase
        signals = {k: v.copy().loc[start:stop, :] for k, v in self.items()}
        return WholeBody(**signals)

    def __init__(
        self,
        box_height_cm: float,
        bodymass_kg: float,
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
        """Initialize a DropJump object."""
        super().__init__(
            bodymass_kg=bodymass_kg,
            free_hands=free_hands,
            straight_legs=False,
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
        """
        # check box height
        if not isinstance(box_height_cm, (float, int)):
            raise ValueError("box_height_cm must be a float or int")
        self._box_height_cm = float(box_height_cm)

    @property
    def box_height_cm(self):
        """
        Returns the box height in centimeters.
        """
        return self._box_height_cm

    @classmethod
    def from_tdf(
        cls,
        file: str,
        box_height_cm: float,
        bodymass_kg: float | int,
        free_hands: bool = False,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
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
        """Create a DropJump object from a TDF file."""
        if left_foot_ground_reaction_force is None and right_foot_ground_reaction_force is None:
            raise ValueError(
                "at least one of left_foot_ground_reaction_force or "
                "right_foot_ground_reaction_force must be provided."
            )
        record = WholeBody.from_tdf(
            file,
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
            **record._data,
        )

    def copy(self):
        return DropJump(
            box_height_cm=self.box_height_cm,
            bodymass_kg=self.bodymass_kg,
            free_hands=self.free_hands,
            **{i: v.copy() for i, v in self.items()},
        )


__all__ = ["DropJump"]
