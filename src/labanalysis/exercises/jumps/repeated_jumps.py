"""Repeated jumps exercise module."""

import numpy as np

from ...constants import MINIMUM_CONTACT_FORCE_N, MINIMUM_FLIGHT_TIME_S
from ...signalprocessing import continuous_batches, fillna, butterworth_filt
from ...records.body import WholeBody
from ...records import ForcePlatform
from ...timeseries import Signal1D, Signal3D, EMGSignal, Point3D
from .single_jump import SingleJump


class RepeatedJumps(WholeBody):
    """
    Repeated jumps exercise for fatigue assessment and endurance evaluation.

    RepeatedJumps analyzes continuous jumping sequences to assess neuromuscular
    fatigue, mechanical power decline, and coordination degradation over multiple
    jump repetitions. The class automatically detects individual jumps from
    continuous data and tracks performance changes across the sequence.

    The exercise is used for:
    - Anaerobic fatigue profiling
    - Jump endurance assessment
    - Training load monitoring
    - Return-to-play testing
    - Coordination stability evaluation

    Parameters
    ----------
    bodymass_kg : float
        Participant's body mass in kilograms.
    straight_legs : bool, optional
        Whether jumps performed with straight legs (true) or knee flexion allowed
        (false). Affects jump mechanics and fatigue patterns. Default is False.
    free_hands : bool, optional
        Whether arm swing is allowed (true) or hands on hips (false).
        Default is False.
    excluded_jumps : list of int, optional
        Indices of jumps to exclude from analysis (e.g., failed attempts).
        Default is empty list.
    left_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for left foot. Default is None.
    right_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for right foot. Default is None.
    left_hand_ground_reaction_force : ForcePlatform, optional
        Force platform for left hand (for prone jump variations). Default is None.
    right_hand_ground_reaction_force : ForcePlatform, optional
        Force platform for right hand (for prone jump variations). Default is None.
    **markers : Point3D
        Biomechanical markers for full-body kinematics (same as WholeBody).

    Attributes
    ----------
    bodymass_kg : float
        Participant's body mass.
    straight_legs : bool
        Whether straight-leg jumps protocol.
    free_hands : bool
        Whether arm swing allowed.
    excluded_jumps : list of int
        Indices of excluded jumps.
    jumps : list of SingleJump
        Individual jump objects extracted from continuous data.
    fatigue_index : float
        Performance decline percentage: 100 * (best - worst) / best.

    Properties
    ----------
    bodymass_kg : float
        Participant body mass in kg.
    straight_legs : bool
        Straight-leg jump protocol flag.
    free_hands : bool
        Free arm swing flag.
    excluded_jumps : list of int
        Excluded jump indices.
    jumps : list of SingleJump
        Detected individual jumps (excluding specified indices).

    Methods
    -------
    copy()
        Return independent copy of repeated jumps.
    from_tdf(file, bodymass_kg, ...)
        Load repeated jumps from BTS TDF file.
    set_bodymass_kg(bodymass_kg)
        Set participant body mass.
    set_straight_legs(straight)
        Set straight-leg protocol flag.
    set_free_hands(free)
        Set free arm swing flag.
    set_excluded_jumps(jumps)
        Set indices of jumps to exclude from analysis.

    Notes
    -----
    Jump Detection:
    Individual jumps automatically detected from continuous force data using:
    - Contact detection: Vertical force > 30N threshold
    - Flight detection: Minimum flight time > 50ms
    - Separation: Adjacent jumps split at force minima

    Performance Metrics (per jump):
    - Jump height (cm): Calculated from flight time
    - Contact time (ms): Ground contact duration
    - Flight time (ms): Aerial phase duration
    - Reactive strength index: height / contact_time
    - Peak power (W): Maximum mechanical power output

    Fatigue Analysis:
    - Track jump height decline over sequence
    - Monitor contact time increase (fatigue sign)
    - Calculate fatigue index: (max - min) / max * 100
    - Identify drop-off point (>10% decline threshold)

    Protocol Variations:
    - Straight-leg jumps: Emphasize ankle plantarflexors, minimize knee contribution
    - Bent-knee jumps: Allow full lower-limb coordination
    - Hands-on-hips: Isolate lower limb contribution
    - Free arm swing: Maximize jump performance

    Examples
    --------
    >>> import labanalysis as laban
    >>>
    >>> # Load 15-second repeated jump test
    >>> rj = laban.RepeatedJumps.from_tdf(
    ...     file="repeated_jumps_15s.tdf",
    ...     bodymass_kg=75.0,
    ...     straight_legs=False,
    ...     free_hands=False,
    ...     left_foot_ground_reaction_force="left_fp"
    ... )
    >>>
    >>> # Access individual jumps
    >>> print(f"Total jumps: {len(rj.jumps)}")
    >>> for i, jump in enumerate(rj.jumps, 1):
    ...     print(f"Jump {i}: {jump.jump_height:.1f} cm, CT: {jump.contact_time*1000:.0f} ms")
    >>>
    >>> # Fatigue analysis
    >>> heights = [j.jump_height for j in rj.jumps]
    >>> fatigue_index = (max(heights) - min(heights)) / max(heights) * 100
    >>> print(f"Fatigue index: {fatigue_index:.1f}%")
    >>>
    >>> # Exclude failed jump (e.g., jump 5)
    >>> rj.set_excluded_jumps([4])  # 0-indexed
    >>> print(f"Valid jumps: {len(rj.jumps)}")

    See Also
    --------
    SingleJump : Base class for single jump analysis.
    DropJump : Drop jump for plyometric assessment.
    JumpTest : Complete jump testing protocol.
    WholeBody : Full-body biomechanical model.
    """

    @property
    def bodymass_kg(self):
        """
        Returns the subject's body mass in kilograms.

        Returns
        -------
        float
            Body mass in kg.
        """
        return self._bodymass_kg

    def set_bodymass_kg(self, bodymass_kg: float):
        if not isinstance(bodymass_kg, (float, int)) or bodymass_kg <= 0:
            raise ValueError("bodymass_kg must be a float or int > 0.")
        self._bodymass_kg = bodymass_kg

    @property
    def excluded_jumps(self):
        return self._excluded_jumps

    def set_excluded_jumps(self, jumps: list[int]):
        if not isinstance(jumps, list) or not all([isinstance(i, int) for i in jumps]):
            raise ValueError("jumps must be a list of int")
        self._excluded_jumps = jumps

    @property
    def straight_legs(self):
        return self._straight_legs

    def set_straight_legs(self, straight: bool):
        if not isinstance(straight, bool):
            raise ValueError("straight must be True or False.")
        self._straight_legs = straight

    @property
    def free_hands(self):
        return self._free_hands

    def set_free_hands(self, free: bool):
        if not isinstance(free, bool):
            raise ValueError("free must be True or False.")
        self._free_hands = free

    def __init__(
        self,
        bodymass_kg: float,
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
        exclude_jumps: list[int] = [0, -1],
        straight_legs: bool = False,
        free_hands: bool = False,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """Initialize a RepeatedJumps object."""
        all_signals = {
            **signals,
            **dict(
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
            ),
        }
        if (
            left_foot_ground_reaction_force is None
            and right_foot_ground_reaction_force is None
        ):
            raise ValueError(
                "at least one of 'left_foot_ground_reaction_force' or "
                "'right_foot_ground_reaction_force' must be ForcePlatform instances."
            )
        super().__init__(**{i: v for i, v in all_signals.items() if v is not None})  # type: ignore
        self.set_bodymass_kg(bodymass_kg)
        self.set_excluded_jumps(exclude_jumps)
        self.set_straight_legs(straight_legs)
        self.set_free_hands(free_hands)

    @classmethod
    def from_tdf(
        cls,
        file: str,
        bodymass_kg: float | int,
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
        exclude_jumps: list[int] = [],
        straight_legs: bool = False,
        free_hands: bool = False,
    ):
        """Create a RepeatedJumps object from a TDF file."""
        if (
            left_foot_ground_reaction_force is None
            and right_foot_ground_reaction_force is None
        ):
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
            bodymass_kg=bodymass_kg,
            exclude_jumps=exclude_jumps,
            straight_legs=straight_legs,
            free_hands=free_hands,
            **record._data,
        )

    def _get_constructor_args(self):
        """
        Return custom constructor arguments for loc/iloc slicing.

        Returns dict with custom attributes needed to reconstruct RepeatedJumps
        after slicing operations.
        """
        return {
            "bodymass_kg": self.bodymass_kg,
            "free_hands": self.free_hands,
            "exclude_jumps": self.excluded_jumps,
            "straight_legs": self.straight_legs,
        }

    def copy(self):
        return RepeatedJumps(
            bodymass_kg=self.bodymass_kg,
            free_hands=self.free_hands,
            exclude_jumps=self.excluded_jumps,
            straight_legs=self.straight_legs,
            **{i: v.copy() for i, v in self.items()},
        )

    @property
    def jumps(self):
        vgrf = self.resultant_force.copy()
        time = vgrf.index
        vgrf = vgrf.force[self.vertical_axis].to_numpy().flatten()
        vgrf = fillna(arr=vgrf, value=0).flatten()
        fsamp = float(1 / np.mean(np.diff(time)))
        vgrf = butterworth_filt(
            arr=vgrf,
            fsamp=fsamp,
            fcut=50.0,
            order=4,
            ftype="lowpass",
            phase_corrected=True,
        )

        # get the batches with grf lower than 30N (i.e flight phases)
        flight_batches = continuous_batches(vgrf <= float(MINIMUM_CONTACT_FORCE_N))

        # remove those batches resulting in too short flight phases
        # (i.e. ~0.2s flight time)
        fsamp = 1 / np.mean(np.diff(time))
        min_samples = int(round(MINIMUM_FLIGHT_TIME_S * fsamp))
        flight_batches = [i for i in flight_batches if len(i) >= min_samples]

        # ensure that the first jump does not start with a flight
        if flight_batches[0][0] == 0:
            flight_batches = flight_batches[1:]

        # ensure that the last jump does not end in flight
        if flight_batches[-1][-1] == len(vgrf) - 1:
            flight_batches = flight_batches[:-1]

        # get the contact peaks
        contact_peaks = []
        for b0, b1 in zip(flight_batches[:-1], flight_batches[1:]):
            contact_peaks.append(np.argmax(vgrf[b0[-1] : b1[0]]) + b0[-1])
        contact_peaks.append(
            np.argmax(vgrf[flight_batches[-1][-1] :]) + flight_batches[-1][-1]
        )

        # get the contact starts
        contact_starts = []
        contact_batches = continuous_batches(vgrf > float(MINIMUM_CONTACT_FORCE_N))
        for i, batch in enumerate(flight_batches):
            pre = [c for c in contact_batches if c[-1] <= batch[0]]
            if len(pre) == 0:
                raise RuntimeError("no contact phase found")
            pre = pre[-1]
            contact_starts.append(pre[0])

        # separate each jump
        jumps: list[SingleJump] = []
        for i, (pre, post) in enumerate(zip(contact_starts, contact_peaks)):
            start = float(time[pre])
            stop = float(time[post])
            jumps.append(
                SingleJump(
                    bodymass_kg=self.bodymass_kg,
                    straight_legs=self.straight_legs,
                    free_hands=self.free_hands,
                    **{i: v.copy().loc[start:stop, :] for i, v in self.items()},
                )
            )

        # exclude unnecessary jumps
        sanitized_indices = [
            i + (0 if i >= 0 else len(jumps)) for i in self.excluded_jumps
        ]
        sanitized_indices = sorted(set(sanitized_indices), reverse=True)
        for i in sanitized_indices:
            jumps.pop(i)

        return jumps


__all__ = ["RepeatedJumps"]
