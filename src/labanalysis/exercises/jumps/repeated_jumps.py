"""
Repeated jumps exercise analysis module.

This module provides the RepeatedJumps class for analyzing continuous jumping
sequences to assess neuromuscular fatigue, mechanical power decline, and
coordination degradation across multiple jump repetitions.
"""

from pathlib import Path
from typing import Literal

import numpy as np

from ...constants import MINIMUM_CONTACT_FORCE_N, MINIMUM_FLIGHT_TIME_S
from ...records import ForcePlatform
from ...records.body import WholeBody
from ...signalprocessing import butterworth_filt, continuous_batches
from ...timeseries import EMGSignal, Point3D, Signal1D, Signal3D
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
        """
        Set the participant's body mass.

        Parameters
        ----------
        bodymass_kg : float or int
            Body mass in kilograms. Must be positive.

        Raises
        ------
        ValueError
            If bodymass_kg is not numeric or not positive.
        """
        if not isinstance(bodymass_kg, (float, int)) or bodymass_kg <= 0:
            raise ValueError("bodymass_kg must be a float or int > 0.")
        self._bodymass_kg = bodymass_kg

    @property
    def excluded_jumps(self):
        """
        Get the list of excluded jump indices.

        Returns
        -------
        list of int
            Indices of jumps to exclude from analysis. Supports negative
            indexing (e.g., -1 for last jump).

        See Also
        --------
        set_excluded_jumps : Set excluded jump indices.
        jumps : Access individual jumps (excluding specified indices).
        """
        return self._excluded_jumps

    def set_excluded_jumps(self, jumps: list[int]):
        """
        Set indices of jumps to exclude from analysis.

        Parameters
        ----------
        jumps : list of int
            Indices of jumps to exclude. Supports negative indexing.
            Common exclusions: [0] (warm-up jump), [-1] (incomplete jump),
            [0, -1] (first and last jumps).

        Raises
        ------
        ValueError
            If jumps is not a list of integers.

        Notes
        -----
        Excluded jumps are removed from the `jumps` property but remain
        in the raw continuous data. This allows selective analysis
        excluding failed attempts or protocol variations.

        Examples
        --------
        >>> rj.set_excluded_jumps([0])  # Exclude first (warm-up) jump
        >>> rj.set_excluded_jumps([0, -1])  # Exclude first and last
        >>> rj.set_excluded_jumps([4, 5])  # Exclude specific failed jumps
        """
        if not isinstance(jumps, list) or not all([isinstance(i, int) for i in jumps]):
            raise ValueError("jumps must be a list of int")
        self._excluded_jumps = jumps

    @property
    def straight_legs(self):
        """
        Get whether straight-leg jump protocol was used.

        Returns
        -------
        bool
            True if straight-leg jumps, False if normal knee flexion allowed.

        See Also
        --------
        set_straight_legs : Set straight-leg protocol flag.
        """
        return self._straight_legs

    def set_straight_legs(self, straight: bool):
        """
        Set the straight-leg jump protocol flag.

        Parameters
        ----------
        straight : bool
            True for straight-leg jumps (ankle emphasis), False for
            normal knee flexion allowed.

        Raises
        ------
        ValueError
            If straight is not a boolean.

        Notes
        -----
        Straight-leg repeated jumps emphasize ankle plantarflexor
        endurance and minimize knee extensor contribution, useful for
        calf-specific fatigue assessment.
        """
        if not isinstance(straight, bool):
            raise ValueError("straight must be True or False.")
        self._straight_legs = straight

    @property
    def free_hands(self):
        """
        Get whether free arm swing was allowed.

        Returns
        -------
        bool
            True if arm swing allowed, False if hands restricted.

        See Also
        --------
        set_free_hands : Set free hands protocol flag.
        """
        return self._free_hands

    def set_free_hands(self, free: bool):
        """
        Set the free arm swing protocol flag.

        Parameters
        ----------
        free : bool
            True if arm swing allowed, False if hands restricted
            (e.g., on hips).

        Raises
        ------
        ValueError
            If free is not a boolean.

        Notes
        -----
        Hands-on-hips protocol isolates lower limb contribution and
        is commonly used to standardize repeated jump testing and
        minimize upper body fatigue effects.
        """
        if not isinstance(free, bool):
            raise ValueError("free must be True or False.")
        self._free_hands = free

    def set_side(self, value: Literal["left", "right", "bilateral"] | None):
        """
        Set the side of the jump.

        Parameters
        ----------
        value : Literal["left", "right", "bilateral"] | None
            The side of the jump.

        Raises
        ------
        ValueError
            In case a value different from "left", "right", "bilateral" or None is provided.
        """
        if value is None:
            self._side = value
        if value not in ["left", "right", "bilateral"]:
            raise ValueError("side must be 'left', 'right', 'bilateral' or None")
        self._side = value

    @property
    def side(self):
        """
        Determine jump laterality from available force platforms.

        Returns
        -------
        str
            "left", "right", or "bilateral"
        """

        # use the provided label
        if self._side is not None:
            return self._side

        # evaluate based on available force platforms
        has_left = self.left_foot_ground_reaction_force is not None
        has_right = self.right_foot_ground_reaction_force is not None

        if has_left and has_right:
            return "bilateral"
        elif has_left:
            return "left"
        elif has_right:
            return "right"
        else:
            raise RuntimeError("No force platforms available to determine jump side.")

    def __init__(
        self,
        bodymass_kg: float,
        side: Literal["bilateral", "left", "right"] | None = None,
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
        """
        Initialize a RepeatedJumps instance.

        Parameters
        ----------
        bodymass_kg : float
            Participant's body mass in kilograms (required).
        side : {"bilateral", "left", "right"} or None, optional
            Jump execution side. Default is None (auto-detected).
        exclude_jumps : list of int, optional
            Indices of jumps to exclude from analysis. Default is [0, -1]
            (exclude first and last jumps).
        straight_legs : bool, optional
            Whether straight-leg protocol used. Default is False.
        free_hands : bool, optional
            Whether arm swing allowed. Default is False.
        left_foot_ground_reaction_force : ForcePlatform or None, optional
            Left foot force platform data. Default is None.
        right_foot_ground_reaction_force : ForcePlatform or None, optional
            Right foot force platform data. Default is None.
        **force_platforms : ForcePlatform or None, optional
            Additional force platform data (hands).
        **markers : Point3D or None, optional
            3D marker trajectories for kinematic analysis.
        **signals : Signal1D, Signal3D, EMGSignal, Point3D, or ForcePlatform
            Additional biomechanical signals.

        Notes
        -----
        Default exclusion [0, -1] removes the first jump (warm-up) and last
        jump (potentially incomplete) from analysis, which is standard practice
        for repeated jump testing.

        See Also
        --------
        from_tdf : Create instance from TDF file.
        """
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
        super().__init__(**{i: v for i, v in all_signals.items() if v is not None})  # type: ignore
        self.set_bodymass_kg(bodymass_kg)
        self.set_excluded_jumps(exclude_jumps)
        self.set_straight_legs(straight_legs)
        self.set_free_hands(free_hands)
        self.set_side(side)

    @classmethod
    def from_tdf(
        cls,
        file: str | Path,
        bodymass_kg: float | int,
        side: Literal["bilateral", "left", "right"] | None = None,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
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
        """
        Create RepeatedJumps instance from BTS Bioengineering TDF file.

        Parameters
        ----------
        file : str
            Path to the TDF file containing repeated jump trial data.
        bodymass_kg : float or int
            Participant's body mass in kilograms (required).
        side : {"bilateral", "left", "right"} or None, optional
            Jump execution side. Default is None (auto-detected).
        exclude_jumps : list of int, optional
            Indices of jumps to exclude from analysis. Default is [] (empty,
            no exclusions). Common: [0] (first only), [0, -1] (first and last).
        straight_legs : bool, optional
            Whether straight-leg protocol used. Default is False.
        free_hands : bool, optional
            Whether arm swing allowed. Default is False (hands on hips).
        left_foot_ground_reaction_force : str or None, optional
            Label for left foot force platform in TDF file. Default is None.
        right_foot_ground_reaction_force : str or None, optional
            Label for right foot force platform in TDF file. Default is None.
        **force_platform_labels : str or None, optional
            Labels for additional force platforms (hands) in TDF file.
        **marker_labels : str or None, optional
            Labels for anatomical markers in TDF file.

        Returns
        -------
        RepeatedJumps
            Initialized RepeatedJumps instance with data loaded from TDF file.

        Notes
        -----
        Common repeated jump protocols:
        - 15-second test: Maximum jumps in 15 seconds
        - 30-second test: Continuous jumping for 30 seconds
        - 60-second test: Extended fatigue assessment
        - Fixed repetitions: e.g., 10, 20, or 30 jumps

        Protocol standardization:
        - Hands-on-hips (free_hands=False) is most common
        - Exclude first jump [0] to remove warm-up effect
        - Exclude last jump [-1] if potentially incomplete
        - Participant instructed to jump "as high and fast as possible"

        See Also
        --------
        WholeBody.from_tdf : Base TDF loading method.
        jumps : Property that segments individual jumps.

        Examples
        --------
        >>> # Standard 15-second test, hands on hips
        >>> rj = RepeatedJumps.from_tdf(
        ...     "rj_15s.tdf",
        ...     bodymass_kg=75,
        ...     exclude_jumps=[0],  # Exclude warm-up jump
        ...     free_hands=False,
        ...     left_foot_ground_reaction_force="FP1",
        ...     right_foot_ground_reaction_force="FP2"
        ... )
        >>> print(f"Valid jumps analyzed: {len(rj.jumps)}")

        >>> # Straight-leg protocol
        >>> rj_sl = RepeatedJumps.from_tdf(
        ...     "rj_straight_legs.tdf",
        ...     bodymass_kg=75,
        ...     straight_legs=True,
        ...     exclude_jumps=[0, -1],
        ...     right_foot_ground_reaction_force="FP2"
        ... )
        """
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
            side=side,
            **record._data,  # type: ignore
        )

    def _get_constructor_args(self):
        """
        Return custom constructor arguments for loc/iloc slicing.

        Returns
        -------
        dict
            Dictionary with custom attributes needed to reconstruct RepeatedJumps
            after slicing operations.

        Notes
        -----
        This method ensures that slicing operations (via loc or iloc) preserve
        the specialized RepeatedJumps attributes that are not part of the base
        WholeBody class.
        """
        return {
            "bodymass_kg": self.bodymass_kg,
            "free_hands": self.free_hands,
            "exclude_jumps": self.excluded_jumps,
            "straight_legs": self.straight_legs,
        }

    @property
    def jumps(self):
        """
        Extract and segment individual jumps from continuous data.

        Returns
        -------
        list of CounterMovementJump
            Individual jump objects, excluding those specified in
            excluded_jumps. Each jump contains complete biomechanical
            data from contact initiation to peak force after landing.

        Notes
        -----
        Automatic jump segmentation algorithm:
        1. Low-pass filter vertical GRF at 50 Hz (4th order Butterworth)
        2. Detect flight phases (force < 30N, duration > 50ms)
        3. Identify contact starts before each flight
        4. Find peak force after each flight
        5. Segment from contact start to peak force for each jump
        6. Remove first/last if they start/end in flight
        7. Exclude specified jumps (via excluded_jumps)

        Each CounterMovementJump object contains:
        - Full force platform data for the jump
        - Marker trajectories during the jump
        - EMG signals during the jump
        - Jump-specific metrics (height, contact time, etc.)

        Processing details:
        - Force signal filtered to remove high-frequency noise
        - Minimum flight time prevents false detection from force dips
        - First jump starting in flight indicates pre-trial activity
        - Last jump ending in flight indicates incomplete sequence

        Common issues:
        - Too few jumps: Check force threshold and flight time parameters
        - Extra jumps detected: Increase minimum flight time or filter cutoff
        - Failed segmentation: Verify force data quality and sampling rate

        Examples
        --------
        >>> rj = RepeatedJumps.from_tdf("trial.tdf", bodymass_kg=75)
        >>> jumps = rj.jumps
        >>> print(f"Detected {len(jumps)} jumps")

        >>> # Analyze each jump
        >>> for i, jump in enumerate(jumps, 1):
        ...     h = jump.jump_height
        ...     ct = jump.contact_time * 1000  # Convert to ms
        ...     ft = jump.flight_time * 1000
        ...     print(f"Jump {i}: {h:.1f} cm, CT: {ct:.0f} ms, FT: {ft:.0f} ms")

        >>> # Fatigue analysis
        >>> heights = [j.jump_height for j in jumps]
        >>> fatigue_idx = (max(heights) - min(heights)) / max(heights) * 100
        >>> print(f"Fatigue index: {fatigue_idx:.1f}%")

        See Also
        --------
        excluded_jumps : Control which jumps are excluded.
        CounterMovementJump : Individual jump class.
        continuous_batches : Batch detection algorithm.
        """
        vgrf = self.resultant_force.copy()
        time = vgrf.index
        vgrf = vgrf.force[self.vertical_axis].fillna(value=0).to_numpy().flatten()
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
                    side=self.side,
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
