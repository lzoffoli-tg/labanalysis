"""
Counter-movement jump exercise analysis module.

This module provides the CounterMovementJump class for analyzing vertical jumps
with pre-stretch (countermovement) from force platform and motion capture data.
Extends SquatJump with additional loading response and propulsion phase detection.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ...records import ForcePlatform
from ...records.body import WholeBody
from ...timeseries import Point3D
from .squat_jump import SquatJump
from ...signalprocessing import find_peaks


class CounterMovementJump(SquatJump):
    """
    Counter-movement jump analysis with stretch-shortening cycle evaluation.

    Analyzes counter-movement jump (CMJ) performance by detecting eccentric
    (loading) and concentric (propulsion) phases, computing kinetic and kinematic
    metrics, and leveraging the stretch-shortening cycle (SSC) to assess elastic
    energy utilization and neuromuscular performance.

    A counter-movement jump involves a rapid downward movement (countermovement)
    followed immediately by an upward propulsive phase. The pre-stretch enhances
    performance through elastic energy storage and the stretch reflex, making CMJ
    typically higher than squat jumps. It is the most commonly used vertical jump
    test in sports performance assessment.

    Parameters
    ----------
    bodymass_kg : float or None, optional
        Participant's body mass in kilograms. Required for kinetic metrics.
        Default is None.
    side : {"left", "right", "bilateral"} or None, optional
        Jump execution side. If None, automatically determined from available
        force platforms. Default is None.
    free_hands : bool, optional
        Whether hands were free to swing during the jump (True) or held on
        hips/restricted (False). Arm swing significantly affects jump height
        and force profiles. Default is False.
    straight_legs : bool, optional
        Whether legs were kept relatively straight during countermovement
        (True) or normal knee flexion allowed (False). Affects stretch-shortening
        cycle mechanics. Default is False.
    left_foot_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for left foot. Default is None.
    right_foot_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for right foot. Default is None.
    left_hand_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for left hand (if applicable). Default is None.
    right_hand_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for right hand (if applicable). Default is None.
    **markers : Point3D or None, optional
        3D marker trajectories for kinematic analysis (same as SquatJump).
    **kwargs
        Additional keyword arguments passed to SquatJump and WholeBody.

    Attributes
    ----------
    free_hands : bool
        Whether hands were free during jump execution.
    straight_legs : bool
        Whether legs were kept straight during countermovement.
    loading_response_phase : WholeBody or None
        Eccentric phase from initial contact to minimum vertical force.
    loading_response_time : float or None
        Duration of loading response phase in seconds.
    propulsion_phase : WholeBody or None
        Concentric phase from minimum vertical force to takeoff.
    propulsion_time : float or None
        Duration of propulsion phase in seconds.

    Attributes (Inherited from SquatJump)
    --------------------------------------
    bodymass_kg : float or None
    side : str
    contact_phase : WholeBody or None
    flight_phase : WholeBody or None
    contact_time : float or None
    flight_time : float or None
    jump_height : float or None
    takeoff_velocity : float or None
    peak_vertical_force : float or None

    Notes
    -----
    Phase Detection:
    - Loading response: From initial ground contact to minimum vertical force
      (deepest point of countermovement)
    - Propulsion: From minimum vertical force to takeoff
    - The minimum force point marks the transition from eccentric to concentric

    Stretch-Shortening Cycle:
    The CMJ exploits the SSC through three mechanisms:
    1. Elastic energy storage in tendons/muscles during loading
    2. Potentiation of the stretch reflex
    3. Optimization of muscle length-tension relationships

    Performance differences between CMJ and SJ (squat jump) indicate SSC
    contribution. Typical CMJ heights are 10-20% greater than SJ.

    Jump Variations:
    - Free hands vs. hands on hips: Free arm swing adds ~10% to jump height
    - Straight legs: Reduces SSC contribution, emphasizes ankle plantarflexion
    - Deep vs. shallow countermovement: Affects force-time characteristics

    Examples
    --------
    >>> from labanalysis.records import ForcePlatform
    >>> # Create CMJ with arm swing
    >>> cmj = CounterMovementJump(
    ...     bodymass_kg=75,
    ...     free_hands=True,
    ...     right_foot_ground_reaction_force=fp
    ... )
    >>> print(f"Loading time: {cmj.loading_response_time:.3f} s")
    >>> print(f"Propulsion time: {cmj.propulsion_time:.3f} s")
    >>> print(f"Jump height: {cmj.jump_height:.3f} m")

    >>> # Compare to squat jump for SSC contribution
    >>> ssc_contribution = cmj.jump_height - squat_jump.jump_height
    >>> print(f"SSC benefit: {ssc_contribution*100:.1f} cm")

    See Also
    --------
    SquatJump : Concentric-only jump without countermovement.
    DropJump : Plyometric jump emphasizing fast SSC.
    """

    def __init__(
        self,
        bodymass_kg: float | None = None,
        side: Literal["left", "right", "bilateral"] | None = None,
        free_hands: bool = False,
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
            bodymass_kg=bodymass_kg,
            side=side,
            **kwargs,
        )
        self.set_free_hands(free_hands)

    def set_free_hands(self, value: bool):
        """
        Set whether hands were free during jump execution.

        Parameters
        ----------
        value : bool
            True if hands were free to swing naturally, False if hands
            were restricted (e.g., on hips, behind back).

        Raises
        ------
        ValueError
            If value is not a boolean.

        Notes
        -----
        Free arm swing typically increases jump height by ~10% and affects
        force-time curve characteristics. This parameter is important for
        standardizing jump protocols and interpreting results.
        """
        if not isinstance(value, bool):
            raise ValueError("free_hands must be True or False")
        self._free_hands = value

    @property
    def free_hands(self):
        """
        Get whether hands were free during jump execution.

        Returns
        -------
        bool
            True if hands were free to swing, False if restricted.

        See Also
        --------
        set_free_hands : Set the free hands status.
        """
        return self._free_hands

    @property
    def counter_movement_phase(self):
        """
        Get the eccentric (loading) phase of the counter-movement.

        Returns
        -------
        WholeBody or None
            Data segment from initial ground contact to the point of minimum
            vertical force (deepest countermovement position), or None if
            contact phase or force data is unavailable.

        Notes
        -----
        The loading response phase corresponds to the eccentric portion of
        the stretch-shortening cycle where:
        - Muscles lengthen under tension (eccentric contraction)
        - Elastic energy is stored in tendons and muscle-tendon units
        - Vertical force decreases below body weight as center of mass descends

        Detection algorithm:
        1. Identify peak vertical force during contact phase
        2. Find minimum vertical force occurring before the peak
        3. Loading phase = initial contact to minimum force

        This phase is critical for SSC performance - faster eccentric loading
        rates generally enhance subsequent concentric performance through
        enhanced stretch reflex and elastic energy return.

        See Also
        --------
        loading_response_time : Duration of this phase.
        propulsion_phase : Subsequent concentric phase.
        """
        # the start of the counter movement phase begins with the contact phase
        cp = self.contact_phase
        if cp is None:
            return None
        t0 = cp.index[0]

        # if we have the s2 marker, we use it to derive the end of the countermovement
        s2 = cp.get("s2")
        if s2 is not None:
            i1 = np.argmin(s2[s2.vertical_axis].to_numpy().flatten())
            t1 = float(s2.index[i1])

        # we use force data to derive the end of the countermovement phase
        else:
            # look for the minimum in vertical force occurring before the peak
            vgrf = cp.resultant_force
            if vgrf is None:
                return None
            time = vgrf.index
            vgrf = vgrf.force[vgrf.vertical_axis].to_numpy().flatten()
            ipk = np.argmax(vgrf)
            mns = find_peaks(-vgrf[:ipk])
            if len(mns) == 0:
                raise RuntimeError(
                    "No local minima was found in vertical ground reaction force."
                )
            t1 = time[mns[np.argmin(vgrf[mns])]]

        return WholeBody(
            **{
                i: v.loc[(v.index >= t0) & (v.index <= t1), :].copy()
                for i, v in self.items()
            }
        )

    @property
    def propulsion_phase(self):
        """
        Get the concentric (propulsion) phase of the jump.

        Returns
        -------
        WholeBody or None
            Data segment from minimum vertical force (deepest countermovement)
            to takeoff, or None if contact phase or force data is unavailable.

        Notes
        -----
        The propulsion phase corresponds to the concentric portion of the
        stretch-shortening cycle where:
        - Muscles shorten while generating force (concentric contraction)
        - Stored elastic energy is released
        - Vertical force increases and exceeds body weight
        - Center of mass accelerates upward

        Detection algorithm:
        1. Identify peak vertical force during contact phase
        2. Find minimum vertical force occurring before the peak
        3. Propulsion phase = minimum force to takeoff

        This phase represents the active propulsive performance enhanced by
        the preceding eccentric loading. The force-time characteristics during
        propulsion determine takeoff velocity and jump height.

        Performance metrics during this phase include:
        - Peak force and rate of force development
        - Impulse (force × time integral)
        - Propulsion duration

        See Also
        --------
        propulsion_time : Duration of this phase.
        loading_response_phase : Preceding eccentric phase.
        takeoff_velocity : Result of propulsive performance.
        """
        # we get the end of the propulsion phase as the end of the contact phase
        cp = self.contact_phase
        if cp is None:
            return None
        t1 = cp.index[-1]

        # we get the start of the propulsion phase as the end of the counter
        # movement phase
        cm = self.counter_movement_phase
        if cm is None:
            return None
        t0 = cm.index[-1]

        # return the segment
        return WholeBody(
            **{
                i: v.loc[(v.index > t0) & (v.index <= t1), :].copy()
                for i, v in self.items()
            }
        )

    @property
    def muscular_reactivity_index(self):
        """
        Calculate reactive neuromuscular index (RNI) for each muscle.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: muscle, limb, metric, unit, value.
            Returns empty DataFrame if EMG data or phases are unavailable.

        Notes
        -----
        The reactive neuromuscular index quantifies the muscle activation
        during the propulsion phase relative to the loading phase:

            RNI = 100 × (∫ EMG_propulsion dt) / (∫ EMG_loading dt)

        Higher RNI values indicate greater muscle activation during the
        concentric propulsion phase compared to the eccentric loading phase,
        reflecting neuromuscular efficiency in the stretch-shortening cycle.

        This metric is calculated separately for each available EMG channel.

        See Also
        --------
        counter_movement_phase : Eccentric loading phase (denominator).
        propulsion_phase : Concentric propulsion phase (numerator).
        """

        # check if the necessary phases can be obtained
        cmp = self.counter_movement_phase
        ppp = self.propulsion_phase
        if cmp is None or ppp is None:
            return pd.DataFrame()

        # get the integral of each muscle for both phases
        def get_rni(x):
            brake = cmp.get(x)
            push = ppp.get(x)
            if brake is None or push is None:
                return np.nan
            return float(
                100
                * np.trapezoid(push.to_numpy().flatten(), push.index)
                / np.trapezoid(brake.to_numpy().flatten(), brake.index)
            )

        rni = [
            {
                "muscle": m.muscle_name,
                "limb": m.side,
                "metric": "reactive neuromuscular index",
                "unit": "%",
                "value": get_rni(m),
            }
            for m in self.emgsignals.keys()
        ]

        return pd.DataFrame(rni)

    @property
    def output_metrics(self):
        """
        Get all relevant performance metrics for the counter-movement jump.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all jump metrics including elevation, forces,
            EMG activity, and reactive neuromuscular indices.

        Notes
        -----
        Combines base jump metrics from SquatJump with CMJ-specific metrics
        (muscular reactivity index). All metrics are labeled with jump type,
        side, and free hands status for easy filtering and comparison.

        See Also
        --------
        muscular_reactivity_index : CMJ-specific EMG metric.
        SquatJump.output_metrics : Base jump metrics.
        """
        out = pd.concat(
            [super().output_metrics, self.muscular_reactivity_index], ignore_index=True
        )
        out.insert(1, "free hands", self.free_hands)
        out.loc[out.index, "side"] = self.side
        out.loc[out.index, "type"] = self.name
        return out.sort_values(["metric", "side", "free hands"])

    def to_dataframe(self):
        """
        Convert the counter-movement jump to a pandas DataFrame with phase labels.

        Returns
        -------
        pd.DataFrame
            DataFrame with time series data and phase column indicating
            "contact", "flight", "landing", "counter movement", and "propulsion".

        Raises
        ------
        RuntimeError
            If counter movement or propulsion phase cannot be determined.

        Notes
        -----
        Extends the base SquatJump DataFrame by adding specific phase labels
        for the counter-movement (eccentric) and propulsion (concentric) phases.

        See Also
        --------
        SquatJump.to_dataframe : Base DataFrame conversion.
        counter_movement_phase : Eccentric phase data.
        propulsion_phase : Concentric phase data.
        """

        # convert to dataframe
        df = super().to_dataframe()

        # add counter movement phase
        cmp = self.counter_movement_phase
        if cmp is None:
            raise RuntimeError("counter movement phase not found.")
        cmask = (df.index >= cmp.index[0]) & (df.index <= cmp.index[-1])
        df.loc[cmask, "phase"] = "counter movement"

        # add propulsion phase
        pp = self.propulsion_phase
        if pp is None:
            raise RuntimeError("propulsion phase not found.")
        pmask = (df.index >= pp.index[0]) & (df.index <= pp.index[-1])
        df.loc[pmask, "phase"] = "propulsion"

        return df

    @classmethod
    def from_tdf(
        cls,
        filename: str | Path,
        bodymass_kg: float | int,
        side: Literal["left", "right", "bilateral"] | None = None,
        free_hands: bool = False,
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
        Create CounterMovementJump instance from BTS Bioengineering TDF file.

        Parameters
        ----------
        filename : str or Path
            Path to the TDF file containing CMJ trial data.
        bodymass_kg : float or int
            Participant's body mass in kilograms (required for kinetic analysis).
        side : {"left", "right", "bilateral"} or None, optional
            Jump execution side. If None, automatically determined from
            available force platform data. Default is None.
        free_hands : bool, optional
            Whether hands were free to swing during the jump. Default is False
            (hands restricted, e.g., on hips).
        left_foot_ground_reaction_force : str or None, optional
            Label for left foot force platform in TDF file. Default is None.
        right_foot_ground_reaction_force : str or None, optional
            Label for right foot force platform in TDF file. Default is None.
        **force_platform_labels : str or None, optional
            Labels for additional force platforms (hands) in TDF file.
        **marker_labels : str or None, optional
            Labels for anatomical markers in TDF file (same as SquatJump).

        Returns
        -------
        CounterMovementJump
            Initialized CounterMovementJump instance with data loaded from TDF file.

        Notes
        -----
        The `free_hands` parameter is critical for:
        - Standardizing jump protocols across test sessions
        - Comparing results to normative data
        - Interpreting performance metrics correctly

        Common CMJ protocols:
        - Standard: free_hands=True
        - Hands on hips: free_hands=False

        See Also
        --------
        SquatJump.from_tdf : Base class TDF loading method.
        WholeBody.from_tdf : Low-level TDF file parsing.

        Examples
        --------
        >>> # Standard CMJ with arm swing
        >>> cmj = CounterMovementJump.from_tdf(
        ...     "cmj_trial01.tdf",
        ...     bodymass_kg=75,
        ...     free_hands=True,
        ...     left_foot_ground_reaction_force="FP1",
        ...     right_foot_ground_reaction_force="FP2",
        ...     s2="S2"
        ... )
        >>> print(f"CMJ height: {cmj.jump_height:.3f} m")

        >>> # Hands-on-hips protocol
        >>> cmj_restricted = CounterMovementJump.from_tdf(
        ...     "cmj_trial02.tdf",
        ...     bodymass_kg=75,
        ...     free_hands=False,
        ...     right_foot_ground_reaction_force="FP2"
        ... )
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
            free_hands=free_hands,
            **{i: v for i, v in record.items()},  # type: ignore
        )


__all__ = ["CounterMovementJump"]
