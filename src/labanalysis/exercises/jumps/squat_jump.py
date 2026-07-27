"""
Squat jump exercise analysis module.

This module provides the SquatJump class for analyzing vertical jump
performance from force platform and motion capture data. Supports phase
detection, kinetic analysis, and multiple methods for jump height estimation.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ...constants import MINIMUM_CONTACT_FORCE_N, MINIMUM_FLIGHT_TIME_S, G
from ...records import ForcePlatform, Record
from ...records.body import WholeBody
from ...signalprocessing import continuous_batches
from ...timeseries import Point3D, EMGSignal


class SquatJump(WholeBody):
    """
    Vertical jump analysis from force platform and motion capture data.

    Analyzes squat jump performance by detecting contact and flight phases,
    computing kinetic and kinematic metrics, and estimating jump height using
    multiple methods (flight time, takeoff velocity, marker trajectory).

    A squat jump is a concentric-only vertical jump initiated from a static
    semi-squat position without pre-stretch (countermovement). It isolates
    concentric muscle performance and is commonly used to assess lower limb
    power production.

    Parameters
    ----------
    bodymass_kg : float or None, optional
        Participant's body mass in kilograms. Required for kinetic metrics
        (takeoff velocity, forces relative to body weight). Default is None.
    side : {"left", "right", "bilateral"} or None, optional
        Jump execution side. If None, automatically determined from available
        force platforms. Default is None.
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
    contact_phase : WholeBody or None
        Data segment from ground contact to takeoff.
    flight_phase : WholeBody or None
        Data segment during aerial phase.
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

    Raises
    ------
    ValueError
        If neither left nor right foot force platform is provided.

    Notes
    -----
    Phase Detection:
    - Contact phase: Force >= 30N threshold from first contact to takeoff
    - Flight phase: Force < 30N for minimum duration (typically 0.1s)
    - Longest valid flight phase is selected if multiple are detected

    Jump Height Methods:
    1. S2 marker: Direct measurement of center of mass elevation
    2. Flight time: h = (g × t²) / 8 (tends to overestimate)
    3. Takeoff velocity: h = v² / (2g) (from force integration)

    The `jump_height` property returns the S2 method if available,
    otherwise the minimum of flight time and takeoff velocity methods
    for a conservative estimate.

    Examples
    --------
    >>> from labanalysis.records import ForcePlatform
    >>> from labanalysis.timeseries import Point3D, Signal3D
    >>> # Create mock data
    >>> fp = ForcePlatform(
    ...     origin=Point3D(...),
    ...     force=Signal3D(...),
    ...     torque=Signal3D(...)
    ... )
    >>> jump = SquatJump(
    ...     bodymass_kg=75,
    ...     right_foot_ground_reaction_force=fp,
    ...     s2=Point3D(...)
    ... )
    >>> print(f"Jump height: {jump.jump_height:.3f} m")
    >>> print(f"Takeoff velocity: {jump.takeoff_velocity:.2f} m/s")
    >>> print(f"Peak force: {jump.peak_vertical_force:.1f} N")

    See Also
    --------
    CounterMovementJump : Jump with pre-stretch movement.
    DropJump : Plyometric jump from elevated surface.
    WholeBody : Base class for whole-body biomechanical data.
    ForcePlatform : Force platform data structure.
    """

    def __init__(
        self,
        bodymass_kg: float | None = None,
        side: Literal["left", "right", "bilateral"] | None = None,
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
            **kwargs,
        )
        self.set_bodymass_kg(bodymass_kg)
        self.set_side(side)

    def set_bodymass_kg(self, value: float | None):
        """
        Set the bodymass in kg of the participant.

        Parameters
        ----------
        value : float | None
            The bodymass in kg.

        Raises
        ------
        ValueError
            IIn case a non positive float is provided.
        """
        if value is None:
            self._bodymass_kg = np.nan
        try:
            self._bodymass_kg = float(value)  # type: ignore
            assert self._bodymass_kg > 0
        except Exception:
            raise ValueError("bodymass must be a positive float or None")

    @property
    def bodymass_kg(self):
        """
        Get the participant's body mass in kilograms.

        Returns
        -------
        float or None
            Body mass in kg, or None if not set.
        """
        return self._bodymass_kg

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
        if not (value is None or value in ["left", "right", "bilateral"]):
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

    @property
    def flight_time(self):
        """
        Get the duration of the flight (aerial) phase.

        Returns
        -------
        float or None
            Flight time in seconds, or None if flight phase cannot be determined.

        Notes
        -----
        Calculated as the time difference between the end and start of
        the flight phase. Requires successful detection of a valid flight
        phase meeting minimum duration criteria.
        """
        index = self.flight_phase
        if index is None:
            return np.nan
        index = index.index
        return float(index[-1] - index[0])

    @property
    def contact_time(self):
        """
        Get the duration of ground contact phase.

        Returns
        -------
        float or None
            Contact time in seconds, or None if contact phase cannot be determined.

        Notes
        -----
        Calculated as the time difference between the end and start of
        the contact phase (first ground contact to takeoff).
        """
        index = self.contact_phase
        if index is None:
            return np.nan
        index = index.index
        return float(index[-1] - index[0])

    @property
    def contact_phase(self):
        """
        Returns the ground contact phase as a WholeBody record.

        Returns
        -------
        WholeBody or SingleJump (or subclass)
            Data segment from first ground contact to takeoff.

        Notes
        -----
        For jumps with multiple contact phases (e.g., drop jumps), this returns
        the primary propulsive contact phase (typically the last one).
        """
        vgrf = self.resultant_force
        if vgrf is None:
            return None
        module = vgrf.force[self.vertical_axis].to_numpy().flatten()

        # definisco il primo batch con contatto a terra
        contact = continuous_batches(module >= MINIMUM_CONTACT_FORCE_N)
        if len(contact) < 2:
            raise ValueError("No contact was found")
        t0 = vgrf.index[contact[0][0]]

        # fine del tempo di contatto
        fp = self.flight_phase
        if fp is None:
            return None
        t1 = fp.index[0]

        # ritorno l'oggetto corrispondente all'intervallo di tempo tra l'avvio
        # della fase di contatto e l'inizio della fase di volo
        return WholeBody(
            **{
                i: v.copy().loc[(v.index >= t0) & (v.index < t1), :]
                for i, v in self.items()
            }
        )

    @property
    def flight_phase(self):
        """
        Returns the flight (aerial) phase as a WholeBody record.

        Returns
        -------
        WholeBody or SingleJump (or subclass)
            Data segment during aerial phase after takeoff.

        Notes
        -----
        For jumps with multiple flight phases, this returns the primary
        flight phase (typically the longest one meeting minimum duration).
        """
        vgrf = self.resultant_force
        if vgrf is None:
            return None
        module = vgrf.force[self.vertical_axis].to_numpy().flatten()

        # identifico i batch relativi alla fase di volo e di contatto
        flight = module < MINIMUM_CONTACT_FORCE_N
        contact = ~flight
        f_batches = continuous_batches(flight)
        c_batches = continuous_batches(contact)
        if len(f_batches) < 1:
            raise ValueError("No flight phase was found")
        if len(c_batches) < 2:
            raise ValueError("No contact was found")

        # rimuovo i flight batch che intervengono prima di un primo contatto
        f_batches = [i for i in f_batches if i[0] > c_batches[0][-1]]

        # rimuovo i flight batch che intervengono oltre l'ultimo contatto
        f_batches = [i for i in f_batches if i[-1] < c_batches[-1][0]]

        # rimuovo i batch di durata non ragionevole
        i = 0
        while i < len(f_batches):
            batch = f_batches[i]
            duration = vgrf.index[batch[-1]] - vgrf.index[batch[0]]
            if duration < MINIMUM_FLIGHT_TIME_S:
                f_batches.pop(i)
            else:
                i += 1

        # verifico che vi siano flight batches ragionevoli
        if len(f_batches) == 0:
            raise ValueError("No valid flight phases have been discovered.")

        # ritorno l'oggetto corrispondente al batch più lungo
        index = np.argsort([len(i) for i in f_batches])
        batch = f_batches[index[-1]]
        t0, t1 = vgrf.iloc[batch].index[[0, -1]]
        return WholeBody(
            **{
                i: v.loc[(v.index >= t0) & (v.index <= t1), :].copy()
                for i, v in self.items()
            }
        )

    @property
    def jump_height_from_s2(self):
        """
        Calculate jump height from S2 (sacral) marker trajectory.

        Returns
        -------
        float or None
            Jump height in meters measured from S2 marker vertical displacement,
            or None if S2 marker or flight phase data is unavailable.

        Notes
        -----
        Computed as the difference between maximum and initial vertical
        position of the S2 marker during the flight phase. This method
        provides a direct kinematic measurement of center of mass elevation.
        """
        ff = self.flight_phase
        if ff is None:
            return np.nan
        s2: Point3D = ff["s2"]  # type: ignore
        if s2 is None:
            return np.nan
        vt = s2[self.vertical_axis].to_numpy().flatten()
        return float(vt.max() - vt[0])

    @property
    def jump_height_from_ft(self):
        """
        Calculate jump height from flight time using ballistic equations.

        Returns
        -------
        float or None
            Jump height in meters calculated from flight time,
            or None if flight time cannot be determined.

        Notes
        -----
        Uses the kinematic equation for projectile motion:
            h = (g × t²) / 8

        where g is gravitational acceleration (9.81 m/s²) and t is flight time.

        This assumes symmetric parabolic trajectory and neglects air resistance.
        The method tends to overestimate actual jump height due to body
        configuration changes during flight.

        See Also
        --------
        jump_height_from_s2 : Direct kinematic measurement from marker.
        jump_height_from_tov : Calculation from takeoff velocity.
        """
        flight_time = self.flight_time
        if flight_time is None:
            return np.nan
        return float(G / 8 * flight_time**2)

    @property
    def takeoff_velocity(self):
        """
        Calculate vertical takeoff velocity from ground reaction force.

        Returns
        -------
        float or None
            Vertical velocity at takeoff in m/s, or None if required data
            (body mass, contact phase, force data) is unavailable.

        Notes
        -----
        Computed by integrating vertical acceleration during the contact phase:
        1. Net force = GRF - body weight
        2. Acceleration = net force / body mass
        3. Velocity = ∫ acceleration dt (assuming v₀ = 0)

        Uses trapezoidal rule for numerical integration. Initial acceleration
        is subtracted to enforce zero initial velocity assumption.

        This method is sensitive to force platform noise and requires proper
        filtering of GRF data for accurate results.

        See Also
        --------
        jump_height_from_tov : Jump height calculated from this velocity.
        """
        if self.bodymass_kg is None:
            return np.nan
        cf = self.contact_phase
        if cf is None:
            return np.nan
        rf = cf.resultant_force
        if rf is None:
            return np.nan
        vgrf = rf.force[self.vertical_axis].copy().fillna()
        body_weight = self.bodymass_kg * G
        vgrf = vgrf - body_weight  # type: ignore
        vacc = vgrf.to_numpy().flatten() / self.bodymass_kg
        vacc -= vacc[0]  # assumo che v0 = 0
        time = vgrf.index
        return float(np.trapezoid(vacc, time))  # type: ignore

    @property
    def jump_height_from_tov(self):
        """
        Calculate jump height from takeoff velocity.

        Returns
        -------
        float or None
            Jump height in meters calculated from takeoff velocity,
            or None if takeoff velocity cannot be determined.

        Notes
        -----
        Uses the kinematic equation:
            h = v² / (2g)

        where v is takeoff velocity and g is gravitational acceleration.

        This method is generally more accurate than flight time method
        as it accounts for actual propulsive performance during contact.

        See Also
        --------
        takeoff_velocity : Velocity calculation from force integration.
        jump_height_from_ft : Alternative calculation from flight time.
        """
        tv = self.takeoff_velocity
        if tv is None:
            return np.nan
        return (tv**2) / (2 * G)

    @property
    def elevation(self):
        """
        Get the jump height.

        Returns
        -------
        float or None
            Jump height in meters, or None if no valid estimates available.

        Notes
        -----
        Selection priority:
        1. S2 marker method (if available) - most direct measurement
        2. Minimum of flight time and takeoff velocity methods - conservative
           estimate that tends to be more accurate

        The flight time method typically overestimates due to body configuration
        changes during flight, while the takeoff velocity method can be affected
        by force platform noise. Taking the minimum reduces overestimation.

        See Also
        --------
        jump_height_from_s2 : Direct kinematic measurement.
        jump_height_from_ft : Calculation from flight time.
        jump_height_from_tov : Calculation from takeoff velocity.
        """
        s2 = self.jump_height_from_s2
        if s2:
            return s2
        vals = [self.jump_height_from_ft, self.jump_height_from_tov]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan
        return float(np.min(vals))

    @property
    def peak_vertical_force(self):
        """
        Get the maximum vertical ground reaction force as percentage
        of the bodyweight.

        Returns
        -------
        float or None
            Peak vertical force in Newtons, or None if contact phase or
            force data is unavailable.

        Notes
        -----
        Identifies the maximum vertical GRF value during the contact phase
        (ground contact to takeoff). This metric indicates the maximum
        loading on the lower extremities during propulsion.

        Typically expressed relative to body weight for normalization:
            Peak force / (body mass × g)

        Common values for vertical jumps range from 2-3× body weight
        for recreational athletes to >3× for elite jumpers.

        See Also
        --------
        contact_phase : The data segment analyzed.
        """
        cp = self.contact_phase
        if cp is None:
            return np.nan
        grf = cp.resultant_force
        if grf is None:
            return np.nan
        return float(
            grf.force[self.vertical_axis].to_numpy().max() / self.bodymass_kg / G
        )

    @property
    def rate_of_force_development(self):
        """
        Get the rate of force development during contact.

        Returns
        -------
        float or None
            Rate of force development in kN/s, or None if contact phase or
            force data is unavailable.

        Notes
        -----
        RFD is calculated as the slope from body weight to peak force:

            RFD = (Peak Force - Body Weight) / (Time to Peak - Time at BW)

        Measured from the last point where force equals body weight before
        peak force to the peak force itself during the contact phase.

        This metric reflects neuromuscular explosiveness and is critical for
        power performance assessment.

        See Also
        --------
        contact_phase : The data segment analyzed.
        peak_vertical_force : Maximum force during contact.
        """

        # extract vertical grf and time
        cp = self.contact_phase
        if cp is None:
            return np.nan
        grf = cp.resultant_force
        if grf is None:
            return np.nan
        vgrf = grf.force[self.vertical_axis]
        time = vgrf.index
        vgrf = vgrf.to_numpy().flatten()

        # get the index of the peak force
        i_peak = np.argmax(vgrf)

        # get the index of the last sample before the peak with force
        # equal to the participant's bodyweight
        bw = self.bodymass_kg * G
        i_bw = np.where(vgrf[:i_peak] <= bw)[0]
        if len(i_bw) == 0:
            return np.nan
        i_bw = i_bw[-1]

        # get the rate of force development
        dt = time[i_peak] - time[i_bw]
        df = vgrf[i_peak] - bw
        return float(df / dt / 1000)

    @property
    def force_asymmetry(self):
        """
        Get the asymmetry in force generation between left and right foot.

        Returns
        -------
        float or None
            Force asymmetry percentage, or None if contact phase or
            force data for left and right foot are unavailable.

        Notes
        -----
        Calculated as:

            Asymmetry = 100 × (Right - Left) / (Right + Left)

        Positive values indicate greater right-side force production,
        negative values indicate greater left-side force production.

        Typical asymmetry values:
        - <10%: Normal/acceptable for most populations
        - 10-15%: Moderate asymmetry, may warrant attention
        - >15%: Significant asymmetry, potential injury risk

        Only applicable to bilateral jumps with separate force platforms
        for each foot.

        See Also
        --------
        contact_phase : The data segment analyzed.
        """
        # extract contact phase
        cp = self.contact_phase
        if cp is None:
            return np.nan

        # extract left and right force
        left = cp.get("left_foot_ground_reaction_force")
        right = cp.get("right_foot_ground_reaction_force")
        if (
            left is None
            or right is None
            or not isinstance(left, ForcePlatform)
            or not isinstance(right, ForcePlatform)
        ):
            return np.nan

        # get vertical forces
        left = left.force[left.vertical_axis].to_numpy().flatten().mean()
        right = right.force[right.vertical_axis].to_numpy().flatten().mean()

        # symmetry
        return float((right - left) / (right + left) * 100)

    @property
    def muscles_activity(self):
        """
        Get the muscle EMG activity for all available muscles.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: muscle, side, limb, unit, value.
            Returns empty DataFrame if contact phase or EMG data is unavailable.

        Notes
        -----
        Calculates mean EMG amplitude for each muscle during the contact phase.
        Only includes muscles matching the jump side (bilateral jumps include
        both left and right muscles).

        The returned DataFrame contains one row per muscle channel with:
        - muscle: Muscle name (e.g., "vastus_lateralis")
        - side: Jump side ("left", "right", or "bilateral")
        - limb: Muscle side ("left" or "right")
        - unit: EMG signal unit (typically "mV" or "µV")
        - value: Mean EMG amplitude during contact

        See Also
        --------
        contact_phase : The data segment analyzed.
        muscles_asymmetry : EMG asymmetry between sides.
        """

        cp = self.contact_phase
        if cp is None:
            return pd.DataFrame()
        out = []
        for emg in cp.emgsignals.values():
            if (emg.side == self.side or self.side == "bilateral") and isinstance(
                emg, EMGSignal
            ):
                out.append(
                    {
                        "muscle": emg.muscle_name,
                        "side": self.side,
                        "limb": emg.side,
                        "unit": emg.unit,
                        "value": float(emg.mean()),
                    }
                )
        if len(out) == 0:
            return pd.DataFrame()
        return pd.DataFrame(out)

    @property
    def muscles_asymmetry(self):
        """
        Get the muscle EMG asymmetry between left and right sides.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: muscle, unit, value.
            Returns empty DataFrame if muscles activity is not available or
            jump is not bilateral.

        Notes
        -----
        Only applicable to bilateral jumps with EMG data from both sides.

        Asymmetry is calculated as:

            Asymmetry = 100 × (Right - Left) / (Right + Left)

        for each muscle pair (e.g., left vs right vastus lateralis).

        Positive values indicate greater right-side activation,
        negative values indicate greater left-side activation.

        The returned DataFrame contains one row per muscle with:
        - muscle: Muscle name
        - unit: "%"
        - value: Asymmetry percentage

        See Also
        --------
        muscles_activity : The muscle activity data.
        force_asymmetry : Force-based asymmetry metric.
        """
        if self.side != "bilateral":
            return pd.DataFrame()
        act = self.muscles_activity
        if act.empty:
            return pd.DataFrame()

        def get_asymmetry(x: pd.Series):
            if "left" in x.index and "right" in x.index:
                return float(100 * (x.right - x.left) / (x.right + x.left))
            return None

        out = (
            act.drop(["unit", "side"], axis=1)
            .pivot_table(index="muscle", columns="limb", values="value")
            .apply(get_asymmetry)
        )
        out = pd.DataFrame(out)
        out.columns = pd.Index(["value"])
        out.insert(0, "unit", "%")
        out.insert(0, "muscle", out.index)
        out.reset_index(inplace=True, drop=True)
        return out

    @property
    def output_metrics(self):
        """
        Get all relevant performance metrics for the squat jump.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all jump metrics including kinetic, kinematic,
            and EMG-derived metrics.

        Notes
        -----
        Standard metrics included:
        - elevation (cm): Jump height
        - takeoff velocity (cm/s): Vertical velocity at takeoff
        - flight time (ms): Aerial phase duration
        - peak force (BW): Maximum vertical force relative to body weight
        - rate of force development (kN/s): Force generation speed
        - force asymmetry (%): Left-right force imbalance
        - muscles activity: Mean EMG for each muscle
        - muscles asymmetry (%): Left-right EMG imbalance

        All metrics are labeled with jump type and side for easy filtering.

        See Also
        --------
        elevation : Jump height calculation.
        muscles_activity : EMG metrics.
        force_asymmetry : Kinetic asymmetry.
        """

        # get standard metrics
        out = [
            {
                "metric": "elevation",
                "unit": "cm",
                "value": self.elevation * 100,
            },
            {
                "metric": "takeoff velocity",
                "unit": "cm/s",
                "value": self.takeoff_velocity * 100,
            },
            {
                "metric": "flight time",
                "unit": "ms",
                "value": self.flight_time * 1000,
            },
            {
                "metric": "peak force",
                "unit": "BW",
                "value": self.peak_vertical_force,
            },
            {
                "metric": "rate of force development",
                "unit": "kN/s",
                "value": self.rate_of_force_development,
            },
            {
                "metric": "force asymmetry",
                "unit": "%",
                "value": self.force_asymmetry,
            },
        ]
        out = pd.DataFrame(out)

        # add emg metrics
        out = pd.concat(
            [out, self.muscles_activity, self.muscles_asymmetry],
            ignore_index=True,
        )

        # add jump name and side
        out.insert(0, "side", self.side)
        out.insert(0, "type", self.name)

        return out.sort_values(["metric", "side"])

    def to_dataframe(self):
        """
        Convert the squat jump to a pandas DataFrame with phase labels.

        Returns
        -------
        pd.DataFrame
            DataFrame with time series data, phase column, and relative time.

        Raises
        ------
        ValueError
            If no flight phase can be detected.

        Notes
        -----
        The DataFrame includes:
        - time_s: Relative time from start (seconds)
        - phase: Jump phase ("contact", "flight", or "landing")
        - All biomechanical signals (forces, markers, EMG)

        The returned data spans from 0.5s before flight to 0.5s after landing
        (or to data boundaries if shorter). Phase labels enable easy filtering
        for phase-specific analysis.

        See Also
        --------
        flight_phase : Aerial phase data used for segmentation.
        contact_phase : Ground contact phase.
        """

        # convert to dataframe
        objs = {}
        if "s2" in self.keys():
            objs["s2"] = self.s2.copy()
        objs.update(**{i: v for i, v in self.forceplatforms.items()})
        objs.update(**{i: v for i, v in self.emgsignals.items()})
        df = Record(**objs).to_dataframe()

        # get the flight phase start and end
        ff = self.flight_phase
        if ff is None:
            raise ValueError("no flight phase detected.")

        f0, f1 = ff.index[0], ff.index[-1]

        # wrap half second before and after the jump (if available)
        t0 = max(self.index[0], f0 - 0.5)
        t1 = min(self.index[-1], f1 + 0.5)
        mask = (self.index >= t0) & (self.index <= t1)
        new = pd.DataFrame(df.loc[mask])

        # append a column mapping the phases
        new.insert(0, "phase", None)

        # contact
        cmask = new.index < f0
        new.loc[cmask, "phase"] = "contact"

        # flight
        fmask = (new.index >= f0) & (new.index <= f1)
        new.loc[fmask, "phase"] = "flight"

        # landing
        lmask = new.index > f1
        new.loc[lmask, "phase"] = "landing"

        # add the time as column
        new.insert(0, "time", new.index - new.index[0])

        return new

    @classmethod
    def from_tdf(
        cls,
        filename: str | Path,
        bodymass_kg: float | int,
        side: Literal["left", "right", "bilateral"] | None = None,
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
        Create SquatJump instance from BTS Bioengineering TDF file.

        Parameters
        ----------
        filename : str or Path
            Path to the TDF file containing jump trial data.
        bodymass_kg : float or int
            Participant's body mass in kilograms (required for kinetic analysis).
        side : {"left", "right", "bilateral"} or None, optional
            Jump execution side. If None, automatically determined from
            available force platform data. Default is None.
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
        SquatJump
            Initialized SquatJump instance with data loaded from TDF file.

        Notes
        -----
        TDF (Technogym Data Format) files are binary files from BTS
        Bioengineering systems containing synchronized force platform
        and 3D motion capture data.

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
        >>> jump = SquatJump.from_tdf(
        ...     "trial01.tdf",
        ...     bodymass_kg=75,
        ...     left_foot_ground_reaction_force="FP1",
        ...     right_foot_ground_reaction_force="FP2",
        ...     s2="S2"
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
            **{i: v for i, v in record.items()},  # type: ignore
        )


__all__ = ["SquatJump"]
