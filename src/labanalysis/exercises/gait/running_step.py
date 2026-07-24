"""Running step (gait cycle) module."""

from typing import Literal

import numpy as np

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...records.body import WholeBody
from ...records.forceplatform import ForcePlatform
from ...records.record import Record
from ...timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from .gait_cycle import GaitCycle

__all__ = ["RunningStep"]


class RunningStep(GaitCycle):
    """
    Represents a single running step (one gait cycle during running).

    RunningStep extends GaitCycle with running-specific phases and metrics.
    A running step is characterized by a flight phase (no ground contact)
    followed by a contact phase (ground contact).

    The contact phase is further subdivided into:
    - Loading response: From footstrike to midstance
    - Propulsion: From midstance to toe-off

    Parameters
    ----------
    Inherits all parameters from GaitCycle.

    Attributes
    ----------
    flight_phase : WholeBody
        Data during the flight phase (toeoff to footstrike).
    contact_phase : WholeBody
        Data during the contact phase (footstrike to next toeoff).
    loading_response_phase : WholeBody
        Data during loading response (footstrike to midstance).
    propulsion_phase : WholeBody
        Data during propulsion (midstance to toeoff).
    flight_time_s : float
        Duration of flight phase in seconds.
    contact_time_s : float
        Duration of contact phase in seconds.
    loadingresponse_time_s : float
        Duration of loading response phase in seconds.
    propulsion_time_s : float
        Duration of propulsion phase in seconds.

    Notes
    -----
    Unlike walking, running is characterized by a flight phase where
    neither foot is in contact with the ground. This class provides
    properties to extract and analyze both the aerial and ground contact
    phases of the running gait cycle.

    The cycle timing follows the pattern:
    init_s (toeoff) -> flight -> footstrike_s -> loading response ->
    midstance_s -> propulsion -> end_s (next toeoff)

    See Also
    --------
    GaitCycle : Parent class for general gait cycles.
    WalkingStride : Gait cycle class for walking.
    RunningExercise : Exercise class for running analysis.
    """

    @property
    def _temporal_metrics(self):
        """private property to describe temporal metrics"""
        metrics = super()._temporal_metrics
        metrics.extend(
            ["contact_time", "flight_time", "loadingresponse_time", "propulsion_time"]
        )
        return metrics

    @property
    def _spatial_metrics(self):
        """private property to describe spatial metrics"""
        metrics = super()._spatial_metrics
        metrics.extend(["overstride"])
        return metrics

    @property
    def flight_phase(self):
        """
        Get data during the flight phase (toe-off to footstrike).
        """
        sliced = self.copy()[self.init_time : self.footstrike_time]
        out = WholeBody()
        if isinstance(sliced, Record):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def contact_phase(self):
        """
        Get data during the contact phase (footstrike to toe-off).
        """
        sliced = self.copy()[self.footstrike_time : self.end_time]
        out = WholeBody()
        if isinstance(sliced, Record):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def loading_response_phase(self):
        """
        Get data during the loading response phase (footstrike to midstance).
        """
        sliced = self.copy()[self.footstrike_time : self.midstance_time]
        out = WholeBody()
        if isinstance(sliced, Record):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def propulsion_phase(self):
        """
        Get data during the propulsion phase (midstance to toe-off).
        """
        sliced = self.copy()[self.midstance_time : self.end_time]
        out = WholeBody()
        if isinstance(sliced, Record):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def flight_time(self):
        """
        Get the flight phase duration in seconds.
        """
        return self.footstrike_time - self.init_time

    @property
    def loadingresponse_time(self):
        """
        Get the loading response phase duration in seconds.
        """
        return self.midstance_time - self.footstrike_time

    @property
    def propulsion_time(self):
        """
        Get the propulsion phase duration in seconds.
        """
        return self.end_time - self.midstance_time

    @property
    def contact_time(self):
        """
        Get the contact phase duration in seconds.
        """
        return self.end_time - self.footstrike_time

    @property
    def overstride(self):
        """
        Get the overstride distance defined as the difference between the
        horizontal position of the center of mass and the horizontal position
        of the foot at footstrike.

        Returns
        -------
        float
            The overstride distance.
        """
        res = self.resultant_force
        plv = self.pelvis
        fs = self.footstrike_time
        if res is None or plv is None or fs is None:
            raise ValueError("Required data for overstride calculation is missing.")
        mask = plv.index >= fs
        com = plv.loc[mask].center[plv.anteroposterior_axis]
        res = res.loc[com.index].origin[res.anteroposterior_axis]
        return float((res.to_numpy()[0][0] - com.to_numpy()[0][0]))

    def _footstrike_kinetics(self):
        """
        Detect footstrike time using ground reaction force data.

        Identifies footstrike as the last sample below the height threshold
        before the peak vertical ground reaction force.

        Raises
        ------
        ValueError
            If no ground reaction force data is available or no footstrike is found.
        """

        # get the contact phase samples
        grf = self.resultant_force
        if grf is None:
            raise ValueError("no ground reaction force data available.")
        vgrf = grf.force.copy()[self.vertical_axis].to_numpy().flatten()
        time = grf.index
        grfn = vgrf / np.max(vgrf)
        mask = np.where(grfn[: np.argmax(grfn)] < self.height_threshold)[0]

        # extract the first contact time
        if len(mask) == 0:
            raise ValueError("no footstrike has been found.")

        return float(time[mask[-1]])

    def _footstrike_kinematics(self):
        """
        Detect footstrike time using marker trajectory data.

        Identifies footstrike by finding the first sample below the height
        threshold for heel and/or metatarsal head markers. Returns the
        earliest footstrike time if multiple markers are available.

        Raises
        ------
        ValueError
            If no footstrike has been found or required markers are missing.
        """

        # get the relevant vertical coordinates
        contact_foot = self.side.lower()
        fs_time = []
        for marker in ["heel", "metatarsal_head"]:
            val = self.get(f"{contact_foot}_{marker}")
            if val is None:
                continue

            # rescale the signal
            time = val.index
            arr = val.copy()[self.vertical_axis].to_numpy().flatten()  # type: ignore
            arr_min = np.min(arr)
            arr = (arr - arr_min) / (np.max(arr) - arr_min)

            # extract the contact time
            fsi = np.where(arr < self.height_threshold)[0]
            if len(fsi) == 0 or fsi[0] == 0:
                raise ValueError("no footstrike has been found.")
            fs_time += [time[fsi[0]]]

        # get output time
        if len(fs_time) > 0:
            return float(np.min(fs_time))
        raise ValueError("no footstrike has been found.")

    def _midstance_kinetics(self):
        """
        Detect midstance time using ground reaction force data.

        Identifies midstance as the time of peak vertical ground reaction force.

        Raises
        ------
        ValueError
            If no ground reaction force data is available.
        """

        grf = self.resultant_force
        if grf is None:
            raise ValueError("no ground reaction force data available.")
        vgrf = grf.force.copy()[self.vertical_axis].to_numpy().flatten()
        time = grf.index
        return float(time[np.argmax(vgrf)])

    def _midstance_kinematics(self):
        """
        Detect midstance time using marker trajectory data.

        Identifies midstance as the time of minimum vertical position of
        the mean of available foot markers (heel, toe, metatarsal head).

        Raises
        ------
        ValueError
            If none of the required markers are available.
        """

        # get the available markers
        lbls = [f"{self.side.lower()}_{i}" for i in ["heel", "toe"]]
        lbls += [f"{self.side.lower()}_metatarsal_head"]

        # get the mean vertical signal
        time = None
        ref = []
        for lbl in lbls:
            val = self.get(lbl)
            if val is None:
                continue
            if time is None:
                time = val.index
            ref += [val.copy()[self.vertical_axis].to_numpy().flatten()]
        ref = np.mean(np.vstack(np.atleast_2d(ref)), axis=0)  # type: ignore
        if time is None or len(ref) == 0:
            raise ValueError(f"None of {lbls} were found.")

        # return the time corresponding to the minimum value
        return float(time[np.argmin(ref)])

    def __init__(
        self,
        speed: int | float,
        grade: int | float,
        side: Literal["right", "left"],
        algorithm: Literal["kinematics", "kinetics"],
        ground_reaction_force_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
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
        sc: Point3D | None = None,  # sternoclavicular joint
        head_anterior: Point3D | None = None,
        head_posterior: Point3D | None = None,
        head_left: Point3D | None = None,
        head_right: Point3D | None = None,
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a RunningStep instance.

        Parameters
        ----------
        speed : int or float
            Running speed value.
        grade : int or float
            Running grade (incline) value.
        side : {'left', 'right'}
            Side of the body this step represents.
        algorithm : {'kinematics', 'kinetics'}
            Cycle detection algorithm to use.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force (in Newtons) for contact detection.
            Default is DEFAULT_MINIMUM_CONTACT_GRF_N.
        height_threshold : float or int, optional
            Maximum vertical height (as percentage) for contact detection.
            Default is DEFAULT_MINIMUM_HEIGHT_PERCENTAGE.
        left_hand_ground_reaction_force : ForcePlatform or None, optional
            Force platform data for left hand contact.
        right_hand_ground_reaction_force : ForcePlatform or None, optional
            Force platform data for right hand contact.
        left_foot_ground_reaction_force : ForcePlatform or None, optional
            Force platform data for left foot contact.
        right_foot_ground_reaction_force : ForcePlatform or None, optional
            Force platform data for right foot contact.
        left_heel : Point3D or None, optional
            Left heel marker trajectory.
        right_heel : Point3D or None, optional
            Right heel marker trajectory.
        left_toe : Point3D or None, optional
            Left toe marker trajectory.
        right_toe : Point3D or None, optional
            Right toe marker trajectory.
        left_first_metatarsal_head : Point3D or None, optional
            Left first metatarsal head marker.
        left_fifth_metatarsal_head : Point3D or None, optional
            Left fifth metatarsal head marker.
        right_first_metatarsal_head : Point3D or None, optional
            Right first metatarsal head marker.
        right_fifth_metatarsal_head : Point3D or None, optional
            Right fifth metatarsal head marker.
        left_ankle_medial : Point3D or None, optional
            Left ankle medial malleolus marker.
        left_ankle_lateral : Point3D or None, optional
            Left ankle lateral malleolus marker.
        right_ankle_medial : Point3D or None, optional
            Right ankle medial malleolus marker.
        right_ankle_lateral : Point3D or None, optional
            Right ankle lateral malleolus marker.
        left_knee_medial : Point3D or None, optional
            Left knee medial epicondyle marker.
        left_knee_lateral : Point3D or None, optional
            Left knee lateral epicondyle marker.
        right_knee_medial : Point3D or None, optional
            Right knee medial epicondyle marker.
        right_knee_lateral : Point3D or None, optional
            Right knee lateral epicondyle marker.
        left_trochanter : Point3D or None, optional
            Left greater trochanter marker.
        right_trochanter : Point3D or None, optional
            Right greater trochanter marker.
        left_asis : Point3D or None, optional
            Left anterior superior iliac spine marker.
        right_asis : Point3D or None, optional
            Right anterior superior iliac spine marker.
        left_psis : Point3D or None, optional
            Left posterior superior iliac spine marker.
        right_psis : Point3D or None, optional
            Right posterior superior iliac spine marker.
        left_shoulder_anterior : Point3D or None, optional
            Left shoulder anterior marker.
        left_shoulder_posterior : Point3D or None, optional
            Left shoulder posterior marker.
        left_acromion : Point3D or None, optional
            Left acromion (shoulder tip) marker.
        right_shoulder_anterior : Point3D or None, optional
            Right shoulder anterior marker.
        right_shoulder_posterior : Point3D or None, optional
            Right shoulder posterior marker.
        right_acromion : Point3D or None, optional
            Right acromion (shoulder tip) marker.
        left_elbow_medial : Point3D or None, optional
            Left elbow medial epicondyle marker.
        left_elbow_lateral : Point3D or None, optional
            Left elbow lateral epicondyle marker.
        right_elbow_medial : Point3D or None, optional
            Right elbow medial epicondyle marker.
        right_elbow_lateral : Point3D or None, optional
            Right elbow lateral epicondyle marker.
        left_wrist_medial : Point3D or None, optional
            Left wrist medial marker.
        left_wrist_lateral : Point3D or None, optional
            Left wrist lateral marker.
        right_wrist_medial : Point3D or None, optional
            Right wrist medial marker.
        right_wrist_lateral : Point3D or None, optional
            Right wrist lateral marker.
        s2 : Point3D or None, optional
            Second sacral vertebra marker.
        l2 : Point3D or None, optional
            Second lumbar vertebra marker.
        c7 : Point3D or None, optional
            Seventh cervical vertebra marker.
        t5 : Point3D or None, optional
            Fifth thoracic vertebra marker.
        sc : Point3D or None, optional
            Sternoclavicular joint marker.
        head_anterior : Point3D or None, optional
            Head anterior marker.
        head_posterior : Point3D or None, optional
            Head posterior marker.
        head_left : Point3D or None, optional
            Head left side marker.
        head_right : Point3D or None, optional
            Head right side marker.
        **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
            Additional signals (e.g., joint angles, EMG channels, other markers).
        """
        super().__init__(
            speed=speed,
            grade=grade,
            side=side,
            algorithm=algorithm,
            ground_reaction_force_threshold=ground_reaction_force_threshold,
            height_threshold=height_threshold,
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
            c7=c7,
            t5=t5,
            sc=sc,
            l2=l2,
            head_anterior=head_anterior,
            head_posterior=head_posterior,
            head_left=head_left,
            head_right=head_right,
            **extra_signals,
        )
