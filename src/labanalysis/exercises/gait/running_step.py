"""Running step (gait cycle) module."""

from typing import Literal
import warnings
import numpy as np

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...timeseries import Signal1D, Signal3D, EMGSignal, Point3D
from ...records.forceplatform import ForcePlatform
from ...records.timeseriesrecord import TimeseriesRecord
from ...records.body import WholeBody

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
    def flight_phase(self):
        """
        Extract data during the flight phase.

        Returns
        -------
        WholeBody
            All signals sliced from toeoff (init_s) to footstrike.
        """
        sliced = self.copy()[self.init_s : self.footstrike_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def contact_phase(self):
        """
        Extract data during the contact phase.

        Returns
        -------
        WholeBody
            All signals sliced from footstrike to next toeoff (end_s).
        """
        sliced = self.copy()[self.footstrike_s : self.end_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def loading_response_phase(self):
        """
        Extract data during the loading response phase.

        Returns
        -------
        WholeBody
            All signals sliced from footstrike to midstance.
        """
        sliced = self.copy()[self.footstrike_s : self.midstance_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def propulsion_phase(self):
        """
        Extract data during the propulsion phase.

        Returns
        -------
        WholeBody
            All signals sliced from midstance to toeoff (end_s).
        """
        sliced = self.copy()[self.midstance_s : self.end_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def flight_time(self):
        """
        Get the flight time in seconds.

        Returns
        -------
        float
            The flight time in seconds.
        """
        return self.footstrike_s - self.init_s

    @property
    def loadingresponse_time(self):
        """
        Get the loading response time in seconds.

        Returns
        -------
        float
            The loading response time in seconds.
        """
        return self.midstance_s - self.footstrike_s

    @property
    def propulsion_time(self):
        """
        Get the propulsion time in seconds.

        Returns
        -------
        float
            The propulsion time in seconds.
        """
        return self.end_s - self.midstance_s

    @property
    def contact_time(self):
        """
        Get the contact time in seconds.

        Returns
        -------
        float
            The contact time in seconds.
        """
        return self.end_s - self.footstrike_s

    @property
    def peak_braking_force(self):
        """
        Get the peak braking force during loading response phase.

        The braking force is the negative (backward) component of the
        anteroposterior ground reaction force during the loading response
        phase (footstrike to midstance).

        Returns
        -------
        Signal1D or None
            Peak braking force in Newtons, or None if no force data available.
        """
        phase = self.loading_response_phase
        res = phase.resultant_force
        if res is None:
            return None
        try:
            norm: ForcePlatform = self.pelvis.apply(res)  # type: ignore
        except Exception:
            warnings.warn(
                "Error occurred while applying pelvis transformation."
                + "\nData are provided under the global reference frame orientation."
            )
            norm: ForcePlatform = res
        ap_force = norm.force[self.anteroposterior_axis].to_numpy().flatten()

        # Braking = negative values (backward direction)
        braking = ap_force[ap_force < 0]
        if len(braking) == 0:
            return None

        return float(np.abs(np.min(braking)))

    @property
    def peak_propulsion_force(self):
        """
        Get the peak propulsion force during propulsion phase.

        The propulsion force is the positive (forward) component of the
        anteroposterior ground reaction force during the propulsion phase
        (midstance to toe-off).

        Returns
        -------
        Signal1D or None
            Peak propulsion force in Newtons, or None if no force data available.
        """
        phase = self.propulsion_phase
        res = phase.resultant_force
        if res is None:
            return None
        try:
            norm: ForcePlatform = self.pelvis.apply(res)  # type: ignore
        except Exception:
            warnings.warn(
                "Error occurred while applying pelvis transformation."
                + "\nData are provided under the global reference frame orientation."
            )
            norm: ForcePlatform = res
        ap_force = norm.force[self.anteroposterior_axis].to_numpy().flatten()

        # Propulsion = positive values (forward direction)
        propulsion = ap_force[ap_force > 0]
        if len(propulsion) == 0:
            return None

        return float(np.max(propulsion))

    @property
    def vertical_oscillation(self):
        """
        Get the vertical oscillation of the pelvis center during the cycle.

        Vertical oscillation is calculated as the difference between the
        maximum and minimum vertical position of the pelvis center marker
        during the running step.

        Returns
        -------
        Signal1D or None
            Vertical oscillation in meters or millimeters (depending on marker
            unit), or None if pelvis_center is not available.
        """
        pelvis = self.pelvis
        if pelvis is None:
            return None
        com = pelvis.center
        vertical_data = com[self.vertical_axis].to_numpy().flatten()
        return float(np.max(vertical_data) - np.min(vertical_data))

    @property
    def peak_trunk_lateral_flexion(self):
        """
        Get the peak trunk lateral flexion during the cycle.

        Peak lateral flexion is the maximum absolute value of trunk
        lateral flexion angle (in the local reference frame) during
        the running step.

        Returns
        -------
        Signal1D or None
            Peak trunk lateral flexion in degrees, or None if
            trunk_lateralflexion_local is not available.
        """
        trunk = self.trunk
        if trunk is None:
            return None
        trunk_lat = trunk.lateralflexion
        angles = trunk_lat.to_numpy().flatten()
        return float(np.max(np.abs(angles)))

    @property
    def peak_pelvis_lateral_tilt(self):
        """
        Get the peak pelvis lateral tilt during the cycle.

        Peak lateral tilt is the maximum absolute value of pelvis
        lateral tilt angle (in the global reference frame) during
        the running step.

        Returns
        -------
        Signal1D or None
            Peak pelvis lateral tilt in degrees, or None if
            pelvis_lateral_tilt_global is not available.
        """
        pelvis = self.pelvis
        if pelvis is None:
            return None
        angles = pelvis.frontal_plane_tilt.to_numpy().flatten()
        return float(np.max(np.abs(angles)))

    @property
    def peak_trunk_rotation(self):
        """
        Get the peak trunk rotation during the cycle.

        Peak trunk rotation is the maximum absolute value of trunk
        rotation angle during the running step.

        Returns
        -------
        Signal1D or None
            Peak trunk rotation in degrees, or None if
            trunk_rotation is not available.
        """
        trunk = self.trunk
        if trunk is None:
            return None
        angles = trunk.rotation.to_numpy().flatten()
        return float(np.max(np.abs(angles)))

    @property
    def peak_pelvis_rotation(self):
        """
        Get the peak pelvis rotation during the cycle.

        Peak pelvis rotation is the maximum absolute value of pelvis
        rotation angle during the running step.

        Returns
        -------
        Signal1D or None
            Peak pelvis rotation in degrees, or None if
            pelvis_rotation is not available.
        """
        pelvis = self.pelvis
        if pelvis is None:
            return None
        angles = pelvis.transverse_plane_tilt.to_numpy().flatten()
        return float(np.max(np.abs(angles)))

    def _footstrike_kinetics(self):
        """
        Find the footstrike time using the kinetics algorithm.

        Returns
        -------
        float
            The footstrike time in seconds.

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
        Find the footstrike time using the kinematics algorithm.

        Returns
        -------
        float
            The footstrike time in seconds.

        Raises
        ------
        ValueError
            If no footstrike has been found.
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
        Find the midstance time using the kinetics algorithm.

        Returns
        -------
        float
            The midstance time in seconds.

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
        Find the midstance time using the kinematics algorithm.

        Returns
        -------
        float
            The midstance time in seconds.
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
        super().__init__(
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
