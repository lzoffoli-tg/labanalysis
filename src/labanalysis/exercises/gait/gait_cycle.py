"""Gait cycle base class."""

from typing import Literal

import numpy as np

from ...constants import *
from ...records.forceplatform import ForcePlatform
from ...records.record import Record
from ...signalprocessing import *
from ...timeseries import *
from .gait_object import GaitObject

__all__ = ["GaitCycle"]


class GaitCycle(GaitObject):
    """
    Represents a single gait cycle from one foot.

    GaitCycle extends GaitObject to represent an individual gait cycle,
    typically from toe-off to the next toe-off of the same foot. It provides
    automatic detection of key gait events (footstrike, midstance) and computes
    spatiotemporal and kinetic parameters.

    The class supports both kinetics-based (force platform) and kinematics-based
    (marker trajectory) event detection algorithms.

    Parameters
    ----------
    speed : float or int
        Gait speed value.
    grade : float or int
        Gait grade (incline) value.
    side : {'left', 'right'}
        Side of the body this cycle represents.
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
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals (e.g., joint angles, EMG channels, other markers).

    Attributes
    ----------
    side : str
        The side of the cycle ('left' or 'right').
    init_s : float
        Initial toe-off time in seconds.
    end_s : float
        Final toe-off time in seconds.
    cycle_time_s : float
        Cycle duration in seconds.
    footstrike_s : float
        Foot-strike time in seconds.
    midstance_s : float
        Mid-stance time in seconds.
    time_events : dict
        Dictionary of all time events in seconds and percentages.
    output_metrics : pd.DataFrame
        Summary metrics for the cycle.

    Notes
    -----
    The cycle starts from toe-off and ends at the next toe-off of the same foot.
    Subclasses must implement _footstrike_kinetics, _footstrike_kinematics,
    _midstance_kinetics, and _midstance_kinematics methods.

    See Also
    --------
    GaitObject : Parent class providing gait analysis infrastructure.
    GaitExercise : Represents multiple gait cycles.
    """

    def __init__(
        self,
        speed: float | int,
        grade: float | int,
        side: Literal["left", "right"],
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
        Initialize a GaitCycle.

        Parameters
        ----------
        speed : float or int
            Gait speed value.
        grade : float or int
            Gait grade (incline) value.
        side : {'left', 'right'}
            Side of the body this cycle represents.
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
        self.set_side(side)

    @property
    def side(self):
        """
        Get the side of the cycle ('left' or 'right').
        """
        return self._side

    @property
    def init_time(self):
        """
        Get the initial toe-off time in seconds.
        """
        if self.algorithm == "kinetics" and self.resultant_force is not None:
            return float(self.resultant_force.index[0])
        elif self.algorithm == "kinematics" and self.left_heel is not None:
            return float(self.left_heel.index[0])
        raise ValueError(f"'{self.algorithm}' is not a valid algorithm label.")

    @property
    def end_time(self):
        """
        Get the final toe-off time in seconds (end of cycle).
        """
        if self.algorithm == "kinetics" and self.resultant_force is not None:
            return float(self.resultant_force.index[-1])
        elif self.algorithm == "kinematics" and self.left_heel is not None:
            return float(self.left_heel.index[-1])
        raise ValueError(f"'{self.algorithm}' is not a valid algorithm label.")

    @property
    def duration(self):
        """
        Get the cycle duration in seconds.
        """
        return self.end_time - self.init_time

    @property
    def length(self):
        """
        Get the cycle length in meters.
        """
        return self.duration * self.speed / 3.6

    @property
    def cadence(self):
        """
        Get the cadence of the cycle in steps per minute.
        """
        return float(60 / self.duration if self.duration > 0 else 0)

    @property
    def footstrike_time(self):
        """
        Get the foot-strike time in seconds using the selected algorithm.
        """
        if self.algorithm == "kinetics":
            return float(self._footstrike_kinetics())
        elif self.algorithm == "kinematics":
            return float(self._footstrike_kinematics())
        raise ValueError(f"{self.algorithm} not supported")

    @property
    def midstance_time(self):
        """
        Get the mid-stance time in seconds using the selected algorithm.
        """
        if self.algorithm == "kinetics":
            return float(self._midstance_kinetics())
        elif self.algorithm == "kinematics":
            return float(self._midstance_kinematics())
        raise ValueError(f"{self.algorithm} not supported")

    @property
    def vertical_displacement(self):
        """
        Get the vertical displacement of the pelvis center during the step.

        Vertical displacement is calculated as the difference between the
        maximum and minimum vertical position of the pelvis center during
        the running step.
        """
        pelvis = self.pelvis
        if pelvis is None:
            raise RuntimeError("Pelvis data not available")
        com = pelvis.center
        vertical_data = com[self.vertical_axis].to_numpy().flatten()
        return float(np.max(vertical_data) - np.min(vertical_data))

    @property
    def lateral_displacement(self):
        """
        Get the lateral displacement of the pelvis center during the step.

        Lateral displacement is calculated as the difference between the
        maximum and minimum lateral position of the pelvis center during
        the running step.
        """
        pelvis = self.pelvis
        if pelvis is None:
            raise RuntimeError("Pelvis data not available")
        com = pelvis.center
        lateral_data = com[self.lateral_axis].to_numpy().flatten()
        return float(np.max(lateral_data) - np.min(lateral_data))

    @property
    def trunk_lateral_flexion(self):
        """
        Get the peak trunk lateral flexion during the step.

        Peak lateral flexion is the maximum absolute value of trunk
        lateral flexion angle during the running step. Returns None
        if trunk data is not available.
        """
        trunk = self.trunk
        if trunk is None:
            raise RuntimeError("Pelvis data not available")
        angles = trunk.lateralflexion.to_numpy().flatten()
        return float(np.max(np.abs(angles)))

    @property
    def pelvis_lateral_tilt(self):
        """
        Get the peak pelvis lateral tilt during the step.

        Peak lateral tilt is the maximum absolute value of pelvis
        lateral tilt angle during the running step. Returns None
        if pelvis data is not available.
        """
        pelvis = self.pelvis
        if pelvis is None:
            raise RuntimeError("Pelvis data not available")
        angles = pelvis.frontal_plane_tilt.to_numpy().flatten()
        return float(np.max(np.abs(angles)))

    @property
    def trunk_rotation(self):
        """
        Get the peak trunk rotation during the step.

        Peak trunk rotation is the maximum absolute value of trunk
        rotation angle during the running step. Returns None if
        trunk data is not available.
        """
        trunk = self.trunk
        if trunk is None:
            raise RuntimeError("Pelvis data not available")
        angles = trunk.rotation.to_numpy().flatten()
        return float(np.max(np.abs(angles)))

    @property
    def pelvis_rotation(self):
        """
        Get the peak pelvis rotation during the step.

        Peak pelvis rotation is the maximum absolute value of pelvis
        rotation angle during the running step. Returns None if
        pelvis data is not available.
        """
        pelvis = self.pelvis
        if pelvis is None:
            raise RuntimeError("Pelvis data not available")
        angles = pelvis.transverse_plane_tilt.to_numpy().flatten()
        return float(np.max(np.abs(angles)))

    @property
    def trunk_forward_incline(self):
        """
        Get the peak trunk forward incline during the step.

        Peak forward incline is the maximum absolute value of trunk
        forward incline angle during the running step. Returns None if
        trunk data is not available.
        """
        trunk = self.trunk
        if trunk is None:
            raise RuntimeError("Trunk data not available")
        angles = trunk.sagittal_plane_tilt.to_numpy().flatten()
        return float(np.max(np.abs(angles)))

    @property
    def left_arm_abduction(self):
        """
        Get the peak left arm abduction during the step.

        Peak left arm abduction is the maximum absolute value of left arm
        abduction angle during the running step. Returns None if left arm
        data is not available.
        """
        left_shoulder = self.left_shoulder
        if left_shoulder is None:
            raise RuntimeError("Left shoulder data not available")
        angles = left_shoulder.adductionabduction.to_numpy().flatten()
        return float(np.max(angles))

    @property
    def right_arm_abduction(self):
        """
        Get the peak right arm abduction during the step.

        Peak right arm abduction is the maximum absolute value of right arm
        abduction angle during the running step. Returns None if right arm
        data is not available.
        """
        right_shoulder = self.right_shoulder
        if right_shoulder is None:
            raise RuntimeError("Right shoulder data not available")
        angles = right_shoulder.adductionabduction.to_numpy().flatten()
        return float(np.max(angles))

    @property
    def left_arm_flexion(self):
        """
        Get the peak left arm flexion during the step.

        Peak left arm flexion is the maximum absolute value of left arm
        flexion angle during the running step. Returns None if left arm
        data is not available.
        """
        left_shoulder = self.left_shoulder
        if left_shoulder is None:
            raise RuntimeError("Left shoulder data not available")
        angles = left_shoulder.flexionextension.to_numpy().flatten()
        return float(np.max(angles))

    @property
    def right_arm_flexion(self):
        """
        Get the peak right arm flexion during the step.

        Peak right arm flexion is the maximum absolute value of right arm
        flexion angle during the running step. Returns None if right arm
        data is not available.
        """
        right_shoulder = self.right_shoulder
        if right_shoulder is None:
            raise RuntimeError("Right shoulder data not available")
        angles = right_shoulder.flexionextension.to_numpy().flatten()
        return float(np.max(angles))

    @property
    def left_hip_flexion(self):
        """
        Get the peak left hip flexion during the step.

        Peak left hip flexion is the maximum absolute value of left hip
        flexion angle during the running step. Returns None if left hip
        data is not available.
        """
        left_hip = self.left_hip
        if left_hip is None:
            raise RuntimeError("Left hip data not available")
        angles = left_hip.flexionextension.to_numpy().flatten()
        return float(np.max(angles))

    @property
    def right_hip_flexion(self):
        """
        Get the peak right hip flexion during the step.

        Peak right hip flexion is the maximum absolute value of right hip
        flexion angle during the running step. Returns None if right hip
        data is not available.
        """
        right_hip = self.right_hip
        if right_hip is None:
            raise RuntimeError("Right hip data not available")
        angles = right_hip.flexionextension.to_numpy().flatten()
        return float(np.max(angles))

    @property
    def left_knee_flexion(self):
        """
        Get the peak left knee flexion during the step.

        Peak left knee flexion is the maximum absolute value of left knee
        flexion angle during the running step. Returns None if left knee
        data is not available.
        """
        left_knee = self.left_knee
        if left_knee is None:
            raise RuntimeError("Left knee data not available")
        angles = left_knee.flexionextension.to_numpy().flatten()
        return float(np.max(angles))

    @property
    def right_knee_flexion(self):
        """
        Get the peak right knee flexion during the step.

        Peak right knee flexion is the maximum absolute value of right knee
        flexion angle during the running step. Returns None if right knee
        data is not available.
        """
        right_knee = self.right_knee
        if right_knee is None:
            raise RuntimeError("Right knee data not available")
        angles = right_knee.flexionextension.to_numpy().flatten()
        return float(np.max(angles))

    @property
    def peak_force(self):
        """
        Get peak vertical ground reaction force during the cycle.

        Returns None if force platform data is not available.
        """
        grf = self.ground_reaction_force
        if grf is None:
            raise RuntimeError("Resultant force not available")
        return float(np.max(np.asarray(grf)))

    @property
    def ground_reaction_force(self):
        """
        Get the ground reaction force signal during the cycle.

        Returns None if force platform data is not available.
        """
        grf = self.resultant_force
        if grf is None:
            raise RuntimeError("Resultant force not available")
        out = grf.force[grf.vertical_axis]
        if not isinstance(out, Signal1D):
            raise RuntimeError("Ground reaction force is not a valid Signal1D")
        return out

    @property
    def centre_of_pressure(self):
        """
        Get the centre of pressure signal during the cycle.

        Returns None if force platform data is not available.
        """
        cop = self.resultant_force
        if cop is None:
            raise RuntimeError("Resultant force not available")
        out = cop.origin.drop(cop.vertical_axis)
        if not isinstance(out, Timeseries):
            raise RuntimeError("Centre of pressure is not a valid Timeseries")
        return out

    @property
    def _absolute_time_metrics(self):
        """private property to describe absolute rather than relative time events"""
        return [
            "footstrike_time",
            "midstance_time",
            "init_time",
            "end_time",
        ]

    @property
    def _temporal_metrics(self):
        """private property to describe temporal metrics"""
        return [
            "duration",
        ]

    @property
    def _frequency_metrics(self):
        """private property to describe frequency metrics"""
        return [
            "cadence",
        ]

    @property
    def _spatial_metrics(self):
        """private property to describe spatial metrics"""
        return [
            "lateral_displacement",
            "length",
            "vertical_displacement",
        ]

    @property
    def _angular_metrics(self):
        """private property to describe angular metrics"""
        return [
            "left_arm_abduction",
            "left_arm_flexion",
            "left_hip_flexion",
            "left_knee_flexion",
            "right_arm_abduction",
            "right_arm_flexion",
            "right_hip_flexion",
            "right_knee_flexion",
            "pelvis_lateral_tilt",
            "pelvis_rotation",
            "trunk_forward_incline",
            "trunk_lateral_flexion",
            "trunk_rotation",
        ]

    @property
    def _kinetic_metrics(self):
        """private property to describe kinetic metrics"""
        return ["peak_force"]

    @property
    def _kinetic_signals(self):
        """private property to describe kinetic 1D signals"""
        return ["ground_reaction_force"]

    @property
    def _displacement_signals(self):
        """private property to describe kinetic 1D signals"""
        return ["centre_of_pressure"]

    @property
    def _angular_signals(self):
        """private property to describe kinetic 1D signals"""
        return [
            "left_arm_abduction_adduction",
            "left_arm_flexion_extension",
            "left_hip_flexion_extension",
            "left_knee_flexion_extension",
            "right_arm_abduction_adduction",
            "right_arm_flexion_extension",
            "right_hip_flexion_extension",
            "right_knee_flexion_extension",
            "pelvis_lateral_tilt",
            "pelvis_rotation",
            "trunk_flexion_extension",
            "trunk_lateral_flexion_extension",
            "trunk_rotation",
        ]

    def get_output_metrics(self, include_emg: bool = True):
        """
        Get summary metrics for the gait cycle.

        Includes spatiotemporal parameters, kinetic metrics (if force platform
        data is available), kinematic parameters (joint angles min/max), and
        EMG mean activation values.

        Parameters
        ----------
        include_emg: bool (default=True)
            Whether to include EMG mean activation values in the output.

        Returns
        -------
        dict[tuple[str, str], str | int | float]
            A dictionary containing the summary metrics for the gait cycle.
        """

        # type checking
        if not isinstance(include_emg, bool):
            raise TypeError("include_emg must be a boolean")

        # get basic parameters
        results: dict[tuple[str, str], str | int | float] = {
            ("speed", "km/h"): self.speed,
            ("pace", "min/km"): self.pace,
            ("grade", "%"): self.speed,
            ("side", "left/right"): self.side,
        }

        # get time events
        for attr_name in self._absolute_time_metrics:
            try:
                value = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(value, (int, float)):
                results[(attr_name, "s")] = round(value, 3)

        # get temporal metrics
        for attr_name in self._temporal_metrics:
            try:
                value = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(value, (int, float)):
                results[(attr_name, "ms")] = int(round(value * 1000, 0))

        # get frequency metrics
        for attr_name in self._frequency_metrics:
            try:
                value = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(value, (int, float)):
                results[(attr_name, "spm")] = int(round(value, 0))

        # get spatial metrics
        for attr_name in self._spatial_metrics:
            try:
                value = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(value, (int, float)):
                results[(attr_name, "mm")] = int(round(value * 1000, 0))

        # get angular metrics
        for attr_name in self._angular_metrics:
            try:
                value = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(value, (int, float)):
                results[(attr_name, "deg")] = int(round(value, 0))

        # get kinetic metrics
        for attr_name in self._kinetic_metrics:
            try:
                value = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(value, (int, float)):
                results[(attr_name, "N")] = int(round(value, 0))

        # add emg mean activation
        if include_emg:
            for emgsignal in self.emgsignals.values():
                if isinstance(emgsignal, EMGSignal):
                    avg = float(np.mean(emgsignal.to_numpy()))
                    name = " ".join([emgsignal.side, emgsignal.muscle_name])
                    unit = emgsignal.unit
                    results[(name, unit)] = avg

        return results

    def get_output_signals(self, include_emg: bool = True):
        """
        Get summary signals for the gait cycle.

        Parameters
        ----------
        include_emg: bool (default=True)
            Whether to include EMG mean activation values in the output.

        Returns
        -------
        Record
            A dictionary containing the signals representing the gait cycle.
        """

        # type checking
        if not isinstance(include_emg, bool):
            raise TypeError("include_emg must be a boolean")

        # get signals
        signals = self._kinetic_signals
        signals += self._displacement_signals
        signals += self._angular_signals
        results = Record()
        for attr_name in signals:
            try:
                value = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(value, Signal1D):
                results[(attr_name, value.unit)] = value

        # add emg mean activation
        if include_emg:
            for emgsignal in self.emgsignals.values():
                if isinstance(emgsignal, EMGSignal):
                    name = " ".join([emgsignal.side, emgsignal.muscle_name])
                    results[name] = emgsignal

        return results

    def _footstrike_kinetics(self):
        """
        Detect foot-strike time using force platform data.

        Must be implemented by subclasses to detect the foot-strike event
        from ground reaction force measurements.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _footstrike_kinematics(self):
        """
        Detect foot-strike time using marker trajectory data.

        Must be implemented by subclasses to detect the foot-strike event
        from heel and toe marker positions.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _midstance_kinetics(self):
        """
        Detect mid-stance time using force platform data.

        Must be implemented by subclasses to detect the mid-stance event
        from ground reaction force measurements.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _midstance_kinematics(self):
        """
        Detect mid-stance time using marker trajectory data.

        Must be implemented by subclasses to detect the mid-stance event
        from heel and toe marker positions.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def set_side(self, side: Literal["right", "left"]):
        """
        Set the cycle side.

        Parameters
        ----------
        side : {'left', 'right'}
            Side of the body this cycle represents.

        Raises
        ------
        ValueError
            If side is not 'left' or 'right'.
        """
        if not isinstance(side, str):
            raise ValueError("'side' must be 'left' or 'right'.")
        if side not in ["left", "right"]:
            raise ValueError("'side' must be 'left' or 'right'.")
        self._side = side

    def _get_constructor_args(self):
        """
        Get constructor arguments for copy/slice operations.

        Used internally by copy() method to preserve gait cycle parameters.
        """
        return {
            "speed": self.speed,
            "grade": self.grade,
            "side": self.side,
            "algorithm": self.algorithm,
            "ground_reaction_force_threshold": self.ground_reaction_force_threshold,
            "height_threshold": self.height_threshold,
        }

    def copy(self):
        """
        Create an independent copy of this GaitCycle.

        Creates a deep copy with all data and parameters (side, algorithm,
        thresholds, speed, grade) preserved.
        """
        # Get constructor args
        constructor_args = self._get_constructor_args()

        # Copy all timeseries data
        data_copy = {i: v.copy() for i, v in self._data.items()}

        # Merge and create new instance
        return self.__class__(**{**constructor_args, **data_copy})  # type: ignore
