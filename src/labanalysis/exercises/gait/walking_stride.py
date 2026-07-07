"""Walking-specific gait classes."""

from typing import Literal

import numpy as np

from ...constants import *
from ...records.body import WholeBody
from ...records.forceplatform import ForcePlatform
from ...records.timeseriesrecord import TimeseriesRecord
from ...signalprocessing import *
from ...timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from .gait_cycle import GaitCycle

__all__ = ["WalkingStride"]


class WalkingStride(GaitCycle):
    """
    Represents a single walking stride.

    Parameters
    ----------
    side : Literal['left', 'right']
        The side of the cycle.
    frame : StateFrame
        A stateframe object containing all the available kinematic, kinetic and EMG data related to the cycle.
    algorithm : Literal['kinematics', 'kinetics'], optional
        Algorithm used for gait cycle detection. 'kinematics' uses marker data, 'kinetics' uses force platform data.
    left_heel : Point3D or None, optional
        The left heel marker data.
    right_heel : Point3D or None, optional
        The right heel marker data.
    left_toe : Point3D or None, optional
        The left toe marker data.
    right_toe : Point3D or None, optional
        The right toe marker data.
    left_first_metatarsal_head : Point3D or None
        Left first metatarsal head marker.
    left_fifth_metatarsal_head : Point3D or None
        Left fifth metatarsal head marker.
    right_first_metatarsal_head : Point3D or None
        Right first metatarsal head marker.
    right_fifth_metatarsal_head : Point3D or None, optional
        The left metatarsal head marker data.
    right_metatarsal_head : Point3D or None, optional
        The right metatarsal head marker data.
    ground_reaction_force : ForcePlatform or None, optional
        Ground reaction force data.
    ground_reaction_force_threshold : float or int, optional
        Minimum ground reaction force for contact detection.
    height_threshold : float or int, optional
        Maximum vertical height for contact detection.
    vertical_axis : Literal['X', 'Y', 'Z'], optional
        The vertical axis.
    antpos_axis : Literal['X', 'Y', 'Z'], optional
        The anterior-posterior axis.

    Note
    ----
    The cycle starts from the toe-off and ends at the next toe-off of the same foot.
    """

    _opposite_footstrike_s: float
    _absolute_time_events = [
        "footstrike_s",
        "opposite_footstrike_s",
        "midstance_s",
        "init_s",
        "end_s",
    ]

    @property
    def swing_phase(self):
        """
        Get the TimeseriesRecord corresponding to the swing phase of the step.

        Returns
        -------
        TimeseriesRecord
            The TimeseriesRecord for the swing phase.
        """
        sliced = self.copy()[self.init_s : self.footstrike_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def stance_phase(self):
        """
        Get the TimeseriesRecord corresponding to the contact phase.

        Returns
        -------
        TimeseriesRecord
            The TimeseriesRecord for the contact phase.
        """
        sliced = self.copy()[self.footstrike_s : self.end_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def swing_time_s(self):
        """
        Get the swing time in seconds.

        Returns
        -------
        float
            The swing time in seconds.
        """
        return self.footstrike_s - self.init_s

    @property
    def stance_time_s(self):
        """
        Get the stance time in seconds.

        Returns
        -------
        float
            The stance time in seconds.
        """
        return self.end_s - self.footstrike_s

    @property
    def opposite_footstrike_s(self):
        """
        Get the time corresponding to the footstrike of the opposite leg.

        Returns
        -------
        float
            The time of the opposite footstrike in seconds.
        """
        if self.algorithm == "kinetics":
            return self._opposite_footstrike_kinetics()
        elif self.algorithm == "kinematics":
            return self._opposite_footstrike_kinematics()
        raise ValueError(f"{self.algorithm} not supported")

    @property
    def first_double_support_phase(self):
        """
        Get the TimeseriesRecord corresponding to the first double support phase.

        Returns
        -------
        TimeseriesRecord
            The TimeseriesRecord for the first double support phase.
        """
        sliced = self.copy()[self.footstrike_s : self.midstance_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def first_double_support_time_s(self):
        """
        Get the first double support time in seconds.

        Returns
        -------
        float
            The first double support time in seconds.
        """
        return self.midstance_s - self.footstrike_s

    @property
    def second_double_support_phase(self):
        """
        Get the TimeseriesRecord corresponding to the second double support phase.

        Returns
        -------
        TimeseriesRecord
            The TimeseriesRecord for the second double support phase.
        """
        sliced = self.copy()[self.opposite_footstrike_s : self.end_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def second_double_support_time_s(self):
        """
        Get the second double support time in seconds.

        Returns
        -------
        float
            The second double support time in seconds.
        """
        return self.end_s - self.opposite_footstrike_s

    @property
    def single_support_phase(self):
        """
        Get the TimeseriesRecord corresponding to the single support phase.

        Returns
        -------
        TimeseriesRecord
            The TimeseriesRecord for the single support phase.
        """
        sliced = self.copy()[self.midstance_s : self.opposite_footstrike_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def single_support_time_s(self):
        """
        Get the single support time in seconds.

        Returns
        -------
        float
            The single support time in seconds.
        """
        return self.opposite_footstrike_s - self.midstance_s

    def _get_grf_positive_crossing_times(self):
        """
        Find the positive crossings over the mean force.

        Returns
        -------
        np.ndarray
            Array of positive crossing times.

        Raises
        ------
        ValueError
            If no ground reaction force data is available.
        """
        # get the ground reaction force
        res = self.resultant_force
        if res is None:
            raise ValueError("no ground reaction force data available.")
        time = res.index
        vres = res.force.copy()[self.vertical_axis].to_numpy().flatten()
        vres -= np.nanmean(vres)
        vres /= np.max(vres)

        # get the zero-crossing points
        zeros, signs = crossings(vres, 0)
        return time[zeros[signs > 0]].astype(float)

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
            If no footstrike has been found.
        """
        positive_zeros = self._get_grf_positive_crossing_times()
        if len(positive_zeros) == 0:
            raise ValueError("no footstrike has been found.")
        return float(positive_zeros[0])

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
        vcoords = {}
        time = self.index
        for marker in ["heel", "metatarsal_head"]:
            lbl = f"{self.side.lower()}_{marker}"
            dfr = getattr(self, lbl)
            if dfr is not None:
                vcoords[lbl] = dfr.copy()[self.vertical_axis].to_numpy().flatten()

        # filter the signals and extract the first contact time
        fs_time = []
        for val in vcoords.values():
            val = val / np.max(val)
            fsi = np.where(val < self.height_threshold)[0]
            if len(fsi) > 0:
                fs_time += [time[fsi[0]]]

        # return
        if len(fs_time) == 0:
            raise ValueError("not footstrike has been found")
        return float(np.min(fs_time))

    def _opposite_footstrike_kinematics(self):
        """
        Find the opposite footstrike time using the kinematics algorithm.

        Returns
        -------
        float
            The opposite footstrike time in seconds.

        Raises
        ------
        ValueError
            If no opposite footstrike has been found.
        """

        # get the opposite leg
        noncontact_foot = "left" if self.side == "right" else "right"

        # get the relevant vertical coordinates
        vcoords = {}
        for marker in ["heel", "metatarsal_head"]:
            lbl = f"{noncontact_foot}_{marker}"
            dfr = getattr(self, lbl)
            if dfr is not None:
                vcoords[lbl] = dfr.copy()[self.vertical_axis].to_numpy().flatten()

        # filter the signals and extract the first contact time
        time = self.index
        fs_time = []
        for val in vcoords.values():
            val = val / np.max(val)
            fsi = np.where(val >= self.height_threshold)[0]
            if len(fsi) > 0 and fsi[-1] + 1 < len(time):
                fs_time += [time[fsi[-1] + 1]]

        # return
        if len(fs_time) == 0:
            raise ValueError("not opposite footstrike has been found")
        return float(np.min(fs_time))

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
            If resultant_force not found or no valid mid-stance was found.
        """

        # get the anterior-posterior resultant force
        res = self.resultant_force
        if res is None:
            raise ValueError("resultant_force not found")
        time = res.index
        res_ap = res.copy()[self.anteroposterior_axis].to_numpy().flatten()  # type: ignore
        res_ap -= np.nanmean(res_ap)

        # get the dominant frequency
        fsamp = float(1 / np.mean(np.diff(time)))
        frq, pwr = psd(res_ap, fsamp)
        ffrq = frq[np.argmax(pwr)]

        # find the local minima
        min_samp = int(fsamp / ffrq / 2)
        mns = find_peaks(-res_ap, 0, min_samp)
        if len(mns) != 2:
            raise ValueError("no valid mid-stance was found.")
        pk = np.argmax(res_ap[mns[0] : mns[1]]) + mns[0]

        # get the range and obtain the toe-off
        # as the last value occurring before the peaks within the
        # 1 - height_threshold of that range
        thresh = (1 - self.height_threshold) * res_ap[pk]
        loc = np.where(res_ap[pk:] < thresh)[0] + pk
        if len(loc) > 0:
            return float(time[loc[0]])
        raise ValueError("no valid mid-stance was found.")

    def _midstance_kinematics(self):
        """
        Find the midstance time using the kinematics algorithm.

        Returns
        -------
        float
            The midstance time in seconds.
        """

        # get the minimum height across all foot markers
        vcoord = []
        time = self.index
        for lbl in ["heel", "metatarsal_head", "toe"]:
            name = f"{self.side.lower()}_{lbl}"
            val = getattr(self, name)
            if val is not None:
                val = val.copy()[self.vertical_axis].to_numpy().flatten()
                val = val / np.max(val)
                vcoord += [val]
        vcoord = np.vstack(np.atleast_2d(*vcoord)).mean(axis=0)
        idx = np.argmin(vcoord)
        return float(time[idx])

    def _opposite_footstrike_kinetics(self):
        """
        Find the opposite footstrike time using the kinetics algorithm.

        Returns
        -------
        float
            The opposite footstrike time in seconds.

        Raises
        ------
        ValueError
            If no opposite footstrike has been found.
        """
        positive_zeros = self._get_grf_positive_crossing_times()
        if len(positive_zeros) < 2:
            raise ValueError("no opposite footstrike has been found.")
        return float(positive_zeros[1])

    def __init__(
        self,
        side: Literal["left", "right"],
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
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
            Initialize a WalkingStride instance.

            Parameters
            ----------
            side : {'left', 'right'}
                The side of the cycle.
            algorithm : {'kinematics', 'kinetics'}
                The cycle detection algorithm.
            left_heel : Point3D or None, optional
                Marker data for the left heel.
            right_heel : Point3D or None, optional
                Marker data for the right heel.
            left_toe : Point3D or None, optional
                Marker data for the left toe.
            right_toe : Point3D or None, optional
                Marker data for the right toe.
            left_first_metatarsal_head : Point3D or None
            Left first metatarsal head marker.
        left_fifth_metatarsal_head : Point3D or None
            Left fifth metatarsal head marker.
        right_first_metatarsal_head : Point3D or None
            Right first metatarsal head marker.
        right_fifth_metatarsal_head : Point3D or None, optional
                Marker data for the left metatarsal head.
            right_metatarsal_head : Point3D or None, optional
                Marker data for the right metatarsal head.
            ground_reaction_force : ForcePlatform or None, optional
                Ground reaction force data.
            ground_reaction_force_threshold : float or int, optional
                Minimum ground reaction force for contact detection.
            height_threshold : float or int, optional
                Maximum vertical height for contact detection.
            vertical_axis : {'X', 'Y', 'Z'}, optional
                The vertical axis.
            antpos_axis : {'X', 'Y', 'Z'}, optional
                The anterior-posterior axis.
            **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
                Additional signals to include.
        """
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
