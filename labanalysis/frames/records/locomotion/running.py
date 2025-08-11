"""
Module for running test analysis.

This module defines classes for performing running test analysis,
including step detection and summary plots.
"""

#! IMPORTS


import warnings
from typing import Literal

import numpy as np
import pandas as pd

from ....constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ....signalprocessing import find_peaks
from ...timeseries.emgsignal import EMGSignal
from ...timeseries.point3d import Point3D
from ...timeseries.signal1d import Signal1D
from ...timeseries.signal3d import Signal3D
from ..bodies import WholeBody
from ..forceplatforms import ForcePlatform
from ..timeseriesrecord import TimeseriesRecord
from .gait import GaitCycle, GaitExercise

__all__ = ["RunningExercise", "RunningStep"]


class RunningStep(GaitCycle):

    @property
    def flight_phase(self):
        sliced = self.copy()[self.init_s : self.footstrike_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def contact_phase(self):
        sliced = self.copy()[self.footstrike_s : self.end_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def loading_response_phase(self):
        sliced = self.copy()[self.footstrike_s : self.midstance_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def propulsion_phase(self):
        sliced = self.copy()[self.midstance_s : self.end_s]
        out = WholeBody()
        if isinstance(sliced, TimeseriesRecord):
            for i, v in sliced.items():
                out[i] = v
        return out

    @property
    def flight_time_s(self):
        """
        Get the flight time in seconds.

        Returns
        -------
        float
            The flight time in seconds.
        """
        return self.footstrike_s - self.init_s

    @property
    def loadingresponse_time_s(self):
        """
        Get the loading response time in seconds.

        Returns
        -------
        float
            The loading response time in seconds.
        """
        return self.midstance_s - self.footstrike_s

    @property
    def propulsion_time_s(self):
        """
        Get the propulsion time in seconds.

        Returns
        -------
        float
            The propulsion time in seconds.
        """
        return self.end_s - self.midstance_s

    @property
    def contact_time_s(self):
        """
        Get the contact time in seconds.

        Returns
        -------
        float
            The contact time in seconds.
        """
        return self.end_s - self.footstrike_s

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
        vcoords = {}
        contact_foot = self.side.lower()
        for marker in ["heel", "metatarsal_head"]:
            lbl = f"{contact_foot}_{marker}"
            val = self[f"{contact_foot}_{marker}"]
            if val is None:
                continue
            vcoords[lbl] = val.copy()[self.vertical_axis].to_numpy().flatten()  # type: ignore

        # filter the signals and extract the first contact time
        time = self.index
        fs_time = []
        for val in vcoords.values():
            val = val / np.max(val)
            fsi = np.where(val < self.height_threshold)[0]
            if len(fsi) == 0 or fsi[0] == 0:
                raise ValueError("not footstrike has been found.")
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
        time = self.index
        ref = np.zeros_like(time)
        for lbl in lbls:
            val = self.get(lbl)
            if val is None:
                continue
            ref += val.copy()[self.vertical_axis].to_numpy().flatten()
        ref /= len(lbls)

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
        left_metatarsal_head: Point3D | None = None,
        right_metatarsal_head: Point3D | None = None,
        left_ankle_medial: Point3D | None = None,
        left_ankle_lateral: Point3D | None = None,
        right_ankle_medial: Point3D | None = None,
        right_ankle_lateral: Point3D | None = None,
        left_knee_medial: Point3D | None = None,
        left_knee_lateral: Point3D | None = None,
        right_knee_medial: Point3D | None = None,
        right_knee_lateral: Point3D | None = None,
        right_throcanter: Point3D | None = None,
        left_throcanter: Point3D | None = None,
        left_asis: Point3D | None = None,
        right_asis: Point3D | None = None,
        left_psis: Point3D | None = None,
        right_psis: Point3D | None = None,
        left_shoulder_anterior: Point3D | None = None,
        left_shoulder_posterior: Point3D | None = None,
        right_shoulder_anterior: Point3D | None = None,
        right_shoulder_posterior: Point3D | None = None,
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
        sc: Point3D | None = None,  # sternoclavicular joint
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
            left_metatarsal_head=left_metatarsal_head,
            right_metatarsal_head=right_metatarsal_head,
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral,
            right_ankle_medial=right_ankle_medial,
            right_ankle_lateral=right_ankle_lateral,
            left_knee_medial=left_knee_medial,
            left_knee_lateral=left_knee_lateral,
            right_knee_medial=right_knee_medial,
            right_knee_lateral=right_knee_lateral,
            left_throcanter=left_throcanter,
            right_throcanter=right_throcanter,
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
            left_shoulder_anterior=left_shoulder_anterior,
            left_shoulder_posterior=left_shoulder_posterior,
            right_shoulder_anterior=right_shoulder_anterior,
            right_shoulder_posterior=right_shoulder_posterior,
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
            sc=sc,
            l2=l2,
            **extra_signals,
        )


class RunningExercise(GaitExercise):
    """
    Represents a running test.

    Parameters
    ----------
    frame : StateFrame
        A stateframe object containing all the available kinematic, kinetic
        and EMG data related to the test.
    algorithm : Literal['kinematics', 'kinetics'], optional
        Algorithm used for gait cycle detection. 'kinematics' uses marker data,
        'kinetics' uses force platform data.
    left_heel : Point3D or None, optional
        The left heel marker data.
    right_heel : Point3D or None, optional
        The right heel marker data.
    left_toe : Point3D or None, optional
        The left toe marker data.
    right_toe : Point3D or None, optional
        The right toe marker data.
    left_metatarsal_head : Point3D or None, optional
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
    """

    def _find_cycles_kinematics(self):
        """
        Find the gait cycles using the kinematics algorithm.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any required marker is missing or no toe-offs are found.
        Warns
        -----
        UserWarning
            If left-right steps alternation is not guaranteed.
        """

        # get toe-off times
        tos = []
        time = self.index
        fsamp = float(1 / np.mean(np.diff(time)))
        for lbl in ["left_toe", "right_toe"]:

            # get the vertical coordinates of the toe markers
            obj = self.get(lbl)
            if obj is None:
                raise ValueError(f"{lbl} is missing.")
            arr = obj.copy()[self.vertical_axis].to_numpy().flatten()

            # filter and rescale
            arr = arr / np.max(arr)

            # get the minimum reasonable contact time for each step
            dsamples = int(round(fsamp / 2))

            # get the peaks at each cycle
            pks = find_peaks(arr, 0.5, dsamples)

            # for each peak obtain the location of the last sample at the
            # required height threshold
            side = lbl.split("_")[0]
            for pk in pks:
                idx = np.where(arr[:pk] <= self.height_threshold)[0]
                if len(idx) > 0:
                    line = pd.Series({"Time": time[idx[-1]], "Side": side})
                    tos += [pd.DataFrame(line).T]

        # wrap the events
        if len(tos) == 0:
            raise ValueError("no toe-offs have been found.")
        tos = pd.concat(tos, ignore_index=True)
        tos = tos.drop_duplicates()
        tos = tos.sort_values("Time")
        tos = tos.reset_index(drop=True)

        # check the alternation of the steps
        sides = tos.Side.values
        if not all(s0 != s1 for s0, s1 in zip(sides[:-1], sides[1:])):
            warnings.warn("Left-Right steps alternation not guaranteed.")

        return [
            self._get_cycle(
                float(tos.Time.values[i0]),
                float(tos.Time.values[i1]),
                tos.Side.values[i1],  # type: ignore
            )
            for i0, i1 in zip(tos.index[:-1], tos.index[1:])
        ]

    def _find_cycles_kinetics(self):
        """
        Find the gait cycles using the kinetics algorithm.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no ground reaction force data is available or no flight phases are found.
        """

        grf = self.resultant_force
        if grf is None:
            raise ValueError("no ground reaction force data available.")

        # get the grf and the latero-lateral COP
        time = grf.index
        cop_ml = grf["origin"].copy()[self.lateral_axis].to_numpy().flatten()  # type: ignore
        vgrf = grf["force"].copy()[self.vertical_axis].to_numpy().flatten()  # type: ignore

        # check if there are flying phases
        flights = vgrf <= self.ground_reaction_force_threshold
        if not any(flights):
            raise ValueError("No flight phases have been found on data.")

        # get the minimum reasonable contact time for each step
        fsamp = float(1 / np.mean(np.diff(time)))
        dsamples = int(round(fsamp / 4))

        # get the peaks in the normalized grf, then return toe-offs and foot
        # strikes
        grfn = vgrf / np.max(vgrf)
        toi = []
        fsi = []
        pks = find_peaks(grfn, 0.5, dsamples)
        for pk in pks:
            to = np.where(grfn[pk:] < self.height_threshold)[0]
            fs = np.where(grfn[:pk] < self.height_threshold)[0]
            if len(fs) > 0 and len(to) > 0:
                toi += [to[0] + pk]
                if len(toi) > 1:
                    fsi += [fs[-1]]
        toi = np.unique(toi)
        fsi = np.unique(fsi)

        # get the mean latero-lateral position of each contact
        contacts = [np.arange(i, j + 1) for i, j in zip(fsi, toi[1:])]
        pos = [np.nanmean(cop_ml[i]) for i in contacts]

        # get the mean value of alternated contacts and set the step sides
        # accordingly
        evens = np.mean(pos[0:-1:2])
        odds = np.mean(pos[1:-1:2])
        sides = []
        for i in np.arange(len(pos)):
            if evens < odds:
                sides += ["left" if i % 2 == 0 else "right"]
            else:
                sides += ["left" if i % 2 != 0 else "right"]

        return [
            self._get_cycle(float(time[to]), float(time[ed]), sd)
            for to, ed, sd in zip(toi[:-1], toi[1:], sides)
        ]

    def _get_cycle(
        self,
        start: float,
        stop: float,
        side: Literal["left", "right"],
    ):
        args = {
            "side": side,
            "ground_reaction_force_threshold": self.ground_reaction_force_threshold,
            "height_threshold": self.height_threshold,
            "algorithm": self.algorithm,
        }
        args.update(**{i: v.copy()[start:stop] for i, v in self.items()})  # type: ignore
        return RunningStep(**args)  # type: ignore

    # * constructor

    def __init__(
        self,
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
        left_metatarsal_head: Point3D | None = None,
        right_metatarsal_head: Point3D | None = None,
        left_ankle_medial: Point3D | None = None,
        left_ankle_lateral: Point3D | None = None,
        right_ankle_medial: Point3D | None = None,
        right_ankle_lateral: Point3D | None = None,
        left_knee_medial: Point3D | None = None,
        left_knee_lateral: Point3D | None = None,
        right_knee_medial: Point3D | None = None,
        right_knee_lateral: Point3D | None = None,
        right_throcanter: Point3D | None = None,
        left_throcanter: Point3D | None = None,
        left_asis: Point3D | None = None,
        right_asis: Point3D | None = None,
        left_psis: Point3D | None = None,
        right_psis: Point3D | None = None,
        left_shoulder_anterior: Point3D | None = None,
        left_shoulder_posterior: Point3D | None = None,
        right_shoulder_anterior: Point3D | None = None,
        right_shoulder_posterior: Point3D | None = None,
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
        sc: Point3D | None = None,  # sternoclavicular joint
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a RunningTest instance.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}, optional
            Algorithm used for gait cycle detection. 'kinematics' uses marker data, 'kinetics' uses force platform data.
        left_heel, right_heel, left_toe, right_toe : Point3D or None, optional
            Marker data for the respective anatomical points.
        left_metatarsal_head : Point3D or None, optional
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
        process_inputs : bool, optional
            If True, process the input data.
        **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
            Additional signals to include.
        """
        super().__init__(
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
            left_metatarsal_head=left_metatarsal_head,
            right_metatarsal_head=right_metatarsal_head,
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral,
            right_ankle_medial=right_ankle_medial,
            right_ankle_lateral=right_ankle_lateral,
            left_knee_medial=left_knee_medial,
            left_knee_lateral=left_knee_lateral,
            right_knee_medial=right_knee_medial,
            right_knee_lateral=right_knee_lateral,
            left_throcanter=left_throcanter,
            right_throcanter=right_throcanter,
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
            left_shoulder_anterior=left_shoulder_anterior,
            left_shoulder_posterior=left_shoulder_posterior,
            right_shoulder_anterior=right_shoulder_anterior,
            right_shoulder_posterior=right_shoulder_posterior,
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
            sc=sc,
            l2=l2,
            **extra_signals,
        )
