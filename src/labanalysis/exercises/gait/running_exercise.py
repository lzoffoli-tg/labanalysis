"""Running exercise module."""

import warnings
from typing import Literal

import numpy as np

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...signalprocessing import find_peaks
from ...timeseries import Signal1D, Signal3D, EMGSignal, Point3D
from ...records.forceplatform import ForcePlatform

from .gait_exercise import GaitExercise
from .running_step import RunningStep

__all__ = ["RunningExercise"]


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
        times = []
        sides = []
        for lbl in ["left_toe", "right_toe"]:

            # get the vertical coordinates of the toe markers
            obj = self.get(lbl)
            if obj is None:
                raise ValueError(f"{lbl} is missing.")
            arr = obj.copy()[self.vertical_axis].to_numpy().flatten()

            # filter and rescale
            arr_min = np.min(arr)
            arr = (arr - arr_min) / (np.max(arr) - arr_min)

            # get the minimum reasonable contact time for each step
            time = obj.index
            fsamp = float(1 / np.mean(np.diff(time)))
            dsamples = int(round(fsamp / 7))

            # get the peaks at each cycle
            pks = find_peaks(arr, 0.5, dsamples)

            # for each peak obtain the location of the last sample at the
            # required height threshold
            side = lbl.split("_")[0]
            for pk in pks:
                idx = np.where(arr[:pk] <= self.height_threshold)[0]
                if len(idx) > 0:
                    times += [time[idx[-1]]]
                    sides += [side]

        # sort the events
        if len(times) == 0:
            raise ValueError("no toe-offs have been found.")
        index = np.argsort(times)
        sorted_times = np.array(times)[index]
        sorted_sides = np.array(sides)[index]
        starts = sorted_times[:-1]
        stops = sorted_times[1:]
        sides = sorted_sides[1:]

        # check the alternation of the steps
        if not all(s0 != s1 for s0, s1 in zip(sides[:-1], sides[1:])):
            warnings.warn("Left-Right steps alternation not guaranteed.")

        # extract the cycles
        cycles = []
        for t0, t1, side in zip(starts, stops, sides):
            cycles += [self._get_cycle(t0, t1, side)]

        # return
        return cycles

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
            Initialize a RunningTest instance.

            Parameters
            ----------
            algorithm : {'kinematics', 'kinetics'}, optional
                Algorithm used for gait cycle detection. 'kinematics' uses marker data, 'kinetics' uses force platform data.
            left_heel, right_heel, left_toe, right_toe : Point3D or None, optional
                Marker data for the respective anatomical points.
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
