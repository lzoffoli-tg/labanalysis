"""Running exercise module."""

import warnings
from typing import Literal

import numpy as np

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...records.forceplatform import ForcePlatform
from ...signalprocessing import find_peaks
from ...timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from .gait_exercise import GaitExercise
from .running_step import RunningStep

__all__ = ["RunningExercise"]


class RunningExercise(GaitExercise):
    """
    Represents a running exercise with automatic step detection.

    RunningExercise extends GaitExercise to provide running-specific cycle
    detection algorithms. It automatically identifies individual running steps
    from continuous data using either kinematic (marker-based) or kinetic
    (force platform-based) methods.

    The class handles flight phases characteristic of running gait and provides
    specialized algorithms for detecting toe-off and footstrike events.

    Parameters
    ----------
    speed : int or float
        Running speed value.
    grade : int or float
        Running grade (incline) value.
    algorithm : {'kinematics', 'kinetics'}, optional
        Cycle detection algorithm to use. Default is 'kinematics'.
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
    steps : list of RunningStep
        Detected running steps extracted from the exercise data.

    See Also
    --------
    GaitExercise : Parent class for gait exercises.
    RunningStep : Represents individual running steps.
    WalkingExercise : Exercise class for walking gait.
    """

    def _find_cycles_kinematics(self):
        """
        Find running steps using toe marker trajectories.

        Detects toe-off events by analyzing vertical position of toe markers.
        For each toe marker, identifies peaks in the vertical trajectory and
        determines toe-off as the last sample below the height threshold before
        each peak. Steps are extracted between consecutive toe-off events.

        Raises
        ------
        ValueError
            If left_toe or right_toe markers are missing, or if no toe-offs are found.

        Warns
        -----
        UserWarning
            If left-right step alternation is not guaranteed.
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
        cycles: list[RunningStep] = []
        for t0, t1, side in zip(starts, stops, sides):
            cycles += [self._get_cycle(t0, t1, side)]

        # return
        return cycles

    def _find_cycles_kinetics(self):
        """
        Find running steps using ground reaction force data.

        Detects contact and flight phases from vertical ground reaction force.
        Identifies toe-off and footstrike events around force peaks, then
        determines step side based on medio-lateral center of pressure position.
        Steps are extracted between consecutive toe-off events.

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
        """
        Extract a single running step from the exercise data.

        Creates a RunningStep instance by slicing all signals between start
        and stop times and preserving algorithm parameters.

        Parameters
        ----------
        start : float
            Start time in seconds (toe-off).
        stop : float
            Stop time in seconds (next toe-off).
        side : {'left', 'right'}
            Side of the body for this step.
        """
        args = {
            "side": side,
            "speed": self.speed,
            "grade": self.grade,
            "ground_reaction_force_threshold": self.ground_reaction_force_threshold,
            "height_threshold": self.height_threshold,
            "algorithm": self.algorithm,
        }
        for i, v in self.items():
            sub = v.copy().loc[(v.index >= start) & (v.index <= stop)]
            args[i] = sub
        return RunningStep(**args)  # type: ignore

    @property
    def steps(self):
        """
        Get the detected running steps.

        Type-safe accessor for cycles that ensures all elements are RunningStep instances.

        Raises
        ------
        TypeError
            If any cycle is not a RunningStep instance.
        """
        steps: list[RunningStep] = []
        for cycle in self.cycles:
            if not isinstance(cycle, RunningStep):
                raise TypeError(
                    f"Element in 'cycles' must be an instance of 'RunningStep', got {type(cycle)}"
                )
            steps.append(cycle)
        return steps

    def __init__(
        self,
        speed: int | float,
        grade: int | float,
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
        Initialize a RunningExercise instance.

        Parameters
        ----------
        speed : int or float
            Running speed value.
        grade : int or float
            Running grade (incline) value.
        algorithm : {'kinematics', 'kinetics'}, optional
            Cycle detection algorithm to use. Default is 'kinematics'.
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
