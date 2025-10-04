"""basic gait module

This module provides classes and utilities for gait analysis, including
GaitObject, GaitCycle, and GaitTest, which support kinematic and kinetic
cycle detection, event extraction, and biofeedback summary generation.
"""

#! IMPORTS


import warnings
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express.colors as plotly_colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..constants import *
from ..signalprocessing import *
from .timeseries import *
from .bodies import WholeBody
from .records import *

#! CONSTANTS


__all__ = [
    "GaitExercise",
    "GaitCycle",
    "GaitObject",
    "RunningExercise",
    "WalkingExercise",
    "RunningStep",
    "WalkingStride",
]


class GaitObject(WholeBody):

    _algorithm: Literal["kinetics", "kinematics"]
    _grf_threshold: float
    _height_threshold: float

    # * constructor

    def __init__(
        self,
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
        """
        Initialize a GaitObject.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            The cycle detection algorithm to use.
        left_heel : Point3D or None, optional
            Left heel marker data.
        right_heel : Point3D or None, optional
            Right heel marker data.
        left_toe : Point3D or None, optional
            Left toe marker data.
        right_toe : Point3D or None, optional
            Right toe marker data.
        left_metatarsal_head : Point3D or None, optional
            Left metatarsal head marker data.
        right_metatarsal_head : Point3D or None, optional
            Right metatarsal head marker data.
        ground_reaction_force : ForcePlatform or None
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
        signals = {
            **extra_signals,
            **dict(
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
            ),
        }
        super().__init__(**{i: v for i, v in signals.items() if v is not None})  # type: ignore

        # set the thresholds
        self.set_height_threshold(height_threshold)
        self.set_grf_threshold(ground_reaction_force_threshold)

        # set the algorithm
        self.set_algorithm(algorithm)

    @property
    def algorithm(self):
        """
        Get the selected cycle detection algorithm.

        Returns
        -------
        str
            The algorithm label.
        """
        return self._algorithm

    @property
    def ground_reaction_force_threshold(self):
        """
        Get the ground reaction force threshold.

        Returns
        -------
        float
        """
        return self._grf_threshold

    @property
    def height_threshold(self):
        """
        Get the height threshold.

        Returns
        -------
        float
        """
        return self._height_threshold

    def set_grf_threshold(self, threshold: float | int):
        """
        Set the ground reaction force threshold.

        Parameters
        ----------
        threshold : float or int
            Threshold value.
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a float or int")
        self._grf_threshold = float(threshold)

    def set_height_threshold(self, threshold: float | int):
        """
        Set the height threshold.

        Parameters
        ----------
        threshold : float or int
            Threshold value.
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a float or int")
        self._height_threshold = float(threshold)

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """
        Set the gait cycle detection algorithm.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            Algorithm label.
        """
        algorithms = ["kinematics", "kinetics"]
        if not isinstance(algorithm, str) or algorithm not in algorithms:
            msg = "'algorithm' must be any between 'kinematics' or 'kinetics'."
            raise ValueError(msg)
        algo = algorithm
        if (
            algo == "kinetics"
            and self.resultant_force is None
            and all(
                [
                    self.left_heel is not None,
                    self.left_toe is not None,
                    self.right_heel is not None,
                    self.right_toe is not None,
                ]
            )
        ):
            msg = f"'forceplatforms data' not found. The 'algorithm' option"
            msg += " has been set to 'kinematics'."
            warnings.warn(msg)
            algo = "kinematics"
        elif (
            algo == "kinematics"
            and self.resultant_force is not None
            and not all(
                [
                    self.left_heel is not None,
                    self.left_toe is not None,
                    self.right_heel is not None,
                    self.right_toe is not None,
                ]
            )
        ):
            msg = f"Not all left_heel, right_heel, left_toe and right_toe"
            msg += " markers have been found to run the 'kinematics' algorithm."
            msg += " The 'kinetics' algorithm has therefore been selected."
            warnings.warn(msg)
            algo = "kinetics"
        elif self.resultant_force is None and any(
            [
                self.left_heel is None,
                self.left_toe is None,
                self.right_heel is None,
                self.right_toe is None,
            ]
        ):
            msg = "Neither ground reaction force nor left_heel, right_heel, "
            msg += "left_toe and right_toe markers have been found."
            msg += " Therefore none of the available algorithms can be used."
            raise ValueError(msg)

        self._algorithm = algo


class GaitCycle(GaitObject):
    """
    Basic gait cycle class.

    Parameters
    ----------
    side : {'left', 'right'}
        The side of the cycle.
    algorithm : {'kinematics', 'kinetics'}
        The cycle detection algorithm.
    left_heel : Point3D or None
        Marker data for the left heel.
    right_heel : Point3D or None
        Marker data for the right heel.
    left_toe : Point3D or None
        Marker data for the left toe.
    right_toe : Point3D or None
        Marker data for the right toe.
    left_metatarsal_head : Point3D or None
        Marker data for the left metatarsal head.
    right_metatarsal_head : Point3D or None
        Marker data for the right metatarsal head.
    ground_reaction_force : ForcePlatform or None
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

    Notes
    -----
    The cycle starts from the toeoff and ends at the next toeoff of the same foot.
    """

    # * class variables

    _side: Literal["left", "right"]
    _absolute_time_events: list[str] = [
        "footstrike_s",
        "midstance_s",
        "init_s",
        "end_s",
    ]

    # * constructor

    def __init__(
        self,
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
        Initialize a GaitCycle.

        Parameters
        ----------
        side : {'left', 'right'}
            The side of the cycle.
        algorithm : {'kinematics', 'kinetics'}
            The cycle detection algorithm.
        left_heel : Point3D or None
            Marker data for the left heel.
        right_heel : Point3D or None
            Marker data for the right heel.
        left_toe : Point3D or None
            Marker data for the left toe.
        right_toe : Point3D or None
            Marker data for the right toe.
        left_metatarsal_head : Point3D or None
            Marker data for the left metatarsal head.
        right_metatarsal_head : Point3D or None
            Marker data for the right metatarsal head.
        ground_reaction_force : ForcePlatform or None
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
        self.set_side(side)

    # * attributes

    @property
    def side(self):
        """
        Return the side of the cycle.

        Returns
        -------
        str
        """
        return self._side

    @property
    def init_s(self):
        """
        Return the first toeoff time in seconds.

        Returns
        -------
        float
        """
        if self.algorithm == "kinetics" and self.resultant_force is not None:
            return float(self.resultant_force.index[0])
        elif self.algorithm == "kinematics" and self.left_heel is not None:
            return float(self.left_heel.index[0])
        raise ValueError(f"'{self.algorithm}' is not a valid algorithm label.")

    @property
    def end_s(self):
        """
        Return the toeoff time corresponding to the end of the cycle in seconds.

        Returns
        -------
        float
        """
        if self.algorithm == "kinetics" and self.resultant_force is not None:
            return float(self.resultant_force.index[-1])
        elif self.algorithm == "kinematics" and self.left_heel is not None:
            return float(self.left_heel.index[-1])
        raise ValueError(f"'{self.algorithm}' is not a valid algorithm label.")

    @property
    def cycle_time_s(self):
        """
        Return the cycle time in seconds.

        Returns
        -------
        float
        """
        return self.end_s - self.init_s

    @property
    def footstrike_s(self):
        """
        Return the foot-strike time in seconds.

        Returns
        -------
        float
        """
        if self.algorithm == "kinetics":
            return self._footstrike_kinetics()
        elif self.algorithm == "kinematics":
            return self._footstrike_kinematics()
        raise ValueError(f"{self.algorithm} not supported")

    @property
    def midstance_s(self):
        """
        Return the mid-stance time in seconds.

        Returns
        -------
        float
        """
        if self.algorithm == "kinetics":
            return self._midstance_kinetics()
        elif self.algorithm == "kinematics":
            return self._midstance_kinematics()
        raise ValueError(f"{self.algorithm} not supported")

    @property
    def time_events(self):
        """
        Return all the time events defining the cycle.

        Returns
        -------
        pd.DataFrame
        """
        evts: dict[str, float] = {}
        for lbl in dir(self):
            if lbl.endswith("_s") and not lbl.startswith("_"):
                name = lbl.rsplit("_", 1)[0].strip().split(" ")[0].lower()
                time = getattr(self, lbl)
                perc = time
                if lbl in self._absolute_time_events:
                    perc -= self.init_s
                perc = perc / self.cycle_time_s * 100
                evts[f"{name.lower().replace("_time", "")}_s"] = float(time)
                evts[f"{name.lower().replace("_time", "")}_%"] = float(perc)
        return evts

    @property
    def lateral_displacement(self):
        try:
            obj = self.pelvis_center
        except Exception as exc:
            obj = self.resultant_force
            if obj is None:
                return np.nan
            obj = obj.origin.copy()
        arr = obj[self.lateral_axis].to_numpy()
        return float(np.max(arr) - np.min(arr))

    @property
    def vertical_displacement(self):
        try:
            vt = np.asarray(self.pelvis_center[self.vertical_axis])
            return float(np.max(vt) - np.min(vt))
        except Exception:
            return np.nan

    @property
    def peak_force(self):
        grf = self.resultant_force
        if grf is None:
            return np.nan
        return float(np.max(np.asarray(grf["force"][self.vertical_axis])))

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """

        # get spatio-temporal parameters
        new = {
            "type": self.__class__.__name__,
            "side": self.side,
            **self.time_events,
        }

        # add kinetic parameters
        res = self.resultant_force
        if res is not None:
            cop_unit = res.origin.unit
            grf_unit = res.force.unit
            new.update(
                **{
                    f"vertical_displacement_{cop_unit}": self.vertical_displacement,
                    f"lateral_displacement_{cop_unit}": self.lateral_displacement,
                    f"peak_vertical_force_{grf_unit}": self.peak_force,
                }
            )

        # add kinematic parameters
        for key in self._angular_measures:
            try:
                val = getattr(self, key).copy().to_numpy().flatten()
                new.update(
                    **{
                        f"{key}_min": float(val.min()),
                        f"{key}_max": float(val.max()),
                    }
                )
            except Exception as exc:
                continue

        # add emg mean activation
        for muscle, emgsignal in self.emgsignals.items():
            if isinstance(emgsignal, EMGSignal):
                if emgsignal.side == self.side:
                    avg = float(np.mean(emgsignal.to_numpy()))
                    new[emgsignal.muscle_name] = avg

        return pd.DataFrame(pd.Series(new)).T

    def _footstrike_kinetics(self) -> float:
        """
        Return the foot-strike time in seconds using the kinetics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _footstrike_kinematics(self) -> float:
        """
        Return the foot-strike time in seconds using the kinematics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _midstance_kinetics(self) -> float:
        """
        Return the mid-stance time in seconds using the kinetics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _midstance_kinematics(self) -> float:
        """
        Return the mid-stance time in seconds using the kinematics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def set_side(self, side: Literal["right", "left"]):
        """
        Set the cycle side.

        Parameters
        ----------
        side : {'left', 'right'}
        """
        if not isinstance(side, str):
            raise ValueError("'side' must be 'left' or 'right'.")
        if side not in ["left", "right"]:
            raise ValueError("'side' must be 'left' or 'right'.")
        self._side = side


class GaitExercise(GaitObject):

    @property
    def cycles(self):
        """
        Get the detected gait cycles.

        Returns
        -------
        list of GaitCycle
        """
        if self.algorithm == "kinematics":
            return self._find_cycles_kinematics()
        elif self.algorithm == "kinetics":
            return self._find_cycles_kinetics()
        else:
            raise ValueError(f"{self.algorithm} currently not supported.")

    def _find_cycles_kinetics(self) -> list[GaitCycle]:
        """
        Find the gait cycles using the kinetics algorithm.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def _find_cycles_kinematics(self) -> list[GaitCycle]:
        """
        Find the gait cycles using the kinematics algorithm.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def __init__(
        self,
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
        """
        Initialize a GaitTest.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            The cycle detection algorithm.
        left_heel : Point3D or None
            Marker data for the left heel.
        right_heel : Point3D or None
            Marker data for the right heel.
        left_toe : Point3D or None
            Marker data for the left toe.
        right_toe : Point3D or None
            Marker data for the right toe.
        left_metatarsal_head : Point3D or None
            Marker data for the left metatarsal head.
        right_metatarsal_head : Point3D or None
            Marker data for the right metatarsal head.
        ground_reaction_force : ForcePlatform or None
            Ground reaction force data.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force for contact detection.
        height_threshold : float or int, optional
            Maximum vertical height for contact detection.
        vertical_axis : {'X', 'Y', 'Z'}, optional
            The vertical axis.
        antpos_axis : {'X', 'Y', 'Z'}, optional
            The anterior-posterior axis.
        process_inputs: bool, optional
            If True, the ProcessPipeline integrated within this instance is
            applied. Otherwise raw data are retained.
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

    @classmethod
    def from_tdf(
        cls,
        file: str,
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
        ground_reaction_force_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        left_heel: str | None = None,
        right_heel: str | None = None,
        left_toe: str | None = None,
        right_toe: str | None = None,
        left_metatarsal_head: str | None = None,
        right_metatarsal_head: str | None = None,
        left_ankle_medial: str | None = None,
        left_ankle_lateral: str | None = None,
        right_ankle_medial: str | None = None,
        right_ankle_lateral: str | None = None,
        left_knee_medial: str | None = None,
        left_knee_lateral: str | None = None,
        right_knee_medial: str | None = None,
        right_knee_lateral: str | None = None,
        right_throcanter: str | None = None,
        left_throcanter: str | None = None,
        left_asis: str | None = None,
        right_asis: str | None = None,
        left_psis: str | None = None,
        right_psis: str | None = None,
        left_shoulder_anterior: str | None = None,
        left_shoulder_posterior: str | None = None,
        right_shoulder_anterior: str | None = None,
        right_shoulder_posterior: str | None = None,
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
        sc: str | None = None,  # sternoclavicular joint
    ):
        """
        Generate a GaitTest object directly from a .tdf file.

        Parameters
        ----------
        file : str
            Path to a ".tdf" file.
        algorithm : {'kinematics', 'kinetics'}, optional
            The cycle detection algorithm.
        left_heel : str or None, optional
            Name of the left heel marker in the tdf file.
        right_heel : str or None, optional
            Name of the right heel marker in the tdf file.
        left_toe : str or None, optional
            Name of the left toe marker in the tdf file.
        right_toe : str or None, optional
            Name of the right toe marker in the tdf file.
        left_metatarsal_head : str or None, optional
            Name of the left metatarsal head marker in the tdf file.
        right_metatarsal_head : str or None, optional
            Name of the right metatarsal head marker in the tdf file.
        ground_reaction_force : str or None, optional
            Name of the ground reaction force data in the tdf file.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force for contact detection.
        height_threshold : float or int, optional
            Maximum vertical height for contact detection.
        vertical_axis : {'X', 'Y', 'Z'}, optional
            The vertical axis.
        antpos_axis : {'X', 'Y', 'Z'}, optional
            The anterior-posterior axis.
        process_inputs: bool, optional
            If True, the ProcessPipeline integrated within this instance is
            applied. Otherwise raw data are retained.

        Returns
        -------
        GaitTest
        """
        record = TimeseriesRecord.from_tdf(file)
        labels = {
            "left_hand_ground_reaction_force": left_hand_ground_reaction_force,
            "right_hand_ground_reaction_force": right_hand_ground_reaction_force,
            "left_foot_ground_reaction_force": left_foot_ground_reaction_force,
            "right_foot_ground_reaction_force": right_foot_ground_reaction_force,
            "left_heel": left_heel,
            "right_heel": right_heel,
            "left_toe": left_toe,
            "right_toe": right_toe,
            "left_metatarsal_head": left_metatarsal_head,
            "right_metatarsal_head": right_metatarsal_head,
            "left_ankle_medial": left_ankle_medial,
            "left_ankle_lateral": left_ankle_lateral,
            "right_ankle_medial": right_ankle_medial,
            "right_ankle_lateral": right_ankle_lateral,
            "left_knee_medial": left_knee_medial,
            "left_knee_lateral": left_knee_lateral,
            "right_knee_medial": right_knee_medial,
            "right_knee_lateral": right_knee_lateral,
            "left_throcanter": left_throcanter,
            "right_throcanter": right_throcanter,
            "left_asis": left_asis,
            "right_asis": right_asis,
            "left_psis": left_psis,
            "right_psis": right_psis,
            "left_shoulder_anterior": left_shoulder_anterior,
            "left_shoulder_posterior": left_shoulder_posterior,
            "right_shoulder_anterior": right_shoulder_anterior,
            "right_shoulder_posterior": right_shoulder_posterior,
            "left_elbow_medial": left_elbow_medial,
            "left_elbow_lateral": left_elbow_lateral,
            "right_elbow_medial": right_elbow_medial,
            "right_elbow_lateral": right_elbow_lateral,
            "left_wrist_medial": left_wrist_medial,
            "left_wrist_lateral": left_wrist_lateral,
            "right_wrist_medial": right_wrist_medial,
            "right_wrist_lateral": right_wrist_lateral,
            "s2": s2,
            "c7": c7,
            "sc": sc,
            "l2": l2,
        }
        objects = {}
        for key, val in labels.items():
            if val is not None:
                read = record.get(val)
                if read is not None:
                    objects[key] = read
        extras = {i: v for i, v in record.items() if i not in list(labels.values())}
        objects.update(**extras)

        return cls(
            algorithm=algorithm,
            ground_reaction_force_threshold=ground_reaction_force_threshold,
            height_threshold=height_threshold,
            **objects,  # type: ignore
        )

    def to_plotly_figure(self):

        # get the relevant data
        data = {}
        res = self.resultant_force
        if res is not None:
            data["GRF"] = res["force"].copy()[self.vertical_axis]
            data["COP<sub>ML</sub>"] = res["origin"].copy()[self.lateral_axis]
            data["COP<sub>AP</sub>"] = res["origin"].copy()[self.anteroposterior_axis]
        markers = ["left_heel", "left_metatarsal_head", "left_toe"]
        markers += ["right_heel", "right_metatarsal_head", "right_toe"]
        for marker in markers:
            obj = self.get(marker)
            if obj is not None:
                data[f"{marker}<sub>VT</sub>"] = obj.copy()[self.vertical_axis]

        # extract the time events from each cycle
        target_events = ["init_s", "footstrike_s", "midstance_s", "end_s"]
        events = {}
        cycles = self.cycles
        for cycle in cycles:
            cycle_events = cycle.time_events
            for event in target_events:
                lbl = f"{cycle.side} {event[:-2]} ({event[-1]})"
                if lbl not in list(events.keys()):
                    events[lbl] = []
                events[lbl] += [cycle_events[event]]

        # generate the figure
        fig = make_subplots(
            rows=len(data),
            cols=1,
            shared_xaxes=True,
            shared_yaxes=False,
            row_titles=list(data.keys()),
        )
        fig.update_layout(
            title=fig.__class__.__name__ + f" ('{self.algorithm}' algorithm)",
            template="simple_white",
            height=300 * len(data),
        )
        cmap = plotly_colors.qualitative.Plotly

        # populate with the available data
        for i, (key, value) in enumerate(data.items()):
            y = value.to_numpy().flatten()
            fig.add_trace(
                row=i + 1,
                col=1,
                trace=go.Scatter(
                    x=value.index,
                    y=y,
                    name=key,
                    mode="lines",
                    showlegend=False,
                    legendgroup="signals",
                    legendgrouptitle_text="signals",
                    line_color=cmap[0],
                    line_width=4,
                ),
            )
            fig.update_yaxes(row=i + 1, col=1, title=value.unit)

            # highlight the time events of each cycle
            yrange = [np.min(y), np.max(y)]
            for j, (lbl, values) in enumerate(events.items()):
                for e, val in enumerate(values):
                    fig.add_trace(
                        row=i + 1,
                        col=1,
                        trace=go.Scatter(
                            x=[val, val],
                            y=yrange,
                            name=lbl,
                            showlegend=bool(e == 0) & bool(i == 0),
                            legendgroup=lbl,
                            line_color=cmap[j + 1],
                            opacity=0.5,
                            mode="lines",
                            line_width=2,
                            line_dash="dash",
                        ),
                    )

        return fig


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
            frq, pwr = psd(arr, fsamp)
            ffreq = frq[np.argmax(pwr)]
            dsamples = int(round(fsamp / ffreq * 0.8))

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


class WalkingExercise(GaitExercise):
    """
    Represents a walking test.

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
            If any required marker is missing or no toe-offs have been found.
        """

        # get toe-off times
        time = self.index
        fsamp = float(1 / np.mean(np.diff(time)))
        cycles: list[WalkingStride] = []
        for lbl in ["left_toe", "right_toe"]:

            # get the vertical coordinates of the toe markers
            arr = self[lbl].copy()[self.vertical_axis].to_numpy().flatten()  # type: ignore

            # filter and rescale
            ftoe = arr / np.max(arr)

            # get the minimum reasonable contact time for each step
            frq, pwr = psd(ftoe, fsamp)
            ffrq = frq[np.argmax(pwr)]
            dsamples = int(round(fsamp / ffrq / 2))

            # get the peaks at each cycle
            pks = find_peaks(ftoe, 0.5, dsamples)

            # for each peak obtain the location of the last sample at the
            # required height threshold
            tos = []
            side = lbl.split("_")[0]
            for pk in pks:
                idx = np.where(ftoe[:pk] <= self.height_threshold)[0]
                if len(idx) > 0:
                    line = pd.Series({"time": time[idx[-1]], "side": side})
                    tos += [pd.DataFrame(line).T]

            # wrap the events
            if len(tos) == 0:
                raise ValueError("no toe-offs have been found.")
            tos = pd.concat(tos, ignore_index=True)
            tos = tos.drop_duplicates()
            tos = tos.sort_values("time")
            tos = tos.reset_index(drop=True)

            # check the alternation of the steps
            for i0, i1 in zip(tos.index[:-1], tos.index[1:]):  # type: ignore
                t0 = float(tos.time.values[i0])
                t1 = float(tos.time.values[i1])
                cycles += [
                    self._get_cycle(
                        t0,
                        t1,
                        side,  # type: ignore
                    )
                ]

        # sort the cycles
        cycle_index = np.argsort([i.init_s for i in cycles])
        return [cycles[i] for i in cycle_index]

    def _find_cycles_kinetics(self):
        """
        Find the gait cycles using the kinetics algorithm.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ground_reaction_force not found.
        """

        # get the relevant data
        res = self.resultant_force
        if res is None:
            raise ValueError("ground_reaction_force not found")
        time = res.index
        res_ap = res["force"].copy()[self.anteroposterior_axis].to_numpy().flatten()  # type: ignore
        res_ap -= np.nanmean(res_ap)

        # get the dominant frequency
        fsamp = float(1 / np.mean(np.diff(time)))
        frq, pwr = psd(res_ap, fsamp)
        ffrq = frq[np.argmax(pwr)]

        # find peaks
        min_samp = int(fsamp / ffrq / 2)
        pks = find_peaks(res_ap, 0, min_samp)

        # for each peak pair get the range and obtain the toe-off
        # as the last value occurring before the peaks within the
        # 1 - height_threshold of that range
        toi = []
        for pk in pks:
            thresh = (1 - self.height_threshold) * res_ap[pk]
            loc = np.where(res_ap[:pk] < thresh)[0]
            if len(loc) > 0:
                toi += [loc[-1]]

        # get the latero-lateral centre of pressure
        cop_ml = res["origin"].copy()[self.lateral_axis].to_numpy().flatten()  # type: ignore
        cop_ml -= np.nanmean(cop_ml)

        # get the sin function best fitting the cop_ml
        def _sin_fitted(arr: np.ndarray):
            """fit a sine over arr"""
            rfft = np.fft.rfft(arr - np.mean(arr))
            pwr = psd(arr)[1]
            rfft[pwr < np.max(pwr)] = 0
            return np.fft.irfft(rfft, len(arr))

        sin_ml = _sin_fitted(cop_ml)

        # get the mean latero-lateral position of each toe-off interval
        cnt = [np.arange(i, j + 1) for i, j in zip(toi[:-1], toi[1:])]
        pos = [np.nanmean(sin_ml[i]) for i in cnt]

        # get the sides
        sides = ["left" if i > 0 else "right" for i in pos]

        # generate the steps
        toi_evens = toi[0:-1:2]
        sides_evens = sides[0:-1:2]
        toi_odds = toi[1:-1:2]
        sides_odds = sides[1:-1:2]
        cycles: list[WalkingStride] = []
        for ti, si in zip([toi_evens, toi_odds], [sides_evens, sides_odds]):
            for to, ed, side in zip(ti[:-1], ti[1:], si):
                cycles += [
                    self._get_cycle(
                        float(self.index[to]),
                        float(self.index[ed]),
                        side,  # type: ignore
                    )
                ]

        # sort the cycles
        idx = np.argsort([i.init_s for i in cycles])
        return [cycles[i] for i in idx]

    def _get_cycle(
        self,
        start: float,
        stop: float,
        side: Literal["left", "right"],
    ):
        step = self.copy()[start:stop]
        args = {
            "side": side,
            "ground_reaction_force_threshold": self.ground_reaction_force_threshold,
            "height_threshold": self.height_threshold,
            "algorithm": self.algorithm,
        }
        if step is not None:
            args.update(**{i: v for i, v in step.items()})  # type: ignore
        return WalkingStride(**args)  # type: ignore

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
        Initialize a WalkingTest instance.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}, optional
            Algorithm used for gait cycle detection. 'kinematics' uses marker
            data, 'kinetics' uses force platform data.
        left_heel : Point3D or None, optional
            Marker data for the left heel.
        right_heel : Point3D or None, optional
            Marker data for the right heel.
        left_toe : Point3D or None, optional
            Marker data for the left toe.
        right_toe : Point3D or None, optional
            Marker data for the right toe.
        left_metatarsal_head : Point3D or None, optional
            Marker data for the left metatarsal head.
        right_metatarsal_head : Point3D or None, optional
            Marker data for the right metatarsal head.
        grf : ForcePlatform or None, optional
            Ground reaction force data.
        grf_threshold : float or int, optional
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
