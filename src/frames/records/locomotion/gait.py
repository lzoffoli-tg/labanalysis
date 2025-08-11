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
import plotly.graph_objects as go
import plotly.express.colors as plotly_colors
from plotly.subplots import make_subplots

from ....constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...timeseries.emgsignal import EMGSignal
from ...timeseries.point3d import Point3D
from ...timeseries.signal1d import Signal1D
from ...timeseries.signal3d import Signal3D
from ..bodies import WholeBody
from ..timeseriesrecord import ForcePlatform

#! CONSTANTS


__all__ = ["GaitExercise", "GaitCycle", "GaitObject"]


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
    _footstrike_s: float
    _midstance_s: float
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
        return self._footstrike_s

    @property
    def midstance_s(self):
        """
        Return the mid-stance time in seconds.

        Returns
        -------
        float
        """
        return self._midstance_s

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

    def _update_events(self):
        """
        Update gait events.
        """
        if self.algorithm == "kinetics":
            try:
                self._midstance_s = self._midstance_kinetics()
            except Exception:
                self._midstance_s = np.nan
            try:
                self._footstrike_s = self._footstrike_kinetics()
            except Exception:
                self._footstrike_s = np.nan
        elif self.algorithm == "kinematics":
            try:
                self._midstance_s = self._midstance_kinematics()
            except Exception:
                self._midstance_s = np.nan
            try:
                self._footstrike_s = self._footstrike_kinematics()
            except Exception:
                self._footstrike_s = np.nan

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """
        Set the gait cycle detection algorithm.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            Algorithm label.
        """
        super().set_algorithm(algorithm)
        self._update_events()

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
        record = super().from_tdf(file)
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
                read = record.get(key)
                if read is None:
                    raise ValueError(f"{key} not found in the provided file.")
                objects[key] = read

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
        events = {}
        target_events = ["init_s", "footstrike_s", "midstance_s", "end_s"]
        for cycle in self.cycles:
            for event, value in cycle.time_events.items():
                if event in target_events:
                    if not any([i == event for i in events.keys()]):
                        events[event] = []
                    events[event] += [value]

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
                            showlegend=bool(e == 0),
                            legendgroup="events",
                            legendgrouptitle_text="events",
                            line_color=cmap[j + 1],
                            opacity=0.7,
                            mode="lines",
                            line_width=3,
                        ),
                    )

        return fig
