"""Gait cycle base class."""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express.colors as plotly_colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...constants import *
from ...signalprocessing import *
from ..timeseries import *
from ..bodies import WholeBody
from ..records import ForcePlatform, TimeseriesRecord

from ._base import GaitObject

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
    left_first_metatarsal_head : Point3D or None
        Left first metatarsal head marker.
    left_fifth_metatarsal_head : Point3D or None
        Left fifth metatarsal head marker.
    right_first_metatarsal_head : Point3D or None
        Right first metatarsal head marker.
    right_fifth_metatarsal_head : Point3D or None
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
        left_first_metatarsal_head : Point3D or None
        Left first metatarsal head marker.
    left_fifth_metatarsal_head : Point3D or None
        Left fifth metatarsal head marker.
    right_first_metatarsal_head : Point3D or None
        Right first metatarsal head marker.
    right_fifth_metatarsal_head : Point3D or None
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

    def _footstrike_kinetics(self):
        """
        Return the foot-strike time in seconds using the kinetics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _footstrike_kinematics(self):
        """
        Return the foot-strike time in seconds using the kinematics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _midstance_kinetics(self):
        """
        Return the mid-stance time in seconds using the kinetics algorithm.

        Returns
        -------
        float
        """
        raise NotImplementedError

    def _midstance_kinematics(self):
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


