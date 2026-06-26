"""Gait exercise base class."""

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

class GaitExercise(GaitObject):
    """
    Represents a complete gait exercise containing multiple gait cycles.

    GaitExercise extends GaitObject to automatically detect and extract
    individual gait cycles from continuous locomotion data. Subclasses
    implement specific cycle detection algorithms for different locomotion
    types (e.g., running, walking).

    The class provides the `cycles` property which returns a list of detected
    GaitCycle objects. The detection algorithm used depends on the inherited
    `algorithm` attribute ('kinetics' or 'kinematics').

    Parameters
    ----------
    Inherits all parameters from GaitObject.

    Attributes
    ----------
    cycles : list of GaitCycle
        Detected gait cycles extracted from the exercise data.

    Notes
    -----
    This is an abstract base class. Subclasses must implement:
    - _find_cycles_kinetics() : Detect cycles using force platform data
    - _find_cycles_kinematics() : Detect cycles using marker trajectories

    The cycles property automatically calls the appropriate detection method
    based on the selected algorithm.

    See Also
    --------
    GaitObject : Parent class providing gait analysis infrastructure.
    RunningExercise : Exercise class for running-specific cycle detection.
    WalkingExercise : Exercise class for walking-specific cycle detection.
    GaitCycle : Represents individual gait cycles.
    """

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

    def _find_cycles_kinetics(self):
        """
        Find the gait cycles using the kinetics algorithm.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def _find_cycles_kinematics(self):
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
        left_first_metatarsal_head: str | None = None,
        left_fifth_metatarsal_head: str | None = None,
        right_first_metatarsal_head: str | None = None,
        right_fifth_metatarsal_head: str | None = None,
        left_ankle_medial: str | None = None,
        left_ankle_lateral: str | None = None,
        right_ankle_medial: str | None = None,
        right_ankle_lateral: str | None = None,
        left_knee_medial: str | None = None,
        left_knee_lateral: str | None = None,
        right_knee_medial: str | None = None,
        right_knee_lateral: str | None = None,
        right_trochanter: str | None = None,
        left_trochanter: str | None = None,
        left_asis: str | None = None,
        right_asis: str | None = None,
        left_psis: str | None = None,
        right_psis: str | None = None,
        left_shoulder_anterior: str | None = None,
        left_shoulder_posterior: str | None = None,
        left_acromion: str | None = None,
        right_shoulder_anterior: str | None = None,
        right_shoulder_posterior: str | None = None,
        right_acromion: str | None = None,
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
        t5: str | None = None,
        sc: str | None = None,  # sternoclavicular joint
        head_anterior: str | None = None,
        head_posterior: str | None = None,
        head_left: str | None = None,
        head_right: str | None = None,
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
        left_first_metatarsal_head : str or None, optional
            Name of the left first metatarsal head marker in the tdf file.
        left_fifth_metatarsal_head : str or None, optional
            Name of the left fifth metatarsal head marker in the tdf file.
        right_first_metatarsal_head : str or None, optional
            Name of the right first metatarsal head marker in the tdf file.
        right_fifth_metatarsal_head : str or None, optional
            Name of the right fifth metatarsal head marker in the tdf file.
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
            "left_first_metatarsal_head": left_first_metatarsal_head,
            "left_fifth_metatarsal_head": left_fifth_metatarsal_head,
            "right_first_metatarsal_head": right_first_metatarsal_head,
            "right_fifth_metatarsal_head": right_fifth_metatarsal_head,
            "left_ankle_medial": left_ankle_medial,
            "left_ankle_lateral": left_ankle_lateral,
            "right_ankle_medial": right_ankle_medial,
            "right_ankle_lateral": right_ankle_lateral,
            "left_knee_medial": left_knee_medial,
            "left_knee_lateral": left_knee_lateral,
            "right_knee_medial": right_knee_medial,
            "right_knee_lateral": right_knee_lateral,
            "left_trochanter": left_trochanter,
            "right_trochanter": right_trochanter,
            "left_asis": left_asis,
            "right_asis": right_asis,
            "left_psis": left_psis,
            "right_psis": right_psis,
            "left_shoulder_anterior": left_shoulder_anterior,
            "left_shoulder_posterior": left_shoulder_posterior,
            "left_acromion": left_acromion,
            "right_shoulder_anterior": right_shoulder_anterior,
            "right_shoulder_posterior": right_shoulder_posterior,
            "right_acromion": right_acromion,
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
            "t5": t5,
            "sc": sc,
            "l2": l2,
            "head_anterior": head_anterior,
            "head_posterior": head_posterior,
            "head_left": head_left,
            "head_right": head_right,
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
        markers = ["left_heel", "left_first_metatarsal_head", "left_fifth_metatarsal_head", "left_toe"]
        markers += ["right_heel", "right_first_metatarsal_head", "right_fifth_metatarsal_head", "right_toe"]
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


