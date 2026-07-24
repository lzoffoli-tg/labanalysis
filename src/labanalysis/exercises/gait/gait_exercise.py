"""Gait exercise base class."""

from pathlib import Path
from typing import Literal

import numpy as np
import plotly.express.colors as plotly_colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...constants import *
from ...records.forceplatform import ForcePlatform
from ...records.record import Record
from ...signalprocessing import *
from ...timeseries import *
from .gait_object import GaitObject

__all__ = ["GaitExercise"]


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
        Get the detected gait cycles using the selected algorithm.
        """
        if self.algorithm == "kinematics":
            return self._find_cycles_kinematics()
        elif self.algorithm == "kinetics":
            return self._find_cycles_kinetics()
        else:
            raise ValueError(f"{self.algorithm} currently not supported.")

    def _find_cycles_kinetics(self):
        """
        Find gait cycles using force platform data.

        Must be implemented by subclasses to provide kinetics-based
        cycle detection using ground reaction force data.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def _find_cycles_kinematics(self):
        """
        Find gait cycles using marker trajectory data.

        Must be implemented by subclasses to provide kinematics-based
        cycle detection using heel and toe marker positions.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def __init__(
        self,
        speed: int | float,
        grade: int | float,
        algorithm: Literal["kinematics", "kinetics"] = "kinetics",
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
        Initialize a GaitExercise.

        Parameters
        ----------
        speed : int or float
            Gait speed value.
        grade : int or float
            Gait grade (incline) value.
        algorithm : {'kinematics', 'kinetics'}, optional
            Cycle detection algorithm to use. Default is 'kinetics'.
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

    @classmethod
    def from_tdf(
        cls,
        file: str | Path,
        speed: int | float,
        grade: int | float,
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
        Create a GaitExercise object from a .tdf file.

        Reads marker trajectories and force platform data from a BTS Bioengineering
        .tdf file and creates a GaitExercise instance with the specified parameters.

        Parameters
        ----------
        file : str
            Path to the .tdf file.
        speed : int or float
            Gait speed value.
        grade : int or float
            Gait grade (incline) value.
        algorithm : {'kinematics', 'kinetics'}, optional
            Cycle detection algorithm to use. Default is 'kinematics'.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force (in Newtons) for contact detection.
            Default is DEFAULT_MINIMUM_CONTACT_GRF_N.
        height_threshold : float or int, optional
            Maximum vertical height (as percentage) for contact detection.
            Default is DEFAULT_MINIMUM_HEIGHT_PERCENTAGE.
        left_hand_ground_reaction_force : str or None, optional
            Name of the left hand force platform signal in the tdf file.
        right_hand_ground_reaction_force : str or None, optional
            Name of the right hand force platform signal in the tdf file.
        left_foot_ground_reaction_force : str or None, optional
            Name of the left foot force platform signal in the tdf file.
        right_foot_ground_reaction_force : str or None, optional
            Name of the right foot force platform signal in the tdf file.
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
        left_ankle_medial : str or None, optional
            Name of the left ankle medial marker in the tdf file.
        left_ankle_lateral : str or None, optional
            Name of the left ankle lateral marker in the tdf file.
        right_ankle_medial : str or None, optional
            Name of the right ankle medial marker in the tdf file.
        right_ankle_lateral : str or None, optional
            Name of the right ankle lateral marker in the tdf file.
        left_knee_medial : str or None, optional
            Name of the left knee medial marker in the tdf file.
        left_knee_lateral : str or None, optional
            Name of the left knee lateral marker in the tdf file.
        right_knee_medial : str or None, optional
            Name of the right knee medial marker in the tdf file.
        right_knee_lateral : str or None, optional
            Name of the right knee lateral marker in the tdf file.
        left_trochanter : str or None, optional
            Name of the left trochanter marker in the tdf file.
        right_trochanter : str or None, optional
            Name of the right trochanter marker in the tdf file.
        left_asis : str or None, optional
            Name of the left ASIS marker in the tdf file.
        right_asis : str or None, optional
            Name of the right ASIS marker in the tdf file.
        left_psis : str or None, optional
            Name of the left PSIS marker in the tdf file.
        right_psis : str or None, optional
            Name of the right PSIS marker in the tdf file.
        left_shoulder_anterior : str or None, optional
            Name of the left shoulder anterior marker in the tdf file.
        left_shoulder_posterior : str or None, optional
            Name of the left shoulder posterior marker in the tdf file.
        left_acromion : str or None, optional
            Name of the left acromion marker in the tdf file.
        right_shoulder_anterior : str or None, optional
            Name of the right shoulder anterior marker in the tdf file.
        right_shoulder_posterior : str or None, optional
            Name of the right shoulder posterior marker in the tdf file.
        right_acromion : str or None, optional
            Name of the right acromion marker in the tdf file.
        left_elbow_medial : str or None, optional
            Name of the left elbow medial marker in the tdf file.
        left_elbow_lateral : str or None, optional
            Name of the left elbow lateral marker in the tdf file.
        right_elbow_medial : str or None, optional
            Name of the right elbow medial marker in the tdf file.
        right_elbow_lateral : str or None, optional
            Name of the right elbow lateral marker in the tdf file.
        left_wrist_medial : str or None, optional
            Name of the left wrist medial marker in the tdf file.
        left_wrist_lateral : str or None, optional
            Name of the left wrist lateral marker in the tdf file.
        right_wrist_medial : str or None, optional
            Name of the right wrist medial marker in the tdf file.
        right_wrist_lateral : str or None, optional
            Name of the right wrist lateral marker in the tdf file.
        s2 : str or None, optional
            Name of the S2 vertebra marker in the tdf file.
        l2 : str or None, optional
            Name of the L2 vertebra marker in the tdf file.
        c7 : str or None, optional
            Name of the C7 vertebra marker in the tdf file.
        t5 : str or None, optional
            Name of the T5 vertebra marker in the tdf file.
        sc : str or None, optional
            Name of the sternoclavicular joint marker in the tdf file.
        head_anterior : str or None, optional
            Name of the head anterior marker in the tdf file.
        head_posterior : str or None, optional
            Name of the head posterior marker in the tdf file.
        head_left : str or None, optional
            Name of the head left marker in the tdf file.
        head_right : str or None, optional
            Name of the head right marker in the tdf file.
        """
        record = Record.from_tdf(file)
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
            speed=speed,
            grade=grade,
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
            data["GRF"] = res.force.copy()[self.vertical_axis]
            data["COP<sub>ML</sub>"] = res.origin.copy()[self.lateral_axis]
            data["COP<sub>AP</sub>"] = res.origin.copy()[self.anteroposterior_axis]
        markers = [
            "left_heel",
            "left_first_metatarsal_head",
            "left_fifth_metatarsal_head",
            "left_toe",
        ]
        markers += [
            "right_heel",
            "right_first_metatarsal_head",
            "right_fifth_metatarsal_head",
            "right_toe",
        ]
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
