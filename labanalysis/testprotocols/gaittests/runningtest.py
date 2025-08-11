"""Running test module"""

#! IMPORTS


from typing import Literal

import pandas as pd

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...frames.records.forceplatforms import ForcePlatform
from ...frames.records.locomotion.running import RunningExercise
from ...frames.records.timeseriesrecord import TimeseriesRecord
from ...frames.timeseries.emgsignal import EMGSignal
from ...frames.timeseries.point3d import Point3D
from ...frames.timeseries.signal1d import Signal1D
from ...frames.timeseries.signal3d import Signal3D
from ..protocols import Participant, TestProtocol

__all__ = ["RunningTest"]


#! CLASSESS


class RunningTest(RunningExercise, TestProtocol):

    @property
    def results(self):
        cop_list = []
        grf_list = []
        metrics = []
        horizontal_axes = [self.lateral_axis, self.anteroposterior_axis]
        for i, cycle in enumerate(self.cycles):

            # add cop
            res = cycle.resultant_force
            if res is None:
                continue
            cop = res["origin"].copy().reset_time(inplace=False).to_dataframe()[horizontal_axes]  # type: ignore
            cop.insert(0, "Time", cop.index)
            cop.insert(0, "Cycle", i + 1)
            cop.insert(0, "Side", cycle.side)
            cop_list += [cop]

            # add grf
            grf = res["force"].copy().reset_time().to_dataframe()[[self.vertical_axis]]  # type: ignore
            grf.insert(0, "Time", grf.index)
            grf.insert(0, "Cycle", i + 1)
            grf.insert(0, "Side", cycle.side)
            grf_list += [grf]

            # add summary metrics
            metrics_cycle = cycle.output_metrics
            metrics_cycle.insert(0, "cycle", i + 1)
            metrics += [metrics_cycle]

        # outcomes
        out = {
            "summary": pd.concat(metrics, ignore_index=True),
            "analytics": {
                "centre_of_pressure": pd.concat(cop_list, ignore_index=True),
                "ground_reaction_force": pd.concat(grf_list, ignore_index=True),
            },
        }
        return out

    def __init__(
        self,
        participant: Participant,
        normative_data_path: str = "",
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
        self.set_participant(participant)
        self.set_normative_data_path(normative_data_path)

    @classmethod
    def from_tdf(
        cls,
        file: str,
        participant: Participant,
        normative_data_path: str = "",
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
                if read is None:
                    raise ValueError(f"{key} not found in the provided file.")
                objects[key] = read
                record.drop(val, True)
        others = {i: v for i, v in record.items()}

        return cls(
            participant=participant,
            normative_data_path=normative_data_path,
            algorithm=algorithm,
            ground_reaction_force_threshold=ground_reaction_force_threshold,
            height_threshold=height_threshold,
            **objects,  # type: ignore
            **others,  # type: ignore
        )
