"""Running test implementation."""

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...records import ForcePlatform, TimeseriesRecord
from ...exercises.gait import RunningExercise
from ...timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from ...pipelines import get_default_processing_pipeline
from ..participant import Participant
from ..test_protocol import TestProtocol


class RunningTest(RunningExercise, TestProtocol):
    """
    Test protocol for running gait analysis and biomechanical assessment.

    RunningTest extends RunningExercise with test protocol capabilities,
    enabling systematic running gait analysis with participant tracking,
    cycle detection, and automated metrics extraction. The class combines
    biomechanical gait analysis with clinical test protocol structure.

    The test automatically detects individual running steps from continuous
    data, extracts spatiotemporal parameters, and organizes results into
    structured summaries and time-series analytics suitable for clinical
    reporting and research analysis.

    Parameters
    ----------
    participant : Participant
        Participant information including demographics and anthropometrics.
    normative_data : pd.DataFrame, optional
        Reference data for performance comparison. Default is empty DataFrame.
    algorithm : {'kinematics', 'kinetics'}, optional
        Cycle detection algorithm. 'kinematics' uses marker trajectories,
        'kinetics' uses force platform data. Default is 'kinematics'.
    ground_reaction_force_threshold : float or int, optional
        Minimum vertical ground reaction force (N) for foot contact detection
        when using kinetics algorithm. Default is DEFAULT_MINIMUM_CONTACT_GRF_N.
    height_threshold : float or int, optional
        Maximum normalized height for foot contact detection when using
        kinematics algorithm. Default is DEFAULT_MINIMUM_HEIGHT_PERCENTAGE.
    left_foot_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for left foot. Default is None.
    right_foot_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for right foot. Default is None.
    left_heel : Point3D or None, optional
        Left heel marker trajectory. Default is None.
    right_heel : Point3D or None, optional
        Right heel marker trajectory. Default is None.
    left_toe : Point3D or None, optional
        Left toe marker trajectory. Default is None.
    right_toe : Point3D or None, optional
        Right toe marker trajectory. Default is None.
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional biomechanical signals (joint angles, EMG, etc.).

    Attributes
    ----------
    cycles : list of RunningStep
        Detected running steps extracted from continuous data.
    get_results : dict
        Dictionary containing 'summary' DataFrame with per-cycle metrics
        and 'analytics' dict with time-series data (COP, GRF).
    participant : Participant
        Participant demographics and anthropometrics.
    normative_data : pd.DataFrame
        Reference data for normative comparisons.

    Notes
    -----
    Running Gait Characteristics:
    - Flight phase: Period when neither foot contacts the ground
    - Contact phase: Period from footstrike to toe-off
    - Loading response: Footstrike to midstance (shock absorption)
    - Propulsion: Midstance to toe-off (push-off)

    Extracted Metrics (via get_results):
    - Contact time (ms): Duration of foot-ground contact
    - Flight time (ms): Duration of aerial phase
    - Cycle time (ms): Total duration of one step
    - Peak vertical force (N): Maximum ground reaction force
    - Lateral displacement (mm): Mediolateral COP excursion
    - Vertical displacement (mm): Vertical COP excursion

    Algorithm Selection:
    - Kinematics: Requires left_heel, right_heel, left_toe, right_toe markers
    - Kinetics: Requires force platform data (left/right_foot_ground_reaction_force)

    The test inherits full-body biomechanical model from WholeBody, enabling
    joint angle calculations, segment kinematics, and comprehensive movement
    analysis when appropriate markers are provided.

    See Also
    --------
    RunningExercise : Running gait exercise with cycle detection.
    RunningStep : Individual running step with phase segmentation.
    WalkingTest : Walking gait test protocol.
    TestProtocol : Base class for test protocols.

    Examples
    --------
    >>> participant = Participant(age=30, weight=70, height=175)
    >>> test = RunningTest.from_tdf(
    ...     file='running_trial.tdf',
    ...     participant=participant,
    ...     algorithm='kinematics',
    ...     left_heel='LHEE',
    ...     right_heel='RHEE',
    ...     left_toe='LTOE',
    ...     right_toe='RTOE'
    ... )
    >>> results = test.get_results
    >>> print(results['summary'])  # Spatiotemporal parameters per cycle
    >>> print(results['analytics']['ground_reaction_force'])  # GRF time-series
    """

    @property
    def get_results(self):
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
        normative_data: pd.DataFrame = pd.DataFrame(),
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
        self.set_normative_data(normative_data)

    @classmethod
    def from_tdf(
        cls,
        file: str,
        participant: Participant,
        normative_data: pd.DataFrame = pd.DataFrame(),
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
                if read is None:
                    raise ValueError(f"{key} not found in the provided file.")
                objects[key] = read
                record.drop(val, True)
        others = {i: v for i, v in record.items()}

        return cls(
            participant=participant,
            normative_data=normative_data,
            algorithm=algorithm,
            ground_reaction_force_threshold=ground_reaction_force_threshold,
            height_threshold=height_threshold,
            **objects,  # type: ignore
            **others,  # type: ignore
        )


__all__ = ["RunningTest"]
