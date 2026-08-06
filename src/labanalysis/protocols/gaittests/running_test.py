"""Running test implementation."""

from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from ...constants import (
    DEFAULT_MINIMUM_CONTACT_GRF_N,
    DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
)
from ...exercises.gait import RunningExercise
from ...pipelines.defaults import get_default_processing_pipeline
from ...records import Record
from ...referenceframes.referenceframes import ReferenceFrame
from ..participant import Participant
from ..test_protocol import TestProtocol


class RunningTest(TestProtocol):
    """
    Test protocol for running gait analysis and biomechanical assessment.

    RunningTest extends TestProtocol to provide systematic running gait analysis
    with participant tracking, automated step detection, and comprehensive metrics
    extraction. The class manages multiple running exercise trials and organizes
    results into structured summaries suitable for clinical reporting and research.

    The test automatically detects individual running steps from continuous data
    across multiple trials, extracts spatiotemporal and kinetic parameters, and
    provides interactive visualizations of force profiles.

    Parameters
    ----------
    exercises : list of RunningExercise
        List of running exercise trials to analyze.
    participant : Participant
        Participant information including demographics and anthropometrics.
    normative_data : pd.DataFrame, optional
        Reference data for performance comparison. Default is empty DataFrame.
    emg_normalization_references : Record or str or 'self', optional
        Reference data for EMG normalization. Default is empty Record.
    emg_normalization_function : Callable, optional
        Function to apply for EMG normalization. Default is np.mean.
    emg_activation_references : Record or str or 'self', optional
        Reference data for EMG activation threshold. Default is empty Record.
    emg_activation_threshold : float, optional
        Threshold for EMG activation detection. Default is 3.
    relevant_muscle_map : list of str or None, optional
        List of relevant muscle names to include in analysis. Default is None.

    Attributes
    ----------
    exercises : list of RunningExercise
        Running exercise trials included in the test.
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
    - Peak braking/propulsion forces (N): Anteroposterior force components
    - Vertical oscillation (mm): Pelvis vertical displacement
    - Trunk/pelvis angular metrics (degrees): Peak rotations and tilts

    Algorithm Selection (per exercise):
    - Kinematics: Requires left_heel, right_heel, left_toe, right_toe markers
    - Kinetics: Requires force platform data (left/right_foot_ground_reaction_force)

    See Also
    --------
    RunningExercise : Running gait exercise with cycle detection.
    RunningStep : Individual running step with phase segmentation.
    RunningTestResults : Structured results container.
    TestProtocol : Base class for test protocols.
    """

    def update_results(self, include_emg: bool = False):
        """
        Generate comprehensive running test results.

        Creates structured results including per-step metrics, aggregate statistics,
        time-series analytics, and interactive force profile visualizations.

        Parameters
        ----------
        include_emg : bool, optional
            Include EMG metrics in results. Default is False.

        Notes
        -----
        The returned results include:

        Summary Tables:
        - per_step: Individual metrics for each detected running step
          (contact_time, propulsion_time, flight_time, cadence, peak forces,
          vertical oscillation)
        - aggregate: Mean, standard deviation, coefficient of variation,
          and left-right asymmetry for all metrics

        Analytics:
        - Time-series data in long format with normalized contact phases

        Figures:
        - force_profiles: 2×2 subplot grid showing vertical and anteroposterior
          ground reaction forces for left and right sides, with mean curves
          and standard deviation bands

        Examples
        --------
        >>> test = RunningTest.from_files(['trial.tdf'], [3.0], [0.0], participant=p)
        >>> results = test.get_results()
        >>> print(results.summary['per_step'])
        >>> print(results.summary['aggregate'])
        >>> results.figures['force_profiles'].show()
        """
        from .running_test_results import RunningTestResults

        return RunningTestResults(self, include_emg=include_emg)

    def __init__(
        self,
        exercises: list[RunningExercise],
        participant: Participant,
        normative_data: pd.DataFrame = pd.DataFrame(),
        emg_normalization_references: Record | str | Literal["self"] = Record(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: Record | str | Literal["self"] = Record(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        """
        Initialize a RunningTest instance.

        Parameters
        ----------
        exercises : list of RunningExercise
            List of running exercise trials to analyze.
        participant : Participant
            Participant information including demographics and anthropometrics.
        normative_data : pd.DataFrame, optional
            Reference data for performance comparison. Default is empty DataFrame.
        emg_normalization_references : Record or str or 'self', optional
            Reference data for EMG normalization. Default is empty Record.
        emg_normalization_function : Callable, optional
            Function to apply for EMG normalization. Default is np.mean.
        emg_activation_references : Record or str or 'self', optional
            Reference data for EMG activation threshold. Default is empty Record.
        emg_activation_threshold : float, optional
            Threshold for EMG activation detection. Default is 3.
        relevant_muscle_map : list of str or None, optional
            List of relevant muscle names to include in analysis. Default is None.

        Raises
        ------
        TypeError
            If any exercise in the list is not a RunningExercise instance.
        """
        super().__init__(
            participant=participant,
            normative_data=normative_data,
            emg_normalization_function=emg_normalization_function,
            emg_normalization_references=emg_normalization_references,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
        )
        self.set_participant(participant)
        self.set_normative_data(normative_data)
        self.set_exercises(exercises)

    def set_exercises(self, exercises: list[RunningExercise]):
        """
        Set the running exercises for the test.

        Parameters
        ----------
        exercises : list of RunningExercise
            List of running exercise trials.

        Raises
        ------
        TypeError
            If any exercise is not a RunningExercise instance.
        """
        if not all(isinstance(ex, RunningExercise) for ex in exercises):
            raise TypeError("All exercises must be instances of RunningExercise")
        exes = [e.strip() for e in exercises]
        self._exercises = [e for e in exes if e is not None]

    @property
    def exercises(self):
        """
        Get the running exercises for the test.
        """
        return self._exercises

    @classmethod
    def from_files(
        cls,
        files: list[str | Path],
        speeds: list[float | int],
        grades: list[float | int],
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
        Create a RunningTest from multiple .tdf files.

        Reads biomechanical data from BTS Bioengineering .tdf files and creates
        a complete running test with multiple exercise trials. Each file represents
        one running trial at a specific speed and grade.

        Parameters
        ----------
        files : list of str or Path
            Paths to .tdf files, one per trial.
        speeds : list of float or int
            Running speeds for each trial (same length as files).
        grades : list of float or int
            Running grades (inclines) for each trial (same length as files).
        participant : Participant
            Participant information including demographics and anthropometrics.
        normative_data : pd.DataFrame, optional
            Reference data for performance comparison. Default is empty DataFrame.
        algorithm : {'kinematics', 'kinetics'}, optional
            Cycle detection algorithm. Default is 'kinematics'.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force (in Newtons) for contact detection.
            Default is DEFAULT_MINIMUM_CONTACT_GRF_N.
        height_threshold : float or int, optional
            Maximum vertical height (as percentage) for contact detection.
            Default is DEFAULT_MINIMUM_HEIGHT_PERCENTAGE.
        left_hand_ground_reaction_force : str or None, optional
            Name of left hand force platform signal in tdf files.
        right_hand_ground_reaction_force : str or None, optional
            Name of right hand force platform signal in tdf files.
        left_foot_ground_reaction_force : str or None, optional
            Name of left foot force platform signal in tdf files.
        right_foot_ground_reaction_force : str or None, optional
            Name of right foot force platform signal in tdf files.
        left_heel : str or None, optional
            Name of left heel marker in tdf files.
        right_heel : str or None, optional
            Name of right heel marker in tdf files.
        left_toe : str or None, optional
            Name of left toe marker in tdf files.
        right_toe : str or None, optional
            Name of right toe marker in tdf files.
        left_first_metatarsal_head : str or None, optional
            Name of left first metatarsal head marker in tdf files.
        left_fifth_metatarsal_head : str or None, optional
            Name of left fifth metatarsal head marker in tdf files.
        right_first_metatarsal_head : str or None, optional
            Name of right first metatarsal head marker in tdf files.
        right_fifth_metatarsal_head : str or None, optional
            Name of right fifth metatarsal head marker in tdf files.
        left_ankle_medial : str or None, optional
            Name of left ankle medial marker in tdf files.
        left_ankle_lateral : str or None, optional
            Name of left ankle lateral marker in tdf files.
        right_ankle_medial : str or None, optional
            Name of right ankle medial marker in tdf files.
        right_ankle_lateral : str or None, optional
            Name of right ankle lateral marker in tdf files.
        left_knee_medial : str or None, optional
            Name of left knee medial marker in tdf files.
        left_knee_lateral : str or None, optional
            Name of left knee lateral marker in tdf files.
        right_knee_medial : str or None, optional
            Name of right knee medial marker in tdf files.
        right_knee_lateral : str or None, optional
            Name of right knee lateral marker in tdf files.
        left_trochanter : str or None, optional
            Name of left trochanter marker in tdf files.
        right_trochanter : str or None, optional
            Name of right trochanter marker in tdf files.
        left_asis : str or None, optional
            Name of left ASIS marker in tdf files.
        right_asis : str or None, optional
            Name of right ASIS marker in tdf files.
        left_psis : str or None, optional
            Name of left PSIS marker in tdf files.
        right_psis : str or None, optional
            Name of right PSIS marker in tdf files.
        left_shoulder_anterior : str or None, optional
            Name of left shoulder anterior marker in tdf files.
        left_shoulder_posterior : str or None, optional
            Name of left shoulder posterior marker in tdf files.
        left_acromion : str or None, optional
            Name of left acromion marker in tdf files.
        right_shoulder_anterior : str or None, optional
            Name of right shoulder anterior marker in tdf files.
        right_shoulder_posterior : str or None, optional
            Name of right shoulder posterior marker in tdf files.
        right_acromion : str or None, optional
            Name of right acromion marker in tdf files.
        left_elbow_medial : str or None, optional
            Name of left elbow medial marker in tdf files.
        left_elbow_lateral : str or None, optional
            Name of left elbow lateral marker in tdf files.
        right_elbow_medial : str or None, optional
            Name of right elbow medial marker in tdf files.
        right_elbow_lateral : str or None, optional
            Name of right elbow lateral marker in tdf files.
        left_wrist_medial : str or None, optional
            Name of left wrist medial marker in tdf files.
        left_wrist_lateral : str or None, optional
            Name of left wrist lateral marker in tdf files.
        right_wrist_medial : str or None, optional
            Name of right wrist medial marker in tdf files.
        right_wrist_lateral : str or None, optional
            Name of right wrist lateral marker in tdf files.
        s2 : str or None, optional
            Name of S2 vertebra marker in tdf files.
        l2 : str or None, optional
            Name of L2 vertebra marker in tdf files.
        c7 : str or None, optional
            Name of C7 vertebra marker in tdf files.
        t5 : str or None, optional
            Name of T5 vertebra marker in tdf files.
        sc : str or None, optional
            Name of sternoclavicular joint marker in tdf files.
        head_anterior : str or None, optional
            Name of head anterior marker in tdf files.
        head_posterior : str or None, optional
            Name of head posterior marker in tdf files.
        head_left : str or None, optional
            Name of head left marker in tdf files.
        head_right : str or None, optional
            Name of head right marker in tdf files.

        Raises
        ------
        ValueError
            If files, speeds, and grades lists have different lengths, or if
            any file is not a string or Path object, or if any speed/grade is
            not numeric.
        """
        # input check
        if not all(isinstance(f, (str, Path)) for f in files):
            raise ValueError("All files must be strings or Path objects.")
        if not all(isinstance(s, (float, int)) for s in speeds):
            raise ValueError("All speeds must be numeric values.")
        if not all(isinstance(g, (float, int)) for g in grades):
            raise ValueError("All grades must be numeric values.")
        if len(files) != len(speeds) or len(files) != len(grades):
            raise ValueError(
                "The number of files, speeds, and grades must be the same."
            )

        # read single exercises
        exercises: list[RunningExercise] = []
        for file, speed, grade in zip(files, speeds, grades):
            exercises.append(
                RunningExercise.from_tdf(
                    file,
                    speed,
                    grade,
                    algorithm,
                    ground_reaction_force_threshold,
                    height_threshold,
                    left_hand_ground_reaction_force,
                    right_hand_ground_reaction_force,
                    left_foot_ground_reaction_force,
                    right_foot_ground_reaction_force,
                    left_heel,
                    right_heel,
                    left_toe,
                    right_toe,
                    left_first_metatarsal_head,
                    left_fifth_metatarsal_head,
                    right_first_metatarsal_head,
                    right_fifth_metatarsal_head,
                    left_ankle_medial,
                    left_ankle_lateral,
                    right_ankle_medial,
                    right_ankle_lateral,
                    left_knee_medial,
                    left_knee_lateral,
                    right_knee_medial,
                    right_knee_lateral,
                    right_trochanter,
                    left_trochanter,
                    left_asis,
                    right_asis,
                    left_psis,
                    right_psis,
                    left_shoulder_anterior,
                    left_shoulder_posterior,
                    left_acromion,
                    right_shoulder_anterior,
                    right_shoulder_posterior,
                    right_acromion,
                    left_elbow_medial,
                    left_elbow_lateral,
                    right_elbow_medial,
                    right_elbow_lateral,
                    left_wrist_medial,
                    left_wrist_lateral,
                    right_wrist_medial,
                    right_wrist_lateral,
                    s2,
                    l2,
                    c7,
                    t5,
                    sc,
                    head_anterior,
                    head_posterior,
                    head_left,
                    head_right,
                )
            )

        return cls(
            exercises,
            participant,
            normative_data,
        )

    @property
    def processing_pipeline(self):
        """return the processing pipeline"""
        return get_default_processing_pipeline()

    @property
    def processed_data(self):
        """
        return a Test copy with processed data
        """

        def _change_rf(exe: RunningExercise):
            """
            Change the reference frame for the given running exercise.
            """
            plv = exe.pelvis
            if plv is None:
                raise RuntimeError("pelvis not found within the current exercise.")
            am = plv.asis_midpoint
            am[am.vertical_axis] = 0
            pm = plv.pelvis_midpoint
            pm[pm.vertical_axis] = 0
            ap = (am - pm).to_numpy().mean(axis=0)
            ap = ap / np.linalg.norm(ap)
            vt = np.array([0, 1, 0])
            pc = plv.center.to_numpy().mean(axis=0)
            rf = ReferenceFrame(
                pc,
                None,
                vt,
                ap,
            )
            return rf.apply(exe)

        pro = self.copy()
        pipeline = self.processing_pipeline
        exes = [_change_rf(pipeline(e)) for e in self.exercises]  # type: ignore
        pro.set_exercises(exes)  # type: ignore

        return pro

    def update_results(self, include_emg: bool = True, limit_steps: int | None = None):
        """
        return test results

        Parameters
        ----------
        include_emg: bool
            if True, EMG data is returned (where available)
        limit_steps: int | None
            if provided, limits the number of steps included in the results.
        """
        from .running_test_results import RunningTestResults

        self._results = RunningTestResults(
            self.processed_data,
            include_emg,
            limit_steps,
        )


__all__ = ["RunningTest"]
