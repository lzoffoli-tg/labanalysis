"""Jump test implementation."""

from os.path import exists
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ...constants import MINIMUM_CONTACT_FORCE_N
from ...exercises.jumps import (
    DropJump,
    RepeatedJumps,
    CounterMovementJump,
    SquatJump,
    SingleJump,
)
from ...pipelines.base import ProcessingPipeline
from ...records import ForcePlatform, Record
from ...referenceframes import ReferenceFrame
from ...signalprocessing import butterworth_filt, fillna, rms_filt
from ...timeseries import EMGSignal, Point3D
from ..normativedata import jumps_normative_values
from ..participant import Participant
from ..test_protocol import TestProtocol
from .jump_test_results import JumpTestResults


class JumpTest(TestProtocol):
    """
    Test protocol for vertical jump performance assessment.

    Manages squat jumps (SJ), counter-movement jumps (CMJ), drop jumps (DJ),
    and repeated jumps. Processes force platform and EMG data, applies signal
    processing pipelines, and generates performance reports with normative
    comparisons.

    Parameters
    ----------
    participant : Participant
        Participant information with weight specified.
    normative_data : pd.DataFrame, optional
        Reference data for performance comparison (default: jumps_normative_values).
    emg_normalization_references : Record or str or 'self', optional
        Reference signals for EMG normalization (default: empty Record).
    emg_normalization_function : callable, optional
        Function for normalization value (default: np.mean).
    emg_activation_references : Record or str or 'self', optional
        Reference signals for activation thresholds (default: empty Record).
    emg_activation_threshold : float, optional
        Threshold multiplier for activation onset (default: 3).
    relevant_muscle_map : list of str or None, optional
        Muscle names to include in analysis (default: None, includes all).
    squat_jumps : list of SquatJump, optional
        Squat jump trials (default: empty list).
    counter_movement_jumps : list of CounterMovementJump, optional
        Counter-movement jump trials (default: empty list).
    drop_jumps : list of DropJump, optional
        Drop jump trials (default: empty list).
    repeated_jumps : list of SingleJump, optional
        Repeated jump trials (default: empty list).
    keep_all_data: bool, optional
        if True, all data of each jump is saved. Otherwise, only data
        directly used by the test is retained.

    Attributes
    ----------
    squat_jumps : list of SquatJump
        Concentric-only jumps from static position.
    counter_movement_jumps : list of CounterMovementJump
        Jumps with pre-stretch movement.
    drop_jumps : list of DropJump
        Plyometric jumps from elevated surface.
    repeated_jumps : list of SingleJump
        Continuous jumping sequences.
    jumps : list
        All jump types concatenated.
    processed_data : JumpTest
        Test copy with processed signals.
    processing_pipeline : ProcessingPipeline
        Jump-specific signal processing configuration.

    See Also
    --------
    SquatJump : Concentric-only jump from static position.
    CounterMovementJump : Jump with pre-stretch movement.
    DropJump : Plyometric jump from elevated surface.
    SingleJump : Single vertical jump (base for repeated jumps).
    RepeatedJumps : Continuous jumping sequence.
    JumpTestResults : Results with figures and summaries.

    Examples
    --------
    >>> participant = Participant(weight=75, gender='male')
    >>> test = JumpTest.from_files(
    ...     participant=participant,
    ...     squat_jump_files=['sj_trial1.tdf'],
    ...     counter_movement_jump_files=['cmj_trial1.tdf', 'cmj_trial2.tdf'],
    ...     drop_jump_files=['dj_40cm.tdf'],
    ...     drop_jump_heights_cm=[40]
    ... )
    >>> results = test.get_results(include_emg=True)
    >>> print(results.summary)
    """

    @property
    def repeated_jumps(self):
        """
        Get the list of repeated jump trials.

        Returns
        -------
        list of SingleJump
            All repeated jump trials in the test.

        Notes
        -----
        Repeated jumps are stored as SingleJump instances, which extend
        CounterMovementJump to provide full jump analysis capabilities.
        """
        return self._repeated_jumps

    def add_repeated_jumps(self, *jumps: SingleJump):
        """
        Add one or more repeated jump trials to the test.

        Parameters
        ----------
        *jumps : SingleJump
            Variable number of SingleJump instances to add.

        Raises
        ------
        ValueError
            If any jump is not a SingleJump instance.

        Notes
        -----
        While the parameter type is SingleJump, the validation checks for
        CounterMovementJump as SingleJump extends CounterMovementJump.
        """
        for jump in jumps:
            if not isinstance(jump, SingleJump):
                raise ValueError("jump must be a SingleJump instance.")
            self._repeated_jumps.append(jump.strip())

    def pop_repeated_jumps(self, index: int):
        """
        Remove and return a repeated jump trial at specified index.

        Parameters
        ----------
        index : int
            Zero-based index of the jump to remove.

        Returns
        -------
        SingleJump
            The removed jump trial.

        Raises
        ------
        ValueError
            If index is not an integer or is out of range.
        """
        if not isinstance(index, int):
            raise ValueError("index must be an int.")
        if index < 0 or index > len(self._repeated_jumps) - 1:
            raise ValueError("index out of range.")
        jump = self._repeated_jumps.pop(index)
        return jump

    @property
    def squat_jumps(self):
        """
        Get the list of squat jump trials.

        Returns
        -------
        list of SquatJump
            All squat jump trials in the test.
        """
        return self._squat_jumps

    def add_squat_jumps(self, *jumps: SquatJump):
        """
        Add one or more squat jump trials to the test.

        Parameters
        ----------
        *jumps : SquatJump
            Variable number of SquatJump instances to add.

        Raises
        ------
        ValueError
            If any jump is not a SquatJump instance.
        """
        for jump in jumps:
            if not isinstance(jump, SquatJump):
                raise ValueError("jump must be a SquatJump instance.")
            self._squat_jumps.append(jump.strip())

    def pop_squat_jumps(self, index: int):
        """
        Remove and return a squat jump trial at specified index.

        Parameters
        ----------
        index : int
            Zero-based index of the jump to remove.

        Returns
        -------
        SquatJump
            The removed squat jump trial.

        Raises
        ------
        ValueError
            If index is not an integer or is out of range.
        """
        if not isinstance(index, int):
            raise ValueError("index must be an int.")
        if index < 0 or index > len(self._squat_jumps) - 1:
            raise ValueError("index out of range.")
        squat = self._squat_jumps.pop(index)
        return squat

    @property
    def counter_movement_jumps(self):
        """
        Get the list of counter-movement jump trials.

        Returns
        -------
        list of CounterMovementJump
            All counter-movement jump trials in the test.
        """
        return self._counter_movement_jumps

    def add_counter_movement_jumps(self, *jumps: CounterMovementJump):
        """
        Add one or more counter-movement jump trials to the test.

        Parameters
        ----------
        *jumps : CounterMovementJump
            Variable number of CounterMovementJump instances to add.

        Raises
        ------
        ValueError
            If any jump is not a CounterMovementJump instance.
        """
        for jump in jumps:
            if not isinstance(jump, CounterMovementJump):
                raise ValueError("jump must be a SingleJump instance.")
            self._counter_movement_jumps.append(jump.strip())

    def pop_counter_movement_jumps(self, index: int):
        """
        Remove and return a counter-movement jump trial at specified index.

        Parameters
        ----------
        index : int
            Zero-based index of the jump to remove.

        Returns
        -------
        CounterMovementJump
            The removed counter-movement jump trial.

        Raises
        ------
        ValueError
            If index is not an integer or is out of range.
        """
        if not isinstance(index, int):
            raise ValueError("index must be an int.")
        if index < 0 or index > len(self._counter_movement_jumps) - 1:
            raise ValueError("index out of range.")
        jump = self._counter_movement_jumps.pop(index)
        return jump

    @property
    def drop_jumps(self):
        """
        Get the list of drop jump trials.

        Returns
        -------
        list of DropJump
            All drop jump trials in the test.
        """
        return self._drop_jumps

    def add_drop_jumps(self, *jumps: DropJump):
        """
        Add one or more drop jump trials to the test.

        Parameters
        ----------
        *jumps : DropJump
            Variable number of DropJump instances to add.

        Raises
        ------
        ValueError
            If any jump is not a DropJump instance.
        """
        for jump in jumps:
            if not isinstance(jump, DropJump):
                raise ValueError("jump must be a DropJump instance.")
            self._drop_jumps.append(jump.strip())

    def pop_drop_jumps(self, index: int):
        """
        Remove and return a drop jump trial at specified index.

        Parameters
        ----------
        index : int
            Zero-based index of the jump to remove.

        Returns
        -------
        DropJump
            The removed drop jump trial.

        Raises
        ------
        ValueError
            If index is not an integer or is out of range.
        """
        if not isinstance(index, int):
            raise ValueError("index must be an int.")
        if index < 0 or index > len(self._drop_jumps) - 1:
            raise ValueError("index out of range.")
        jump = self._drop_jumps.pop(index)
        return jump

    @property
    def jumps(self):
        """
        Get all jump trials combined.

        Returns
        -------
        list
            Concatenation of squat jumps, counter-movement jumps, drop jumps,
            and repeated jumps.
        """
        return (
            self.squat_jumps
            + self.counter_movement_jumps
            + self.drop_jumps
            + self.repeated_jumps
        )

    def _filtered_jump_data(
        self, jump: SquatJump | CounterMovementJump | DropJump | SingleJump
    ):
        """
        return the jump according to the 'keep_all_data' spec.

        Parameters
        ----------
        jump: SquatJump | CounterMovementJump | DropJump | SingleJump
            the jump to be filtered

        Returns
        -------
        SquatJump | CounterMovementJump | DropJump | SingleJump
            the filtered jump.

        Note
        ----
        If the keep_all_data attribute is True, the full jump is returned.
        Otherwise, only the data relevant for the test are returned.
        """
        if self.keep_all_data:
            return jump

        # pop all objects that are not relevant
        new = jump.copy()
        for marker in new.points3d.keys():
            if marker != "s2":
                new.drop(marker)
        for signal in new.signals3d.keys():
            new.drop(signal)
        for signal in new.signals1d.keys():
            new.drop(signal)
        for plane in new.planes3d.keys():
            new.drop(plane)
        for ts in new.timeseries.keys():
            new.drop(ts)

        return new.strip().reset_time()

    def __init__(
        self,
        participant: Participant,
        normative_data: pd.DataFrame = jumps_normative_values,
        emg_normalization_references: Record = Record(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: Record = Record(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
        squat_jumps: list[SquatJump] = [],
        counter_movement_jumps: list[CounterMovementJump] = [],
        drop_jumps: list[DropJump] = [],
        repeated_jumps: list[RepeatedJumps] = [],
        keep_all_data: bool = False,
    ):
        """
        Initialize a JumpTest instance.

        Parameters
        ----------
        participant : Participant
            Participant information with weight specified.
        normative_data : pd.DataFrame, optional
            Reference data for performance comparison
            (default: jumps_normative_values).
        emg_normalization_references : Record, optional
            Reference signals for EMG normalization (default: empty Record).
        emg_normalization_function : callable, optional
            Function to compute normalization value from reference signals
            (default: np.mean).
        emg_activation_references : Record, optional
            Reference signals for determining activation thresholds
            (default: empty Record).
        emg_activation_threshold : float, optional
            Threshold multiplier for activation onset detection (default: 3).
        relevant_muscle_map : list of str or None, optional
            Names of muscles to include in analysis. If None, all muscles
            are included (default: None).
        squat_jumps : list of SquatJump, optional
            Squat jump trial instances (default: empty list).
        counter_movement_jumps : list of CounterMovementJump, optional
            Counter-movement jump trial instances (default: empty list).
        drop_jumps : list of DropJump, optional
            Drop jump trial instances (default: empty list).
        repeated_jumps : list of SingleJump, optional
            Repeated jump trial instances (default: empty list).
        keep_all_data: bool, optional
            if True, all data of each jump is saved. Otherwise, only data
            directly used by the test is retained.

        Raises
        ------
        ValueError
            If participant is not a Participant instance.
            If participant's weight is not specified.
        """
        if not isinstance(participant, Participant):
            raise ValueError("participant must be a Participant class instance.")
        if participant.weight is None:
            raise ValueError("participant's weight must be assigned.")
        super().__init__(
            participant=participant,
            normative_data=normative_data,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            relevant_muscle_map=relevant_muscle_map,
            keep_all_data=keep_all_data,
        )
        self._squat_jumps: list[SquatJump] = []
        self._counter_movement_jumps: list[CounterMovementJump] = []
        self._drop_jumps: list[DropJump] = []
        self._repeated_jumps: list[SingleJump] = []
        self.add_squat_jumps(*squat_jumps)
        self.add_counter_movement_jumps(*counter_movement_jumps)
        self.add_drop_jumps(*drop_jumps)
        self.add_repeated_jumps(*repeated_jumps)

    @classmethod
    def from_files(
        cls,
        participant: Participant,
        # EMG
        normative_data: pd.DataFrame = jumps_normative_values,
        emg_normalization_references: Record = Record(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: Record = Record(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
        # Squat jumps
        squat_jump_files: list[str | Path] = [],
        squat_jump_sides: (
            list[Literal["left", "right", "bilateral"] | None] | None
        ) = None,
        # Counter-movement jumps
        counter_movement_jump_files: list[str | Path] = [],
        counter_movement_jump_sides: (
            list[Literal["left", "right", "bilateral"] | None] | None
        ) = None,
        counter_movement_jump_free_hands: list[bool] | None = None,
        # Drop jumps
        drop_jump_files: list[str | Path] = [],
        drop_jump_box_heights_cm: list[int] | None = None,
        drop_jump_sides: (
            list[Literal["left", "right", "bilateral"] | None] | None
        ) = None,
        drop_jump_free_hands: list[bool] | None = None,
        # Repeated jumps
        repeated_jumps_files: list[str | Path] = [],
        repeated_jumps_sides: (
            list[Literal["left", "right", "bilateral"] | None] | None
        ) = None,
        exclude_repeated_jumps: list[list[int]] | None = None,
        repeated_jumps_straight_leg: list[bool] | None = None,
        repeated_jumps_free_hands: list[bool] | None = None,
        # marker/forces labels
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
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
        sc: str | None = None,
        head_anterior: str | None = None,
        head_posterior: str | None = None,
        head_left: str | None = None,
        head_right: str | None = None,
        keep_all_data: bool = False,
    ):
        """
        Create a JumpTest instance from TDF files.

        Parameters
        ----------
        participant : Participant
            Participant information with weight specified.
        normative_data : pd.DataFrame, optional
            Reference data for performance comparison.
        emg_normalization_references : Record, optional
            Reference signals for EMG normalization.
        emg_normalization_function : callable, optional
            Function for normalization value computation.
        emg_activation_references : Record, optional
            Reference signals for activation thresholds.
        emg_activation_threshold : float, optional
            Threshold multiplier for activation onset.
        relevant_muscle_map : list of str or None, optional
            Muscle names to include in analysis.
        squat_jump_files : list of str or Path, optional
            Paths to squat jump TDF files.
        squat_jump_free_hands : list of bool or None, optional
            Whether each squat jump was performed with free hands.
        counter_movement_jump_files : list of str or Path, optional
            Paths to counter-movement jump TDF files.
        counter_movement_jump_free_hands : list of bool or None, optional
            Whether each CMJ was performed with free hands.
        drop_jump_files : list of str or Path, optional
            Paths to drop jump TDF files.
        drop_jump_heights_cm : list of int or None, optional
            Box heights in cm for each drop jump (default: 40cm for all).
        drop_jump_free_hands : list of bool or None, optional
            Whether each drop jump was performed with free hands.
        repeated_jumps_files : list of str or Path, optional
            Paths to repeated jumps TDF files.
        exclude_repeated_jumps : list of list of int or None, optional
            Indices of jumps to exclude from each repeated jumps file.
        repeated_jumps_straight_leg : list of bool or None, optional
            Whether each repeated jump trial used straight leg technique.
        repeated_jumps_free_hands : list of bool or None, optional
            Whether each repeated jump trial was performed with free hands.
        left_foot_ground_reaction_force : str or None, optional
            Label for left foot force platform data.
        right_foot_ground_reaction_force : str or None, optional
            Label for right foot force platform data.
        left_hand_ground_reaction_force : str or None, optional
            Label for left hand force platform data.
        right_hand_ground_reaction_force : str or None, optional
            Label for right hand force platform data.
        left_heel : str or None, optional
            Label for left heel marker.
        right_heel : str or None, optional
            Label for right heel marker.
        left_toe : str or None, optional
            Label for left toe marker.
        right_toe : str or None, optional
            Label for right toe marker.
        left_first_metatarsal_head : str or None, optional
            Label for left first metatarsal head marker.
        left_fifth_metatarsal_head : str or None, optional
            Label for left fifth metatarsal head marker.
        right_first_metatarsal_head : str or None, optional
            Label for right first metatarsal head marker.
        right_fifth_metatarsal_head : str or None, optional
            Label for right fifth metatarsal head marker.
        left_ankle_medial : str or None, optional
            Label for left medial ankle marker.
        left_ankle_lateral : str or None, optional
            Label for left lateral ankle marker.
        right_ankle_medial : str or None, optional
            Label for right medial ankle marker.
        right_ankle_lateral : str or None, optional
            Label for right lateral ankle marker.
        left_knee_medial : str or None, optional
            Label for left medial knee marker.
        left_knee_lateral : str or None, optional
            Label for left lateral knee marker.
        right_knee_medial : str or None, optional
            Label for right medial knee marker.
        right_knee_lateral : str or None, optional
            Label for right lateral knee marker.
        right_trochanter : str or None, optional
            Label for right trochanter marker.
        left_trochanter : str or None, optional
            Label for left trochanter marker.
        left_asis : str or None, optional
            Label for left ASIS marker.
        right_asis : str or None, optional
            Label for right ASIS marker.
        left_psis : str or None, optional
            Label for left PSIS marker.
        right_psis : str or None, optional
            Label for right PSIS marker.
        left_shoulder_anterior : str or None, optional
            Label for left anterior shoulder marker.
        left_shoulder_posterior : str or None, optional
            Label for left posterior shoulder marker.
        left_acromion : str or None, optional
            Label for left acromion marker.
        right_shoulder_anterior : str or None, optional
            Label for right anterior shoulder marker.
        right_shoulder_posterior : str or None, optional
            Label for right posterior shoulder marker.
        right_acromion : str or None, optional
            Label for right acromion marker.
        left_elbow_medial : str or None, optional
            Label for left medial elbow marker.
        left_elbow_lateral : str or None, optional
            Label for left lateral elbow marker.
        right_elbow_medial : str or None, optional
            Label for right medial elbow marker.
        right_elbow_lateral : str or None, optional
            Label for right lateral elbow marker.
        left_wrist_medial : str or None, optional
            Label for left medial wrist marker.
        left_wrist_lateral : str or None, optional
            Label for left lateral wrist marker.
        right_wrist_medial : str or None, optional
            Label for right medial wrist marker.
        right_wrist_lateral : str or None, optional
            Label for right lateral wrist marker.
        s2 : str or None, optional
            Label for S2 vertebra marker.
        l2 : str or None, optional
            Label for L2 vertebra marker.
        c7 : str or None, optional
            Label for C7 vertebra marker.
        t5 : str or None, optional
            Label for T5 vertebra marker.
        sc : str or None, optional
            Label for supraclavicular marker.
        head_anterior : str or None, optional
            Label for anterior head marker.
        head_posterior : str or None, optional
            Label for posterior head marker.
        head_left : str or None, optional
            Label for left head marker.
        head_right : str or None, optional
            Label for right head marker.
        keep_all_data: bool, optional
            if True, all data of each jump is saved. Otherwise, only data
            directly used by the test is retained.

        Returns
        -------
        JumpTest
            A new JumpTest instance with data loaded from the specified files.

        Raises
        ------
        ValueError
            If participant is not a Participant instance.
            If participant's weight is not specified.
            If any file paths are invalid or files do not exist.
            If list lengths do not match corresponding file lists.
            If marker labels are not strings when required.
        """

        # check participant
        if not isinstance(participant, Participant):
            raise ValueError("participant must be a Participant instance.")

        # check bodymass
        bodymass = participant.weight
        if bodymass is None:
            raise ValueError("participant's bodymass must be provided.")

        # check squat jump files
        if not isinstance(squat_jump_files, list) or not all(
            [isinstance(i, (str, Path)) and exists(i) for i in squat_jump_files]
        ):
            msg = "squat_jump_files must be a list of valid tdf file paths "
            msg += "corresponding to SingleJump instances."
            raise ValueError(msg)

        # check squat_jump_sides
        if squat_jump_sides is None:
            squat_jump_sides = [None] * len(squat_jump_files)
        if (
            not isinstance(squat_jump_sides, list)
            or not all(
                [i in ["left", "right", "bilateral", None] for i in squat_jump_sides]
            )
            or len(squat_jump_sides) != len(squat_jump_files)
        ):
            msg = "squat_jump_sides must be a list with the same length as "
            msg += "squat_jump_files, each element being 'left', 'right', 'bilateral', or None."
            raise ValueError(msg)

        # check counter movement jump files
        if not isinstance(counter_movement_jump_files, list) or not all(
            [
                isinstance(i, (str, Path)) and exists(i)
                for i in counter_movement_jump_files
            ]
        ):
            msg = "counter_movement_jump_files must be a list of valid tdf file"
            msg += " paths corresponding to SingleJump instances."
            raise ValueError(msg)

        # check counter_movement_jump_sides
        if counter_movement_jump_sides is None:
            counter_movement_jump_sides = [None] * len(counter_movement_jump_files)
        if (
            not isinstance(counter_movement_jump_sides, list)
            or not all(
                [
                    i in ["left", "right", "bilateral", None]
                    for i in counter_movement_jump_sides
                ]
            )
            or len(counter_movement_jump_sides) != len(counter_movement_jump_files)
        ):
            msg = "counter_movement_jump_sides must be a list with the same length as "
            msg += "counter_movement_jump_files, each element being 'left', 'right', 'bilateral', or None."
            raise ValueError(msg)

        # check counter_movement_jump_free_hands
        if counter_movement_jump_free_hands is None:
            counter_movement_jump_free_hands = [False] * len(
                counter_movement_jump_files
            )
        if (
            not isinstance(counter_movement_jump_free_hands, list)
            or not all([isinstance(i, bool) for i in counter_movement_jump_free_hands])
            or len(counter_movement_jump_free_hands) != len(counter_movement_jump_files)
        ):
            msg = (
                "counter_movement_jump_free_hands must be a list of bool with the same "
            )
            msg += "length as counter_movement_jump_files, each representing whether "
            msg += "the trial was performed with free hands."
            raise ValueError(msg)

        # check drop jump files
        if not isinstance(drop_jump_files, list) or not all(
            [isinstance(i, (str, Path)) and exists(i) for i in drop_jump_files]
        ):
            msg = "drop_jump_files must be a list of valid tdf file"
            msg += " paths corresponding to DropJump instances."
            raise ValueError(msg)

        # check drop jump heights
        if drop_jump_box_heights_cm is None:
            drop_jump_box_heights_cm = [40] * len(drop_jump_files)
        if (
            not isinstance(drop_jump_box_heights_cm, list)
            or not all([isinstance(i, int) for i in drop_jump_box_heights_cm])
            or len(drop_jump_box_heights_cm) != len(drop_jump_files)
        ):
            msg = "drop_jump_box_heights_cm must be a list of int with the same "
            msg += "length as drop_jump_files, each representing the box height "
            msg += "of each drop jump."
            raise ValueError(msg)

        # check drop_jump_sides
        if drop_jump_sides is None:
            drop_jump_sides = [None] * len(drop_jump_files)
        if (
            not isinstance(drop_jump_sides, list)
            or not all(
                [i in ["left", "right", "bilateral", None] for i in drop_jump_sides]
            )
            or len(drop_jump_sides) != len(drop_jump_files)
        ):
            msg = "drop_jump_sides must be a list with the same length as "
            msg += "drop_jump_files, each element being 'left', 'right', 'bilateral', or None."
            raise ValueError(msg)

        # check drop_jump_free_hands
        if drop_jump_free_hands is None:
            drop_jump_free_hands = [True] * len(drop_jump_files)
        if (
            not isinstance(drop_jump_free_hands, list)
            or not all([isinstance(i, bool) for i in drop_jump_free_hands])
            or len(drop_jump_free_hands) != len(drop_jump_files)
        ):
            msg = "drop_jump_free_hands must be a list of bool with the same "
            msg += "length as drop_jump_files, each representing whether "
            msg += "the trial was performed with free hands."
            raise ValueError(msg)

        # check repeated jumps files
        if not isinstance(repeated_jumps_files, list) or not all(
            [isinstance(i, (str, Path)) and exists(i) for i in repeated_jumps_files]
        ):
            msg = "repeated_jumps_files must be a list of valid tdf file"
            msg += " paths corresponding to RepeatedJumps instances."
            raise ValueError(msg)

        # check exclude_repeated_jumps
        if exclude_repeated_jumps is None:
            exclude_repeated_jumps = [[] for _ in repeated_jumps_files]
        if (
            not isinstance(exclude_repeated_jumps, list)
            or not all([isinstance(i, list) for i in exclude_repeated_jumps])
            or not all([isinstance(j, int) for i in exclude_repeated_jumps for j in i])
            or len(exclude_repeated_jumps) != len(repeated_jumps_files)
        ):
            msg = "exclude_repeated_jumps must be a list of lists with the same "
            msg += "length as repeated_jumps_files, each containing int representing "
            msg += "the index of the jumps to exclude from each repeated jump file."
            raise ValueError(msg)

        # check repeated_jumps_straight_leg
        if repeated_jumps_straight_leg is None:
            repeated_jumps_straight_leg = [False] * len(repeated_jumps_files)
        if (
            not isinstance(repeated_jumps_straight_leg, list)
            or not all([isinstance(i, bool) for i in repeated_jumps_straight_leg])
            or len(repeated_jumps_straight_leg) != len(repeated_jumps_files)
        ):
            msg = "repeated_jumps_straight_leg must be a list of bool with the same "
            msg += "length as repeated_jumps_files, each representing whether the "
            msg += "trial used straight leg technique."
            raise ValueError(msg)

        # check repeated_jumps_sides
        if repeated_jumps_sides is None:
            repeated_jumps_sides = [None] * len(repeated_jumps_files)
        if (
            not isinstance(repeated_jumps_sides, list)
            or not all(
                [
                    i in ["left", "right", "bilateral", None]
                    for i in repeated_jumps_sides
                ]
            )
            or len(repeated_jumps_sides) != len(repeated_jumps_files)
        ):
            msg = "repeated_jumps_sides must be a list with the same length as "
            msg += "repeated_jumps_files, each element being 'left', 'right', 'bilateral', or None."
            raise ValueError(msg)

        # check repeated_jumps_free_hands
        if repeated_jumps_free_hands is None:
            repeated_jumps_free_hands = [True] * len(repeated_jumps_files)
        if (
            not isinstance(repeated_jumps_free_hands, list)
            or not all([isinstance(i, bool) for i in repeated_jumps_free_hands])
            or len(repeated_jumps_free_hands) != len(repeated_jumps_files)
        ):
            msg = "repeated_jumps_free_hands must be a list of bool with the same "
            msg += "length as repeated_jumps_files, each representing whether "
            msg += "the trial was performed with free hands."
            raise ValueError(msg)

        # read the files
        sjs = []
        for file, side in zip(squat_jump_files, squat_jump_sides):
            sjs.append(
                SquatJump.from_tdf(
                    filename=file,
                    bodymass_kg=bodymass,
                    side=side,
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
                    l2=l2,
                    c7=c7,
                    t5=t5,
                    sc=sc,
                    head_anterior=head_anterior,
                    head_posterior=head_posterior,
                    head_left=head_left,
                    head_right=head_right,
                )
            )

        cmjs = []
        for file, side, fh in zip(
            counter_movement_jump_files,
            counter_movement_jump_sides,
            counter_movement_jump_free_hands,
        ):
            cmjs.append(
                CounterMovementJump.from_tdf(
                    filename=file,
                    bodymass_kg=bodymass,
                    side=side,
                    free_hands=fh,
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
                    l2=l2,
                    c7=c7,
                    t5=t5,
                    sc=sc,
                    head_anterior=head_anterior,
                    head_posterior=head_posterior,
                    head_left=head_left,
                    head_right=head_right,
                )
            )

        djs = []
        for file, height, side, fh in zip(
            drop_jump_files,
            drop_jump_box_heights_cm,
            drop_jump_sides,
            drop_jump_free_hands,
        ):
            djs.append(
                DropJump.from_tdf(
                    filename=file,
                    bodymass_kg=bodymass,
                    free_hands=fh,
                    box_height_cm=height,
                    side=side,
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
                    l2=l2,
                    c7=c7,
                    t5=t5,
                    sc=sc,
                    head_anterior=head_anterior,
                    head_posterior=head_posterior,
                    head_left=head_left,
                    head_right=head_right,
                )
            )

        rjs = []
        for file, exclude, straight, fh, side in zip(
            repeated_jumps_files,
            exclude_repeated_jumps,
            repeated_jumps_straight_leg,
            repeated_jumps_free_hands,
            repeated_jumps_sides,
        ):
            rjs += RepeatedJumps.from_tdf(
                file=file,
                bodymass_kg=bodymass,
                free_hands=fh,
                exclude_jumps=exclude,
                straight_legs=straight,
                side=side,
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
                l2=l2,
                c7=c7,
                t5=t5,
                sc=sc,
                head_anterior=head_anterior,
                head_posterior=head_posterior,
                head_left=head_left,
                head_right=head_right,
            ).jumps

        return cls(
            participant=participant,
            normative_data=normative_data,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
            squat_jumps=sjs,
            counter_movement_jumps=cmjs,
            drop_jumps=djs,
            repeated_jumps=rjs,
            keep_all_data=keep_all_data,
        )

    def get_results(self, include_emg: bool = True):
        """
        Generate test results with processed data and visualizations.

        Parameters
        ----------
        include_emg : bool, optional
            Whether to include EMG analysis in the results (default: True).

        Returns
        -------
        JumpTestResults
            Results object containing processed data, summary tables,
            and visualization figures.
        """
        return JumpTestResults(self.processed_data, include_emg)

    def _process_record(self, record: Record):
        """
        Apply signal processing pipeline to a record and normalize EMG data.

        Parameters
        ----------
        record : Record
            The record to process.

        Returns
        -------
        Record
            The processed record with filtered signals, normalized EMG data,
            and non-relevant muscles removed.

        Raises
        ------
        ValueError
            If processing pipeline fails to return the correct record type.
        """
        # apply the pipeline to the test data
        pipeline = self.processing_pipeline
        exe = pipeline(record, inplace=False)  # type: ignore
        if not isinstance(exe, type(record)):
            raise ValueError("Something went wrong during data processing.")

        # normalize emg data and remove non-relevant muscles
        norms = self.emg_normalization_values
        to_remove: list[str] = []
        for k, m in exe.emgsignals.items():

            # remove if non relevant
            if self.relevant_muscle_map is not None:
                if not any([i.lower() in k.lower() for i in self.relevant_muscle_map]):
                    to_remove.append(k)
                    continue

            # normalize
            if isinstance(m, EMGSignal):
                for (name, side), val in norms.items():
                    if m.muscle_name == name and m.side == side:
                        exe[k] = m / val * 100
                        exe[k].set_unit("%")  # type: ignore
                        break
        if len(to_remove) > 0:
            exe.drop(to_remove, True)

        return exe

    def _process_jump(
        self, jump: CounterMovementJump | DropJump | SquatJump | SingleJump
    ):
        """
        Process a single jump trial with trimming and reference frame alignment.

        Parameters
        ----------
        jump : SingleJump or DropJump or RepeatedJumps
            The jump trial to process.

        Returns
        -------
        SingleJump or DropJump or RepeatedJumps
            The processed jump with trimmed data, filtered signals, and
            aligned reference frame (for bilateral jumps).

        Raises
        ------
        RuntimeError
            If signal stripping or jump resizing fails.
        ValueError
            If force platform data is invalid or reference frame alignment fails.
        """
        exe = jump.copy().strip()

        # trim the data to the jump duration
        if not isinstance(jump, (SingleJump, DropJump)):
            index = jump.resultant_force.strip()
            if index is None:
                raise RuntimeError("strip failed")
            index = index.index
            for key, val in jump.items():
                idx = (val.index >= index[0]) & (val.index <= index[-1])  # type: ignore
                idx = np.where(idx)[0]
                exe[key] = val.iloc[idx, :]  # type: ignore
            if not isinstance(exe, Record):
                raise RuntimeError("jump resizing failed.")
        exe = self._process_record(exe)  # type: ignore

        # align the reference frame
        if exe.side not in ["right", "left"] and len(exe.forceplatforms) > 1:

            def extract_cop(force: Any):
                if not isinstance(force, ForcePlatform):
                    raise ValueError("force must be a ForcePlatform instance.")
                cop = force.origin
                if not isinstance(cop, Point3D):
                    raise ValueError("force must be a ForcePlatform instance.")
                cop = cop.copy()
                return cop.to_numpy().astype(float).mean(axis=0)

            # on bilateral test, we rotate the system of forces to a
            rt = extract_cop(exe.right_foot_ground_reaction_force)
            lt = extract_cop(exe.left_foot_ground_reaction_force)

            def norm(arr):
                return arr / np.sum(arr**2) ** 0.5

            ml = norm(lt - rt)
            vt = np.array([0, 1, 0])
            ap = np.cross(ml, vt)
            origin = (rt + lt) / 2
            ref_frame = ReferenceFrame(origin, ml, vt, ap)
            exe = ref_frame.apply(exe, inplace=False)
            if exe is None:
                raise ValueError("reference frame alignment returned None")

        return exe

    @property
    def processed_data(self):
        """
        Get a copy of the test with all jumps and references processed.

        Applies signal processing pipeline, EMG normalization, and reference
        frame alignment to all jump trials and reference signals.

        Returns
        -------
        JumpTest
            A new JumpTest instance with all data processed and ready
            for analysis.
        """
        out = self.copy()
        for i, jump in enumerate(out.squat_jumps):
            out.squat_jumps[i] = self._process_jump(jump)  # type: ignore
        for i, jump in enumerate(out.counter_movement_jumps):
            out.counter_movement_jumps[i] = self._process_jump(jump)  # type: ignore
        for i, jump in enumerate(out.drop_jumps):
            out.drop_jumps[i] = self._process_jump(jump)  # type: ignore
        for i, jump in enumerate(out.repeated_jumps):
            out.repeated_jumps[i] = self._process_jump(jump)  # type: ignore
        if len(self.emg_normalization_references) > 0:
            out.set_emg_normalization_references(
                self._process_record(self.emg_normalization_references)  # type: ignore
            )
        if len(self.emg_activation_references) > 0:
            out.set_emg_activation_references(
                self._process_record(self.emg_activation_references)  # type: ignore
            )
        return out

    @property
    def processing_pipeline(self):
        """
        Get the signal processing pipeline for jump test data.

        Creates a custom pipeline with specialized processing functions for:
        - ForcePlatform: NaN filling, lowpass filtering, moment updates
        - EMGSignal: Bandpass filtering, RMS envelope computation
        - Point3D: MICE imputation, lowpass filtering

        Returns
        -------
        ProcessingPipeline
            Configured pipeline for jump-specific signal processing.

        Notes
        -----
        The force platform pipeline handles drop jumps starting outside
        the plates. EMG signals use a 50ms RMS window for envelope extraction.
        """

        # we need a custom force platform processing pipeline due to the
        # drop jump starting condition which might be outside the plates
        def forceplatform_processing_func(fp: ForcePlatform):

            # fill force nans with zeros
            fp.force.iloc[:, :] = fillna(fp.force.to_numpy(), value=0, inplace=False)

            # fill position nans via cubic spline
            fp.origin.iloc[:, :] = fillna(
                fp.origin.to_numpy(), mice=False, inplace=False
            )

            # fill any remaining NaN in torque (can occur from update_moments)
            fp.torque.iloc[:, :] = fillna(fp.torque.to_numpy(), value=0, inplace=False)

            # lowpass filter both origin and force
            fsamp = float(1 / np.mean(np.diff(fp.index)))
            filt_fun = lambda x: butterworth_filt(
                x,
                fcut=30,
                fsamp=fsamp,  # type: ignore
                order=4,
                ftype="lowpass",
                phase_corrected=True,
            )
            fp.origin.apply(filt_fun, axis=0, inplace=True)
            fp.force.apply(filt_fun, axis=0, inplace=True)
            fp.torque.apply(filt_fun, axis=0, inplace=True)

            # update moments
            fp.update_moments(inplace=True)

            # set moments corresponding to the very low vertical force to zero
            module = fp.force.copy().module.to_numpy().flatten()  # type: ignore
            idxs = module < MINIMUM_CONTACT_FORCE_N
            vals = fp.torque.copy().to_numpy()
            vals[idxs, :] = 0
            fp.torque[:, :] = vals

        # we need also a custom EMGSignal processing pipeline to create a
        # short RMS envelope with 50ms rolling window.
        def emgsignal_processing_func(channel: EMGSignal):

            # centering
            channel.iloc[:, :] -= channel.to_numpy().mean()

            # filtering
            fsamp = 1 / np.mean(np.diff(channel.index))
            channel.apply(
                butterworth_filt,
                fcut=[20, 450],
                fsamp=fsamp,
                order=4,
                ftype="bandpass",
                phase_corrected=True,
                inplace=True,
                axis=0,
            )

            # envelope extraction via RMS filter with 50ms window
            channel.apply(
                rms_filt,
                order=int(0.05 * fsamp),
                pad_style="reflect",
                offset=0.5,
                inplace=True,
                axis=0,
            )

        def point3d_processing_func(obj: Point3D):

            # fill missing marker data via MICE imputation
            obj.fillna(mice=True, inplace=True)

            # lowpass filter the marker data
            fsamp = 1 / np.mean(np.diff(obj.index))
            obj.apply(
                butterworth_filt,
                fcut=6,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
                inplace=True,
                axis=0,
            )

        pipeline = ProcessingPipeline(
            ForcePlatform=[forceplatform_processing_func],
            EMGSignal=[emgsignal_processing_func],
            Point3D=[point3d_processing_func],
        )
        return pipeline


__all__ = ["JumpTest"]
