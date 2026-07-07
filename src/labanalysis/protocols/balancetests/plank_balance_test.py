"""Plank balance test implementation."""

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ...exercises import PronePosture
from ...pipelines import get_default_processing_pipeline
from ...records import ForcePlatform, TimeseriesRecord
from ...referenceframes import ReferenceFrame
from ...timeseries.emgsignal import EMGSignal
from ...timeseries.point3d import Point3D
from ..normativedata import plankbalance_normative_values
from ..participant import Participant
from ..test_protocol import TestProtocol
from .plank_balance_test_results import PlankBalanceTestResults


class PlankBalanceTest(TestProtocol):
    """
    Plank balance test protocol for assessing core stability and postural control.

    This class implements a plank (prone) balance test where participants maintain
    a plank position while ground reaction forces and body sway are measured.
    The test can be performed with eyes open or closed to assess visual contribution
    to balance control.

    Parameters
    ----------
    participant : Participant
        Participant information object.
    exercise : PronePosture
        PronePosture exercise object containing force platform and marker data.
    eyes : {'open', 'closed'}
        Visual condition during the test.
    normative_data : pandas.DataFrame, optional
        Normative reference data for comparison (default uses built-in plank balance norms).
    emg_normalization_references : TimeseriesRecord or str or 'self', optional
        Reference data for EMG normalization (default is empty TimeseriesRecord).
    emg_normalization_function : callable, optional
        Function to compute normalization value from reference data (default is np.mean).
    emg_activation_references : TimeseriesRecord or str or 'self', optional
        Reference data for EMG activation threshold (default is empty TimeseriesRecord).
    emg_activation_threshold : float or int, optional
        Threshold multiplier for EMG activation detection (default is 3).
    relevant_muscle_map : list of str or None, optional
        List of relevant muscle names to include in analysis (default is None, includes all).

    Attributes
    ----------
    participant : Participant
        The participant who performed the test.
    exercise : PronePosture
        The recorded prone posture exercise.
    eyes : str
        Visual condition ('open' or 'closed').
    normative_data : pandas.DataFrame
        Reference data for normative comparisons.

    Methods
    -------
    from_files(filename, participant, eyes, normative_data, emg_normalization_references, emg_normalization_function, emg_activation_references, emg_activation_threshold, relevant_muscle_map, left_hand_ground_reaction_force, right_hand_ground_reaction_force, left_foot_ground_reaction_force, right_foot_ground_reaction_force)
        Create PlankBalanceTest instance from TDF file.
    get_results()
        Generate PlankBalanceTestResults from processed data.
    copy()
        Create a deep copy of the test object.
    set_exercise(exercise)
        Set the prone posture exercise.
    set_eyes(eyes)
        Set the visual condition.

    Properties
    ----------
    exercise : PronePosture
        Access to the recorded exercise.
    eyes : str
        Access to the visual condition.
    processed_data : PlankBalanceTest
        Returns a processed copy of the test data.
    processing_pipeline : ProcessingPipeline
        Default signal processing pipeline for plank balance test data.

    Examples
    --------
    >>> from labanalysis import Participant, PlankBalanceTest
    >>> participant = Participant(name="John", surname="Doe", weight=75, height=180)
    >>> plank = PlankBalanceTest.from_files(
    ...     filename="plank_trial.tdf",
    ...     participant=participant,
    ...     eyes="open"
    ... )
    >>> results = plank.get_results()
    >>> print(results.summary)

    See Also
    --------
    PronePosture : Prone posture exercise data.
    PlankBalanceTestResults : Analysis results for plank balance tests.
    UprightBalanceTest : Upright balance test protocol.

    Notes
    -----
    The plank balance test measures postural stability in the prone position with
    hands and feet supported on force platforms. Core stability is assessed through
    center of pressure movement and force distribution analysis.
    """

    @property
    def eyes(self):
        """
        Get the visual condition during the test.

        Returns
        -------
        str
            Visual condition ('open' or 'closed').
        """
        return self._eyes

    def set_eyes(self, eyes: Literal["open", "closed"]):
        """
        Set the visual condition for the test.

        Parameters
        ----------
        eyes : {'open', 'closed'}
            Visual condition during testing.

        Raises
        ------
        ValueError
            If eyes is not 'open' or 'closed'.
        """
        if eyes not in ["open", "closed"]:
            raise ValueError("eyes must be 'open' or 'closed'.")
        self._eyes = eyes

    def __init__(
        self,
        participant: Participant,
        exercise: PronePosture,
        eyes: Literal["open", "closed"],
        normative_data: pd.DataFrame = plankbalance_normative_values,
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float | int = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        super().__init__(
            participant=participant,
            normative_data=normative_data,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            relevant_muscle_map=relevant_muscle_map,
        )
        self.set_eyes(eyes)
        self.set_exercise(exercise)

    def copy(self):
        """
        Create a deep copy of the PlankBalanceTest object.

        Returns
        -------
        PlankBalanceTest
            A new PlankBalanceTest instance with copied data.
        """
        return PlankBalanceTest(
            participant=self.participant,
            exercise=self.exercise,
            eyes=self.eyes,  # type: ignore
            normative_data=self.normative_data,
            emg_normalization_references=self.emg_normalization_references,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
            emg_normalization_function=self.emg_normalization_function,
            relevant_muscle_map=self.relevant_muscle_map,
        )

    def set_exercise(self, exercise: PronePosture):
        """
        Set the prone posture exercise for this test.

        Parameters
        ----------
        exercise : PronePosture
            PronePosture exercise object to assign to this test.

        Raises
        ------
        ValueError
            If exercise is not a PronePosture instance.
        """
        if not isinstance(exercise, PronePosture):
            raise ValueError("exercise must be a PronePosture instance.")
        self._exercise = exercise

    @property
    def exercise(self):
        """
        Get the prone posture exercise.

        Returns
        -------
        PronePosture
            The recorded prone posture exercise for this test.
        """
        return self._exercise

    @classmethod
    def from_files(
        cls,
        filename: str,
        participant: Participant,
        eyes: Literal["open", "closed"],
        left_foot_ground_reaction_force: str = "left_foot",
        right_foot_ground_reaction_force: str = "right_foot",
        left_hand_ground_reaction_force: str = "left_hand",
        right_hand_ground_reaction_force: str = "right_hand",
        normative_data: pd.DataFrame = plankbalance_normative_values,
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float | int = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        return cls(
            participant=participant,
            normative_data=normative_data,
            eyes=eyes,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            relevant_muscle_map=relevant_muscle_map,
            exercise=PronePosture.from_tdf(
                file=filename,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                left_hand_ground_reaction_force=left_hand_ground_reaction_force,
                right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            ),
        )

    @property
    def processed_data(self):

        # apply the pipeline to the test data
        exe = self.processing_pipeline(self.exercise, inplace=False)
        if not isinstance(exe, PronePosture):
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

        # align the reference frame
        def extract_cop(force: Any):
            if not isinstance(force, ForcePlatform):
                raise ValueError("force must be a ForcePlatform instance.")
            cop = force.origin
            if not isinstance(cop, Point3D):
                raise ValueError("force must be a ForcePlatform instance.")
            cop = cop.copy()
            return cop.to_numpy().astype(float).mean(axis=0)

        # calculate reference frame from force platform positions
        rf = extract_cop(exe.right_foot_ground_reaction_force)
        lf = extract_cop(exe.left_foot_ground_reaction_force)
        rh = extract_cop(exe.right_hand_ground_reaction_force)
        lh = extract_cop(exe.left_hand_ground_reaction_force)

        def norm(arr):
            return arr / np.sum(arr**2) ** 0.5

        ml = norm((lf + lh) / 2 - (rf + rh) / 2)
        vt = np.array([0, 1, 0])
        ap = np.cross(ml, vt)
        origin = (rf + lf + rh + lh) / 4
        ref_frame = ReferenceFrame(origin, ml, vt, ap)
        exe = exe.apply(ref_frame, inplace=False)
        if exe is None:
            raise ValueError("reference frame alignment returned None")

        # return processed data
        out = self.copy()
        out.set_exercise(exe)  # type: ignore
        return out

    @property
    def processing_pipeline(self):
        return get_default_processing_pipeline()

    def get_results(self, include_emg: bool = True):
        return PlankBalanceTestResults(
            self.processed_data,
            include_emg,
        )


__all__ = ["PlankBalanceTest"]
