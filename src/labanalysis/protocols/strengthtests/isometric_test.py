"""Isometric test implementation."""

from typing import Callable, Literal

import numpy as np
import pandas as pd

from ...exercises.strength import IsometricExercise
from ...pipelines import get_default_processing_pipeline
from ...records import TimeseriesRecord
from ...signalprocessing import butterworth_filt
from ...timeseries import EMGSignal, Signal1D
from ..participant import Participant
from ..test_protocol import TestProtocol
from .isometric_test_results import IsometricTestResults


class IsometricTest(TestProtocol):
    """
    Test protocol for isometric strength assessment.

    IsometricTest manages and analyzes maximal voluntary isometric contraction
    (MVIC) tests for unilateral and bilateral exercises. The class handles force
    signal acquisition, processes biomechanical data, normalizes EMG signals,
    and generates performance reports with normative comparisons.

    The protocol supports:
    - Unilateral testing (left/right limb separately)
    - Bilateral testing (both limbs simultaneously)
    - EMG normalization and muscle activation analysis
    - Automated signal processing pipelines
    - Force-time curve analysis and peak force extraction

    Parameters
    ----------
    left : IsometricExercise or None
        Left limb isometric exercise data. None if not tested.
    right : IsometricExercise or None
        Right limb isometric exercise data. None if not tested.
    bilateral : IsometricExercise or None
        Bilateral isometric exercise data. None if not tested.
    participant : Participant
        Participant information including demographics and anthropometrics.
    normative_data : pd.DataFrame, optional
        Reference data for performance ranking and comparison.
        Default is empty DataFrame.
    emg_normalization_references : TimeseriesRecord, optional
        Reference signals for EMG amplitude normalization.
        Default is empty TimeseriesRecord.
    emg_normalization_function : callable, optional
        Function to compute normalization value from reference (e.g., np.mean,
        np.max). Default is np.mean.
    emg_activation_references : TimeseriesRecord, optional
        Reference signals for determining muscle activation thresholds.
        Default is empty TimeseriesRecord.
    emg_activation_threshold : float, optional
        Threshold multiplier for detecting muscle activation onset.
        Default is 3 (3x reference level).
    relevant_muscle_map : list of str or None, optional
        List of muscle names to include in analysis. If None, includes all
        detected muscles. Default is None.

    Attributes
    ----------
    left : IsometricExercise or None
        Left limb exercise data.
    right : IsometricExercise or None
        Right limb exercise data.
    bilateral : IsometricExercise or None
        Bilateral exercise data.
    processed_data : IsometricTest
        Copy of test with all signals processed through the pipeline.
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline with isometric-specific configurations.

    Methods
    -------
    copy()
        Return a copy of the test protocol.
    get_results(include_emg=True)
        Process data and return IsometricTestResults.
    set_left_test(test)
        Set left limb exercise data.
    set_right_test(test)
        Set right limb exercise data.
    set_bilateral_test(test)
        Set bilateral exercise data.

    Notes
    -----
    Processing Pipeline:
    - Force signals: 3 Hz lowpass filter (4th order Butterworth, phase-corrected)
    - EMG signals: 20-450 Hz bandpass, RMS envelope
    - Gap filling with cubic spline interpolation
    - Phase-corrected filtering to preserve peak timing

    Isometric testing measures maximum force production without joint movement,
    providing pure strength assessment independent of velocity and power factors.

    Examples
    --------
    >>> from labanalysis.protocols import IsometricTest, Participant
    >>> from labanalysis.exercises.strength import IsometricExercise
    >>>
    >>> # Create participant
    >>> participant = Participant(surname='Athlete', weight=75)
    >>>
    >>> # Load exercise data (analyze first 2 seconds only)
    >>> left_ex = IsometricExercise.from_biostrength("left_leg_press.txt", max_time_s=2)
    >>> right_ex = IsometricExercise.from_biostrength("right_leg_press.txt", max_time_s=2)
    >>>
    >>> # Create test protocol
    >>> test = IsometricTest(
    ...     left=left_ex,
    ...     right=right_ex,
    ...     bilateral=None,
    ...     participant=participant
    ... )
    >>>
    >>> # Get results
    >>> results = test.get_results(include_emg=False)
    >>> print(results.summary)

    See Also
    --------
    IsometricTestResults : Results container for isometric tests.
    IsometricExercise : Exercise data container for isometric contractions.
    TestProtocol : Parent class for test protocols.
    """

    def __init__(
        self,
        left: IsometricExercise | None,
        right: IsometricExercise | None,
        bilateral: IsometricExercise | None,
        participant: Participant,
        normative_data: pd.DataFrame = pd.DataFrame(),
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        super().__init__(
            participant,
            normative_data,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
        )
        self.set_left_test(left)
        self.set_right_test(right)
        self.set_bilateral_test(bilateral)

    def copy(self):
        return IsometricTest(
            participant=self.participant.copy(),
            normative_data=self.normative_data,
            left=self.left,
            right=self.right,
            bilateral=self.bilateral,
            emg_normalization_references=self.emg_normalization_references,
            emg_normalization_function=self.emg_normalization_function,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
            relevant_muscle_map=self.relevant_muscle_map,
        )

    def get_results(self, include_emg: bool = True):
        return IsometricTestResults(
            self.processed_data,
            include_emg,
        )

    def set_left_test(self, test: IsometricExercise | None):
        if test is not None and not isinstance(test, IsometricExercise):
            msg = f"left must be None or an {IsometricExercise.__name__} instance."
            raise ValueError(msg)
        self._left = test

    @property
    def left(self):
        return self._left

    def set_right_test(self, test: IsometricExercise | None):
        if test is not None and not isinstance(test, IsometricExercise):
            msg = f"right must be None or an {IsometricExercise.__name__} instance."
            raise ValueError(msg)
        self._right = test

    @property
    def right(self):
        return self._right

    def set_bilateral_test(self, test: IsometricExercise | None):
        if test is not None and not isinstance(test, IsometricExercise):
            msg = f"bilateral must be None or an {IsometricExercise.__name__} instance."
            raise ValueError(msg)
        self._bilateral = test

    @property
    def bilateral(self):
        return self._bilateral

    def _process_exercise(self, exercise: IsometricExercise):
        # Apply the pipeline to the test data
        # The pipeline will preserve max_time_s via copy()
        exe = self.processing_pipeline(exercise, inplace=False)
        if not isinstance(exe, IsometricExercise):
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

        # return processed data
        return exe

    @property
    def processed_data(self):
        out = self.copy()
        if out.left is not None:
            out.set_left_test(self._process_exercise(out.left))
        if out.right is not None:
            out.set_right_test(self._process_exercise(out.right))
        if out.bilateral is not None:
            out.set_bilateral_test(self._process_exercise(out.bilateral))
        return out

    @property
    def processing_pipeline(self):
        def custom_processing_func(signal: Signal1D):
            # Fill missing values
            signal.fillna(inplace=True)
            # Apply 3Hz lowpass filter to force signals
            fsamp = 1 / np.mean(np.diff(signal.index))
            signal.apply(
                butterworth_filt,
                fcut=3,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
                inplace=True,
            )

        pipeline = get_default_processing_pipeline()
        pipeline.add(Signal1D=[custom_processing_func])
        return pipeline

    @classmethod
    def from_files(
        cls,
        participant: Participant,
        product: Literal[
            "LEG PRESS",
            "LEG PRESS REV",
            "LEG EXTENSION",
            "LEG EXTENSION REV",
            "LEG CURL",
            "LOW ROW",
            "ADJUSTABLE PULLEY REV",
            "CHEST PRESS",
            "SHOULDER PRESS",
        ],
        left_biostrength_filename: str | None = None,
        right_biostrength_filename: str | None = None,
        bilateral_biostrength_filename: str | None = None,
        left_emg_filename: str | None = None,
        right_emg_filename: str | None = None,
        bilateral_emg_filename: str | None = None,
        max_time_s: int | None = None,
        time_points: list[int] = [100, 200, 500, 1000],
        normative_data: pd.DataFrame = pd.DataFrame(),
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
    ):

        # get left data
        left = {}
        if left_biostrength_filename is not None:
            bio = IsometricExercise.from_txt(
                left_biostrength_filename,
                product,
                side="left",
            )
            left.update({i: bio[i] for i in ["force", "position"]})
        if left_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(left_emg_filename)
            left.update(emg.emgsignals)
        if len(left) > 0:
            left = IsometricExercise(
                side="left",
                synchronize_signals=True,
                max_time_s=max_time_s,
                time_points=time_points,
                **left,
            )
        else:
            left = None

        # get right data
        right = {}
        if right_biostrength_filename is not None:
            bio = IsometricExercise.from_txt(
                right_biostrength_filename,
                product,
                side="right",
            )
            right.update({i: bio[i] for i in ["force", "position"]})
        if right_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(right_emg_filename)
            right.update(emg.emgsignals)
        if len(right) > 0:
            right = IsometricExercise(
                side="right",
                synchronize_signals=True,
                max_time_s=max_time_s,
                time_points=time_points,
                **right,
            )
        else:
            right = None

        # get bilateral data
        bilateral = {}
        if bilateral_biostrength_filename is not None:
            bio = IsometricExercise.from_txt(
                bilateral_biostrength_filename,
                product,
                side="bilateral",
            )
            bilateral.update({i: bio[i] for i in ["force", "position"]})
        if bilateral_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(bilateral_emg_filename)
            bilateral.update(emg.emgsignals)
        if len(bilateral) > 0:
            bilateral = IsometricExercise(
                side="bilateral",
                synchronize_signals=True,
                max_time_s=max_time_s,
                time_points=time_points,
                **bilateral,
            )
        else:
            bilateral = None

        return cls(
            participant=participant,
            normative_data=normative_data,
            left=left,
            right=right,
            bilateral=bilateral,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
        )


__all__ = ["IsometricTest"]
