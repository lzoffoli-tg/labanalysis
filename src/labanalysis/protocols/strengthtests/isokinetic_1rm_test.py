"""Isokinetic 1RM test implementation."""

from typing import Callable, Literal

import numpy as np
import pandas as pd

from ...records import TimeseriesRecord
from ...exercises.strength import IsokineticExercise
from ...timeseries import EMGSignal
from ...pipelines import ProcessingPipeline, get_default_emgsignal_processing_func
from ...io.read.biostrength import PRODUCTS
from ..participant import Participant
from ..test_protocol import TestProtocol
from .isokinetic_1rm_test_results import Isokinetic1RMTestResults


class Isokinetic1RMTest(TestProtocol):
    """
    Test protocol for predicting 1-repetition maximum from isokinetic testing.

    Isokinetic1RMTest manages and analyzes constant-velocity strength testing
    to predict maximal strength (1RM). The class uses load-velocity relationships
    from isokinetic dynamometry to extrapolate 1RM values, processes EMG signals,
    and generates performance reports with normative comparisons.

    The protocol supports:
    - Unilateral testing (left/right limb separately)
    - Bilateral testing (both limbs simultaneously)
    - 1RM prediction from velocity-force curves
    - EMG normalization and muscle activation analysis
    - Multiple exercise types (leg press, leg extension, chest press, etc.)

    Parameters
    ----------
    rm1_coefs : dict of str to float
        Regression coefficients for 1RM prediction. Must contain 'beta0'
        (intercept) and 'beta1' (slope) keys mapping to float values.
    left : IsokineticExercise or None
        Left limb isokinetic exercise data. None if not tested.
    right : IsokineticExercise or None
        Right limb isokinetic exercise data. None if not tested.
    bilateral : IsokineticExercise or None
        Bilateral isokinetic exercise data. None if not tested.
    participant : Participant
        Participant information including demographics and anthropometrics.
    normative_data : pd.DataFrame, optional
        Reference data for performance ranking and comparison.
        Default is empty DataFrame.
    emg_normalization_references : TimeseriesRecord or str or 'self', optional
        Reference signals for EMG amplitude normalization. If 'self', uses
        test data for normalization. Default is empty TimeseriesRecord.
    emg_normalization_function : callable, optional
        Function to compute normalization value from reference (e.g., np.mean,
        np.max). Default is np.mean.
    emg_activation_references : TimeseriesRecord or str or 'self', optional
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
    left : IsokineticExercise or None
        Left limb exercise data.
    right : IsokineticExercise or None
        Right limb exercise data.
    bilateral : IsokineticExercise or None
        Bilateral exercise data.
    rm1_coefs : dict of str to float
        1RM prediction coefficients (beta0 = intercept, beta1 = slope).
    processed_data : Isokinetic1RMTest
        Copy of test with all signals processed through the pipeline.
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline with isokinetic-specific configurations.

    Methods
    -------
    copy()
        Return a copy of the test protocol.
    get_results(include_emg=True)
        Process data and return Isokinetic1RMTestResults.
    set_left_test(test)
        Set left limb exercise data.
    set_right_test(test)
        Set right limb exercise data.
    set_bilateral_test(test)
        Set bilateral exercise data.
    set_1rm_coefs(rm1_coefs)
        Set 1RM prediction coefficients.
    from_files(participant, product, ...)
        Load test from BioStrength files for specific exercise equipment.

    Notes
    -----
    1RM Prediction:
    The 1RM is predicted from the load-velocity relationship using linear
    regression: 1RM = beta0 + beta1 * velocity_threshold, where velocity_threshold
    is typically the velocity corresponding to maximal voluntary contraction.

    Supported Exercise Products:
    - LEG PRESS, LEG PRESS REV
    - LEG EXTENSION, LEG EXTENSION REV
    - LEG CURL
    - LOW ROW
    - ADJUSTABLE PULLEY REV
    - CHEST PRESS
    - SHOULDER PRESS

    Each product has pre-calibrated regression coefficients based on equipment
    characteristics and biomechanical validation studies.

    Examples
    --------
    >>> from labanalysis.protocols import Isokinetic1RMTest, Participant
    >>>
    >>> # Create test from BioStrength files
    >>> participant = Participant(surname='Athlete', weight=75)
    >>> test = Isokinetic1RMTest.from_files(
    ...     participant=participant,
    ...     product='LEG PRESS',
    ...     bilateral_biostrength_filename='leg_press_data.txt'
    ... )
    >>>
    >>> # Get results with 1RM prediction
    >>> results = test.get_results(include_emg=False)
    >>> print(f"Predicted 1RM: {results.summary['predicted_1rm_kg'].values[0]:.1f} kg")

    See Also
    --------
    Isokinetic1RMTestResults : Results container for isokinetic 1RM tests.
    IsokineticExercise : Exercise data container for isokinetic contractions.
    TestProtocol : Parent class for test protocols.
    """

    def __init__(
        self,
        rm1_coefs: dict[str, float],
        left: IsokineticExercise | None,
        right: IsokineticExercise | None,
        bilateral: IsokineticExercise | None,
        participant: Participant,
        normative_data: pd.DataFrame = pd.DataFrame(),
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
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
        self.set_1rm_coefs(rm1_coefs)
        self.set_left_test(left)
        self.set_right_test(right)
        self.set_bilateral_test(bilateral)

    def set_left_test(self, test: IsokineticExercise | None):
        if test is not None and not isinstance(test, IsokineticExercise):
            msg = f"left must be None or an {IsokineticExercise.__name__} instance."
            raise ValueError(msg)
        self._left = test

    @property
    def left(self):
        return self._left

    def set_right_test(self, test: IsokineticExercise | None):
        if test is not None and not isinstance(test, IsokineticExercise):
            msg = f"right must be None or an {IsokineticExercise.__name__} instance."
            raise ValueError(msg)
        self._right = test

    @property
    def right(self):
        return self._right

    def set_bilateral_test(self, test: IsokineticExercise | None):
        if test is not None and not isinstance(test, IsokineticExercise):
            msg = (
                f"bilateral must be None or an {IsokineticExercise.__name__} instance."
            )
            raise ValueError(msg)
        self._bilateral = test

    @property
    def bilateral(self):
        return self._bilateral

    def set_1rm_coefs(self, rm1_coefs: dict[str, float]):
        msg = "rm1_coefs must be a dict with keys 'beta0', 'beta1' and floats "
        msg += "as values."
        if not isinstance(rm1_coefs, dict):
            raise ValueError(msg)
        keys = list(rm1_coefs.keys())
        vals = list(rm1_coefs.values())
        if any([i not in ["beta0", "beta1"] for i in keys]):
            raise ValueError(msg)
        if not all([isinstance(i, (int, float)) for i in vals]):
            raise ValueError(msg)
        self._1rm_coefs = rm1_coefs

    @property
    def rm1_coefs(self):
        return self._1rm_coefs

    def copy(self):
        return Isokinetic1RMTest(
            participant=self.participant.copy(),
            normative_data=self.normative_data,
            rm1_coefs=self.rm1_coefs,
            left=self.left,
            right=self.right,
            bilateral=self.bilateral,
            emg_normalization_references=self.emg_normalization_references,
            emg_normalization_function=self.emg_normalization_function,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
        )

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
        normative_data: pd.DataFrame = pd.DataFrame(),
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
    ):

        # get 1RM coefficients
        prod = PRODUCTS[product]
        rm1_coefs = {i: v for i, v in zip(["beta1", "beta0"], prod._rm1_coefs)}  # type: ignore

        # get left data
        left = {}
        if left_biostrength_filename is not None:
            bio = IsokineticExercise.from_txt(
                left_biostrength_filename,
                product,
                side="left",
            )
            left.update({i: bio[i] for i in ["force", "position"]})
        if left_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(left_emg_filename)
            left.update(emg.emgsignals)
        if len(left) > 0:
            left = IsokineticExercise(
                side="left",
                synchronize_signals=True,
                **left,
            )
        else:
            left = None

        # get right data
        right = {}
        if right_biostrength_filename is not None:
            bio = IsokineticExercise.from_txt(
                right_biostrength_filename,
                product,
                side="right",
            )
            right.update({i: bio[i] for i in ["force", "position"]})
        if right_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(right_emg_filename)
            right.update(emg.emgsignals)
        if len(right) > 0:
            right = IsokineticExercise(
                side="right",
                synchronize_signals=True,
                **right,
            )
        else:
            right = None

        # get bilateral data
        bilateral = {}
        if bilateral_biostrength_filename is not None:
            bio = IsokineticExercise.from_txt(
                bilateral_biostrength_filename,
                product,
                side="bilateral",
            )
            bilateral.update({i: bio[i] for i in ["force", "position"]})
        if bilateral_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(bilateral_emg_filename)
            bilateral.update(emg.emgsignals)
        if len(bilateral) > 0:
            bilateral = IsokineticExercise(
                side="bilateral",
                synchronize_signals=True,
                **bilateral,
            )
        else:
            bilateral = None

        return cls(
            participant=participant,
            normative_data=normative_data,
            rm1_coefs=rm1_coefs,
            left=left,
            right=right,
            bilateral=bilateral,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
        )

    def get_results(
        self,
        include_emg: bool = True,
        estimate_1rm: bool = True,
        include_force_balance: bool = True,
    ):
        return Isokinetic1RMTestResults(
            self.processed_data,
            include_emg,
            estimate_1rm,
            include_force_balance,
        )

    def _process_exercise(self, exercise: IsokineticExercise):
        # apply the pipeline to the test data
        exe = self.processing_pipeline(exercise, inplace=False)
        if not isinstance(exe, IsokineticExercise):
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
        """
        def custom_processing_func(signal: Signal1D):
            signal.fillna(inplace=True)
            fsamp = 1 / np.mean(np.diff(signal.index))
            signal.apply(
                butterworth_filt,
                fcut=1,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
                inplace=True,
            )

        pipeline = get_default_processing_pipeline()
        pipeline.add(Signal1D=[custom_processing_func])
        return pipeline
        """
        return ProcessingPipeline(EMGSignal=[get_default_emgsignal_processing_func])


__all__ = ["Isokinetic1RMTest"]
