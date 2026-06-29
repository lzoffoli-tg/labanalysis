"""IsometricExercise module."""

from typing import Literal

import numpy as np

from ....signalprocessing import continuous_batches
from ....timeseries import EMGSignal, Signal1D
from .biostrength_exercise import BiostrengthExercise


class IsometricExercise(BiostrengthExercise):
    """
    Isokinetic Test 1RM instance

    Parameters
    ----------
    time: Iterable[int | float]
        the array containing the time instant of each sample in seconds

    position: Iterable[int | float]
        the array containing the displacement of the handles for each sample

    load: Iterable[int | float]
        the array containing the load measured at each sample in kgf

    coefs_1rm: tuple[int | float, int | float]
        the b0 and b1 coefficients used to estimated the 1RM.

    Attributes
    ----------
    raw: DataFrame
        a DataFrame containing the input data

    repetitions: list[DataFrame]
        a list of dataframes each defining one single repetition

    product: BiostrengthProduct
        the product on which the test has been performed

    peak_load: float
        the peak load measured during the isokinetic repetitions

    rom0: float
        the start of the user's range of movement in meters

    rom1: float
        the end of the user's range of movement in meters

    rom: float
        the range of movement amplitude in meters

    results_table: DataFrame
        a table containing the data obtained during the test

    summary_table: DataFrame
        a table containing summary statistics about the test

    summary_plot: FigureWidget
        a figure representing the results of the test.
    """

    def _get_repetitions_splitting_signal(self):
        arr = self.force.to_numpy().flatten()
        time = self.force.index
        return arr, time

    def _get_repetitions_index(self, force: np.ndarray, time: np.ndarray):

        # check if position has to be inverted
        arr = force.flatten()
        if abs(np.min(arr)) > abs(np.max(arr)):
            arr *= -1

        # get the continuous batches with length higher than 3 seconds
        fsamp = int(1 / np.mean(np.diff(time)))
        batches = continuous_batches(
            arr > np.max(arr) * 0.5,
            tolerance=int(0.1 * fsamp),
        )
        batches = [i for i in batches if len(i) > 3 * fsamp]

        # get the repetitions
        reps: list[list[int]] = []
        for batch in batches:

            # for each batch get the start of the repetition as the last point
            # before the start of the batch without having increments in force
            start = batch[0]
            while start > 0 and arr[start - 1] < arr[start]:
                start -= 1

            # for each batch get the end of the repetition as the first point
            # before the end of the batch without having decrements in force
            stop = batch[-1]
            while stop < len(time) - 1 and arr[stop + 1] < arr[stop]:
                stop += 1

            # add the repetition index
            reps += [list(range(start, stop + 1, 1))]

        return reps

    def _get_repetitions_start_from_biostrength_data(
        self,
        force: np.ndarray,
        position: np.ndarray,
        time: np.ndarray,
    ):
        return self._get_repetitions_index(force, time)[0][0]

    def __init__(
        self,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        synchronize_signals: bool = True,
        **signals: EMGSignal,
    ):
        super().__init__(
            side=side,
            force=force,
            position=position,
            synchronize_signals=synchronize_signals,
            **signals,
        )

    def copy(self):
        return IsometricExercise(
            side=self.side,  # type: ignore
            synchronize_signals=False,
            **{i: v.copy() for i, v in self._data.items()},  # type: ignore
        )
