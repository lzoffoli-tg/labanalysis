"""IsokineticExercise module."""

from typing import Literal

import numpy as np

from ....signalprocessing import *
from ....timeseries import EMGSignal, Signal1D
from .biostrength_exercise import BiostrengthExercise

__all__ = ["IsokineticExercise"]


class IsokineticExercise(BiostrengthExercise):
    """
    Isokinetic resistance Exercise

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
        arr = self.position.to_numpy().flatten()
        time = self.position.index
        return arr, time

    def _get_repetitions_index(self, position: np.ndarray, time: np.ndarray):

        # check if position has to be inverted
        if abs(np.min(position)) > abs(np.max(position)):
            position *= -1

        # get the batches where the position is within the 20-80% of the ROM and
        # positive velocity
        rom0 = np.min(position)
        rom1 = np.max(position)
        rom = (position - rom0) / (rom1 - rom0)
        batches = continuous_batches(rom > 0.5)
        if len(batches) == 0:
            raise RuntimeError("No repetitions have been found")

        # get the 3 (or less) repetitions with the greater ROM
        samples = np.argsort([len(i) for i in batches])[::-1][:3]
        batches = [batches[i] for i in np.sort(samples)]

        # for each batch (i.e. candidate repetition):
        #   - get the start of each repetition looking at the last zero in the
        #     velocity readings occurring before the first sample of the batch
        #   - get the end of each repetition looing at the first zero in the
        #     velocity readings occurring after the last sample of the batch
        reps_idx: list[list[int]] = []
        for batch in batches:
            start = batch[0]
            while start > 0 and rom[start] > rom[start - 1]:
                start -= 1
            stop = np.argmax(position[batch]) + batch[0]
            reps_idx += [list(range(start, stop, 1))]
        return reps_idx

    def _get_repetitions_start_from_biostrength_data(
        self,
        force: np.ndarray,
        position: np.ndarray,
        time: np.ndarray,
    ):
        start = self._get_repetitions_index(position, time)
        if len(start) == 0:
            raise ValueError("No repetitions have been found in data.")
        return int(start[0][0])

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
        return IsokineticExercise(
            side=self.side,  # type: ignore
            synchronize_signals=False,
            **{i: v.copy() for i, v in self._data.items()},  # type: ignore
        )
