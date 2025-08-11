"""
isokinetic exercise module
"""

#! IMPORTS

from typing import Literal

import numpy as np

from .... import signalprocessing as sp
from ....io.read.biostrength import BiostrengthProduct
from ...timeseries.emgsignal import EMGSignal
from ...timeseries.signal1d import Signal1D
from .defaultexercise import DefaultExercise
from .isokineticrepetition import IsokineticRepetition

#! CONSTANTS


__all__ = ["IsokineticExercise"]

#! CLASSES


class IsokineticExercise(DefaultExercise):
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

    _repetition_type = IsokineticRepetition

    def _get_repetitions_splitting_signal(self):
        arr = self.position.to_numpy().flatten()
        time = self.position.index
        return arr, time

    def _get_repetitions_index(self, position: np.ndarray, time: np.ndarray):

        # check if position has to be inverted
        if abs(np.min(position)) > abs(np.max(position)):
            position *= -1

        # get the velocity
        velocity = sp.winter_derivative1(position)
        velocity = np.concatenate([[velocity[0]], velocity, [velocity[-1]]])
        velocity = sp.mean_filt(arr=velocity, order=301, offset=1)

        # get the batches where the position is within the 20-80% of the ROM and
        # positive velocity
        rom0 = np.min(position)
        rom1 = np.max(position)
        rom = (position - rom0) / (rom1 - rom0)
        concentric_phase_condition = (velocity > 0) & (rom > 0.2) & (rom < 0.8)
        batches = sp.continuous_batches(concentric_phase_condition)
        if len(batches) == 0:
            raise RuntimeError("No repetitions have been found")

        # get the 3 (or less) repetitions with the greater ROM
        samples = np.argsort([np.max(position[i]) for i in batches])[::-1][:3]
        batches = [batches[i] for i in np.sort(samples)]

        # for each batch (i.e. candidate repetition):
        #   - get the start of each repetition looking at the last zero in the
        #     velocity readings occurring before the first sample of the batch
        #   - get the end of each repetition looing at the first zero in the
        #     velocity readings occurring after the last sample of the batch
        reps_idx: list[list[int]] = []
        for batch in batches:
            start = np.where(velocity[: batch[0]] <= 0)[0]
            start = 0 if len(start) == 0 else start[-1]
            stop = np.where(velocity[batch[-1] :] <= 0)[0]
            stop = len(velocity) if len(stop) == 0 else (stop[0] + batch[-1] + 1)
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
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        synchronize_signals: bool = True,
        **signals: EMGSignal,
    ):
        super().__init__(
            product=product,
            side=side,
            force=force,
            position=position,
            synchronize_signals=synchronize_signals,
            **signals,
        )
