"""
isokinetic exercise module
"""

#! IMPORTS

from typing import Literal

import numpy as np

from ....io.read.biostrength import BiostrengthProduct
from ....signalprocessing import find_peaks
from ...timeseries.emgsignal import EMGSignal
from ...timeseries.signal1d import Signal1D
from .defaultrepetition import DefaultRepetition

#! CONSTANTS


__all__ = ["IsometricRepetition"]

#! CLASSES


class IsometricRepetition(DefaultRepetition):
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

    @property
    def rate_of_force_development(self):
        force = self.force.to_numpy().flatten()
        time = np.array(self.index)
        peaks = find_peaks(force, height=float(np.max(force) * 0.8))
        peak = np.argmax(force) if len(peaks) == 0 else peaks[0]
        return (force[peak] - force[0]) / (time[peak] - time[0])

    @property
    def time_to_peak_force(self):
        force = self.force.to_numpy().flatten()
        time = np.array(self.index)
        peak = np.argmax(force)
        return time[peak] - time[0]

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """
        new = super().output_metrics
        new.insert(
            new.shape[1],
            "rate_of_force_development_N/s",
            self.rate_of_force_development,
        )
        new.insert(
            new.shape[1],
            "time_to_peak_force_ms",
            self.time_to_peak_force * 1000,
        )
        return new

    def __init__(
        self,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        **signals: EMGSignal,
    ):

        # check the input
        if not issubclass(product.__class__, BiostrengthProduct):
            raise ValueError("'product' must be a valid Biostrength Product.")
        if not side in ["bilateral", "left", "right"]:
            raise ValueError("'side' must be any of 'bilateral', 'left', 'right'")

        # check the required data
        if not isinstance(force, Signal1D) and force.unit != "N":
            raise ValueError("force must be a Signal1D with 'N' as unit")
        if not isinstance(position, Signal1D) and position.unit != "m":
            raise ValueError("position must be a Signal1D with 'm' as unit")
        for key, val in signals.items():
            if not isinstance(val, EMGSignal):
                raise ValueError(f"{key} must be an EMGSignal")

        super().__init__(
            product=product,
            side=side,
            force=force,
            position=position,
            **signals,
        )
