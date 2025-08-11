"""
isokinetic exercise module
"""

#! IMPORTS

from typing import Literal

import numpy as np
import pandas as pd

from ....io.read.biostrength import BiostrengthProduct
from ...timeseries.emgsignal import EMGSignal
from ...timeseries.signal1d import Signal1D
from ..timeseriesrecord import TimeseriesRecord

#! CONSTANTS


__all__ = ["DefaultRepetition"]

#! CLASSES


class DefaultRepetition(TimeseriesRecord):
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

    # * class variables

    _product: BiostrengthProduct
    _side: Literal["bilateral", "left", "right"]

    # * attributes

    @property
    def side(self):
        """get the side of the test"""
        return self._side

    @property
    def product(self):
        """return the product on which the test has been performed"""
        return self._product

    @property
    def peak_force_N(self):
        """return the ending position of the repetitions"""
        return float(self.force.to_numpy().max())

    @property
    def rom_m(self):
        """return the repetition's ROM"""
        position = self.position.to_numpy().flatten()
        return float(np.max(position) - np.min(position))

    @property
    def muscle_activations(self):
        """
        Returns coordination and balance metrics from EMG signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with coordination and balance metrics, or empty if not available.
        """

        # get the muscle activations
        # (if there are no emg data return and empty dataframe)
        emgs = self.emgsignals
        if emgs.shape[1] == 0:
            return pd.DataFrame()

        # check the presence of left and right muscles
        muscles = {}
        for emg in emgs.values():
            if isinstance(emg, EMGSignal):
                side = emg.side
                if side != self.side:
                    continue
                name = emg.muscle_name
                unit = emg.unit
                muscles[f"{side} {name} {unit}"] = emg.to_numpy().mean()

        return pd.DataFrame(pd.Series(muscles)).T

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """
        new = {
            "type": self.product.name,
            "side": self.side,
            "peak_force_N": self.peak_force_N,
            "rom_mm": self.rom_m * 1000,
        }
        new = pd.DataFrame(pd.Series(new)).T
        return pd.concat([new, self.muscle_activations], axis=1)

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

        super().__init__(force=force, position=position, **signals)

        # get the raw data
        self._product = product  # type: ignore
        self._side = side
