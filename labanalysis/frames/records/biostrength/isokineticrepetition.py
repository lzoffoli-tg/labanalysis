"""
isokinetic exercise module
"""

#! IMPORTS

from typing import Literal

from ....constants import G
from ....io.read.biostrength import BiostrengthProduct
from ...timeseries.emgsignal import EMGSignal
from ...timeseries.signal1d import Signal1D
from .defaultrepetition import DefaultRepetition

#! CONSTANTS


__all__ = ["IsokineticRepetition"]

#! CLASSES


class IsokineticRepetition(DefaultRepetition):
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
    def estimated_1rm_kg(self):
        """return the predicted 1RM"""
        b1, b0 = self.product.rm1_coefs
        return self.peak_force_N / G * b1 + b0

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
            "estimated_1rm_kg",
            self.estimated_1rm_kg,
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
