"""
signal1d module
"""

# -*- coding: utf-8 -*-


#! IMPORTS


import numpy as np
import pint

from ...utils import FloatArray1D

from .timeseries import Timeseries

ureg = pint.UnitRegistry()


__all__ = ["Signal1D"]


class Signal1D(Timeseries):
    """
    A 1D signal (single column) time series.
    """

    def __init__(
        self,
        data: np.ndarray,
        index: list[float] | FloatArray1D,
        unit: str | pint.Quantity,
    ):
        """
        Initialize a Signal1D.

        Parameters
        ----------
        data : array-like
            2D data array with one column.
        index : list of float
            Time values.
        unit : str or pint.Quantity
            Unit of measurement for the data.

        Raises
        ------
        ValueError
            If data does not have exactly one column.
        """
        data_array = np.asarray(data, float)
        if data_array.ndim == 1:
            data_array = np.atleast_2d(data_array).T
        if data_array.ndim != 2 or data_array.shape[1] != 1:
            raise ValueError("Signal1D must have exactly one column")
        if not isinstance(unit, (str, pint.Quantity)):
            raise ValueError("unit must be a str or a pint.Quantity")
        if isinstance(unit, str):
            unit = ureg(unit)
        super().__init__(
            data=data_array,
            index=index,
            columns=["amplitude"],
            unit=unit,
        )
