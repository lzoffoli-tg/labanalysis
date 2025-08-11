"""
point3d module
"""

# -*- coding: utf-8 -*-


#! IMPORTS


import numpy as np
import pint

from ...utils import FloatArray1D, FloatArray2D, TextArray1D

from .signal3d import Signal3D

ureg = pint.UnitRegistry()


__all__ = ["Point3D"]


class Point3D(Signal3D):
    """
    A 3D point time series, automatically converted to meters (m).
    """

    def __init__(
        self,
        data: np.ndarray | FloatArray2D,
        index: list[float] | FloatArray1D,
        unit: str | pint.Quantity = "m",
        columns: list[str] | TextArray1D = ["X", "Y", "Z"],
    ):
        """
        Initialize a Point3D.

        Parameters
        ----------
        data : array-like
            2D data array with three columns.
        index : list of float
            Time values.
        unit : str or pint.Quantity, optional
            Unit of measurement for the data, by default "m".
        columns : list, optional
            Column labels, must be 'X', 'Y', 'Z', by default ["X", "Y", "Z"].

        Raises
        ------
        ValueError
            If units are not valid or not unique.
        """
        super().__init__(
            data,
            index,
            unit,
            columns,
        )

        # check the unit
        # check the unit and convert to uV if required
        if not self._unit.check("[length]"):
            raise ValueError("unit must represent length.")
        meters = pint.Quantity("m")
        magnitude = self._unit.to(meters).magnitude
        self[:, :] = self.to_numpy() * magnitude
        self._unit = meters  # type: ignore
