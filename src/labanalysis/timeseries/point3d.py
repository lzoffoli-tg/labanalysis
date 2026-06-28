"""
3D point trajectory time series.
"""

import pint

from .signal3d import Signal3D
from ..utils import FloatArray1D, FloatArray2D, TextArray1D


class Point3D(Signal3D):
    """
    A 3D point time series, automatically converted to meters (m).
    """

    def __init__(
        self,
        data: FloatArray2D,
        index: list[float] | FloatArray1D,
        unit: str | pint.Quantity = "m",
        columns: list[str] | TextArray1D = ["X", "Y", "Z"],
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
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
            vertical_axis,
            anteroposterior_axis,
        )

        if not self._unit.check("[length]"):
            raise ValueError("unit must represent length.")
        meters = pint.Quantity("m")
        magnitude = self._unit.to(meters).magnitude
        self[:, :] = self.to_numpy() * magnitude
        self._unit = meters

    def copy(self):
        return Point3D(
            self._data.copy(),
            self.index.copy(),
            self.unit,
            self.columns.copy(),
            self.vertical_axis,
            self.anteroposterior_axis,
        )


__all__ = ["Point3D"]
