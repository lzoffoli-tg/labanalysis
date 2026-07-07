"""
3D signal time series.
"""

import numpy as np
import pint

from .timeseries import Timeseries
from .signal1d import Signal1D
from ..utils import FloatArray1D, TextArray1D


class Signal3D(Timeseries):
    """
    A 3D signal (three columns: X, Y, Z) time series.
    """

    @property
    def vertical_axis(self):
        return self._vertical_axis

    @property
    def anteroposterior_axis(self):
        return self._anteroposterior_axis

    @property
    def lateral_axis(self):
        bounded = [self.vertical_axis, self.anteroposterior_axis]
        col = [i for i in self.columns if i not in bounded]
        if len(col) == 0:
            raise ValueError("no lateral axis could be found.")
        return str(col[0])

    @property
    def module(self):
        return Signal1D(
            data=(self._data.copy() ** 2).sum(axis=1) ** 0.5,
            index=self.index.copy(),
            unit=self.unit,
        )

    def __init__(
        self,
        data: np.ndarray,
        index: list[float] | FloatArray1D,
        unit: pint.Quantity | str,
        columns: list[str] | TextArray1D = ["X", "Y", "Z"],
        vertical_axis: str = "Y",
        anteroposterior_axis: str = "Z",
    ):
        super().__init__(
            data,
            index,
            columns,
            unit,
        )

        if data.shape[1] != 3:
            raise ValueError("Signal3D must have exactly 3 columns.")

        self.set_vertical_axis(vertical_axis)
        self.set_anteroposterior_axis(anteroposterior_axis)

    def set_vertical_axis(self, axis: str):
        if axis not in self.columns:
            raise ValueError(f"vertical_axis must be any of {self.columns}")
        self._vertical_axis = str(axis)

    def set_anteroposterior_axis(self, axis: str):
        if axis not in self.columns:
            raise ValueError(f"anteroposterior_axis must be any of {self.columns}")
        self._anteroposterior_axis = str(axis)

    def copy(self):
        return Signal3D(
            self._data.copy(),
            self.index.copy(),
            self.unit,
            self.columns.copy(),
            self.vertical_axis,
            self.anteroposterior_axis,
        )

    def _copy_view_attributes(self, view_obj):
        """
        Copy Signal3D-specific attributes to view object.

        This override ensures that vertical_axis and anteroposterior_axis
        are preserved during slicing operations.
        """
        super()._copy_view_attributes(view_obj)
        # Attributes are already handled by parent, but we keep this
        # as an example of how to override if needed in the future


__all__ = ["Signal3D"]
