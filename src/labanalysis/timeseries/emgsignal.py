"""
EMG signal (electromyography) time series.
"""

from typing import Literal

import numpy as np
import pint

from .signal1d import Signal1D
from ..utils import ureg


class EMGSignal(Signal1D):
    """
    A 1D EMG signal, automatically converted to microvolts (uV).
    """

    _muscle_name: str
    _side: Literal["left", "right", "bilateral"]

    def __init__(
        self,
        data,
        index,
        muscle_name: str,
        side: Literal["left", "right", "bilateral"],
        unit: str | pint.Quantity = "uV",
    ):
        """
        Initialize an EMGSignal.

        Parameters
        ----------
        data : array-like
            2D data array with one column.
        index : list of float
            Time values.
        muscle_name : str
            Name of the muscle.
        side : {'left', 'right', 'bilateral'}
            Side of the body.
        unit : str or pint.Quantity, optional
            Unit of measurement for the data, by default "uV".

        Raises
        ------
        ValueError
            If unit is not valid.
        """
        if isinstance(unit, str):
            unit = ureg(unit)
        if unit.check("V"):
            unt = pint.Quantity("uV")
            magnitude = unit.to(unt).magnitude
        elif unit == ureg("%"):
            unt = ureg("%")
            magnitude = 1
        else:
            raise ValueError("unit must represent voltage or percentages.")

        valid_sides = ["left", "right", "bilateral"]
        if (
            not isinstance(side, (str, Literal["left", "right", "bilateral"]))
            or side not in valid_sides
        ):
            raise ValueError(f"side must be any of: {valid_sides}")

        if not isinstance(muscle_name, str):
            raise ValueError("muscle_name must be a str.")

        values = np.squeeze(data) * magnitude
        super().__init__(
            data=values,
            index=index,
            unit=unt,
        )
        self.set_side(side)
        self.set_muscle_name(muscle_name)

    def set_side(self, side: Literal["left", "right", "bilateral"] | str):
        if not isinstance(side, str) or not any(
            [side == i for i in ["left", "right", "bilateral"]]
        ):
            raise ValueError("side must be 'left', 'right' or 'bilateral'.")
        self._side = side

    @property
    def side(self):
        """
        Get the side of the body.

        Returns
        -------
        {'left', 'right', 'bilateral'}
            The side of the body.
        """
        return str(self._side)

    def set_muscle_name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("name must be a string.")
        self._name = name

    @property
    def muscle_name(self):
        """
        Get the name of the muscle.

        Returns
        -------
        str
            The name of the muscle.
        """
        return self._name

    def copy(self):
        return EMGSignal(
            self._data.copy(),
            self.index.copy(),
            self.muscle_name,
            self.side,
            self.unit,
        )


__all__ = ["EMGSignal"]
