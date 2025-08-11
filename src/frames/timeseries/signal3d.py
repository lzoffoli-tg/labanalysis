"""
signal3d module
"""

# -*- coding: utf-8 -*-


#! IMPORTS

from typing import Literal
import numpy as np
import pint

from ...utils import FloatArray1D, TextArray1D

from ...signalprocessing import gram_schmidt

from .timeseries import Timeseries

ureg = pint.UnitRegistry()


__all__ = ["Signal3D"]


class Signal3D(Timeseries):
    """
    A 3D signal (three columns: X, Y, Z) time series.
    """

    _vertical_axis: str
    _anteroposterior_axis: str

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

        # check dimensions
        if data.shape[1] != 3:
            raise ValueError("Signal3D must have exactly 3 columns.")

        # check axex
        if vertical_axis not in self.columns:
            raise ValueError(f"vertical_axis must be any of {self.columns}")
        if anteroposterior_axis not in self.columns:
            raise ValueError(f"anteroposterior_axis must be any of {self.columns}")
        self._vertical_axis = str(vertical_axis)
        self._anteroposterior_axis = str(anteroposterior_axis)

    def change_reference_frame(
        self,
        new_x: (
            np.ndarray
            | list[int | float]
            | tuple[int | float, int | float, int | float]
        ) = [1, 0, 0],
        new_y: (
            np.ndarray
            | list[int | float]
            | tuple[int | float, int | float, int | float]
        ) = [0, 1, 0],
        new_z: (
            np.ndarray
            | list[int | float]
            | tuple[int | float, int | float, int | float]
        ) = [0, 0, 1],
        new_origin: (
            np.ndarray
            | list[int | float]
            | tuple[int | float, int | float, int | float]
        ) = [0, 0, 0],
        inplace: bool = False,
    ):
        """
        Rotate and translate each sample using the new reference frame defined by
        orthonormal versors new_x, new_y, new_z and origin new_origin.

        A point can be aligned to this reference frame by:
            new = np.einsum("nij,nj->ni", R, old - O)

        Where R is the rotation matrix (N, 3, 3) and O (N, 3) is the origin of
        the reference frame.

        Parameters
        ----------
        new_x, new_y, new_z : array-like
            Orthonormal basis vectors.
        new_origin : array-like
            New origin.

        Returns
        -------
        Signal3D
            Transformed signal.

        Raises
        ------
        ValueError
            If input vectors are not valid.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        i = np.atleast_2d(new_x)
        if i.shape[0] == 1:
            i = np.ones(self.shape) * i
        j = np.atleast_2d(new_y)
        if j.shape[0] == 1:
            j = np.ones(self.shape) * j
        k = np.atleast_2d(new_z)
        if k.shape[0] == 1:
            k = np.ones(self.shape) * k
        o = np.atleast_2d(new_origin)
        if o.shape[0] == 1:
            o = np.ones(self.shape) * o
        rmat = gram_schmidt(i, j, k)
        rmat = rmat.transpose([0, 2, 1])
        new = np.einsum("nij,nj->ni", rmat, self._data.copy() - o)
        if inplace:
            self[:, :] = new
        else:
            out = self.copy()
            out[:, :] = new
            return out
