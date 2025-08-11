"""forceplatforms module"""

import numpy as np

from ..timeseries.point3d import Point3D
from ..timeseries.signal3d import Signal3D
from .records import Record


class ForcePlatform(Record):
    """
    Represents a force platform measurement system.

    Parameters
    ----------
    origin : Point3D
        The center of pressure (CoP) location over time.
    force : Signal3D
        The 3D ground reaction force vector over time.
    torque : Signal3D
        The 3D torque vector over time.
    vertical_axis : str, optional
        The label for the vertical axis (default "Y").
    anteroposterior_axis : str, optional
        The label for the anteroposterior axis (default "Z").
    strip : bool, optional
        If True, remove leading/trailing rows or columns that are all NaN from
        all contained objects (default True).
    reset_time : bool, optional
        If True, reset the time index to start at zero for all contained objects
        (default True).

    Methods
    -------
    copy()
        Return a deep copy of the ForcePlatform.
    """

    @property
    def vertical_axis(self):
        origin: Point3D = self["origin"]  # type: ignore
        return origin.vertical_axis

    @property
    def anteroposterior_axis(self):
        origin: Point3D = self["origin"]  # type: ignore
        return origin.anteroposterior_axis

    @property
    def lateral_axis(self):
        origin: Point3D = self["origin"]  # type: ignore
        return origin.lateral_axis

    def __init__(self, origin: Point3D, force: Signal3D, torque: Signal3D):
        """
        Initialize a ForcePlatform.

        Parameters
        ----------
        origin : Point3D
        force : Signal3D
        torque : Signal3D

        Raises
        ------
        TypeError
            If any argument is not of the correct type.
        """
        if not isinstance(origin, Point3D):
            raise TypeError("origin must be an instance of Point3D")
        if not isinstance(force, Signal3D):
            raise TypeError("force must be an instance of Signal3D")
        if not isinstance(torque, Signal3D):
            raise TypeError("torque must be an instance of Signal3D")

        # check the axes
        if (
            origin.vertical_axis != force.vertical_axis
            or origin.vertical_axis != torque.vertical_axis
        ):
            msg = "vertical axes must be the same across origin, "
            msg += "force and torque elements."
            raise ValueError(msg)
        if (
            origin.anteroposterior_axis != force.anteroposterior_axis
            or origin.anteroposterior_axis != torque.anteroposterior_axis
        ):
            msg = "anteroposterior axes must be the same across origin, "
            msg += "force and torque elements."
            raise ValueError(msg)

        super().__init__(origin=origin, force=force, torque=torque)

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__("_data", value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if not key in ["origin", "force", "torque"] or not isinstance(
            value, (Point3D, Signal3D)
        ):
            msg = "only 'origin', 'force' and 'torque' attributes can be "
            msg += " passed to ForcePlatform instances."
            raise ValueError(msg)
        if not isinstance(value, (Signal3D, Point3D)):
            raise ValueError("value must be a Timeseries or Record")
        self._data[key] = value

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

        Parameters
        ----------
        new_x, new_y, new_z : array-like
            Orthonormal basis vectors.
        new_origin : array-like
            New origin.

        Returns
        -------
        ForcePlatform
            Transformed signal.

        Raises
        ------
        ValueError
            If input vectors are not valid.

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if inplace:
            for val in self.values():
                if isinstance(val, (Point3D, Signal3D)):
                    val.change_reference_frame(
                        new_x,
                        new_y,
                        new_z,
                        new_origin,
                        True,
                    )
        else:
            out = self.copy()
            for val in out.values():
                if isinstance(val, (Point3D, Signal3D)):
                    val.change_reference_frame(
                        new_x,
                        new_y,
                        new_z,
                        new_origin,
                        True,
                    )
            return out
