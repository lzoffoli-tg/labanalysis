"""Force platform measurement module."""

import numpy as np

from ..timeseries import Point3D, Signal3D
from ..events.signal import Signal
from .record import Record


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
    strip(axis=0, inplace=False, independent=False)
        Remove leading/trailing rows or columns that are all NaN from all
        contained objects. When independent=False (default), all elements share
        a common timeframe based on the union of non-NaN time points.
    """

    @property
    def vertical_axis(self):
        """return the label defining the vertical axis."""
        origin: Point3D = self["origin"]  # type: ignore
        return origin.vertical_axis

    @property
    def anteroposterior_axis(self):
        """return the label defining the anteroposterior axis."""
        origin: Point3D = self["origin"]  # type: ignore
        return origin.anteroposterior_axis

    @property
    def lateral_axis(self):
        """return the label defining the lateral axis."""
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
        if key in ["_data", "_updated"]:
            super().__setattr__(key, value)
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

    @property
    def free_moment(self):
        """ "return the free moment of the Force Platform"""
        k = np.cross(self.origin.to_numpy(), self.force.to_numpy())
        return self.torque + k

    def update_moments(self, inplace: bool = True):
        """
        update the moments

        Parameters
        ----------
        inplace: bool (default=True)
            if True, the change is applied directly to the object. If False
            a modified copy is returned.

        Return
        ------
        obj: ForcePlatform | None
            the obejct with the change applied if inplace=False.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if inplace:
            self.torque[:, :] = self.free_moment.to_numpy()
        else:
            out = self.copy()
            out.torque[:, :] = out.free_moment.to_numpy()
            return out

    def copy(self):
        """return a copy of the object"""
        return ForcePlatform(
            origin=self.origin.copy(),  # type: ignore
            force=self.force.copy(),  # type: ignore
            torque=self.torque.copy(),  # type: ignore
        )

    @property
    def origin(self):
        """return the coordinated of the force platform CoP"""
        out: Point3D = self["origin"]  # type: ignore
        return out

    @property
    def force(self):
        """return the force attribute of the object"""
        out: Signal3D = self["force"]  # type: ignore
        return out

    @property
    def torque(self):
        """return the torque attribute of the object"""
        out: Signal3D = self["torque"]  # type: ignore
        return out


__all__ = ["ForcePlatform"]
