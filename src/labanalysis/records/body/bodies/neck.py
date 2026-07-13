"""neck joint module"""

import numpy as np

from ....timeseries import Point3D, Signal1D
from .head import Head
from .joint import Joint
from .left_shoulder import LeftShoulder
from .pelvis import Pelvis
from .right_shoulder import RightShoulder

__all__ = ["Neck"]


class Neck(Joint):
    """Neck Joint."""

    def __init__(
        self,
        c7: Point3D,
        sc: Point3D,
        pelvis: Pelvis,
        head: Head,
        left_shoulder: LeftShoulder,
        right_shoulder: RightShoulder,
    ):

        # get the reference frame
        AP = sc - c7
        AP = AP / np.atleast_2d(np.linalg.norm(AP.to_numpy(), axis=1)).T
        C = (sc + c7) / 2
        VT = C - pelvis.center
        Og = c7 + AP * 0.03

        # build the class
        super().__init__(
            Og,  # type: ignore
            None,
            VT,  # type: ignore
            AP,  # type: ignore
        )

        # upper markers
        self["c7"] = c7
        self["sc"] = sc
        self["head"] = head
        self["left_shoulder"] = left_shoulder
        self["right_shoulder"] = right_shoulder

    @property
    def _c7(self):
        """return c7 in the global reference frame"""
        out: Point3D = self["c7"]  # type: ignore
        return out

    @property
    def _sc(self):
        """return sc in the global reference frame"""
        out: Point3D = self["sc"]  # type: ignore
        return out

    @property
    def _head(self):
        """return the head plane"""
        out: Head = self["head"]  # type: ignore
        return out

    @property
    def _left_shoulder(self):
        """return the left_shoulder plane"""
        out: LeftShoulder = self["left_shoulder"]  # type: ignore
        return out

    @property
    def _right_shoulder(self):
        """return the right shoulder plane"""
        out: RightShoulder = self["right_shoulder"]  # type: ignore
        return out

    @property
    def flexionextension(self):
        """
        Calculate neck flexion-extension angle.

        Interpretation
        --------------
        Positive = flexion
        Negative = extension

        Returns
        -------
        Signal1D
            flexion/extension angle in degrees.
        """
        return self.get_angle_by_point(
            self.apply(self._head.center),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )

    @property
    def lateralflexion(self):
        """
        Calculate neck adduction/abduction

        Interpretation
        --------------
        Positive = left
        Negative = right

        Returns
        -------
        Signal1D
            lateral flexion angle in degrees.
        """
        return self.get_angle_by_point(
            self.apply(self._head.center),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.lateral_axis,  # type: ignore
        )

    @property
    def left_shoulder_elevationdepression(self):
        """
        Calculate left shoulder elevation/depression angle.

        Interpretation
        --------------
        Positive = elevation
        Negative = depression

        Returns
        -------
        Signal1D
            elevation/depression angle in degrees.
        """
        return self.get_angle_by_point(
            self.apply(self._left_shoulder.center),  # type: ignore
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

    @property
    def right_shoulder_elevationdepression(self):
        """
        Calculate right shoulder elevation/depression angle.

        Interpretation
        --------------
        Positive = elevation
        Negative = depression

        Returns
        -------
        Signal1D
            elevation/depression angle in degrees.
        """
        out = self.get_angle_by_point(
            self.apply(self._right_shoulder.center),  # type: ignore
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )
        val = out.to_numpy().flatten()
        val[val > 0] = 180 - val[val > 0]
        val[val < 0] = -(180 + val[val < 0])
        out = Signal1D(val.reshape(-1, 1), out.index, "deg")

    @property
    def left_shoulder_protractionretraction(self):
        """
        Calculate left shoulder protraction/retraction angle.

        Interpretation
        --------------
        Positive = protraction
        Negative = retraction

        Returns
        -------
        Signal1D
            protraction/retraction angle in degrees.
        """
        return self.get_angle_by_point(
            self.apply(self._left_shoulder.center),  # type: ignore
            self.lateral_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )

    @property
    def right_shoulder_protractionretraction(self):
        """
        Calculate right shoulder protraction/retraction angle.

        Interpretation
        --------------
        Positive = protraction
        Negative = retraction

        Returns
        -------
        Signal1D
            protraction/retraction angle in degrees.
        """

        out = self.get_angle_by_point(
            self.apply(self._right_shoulder.center),  # type: ignore
            self.lateral_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )
        val = out.to_numpy().flatten()
        val[val > 0] = 180 - val[val > 0]
        val[val < 0] = -180 - val[val < 0]
        out = Signal1D(val.reshape(-1, 1), out.index, "deg")
        return out
