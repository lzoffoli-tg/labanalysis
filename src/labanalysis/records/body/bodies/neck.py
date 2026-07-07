"""neck joint module"""

from ....timeseries import Point3D
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
        pelvis_plane: Pelvis,
        head_plane: Head,
        left_shoulder: LeftShoulder,
        right_shoulder: RightShoulder,
    ):

        # get the reference frame
        AP = sc - c7
        O = (sc + c7) / 2
        VT = O - pelvis_plane.center

        # build the class
        super().__init__(
            O,  # type: ignore
            None,
            VT,  # type: ignore
            AP,  # type: ignore
        )

        # upper markers
        self["c7"] = c7
        self["sc"] = sc
        self["head"] = head_plane
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
    def neck_flexionextension(self):
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
            self.anteriorposterior_axis,  # type: ignore
        )

    @property
    def neck_lateralflexion(self):
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
        out.loc[out.to_numpy() >= 0, :] = 180 - out.loc[out.to_numpy() > 0, :]
        out.loc[out.to_numpy() < 0, :] = 180 + out.loc[out.to_numpy() > 0, :]
        return out

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
        out.loc[out.to_numpy() >= 0, :] = 180 - out.loc[out.to_numpy() > 0, :]
        out.loc[out.to_numpy() < 0, :] = 180 + out.loc[out.to_numpy() > 0, :]
        return out
