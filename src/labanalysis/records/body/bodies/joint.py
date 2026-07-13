"""Composite mixin for all joint center properties."""

from typing import Literal

import numpy as np

from ....referenceframes.referenceframes import ReferenceFrame
from ....timeseries import Signal1D, Timeseries, Point3D, Signal3D
from ...record import Record
from ...timeseriesrecord import TimeseriesRecord
from ....events.signal import Signal

__all__ = ["Joint"]


class Joint(TimeseriesRecord):
    """
    General Joint class
    """

    def __init__(
        self,
        center: Point3D,
        lateral_vector: Signal3D | Point3D | None = None,
        vertical_vector: Signal3D | Point3D | None = None,
        anteroposterior_vector: Signal3D | Point3D | None = None,
    ):
        super().__init__()
        if not isinstance(center, Point3D):
            raise ValueError("center must be a Point3D")
        self["center"] = center

        if lateral_vector is not None:
            if not isinstance(lateral_vector, (Signal3D, Point3D)):
                raise ValueError("lateral_vector must be a Signal3D or a Point3D")

        if vertical_vector is not None:
            if not isinstance(vertical_vector, (Signal3D, Point3D)):
                raise ValueError("vertical_vector must be a Signal3D or a Point3D")

        if anteroposterior_vector is not None:
            if not isinstance(anteroposterior_vector, (Signal3D, Point3D)):
                raise ValueError(
                    "anteroposterior_vector must be a Signal3D or a Point3D"
                )

        rf = ReferenceFrame(
            center,
            lateral_vector,
            vertical_vector,
            anteroposterior_vector,
        )

        # store the versors
        self["lateral_versor"] = Signal3D(
            rf.lateral_axis,
            center.index,
            center.unit,
            center.columns,
            center.vertical_axis,
            center.anteroposterior_axis,
        )
        self["anteroposterior_versor"] = Signal3D(
            rf.anteroposterior_axis,
            center.index,
            center.unit,
            center.columns,
            center.vertical_axis,
            center.anteroposterior_axis,
        )
        self["vertical_versor"] = Signal3D(
            rf.vertical_axis,
            center.index,
            center.unit,
            center.columns,
            center.vertical_axis,
            center.anteroposterior_axis,
        )
        self["center"] = center

    @property
    def center(self):
        """Joint center"""
        out: Point3D = self["center"]  # type: ignore
        return out

    @property
    def lateral_versor(self):
        """the lateral versor"""
        out: Signal3D = self["lateral_versor"]  # type: ignore
        return out

    @property
    def vertical_versor(self):
        """the vertical versor"""
        out: Signal3D = self["vertical_versor"]  # type: ignore
        return out

    @property
    def anteroposterior_versor(self):
        """the anteroposterior versor"""
        out: Signal3D = self["anteroposterior_versor"]  # type: ignore
        return out

    @property
    def reference_frame(self):
        """return the ReferenceFrame instance of the joint"""
        return ReferenceFrame(
            self.center,
            self.lateral_versor,
            self.vertical_versor,
            self.anteroposterior_versor,
        )

    def __call__(self, obj: Timeseries | Record, inplace: bool = False):
        """Make ReferenceFrame callable - delegates to apply()."""
        return self.reference_frame.apply(obj, inplace=inplace)

    def apply_inverse(
        self,
        obj: Timeseries | Record,
        inplace: bool = False,
    ):
        """
        Apply the inverse reference frame transformation to an object.

        Parameters
        ----------
        obj : np.ndarray or pd.DataFrame or Timeseries or Record
            Object to transform. Must represent 3D data.
        inplace : bool, optional
            If True, modify the object in place and return None.
            If False, return a transformed copy. Default is False.

        Returns
        -------
        Timeseries or Record or None
            Transformed object (if inplace=False) or None (if inplace=True).
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if inplace:
            self.reference_frame.apply_inverse(obj, True)
        else:
            return self.reference_frame.apply_inverse(obj, inplace=False)

    def apply(
        self,
        obj: Timeseries | Record,
        inplace: bool = False,
    ):
        """
        Apply the reference frame transformation to an object.

        Parameters
        ----------
        obj : np.ndarray or pd.DataFrame or Timeseries or Record
            Object to transform. Must represent 3D data.
        inplace : bool, optional
            If True, modify the object in place and return None.
            If False, return a transformed copy. Default is False.

        Returns
        -------
        Timeseries or Record or None
            Transformed object (if inplace=False) or None (if inplace=True).
        """

        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if inplace:
            self.reference_frame.apply(obj, True)
        else:
            return self.reference_frame.apply(obj, inplace=False)

    def get_angle_by_point(
        self,
        point: Point3D,
        axis_a: str | Literal["X", "Y", "Z"],
        axis_b: str | Literal["X", "Y", "Z"],
    ):
        """
        return the angle made by the provided point in the current joint
        according to the required plane.

        Parameters
        ----------
        point: Point3D
            the point from which the angle has to be calculated

        axis_a, axis_b: str | Literal
            the label defining the axes of the plane on which the angle has
            to be calculated

        Return
        ------
        Signal1D the angle in degrees.
        """
        if not isinstance(point, Point3D):
            raise ValueError("point must be a Point3D object.")

        # get the relevant axis map
        P: Point3D = self.apply(point)  # type: ignore
        cols = P.columns
        plane_axes = np.array([axis_a, axis_b])
        col_map = [np.where(cols == i)[0][0] for i in plane_axes]
        col_map = np.array(col_map)
        x, y = point.to_numpy()[:, col_map].T

        # get the angle
        return Signal1D(np.degrees(np.arctan2(y, x)), point.index, "deg")
