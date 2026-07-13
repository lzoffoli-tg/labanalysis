"""base segment module"""

import numpy as np

from .joint import Joint
from ....timeseries.point3d import Point3D
from ...timeseriesrecord import TimeseriesRecord
from ....events.signal import Signal


class Segment(TimeseriesRecord):
    """base segment class"""

    def __init__(self, **points: Point3D | None | Joint):
        super().__init__()
        for lbl, obj in points.items():
            if isinstance(obj, Joint):
                self[lbl] = obj.center
            elif isinstance(obj, Point3D):
                self[lbl] = obj

    def _get_distance(self, point1: Point3D | None, point2: Point3D | None):
        """
        Calculate the Euclidean distance between two points in 3D space.

        Parameters
        ----------
        point1 : Point3D
            The first point.
        point2 : Point3D
            The second point.

        Returns
        -------
        Signal1D
            The distance between the two points in meters.
        """
        if point1 is None or point2 is None:
            return None
        v1 = point1.to_numpy()
        v2 = point2.to_numpy()
        return float(np.mean(((v1 - v2) ** 2).sum(axis=1) ** 0.5))
