"""Composite mixin for all joint center properties."""

import numpy as np

from ....events.signal import Signal
from ....timeseries import Point3D, Plane3D
from ...timeseriesrecord import TimeseriesRecord

__all__ = ["BodyPlane"]


class BodyPlane(TimeseriesRecord):
    """
    General Body Plane class
    """

    def __init__(self, **points: Point3D):
        super().__init__()
        for lbl, point in points.items():
            if not isinstance(point, Point3D):
                raise ValueError(f"{lbl} must be a Point3D")
            self[lbl] = point

    def get_projected_point(self, point: Point3D):
        """
        Calculate orthogonal projection of a point onto a plane.

        Finds the point on the plane that minimizes distance to the input point
        (perpendicular projection).

        Parameters
        ----------
        point : Point3D
            3D point to project.

        Returns
        -------
        Point3D
            Projected points on the plane.

        Notes
        -----
        The projection is found by moving from the point along the plane normal
        until reaching the plane. Distance t along normal satisfies:
            t = (ax + by + cz + d) / (a² + b² + c²)
        """
        return self.plane.get_projected_point(point)

    @property
    def plane(self):
        """
        Calculate plane coefficients (a, b, c, d) at each time instant using least squares.

        Fits a plane to at least 3 points in 3D space using principal component
        analysis. The plane normal is determined by the eigenvector corresponding
        to the smallest eigenvalue of the covariance matrix.

        Returns
        -------
        coefficients : Timseries (shape N, 4)
            Plane coefficients [a, b, c, d] at each time instant where
            ax + by + cz + d = 0 defines the plane equation.

        Notes
        -----
        The algorithm:
        1. Computes centroid of input points
        2. Centers points relative to centroid
        3. Computes covariance matrix for each sample
        4. Extracts normal vector (eigenvector of smallest eigenvalue)
        5. Calculates d coefficient from normal and centroid
        """
        return Plane3D.from_points(*self.points_raw.values())  # type: ignore

    @property
    def points_raw(self):
        """return the original points defining the plane"""
        return self.points3d

    @property
    def points_projected(self):
        """return the original points projected on the plane"""
        return TimeseriesRecord(
            **{
                lbl: self.get_projected_point(pnt)  # type: ignore
                for lbl, pnt in self.points_raw.items()
            }
        )

    @property
    def center(self):
        """Joint center"""
        projs = self.points_projected.values()
        sums = projs[0].copy()
        for i in np.arange(1, len(projs)):
            sums = sums + projs[i].to_numpy()
        out: Point3D = sums / len(projs)  # type: ignore
        return out

    def events(self):
        """Return the events associated with the body plane"""
        return self._events
