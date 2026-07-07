"""plane3D module"""

import numpy as np

from ..utils import FloatArray1D
from .point3d import Point3D
from .signal1d import Signal1D
from .timeseries import Timeseries


class Plane3D(Timeseries):
    """
    General Plane 3D class

    This object is a Timeseries with specialized methods and properties to define
    a 3D plane in a least squares sense
    """

    def __init__(
        self,
        a: FloatArray1D | list[float | int] | Signal1D,
        b: FloatArray1D | list[float | int] | Signal1D,
        c: FloatArray1D | list[float | int] | Signal1D,
        d: FloatArray1D | list[float | int] | Signal1D,
        index: FloatArray1D | list[float | int],
    ):
        data = []
        for i in [a, b, c, d]:
            if isinstance(i, Signal1D):
                data.append(i.to_numpy())
            elif isinstance(i, (np.ndarray, list)):
                data.append(np.atleast_2d(np.asarray(i).flatten().astype(float)).T)
        data = np.hstack(data)
        super().__init__(
            data,
            index,
            ["a", "b", "c", "d"],
            "a.u.",
        )

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
        point_array = point.to_numpy()  # [x, y, z]
        planearr = self.to_numpy()  # Nx4: [a, b, c, d]
        a, b, c, d = planearr.T
        normal = np.stack([a, b, c], axis=1)

        numerator = np.sum(normal * point_array, axis=1) + d
        denominator = np.sum(normal**2, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(denominator != 0, numerator / denominator, np.nan)

        projection_points = point_array - normal * t[:, np.newaxis]

        return Point3D(
            data=projection_points,
            index=point.index,
            columns=point.columns,
            unit=point.unit,
        )

    @classmethod
    def from_points(cls, *points: Point3D):
        """
        generate the plane using least squares.

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

        # Calculate centroid for each sample
        mat = np.stack([i.to_numpy() for i in points], axis=1)
        centroid = np.mean(mat, axis=1, keepdims=True)

        # Center points relative to centroid
        centered = mat - centroid

        # Calculate covariance matrix for each sample
        cov = np.einsum("nij,nik->njk", centered, centered) / mat.shape[1]

        # Calculate eigenvalues and eigenvectors with fallback for numerical issues
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Eigenvector associated with smallest eigenvalue is the plane normal
            normals = eigvecs[:, :, 0]

            # Coefficients a, b, c
            a, b, c = normals[:, :3].T

            # Calculate d = -(a*x + b*y + c*z) using centroid
            d = -np.sum(normals * centroid[:, 0, :], axis=1)

            # Stack coefficients
            coefficients = np.stack([a, b, c, d], axis=1)
        except np.linalg.LinAlgError:
            # Fallback: add regularization term to improve numerical stability
            # This handles cases where points are nearly collinear or have numerical issues
            reg_term = 1e-8 * np.eye(3)
            cov_regularized = cov + reg_term[np.newaxis, :, :]
            try:
                eigvals, eigvecs = np.linalg.eigh(cov_regularized)
            except np.linalg.LinAlgError:
                # Second fallback: use SVD which is more robust
                # SVD decomposition: centered = U @ S @ Vt
                # The normal to the plane is the last column of V (last row of Vt)
                normals = np.zeros((centered.shape[0], 3))
                for i in range(centered.shape[0]):
                    try:
                        _, _, vt = np.linalg.svd(centered[i].T)
                        normals[i] = vt[-1]  # Last row of Vt is normal vector
                    except np.linalg.LinAlgError:
                        # Ultimate fallback: use cross product of first two centered points
                        # if available, otherwise use vertical normal
                        if centered.shape[1] >= 2:
                            v1 = centered[i, :, 0]
                            v2 = centered[i, :, 1]
                            normal = np.cross(v1, v2)
                            norm = np.linalg.norm(normal)
                            normals[i] = normal / (norm + 1e-10)
                        else:
                            # Default to vertical plane normal
                            normals[i] = np.array([0, 1, 0])

                # Skip to coefficient calculation
                a, b, c = normals[:, :3].T
                d = -np.sum(normals * centroid[:, 0, :], axis=1)

        # get timeseries
        index = np.unique(np.concatenate([i.index for i in points]))
        return cls(a, b, c, d, index=index)
