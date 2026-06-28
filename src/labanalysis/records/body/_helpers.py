"""Helper methods for geometric calculations and transformations."""

from typing import Literal
import numpy as np

from ...timeseries import Point3D, Signal1D, Signal3D, Timeseries

__all__ = ["HelpersMixin"]


class HelpersMixin:
    """
    Mixin providing geometric helper methods for biomechanical calculations.

    Includes methods for:
    - Plane calculations (least squares fitting, coefficients)
    - Point-plane operations (distance, projection, intersection)
    - Point transformations (translation, projection onto axis)
    - Angle calculations (three points, reference frames)
    - Shoulder joint center estimation (De Leva regression)
    """

    def _get_least_squares_plane_coefs(self, *points: Point3D):
        """
        Calculate plane coefficients (a, b, c, d) at each time instant using least squares.

        Fits a plane to at least 3 points in 3D space using principal component
        analysis. The plane normal is determined by the eigenvector corresponding
        to the smallest eigenvalue of the covariance matrix.

        Parameters
        ----------
        *points : Point3D
            At least 3 Point3D instances defining the plane at each time sample.

        Returns
        -------
        coefficients : np.ndarray, shape (N, 4)
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
        for point in points:
            if not isinstance(point, Point3D):
                raise ValueError("all points must be Point3D instances.")

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
                coefficients = np.stack([a, b, c, d], axis=1)
                return coefficients

        # Eigenvector associated with smallest eigenvalue is the plane normal
        normals = eigvecs[:, :, 0]

        # Coefficients a, b, c
        a, b, c = normals[:, :3].T

        # Calculate d = -(a*x + b*y + c*z) using centroid
        d = -np.sum(normals * centroid[:, 0, :], axis=1)

        # Stack coefficients
        coefficients = np.stack([a, b, c, d], axis=1)

        return coefficients

    def _get_plane_coefs(
        self,
        onplane_a: Point3D,
        onplane_b: Point3D,
        ortogonal_to: Point3D,
    ):
        """
        Calculate plane coefficients passing through two points with orthogonal constraint.

        Computes plane equation (a, b, c, d) for a plane that:
        - Passes through points onplane_a and onplane_b
        - Has normal perpendicular to the line connecting ortogonal_to to the plane

        Parameters
        ----------
        onplane_a : Point3D
            First point on the plane.
        onplane_b : Point3D
            Second point on the plane.
        ortogonal_to : Point3D
            Point defining orthogonal constraint direction.

        Returns
        -------
        coefficients : np.ndarray, shape (N, 4)
            Plane coefficients [a, b, c, d] where ax + by + cz + d = 0.

        Notes
        -----
        The plane normal is found by:
        1. Computing in-plane vector v = b - a
        2. Computing raw normal n_raw = a - ortogonal_to
        3. Removing component of n_raw parallel to v (Gram-Schmidt)
        4. The orthogonal component becomes the plane normal
        """
        # Extract data: shape (T, 3)
        a_data = onplane_a.to_numpy()
        b_data = onplane_b.to_numpy()
        o_data = ortogonal_to.to_numpy()

        # In-plane vector: v = b - a
        v = b_data - a_data

        # Raw normal vector: n_raw = a - ortogonal_to
        n_raw = a_data - o_data

        # Normalize v
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        v_unit = v / v_norm

        # Project n_raw onto v
        proj = np.sum(n_raw * v_unit, axis=1, keepdims=True) * v_unit

        # Component orthogonal to v: plane normal
        normal = n_raw - proj

        # Coefficients a, b, c
        a, b, c = normal[:, 0], normal[:, 1], normal[:, 2]

        # Calculate d using point a_data
        d = -np.sum(normal * a_data, axis=1)

        # Stack coefficients
        coefficients = np.stack([a, b, c, d], axis=1)

        return coefficients

    def _get_point_to_plane_distance(
        self,
        point: Point3D,
        planes: Timeseries,
    ):
        """
        Calculate the perpendicular distance from a point to a plane.

        Parameters
        ----------
        point : Point3D
            3D point coordinates at each time instant.
        planes : Timeseries
            Plane coefficients [a, b, c, d] at each time instant.

        Returns
        -------
        Signal1D
            Perpendicular distance from point to plane at each time instant.

        Notes
        -----
        Uses the point-to-plane distance formula:
            distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
        """
        coefs = planes.to_numpy()
        nums = np.sum(point.to_numpy() * coefs[:, :3], axis=1) + coefs[:, 3]
        dens = np.sum(coefs[:, :3] ** 2, axis=1) ** 0.5
        distances = np.abs(nums) / dens
        return Signal1D(data=distances, index=point.index, unit=point.unit)

    def _get_translated_point_along_plane(
        self,
        point: Point3D,
        local_translations: Signal3D,
        plane: Timeseries,
    ):
        """
        Translate 3D points along local axes defined by a plane.

        Creates a local coordinate system aligned with the plane and applies
        translations expressed in this local frame to the input points.

        Parameters
        ----------
        point : Point3D
            3D points to translate.
        local_translations : Signal3D
            Translation vectors in the plane's local coordinate system.
        plane : Timeseries
            Plane coefficients [a, b, c, d] defining the local axes.

        Returns
        -------
        Point3D
            Translated points in global coordinates.

        Notes
        -----
        Local coordinate system construction:
        1. Plane normal becomes z-axis
        2. Arbitrary vector cross normal gives x-axis
        3. Normal cross x gives y-axis (right-handed system)
        4. Translations in local frame are rotated to global frame
        """
        # Extract plane normal coefficients a, b, c
        normals = plane.iloc[:, :3].to_numpy()
        norm_normals = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_unit = normals / norm_normals  # normalize

        # Arbitrary vector to construct orthonormal basis
        arbitrary = np.tile(np.array([1.0, 0.0, 0.0]), (len(normals_unit), 1))
        parallel_mask = np.isclose(
            np.abs(np.sum(normals_unit * arbitrary, axis=1)), 1.0
        )
        arbitrary[parallel_mask] = np.array([0.0, 1.0, 0.0])

        # Local x-axis
        x_local = np.cross(arbitrary, normals_unit)
        x_local /= np.linalg.norm(x_local, axis=1, keepdims=True)

        # Local y-axis
        y_local = np.cross(normals_unit, x_local)
        y_local /= np.linalg.norm(y_local, axis=1, keepdims=True)

        # Rotation matrix: columns = [x_local, y_local, normal]
        R = np.stack([x_local, y_local, normals_unit], axis=2)  # shape (N, 3, 3)

        # Transform local translations to global coordinates
        l_trans = np.asarray(local_translations)
        T_global = np.einsum("nij,nj->ni", R, l_trans)

        # Apply translation to points
        out: Point3D = point.copy() + T_global
        return out

    def _get_intersection_between_line_to_plane(
        self,
        p1: Point3D,
        p2: Point3D,
        plane: Timeseries,
    ):
        """
        Calculate intersection point between a line and a plane.

        Finds where the line defined by p1 -> p2 intersects the plane
        at each time instant.

        Parameters
        ----------
        p1 : Point3D
            First point defining the line.
        p2 : Point3D
            Second point defining the line.
        plane : Timeseries
            Plane coefficients [a, b, c, d] at each time instant.

        Returns
        -------
        np.ndarray
            Intersection points at each time instant. NaN where line is
            parallel to plane.

        Notes
        -----
        Uses parametric line equation: P = p1 + t*(p2 - p1)
        Solves for t where P satisfies plane equation: ax + by + cz + d = 0
        """
        direction = np.asarray(p2 - p1)
        planearr = np.asarray(plane)
        a, b, c, d = planearr.T
        normal = np.stack([a, b, c], axis=1)

        numerator = -(np.sum(normal * p1, axis=1) + d)
        denominator = np.sum(normal * direction, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(denominator != 0, numerator / denominator, np.nan)

        intersection_points = p1 + direction * t[:, np.newaxis]  # type: ignore
        return intersection_points._data

    def _get_projection_point_on_plane(
        self,
        point: Point3D,
        plane: Timeseries,
    ):
        """
        Calculate orthogonal projection of a point onto a plane.

        Finds the point on the plane that minimizes distance to the input point
        (perpendicular projection).

        Parameters
        ----------
        point : Point3D
            3D point to project.
        plane : Timeseries
            Plane coefficients [a, b, c, d] at each time instant.

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
        planearr = plane.to_numpy()  # Nx4: [a, b, c, d]
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

    def _get_angle_between_three_points(
        self,
        p1: Point3D | np.ndarray,
        p2: Point3D | np.ndarray,
        p3: Point3D | np.ndarray,
    ):
        """
        Calculate angle formed by three 3D points with vertex at the middle point.

        Computes the angle at point p2 formed by the segments p1-p2 and p2-p3
        using the law of cosines.

        Parameters
        ----------
        p1 : Point3D or np.ndarray
            First point (shape (N, 3)).
        p2 : Point3D or np.ndarray
            Vertex point (shape (N, 3)).
        p3 : Point3D or np.ndarray
            Third point (shape (N, 3)).

        Returns
        -------
        np.ndarray
            Angles in degrees at each time instant.

        Notes
        -----
        Uses law of cosines: cos(θ) = (AB² + BC² - AC²) / (2·AB·BC)
        where AB, BC, AC are segment lengths.
        """

        # Get segment lengths
        v1 = p1.to_numpy() if isinstance(p1, Point3D) else p1
        v2 = p2.to_numpy() if isinstance(p2, Point3D) else p2
        v3 = p3.to_numpy() if isinstance(p3, Point3D) else p3
        AB = ((v1 - v2) ** 2).sum(axis=1) ** 0.5
        BC = ((v2 - v3) ** 2).sum(axis=1) ** 0.5
        AC = ((v1 - v3) ** 2).sum(axis=1) ** 0.5

        cos_angle = np.clip((AC**2 - AB**2 - BC**2) / (-2 * AB * BC), -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return np.asarray(angle_deg, float)

    def _get_projection_point_onto_axis(
        self,
        point: Point3D,
        origin: Point3D,
        direction: Point3D,
    ):
        """
        Get the projection of multiple points onto their respective axes defined
        by origin and direction vectors in 3D space.
        """
        # Convert to numpy arrays
        point_np = point.to_numpy()  # shape (n, 3)
        origin_np = origin.to_numpy()  # shape (n, 3)
        direction_np = direction.to_numpy()  # shape (n, 3)

        # Vectors from origin to point
        v = point_np - origin_np  # shape (n, 3)

        # Scalar projection length onto each axis
        proj_length = np.sum(v * direction_np, axis=1)
        proj_length /= np.sum(direction_np**2, axis=1)  # shape (n,)

        # Calculate projected points
        proj_np = origin_np + (
            proj_length[:, np.newaxis] * direction_np
        )  # shape (n, 3)

        # Return new Point3D with same properties as original point
        return Point3D(
            proj_np,
            point.index,
            point.unit,
            point.columns,
        )

    def _calculate_shoulder_from_acromion(
        self, acromion: Point3D, side: str
    ) -> Point3D:
        """
        Calculate shoulder joint center using De Leva (1996) regression from acromion.

        Parameters
        ----------
        acromion : Point3D
            Acromion marker position (left or right)
        side : str
            "left" or "right" to identify which elbow to use for length estimation

        Returns
        -------
        Point3D
            Estimated shoulder joint center position

        Notes
        -----
        Uses De Leva (1996) regression with relative correction:
        SJC is -12.25% of upper arm length proximal to acromion along
        upper arm longitudinal axis.

        This percentage is sex-independent and derived from De Leva's original data:
        - Females: -33.7 mm / 275.1 mm = -12.25%
        - Males: -34.5 mm / 281.7 mm = -12.25%
        """
        # Get reference points for direction calculation
        c7 = self._get_point("c7")  # Cervical spine marker (trunk reference)
        # Note: _get_point() will raise ValueError if c7 is missing

        # Estimate upper arm length from acromion to elbow
        # This approximates the acromion-to-radiale length used by De Leva
        if side == "left":
            elbow = self.left_elbow
        elif side == "right":
            elbow = self.right_elbow
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'left' or 'right'")

        upper_arm_vector = (elbow - acromion).to_numpy()
        upper_arm_length = np.linalg.norm(upper_arm_vector, axis=1, keepdims=True)

        # Calculate direction vector from acromion toward trunk (proximal direction)
        direction = (c7 - acromion).to_numpy()
        direction_normalized = direction / np.linalg.norm(
            direction, axis=1, keepdims=True
        )

        # De Leva (1996) relative offset: -12.25% of upper arm length
        # This is sex-independent when expressed as percentage
        DELEVA_SHOULDER_COEFFICIENT = -0.1225
        offset = upper_arm_length * DELEVA_SHOULDER_COEFFICIENT

        # Apply offset along proximal direction
        # offset is negative, so we move toward trunk
        sjc_position = acromion.to_numpy() + offset * direction_normalized

        return Point3D(
            data=sjc_position,
            index=acromion.index,
            columns=acromion.columns,
            unit=acromion.unit,
        )

    def _get_angle_by_point_on_reference_frame_and_plane(
        self,
        point: Point3D,
        origin: Point3D,
        rotation_matrix: np.ndarray,
        axis_a: str | Literal["X", "Y", "Z"],
        axis_b: str | Literal["X", "Y", "Z"],
    ):

        # get the relevant axis map
        cols = point.columns
        plane_axes = np.array([axis_a, axis_b])
        col_map = [np.where(cols == i)[0][0] for i in plane_axes]
        col_map = np.array(col_map)

        # get the point used to calculate the angle
        centered = (point - origin).to_numpy()
        rotated = np.einsum("nij,nj->ni", rotation_matrix, centered)
        x, y = rotated[:, col_map].T

        # For left-handed reference frames (det(R) = -1), the anteroposterior_axis
        # is negated to keep it pointing forward. When using anteroposterior in
        # angle calculations, we must account for this by negating the component.
        # Check if we're using the anteroposterior axis and if frame is left-handed
        det_R = np.linalg.det(rotation_matrix[0])  # Check first frame
        is_left_handed = det_R < 0

        # Map axis names to indices in rotated coordinates
        axis_map = {"X": 0, "Y": 1, "Z": 2}
        a_idx = axis_map.get(axis_a, -1)
        b_idx = axis_map.get(axis_b, -1)

        # If using Z-axis in left-handed frame, negate it
        if is_left_handed:
            if a_idx == 2:  # axis_a is Z
                x = -x
            if b_idx == 2:  # axis_b is Z
                y = -y

        # get the angle
        angle = np.degrees(np.arctan2(y, x))

        # convert to signal
        angle = Signal1D(angle, point.index, "°")

        return angle, np.asarray(x, float), np.asarray(y, float)

    def _get_projected_pelvis_points(self):
        # get the pelvis references
        l_asis = self._get_point("left_asis")
        r_asis = self._get_point("right_asis")
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")

        # generate the pelvis plane by least squares
        plane = self._get_least_squares_plane_coefs(l_asis, r_asis, l_psis, r_psis)
        plane = Timeseries(
            data=plane,
            index=l_asis.index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )

        # get the projected points
        l_asis_proj = self._get_projection_point_on_plane(l_asis, plane)
        r_asis_proj = self._get_projection_point_on_plane(r_asis, plane)
        l_psis_proj = self._get_projection_point_on_plane(l_psis, plane)
        r_psis_proj = self._get_projection_point_on_plane(r_psis, plane)

        return l_asis_proj, r_asis_proj, l_psis_proj, r_psis_proj
