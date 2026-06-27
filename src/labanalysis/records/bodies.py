"""full body module"""

from typing import Literal

import numpy as np
import pandas as pd

from ..signalprocessing import gram_schmidt
from .records import *
from .referenceframes import ReferenceFrame
from .timeseries import *

__all__ = ["WholeBody"]


def _auto_populate_angular_measures(cls):
    """
    Class decorator to auto-populate _angular_measures.

    Discovers angular measure properties by introspecting the class
    and identifying properties with biomechanical angle keywords in their names.

    Parameters
    ----------
    cls : class
        The class to decorate (WholeBody)

    Returns
    -------
    class
        The decorated class with _angular_measures attribute set
    """
    # Biomechanical keywords that identify angular properties
    angle_keywords = [
        "flexion",
        "extension",
        "rotation",
        "abduction",
        "adduction",
        "tilt",
        "valgus",
        "varus",
        "inversion",
        "eversion",
        "protraction",
        "retraction",
        "lordosis",
        "kyphosis",
    ]

    angular_props = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(cls, name, None)
            if not isinstance(attr, property):
                continue
            if any(kw in name.lower() for kw in angle_keywords):
                angular_props.append(name)
        except:
            pass

    cls._angular_measures = sorted(angular_props)
    return cls


@_auto_populate_angular_measures
class WholeBody(TimeseriesRecord):
    """
    Full body biomechanical model with anatomical landmarks and joint angles.

    This class represents a complete human body model including anatomical landmark
    positions (3D points), ground reaction forces, and computed joint kinematics.
    Automatically calculates joint centers, segment reference frames, and joint
    angles based on provided anatomical markers.

    Parameters
    ----------
    left_hand_ground_reaction_force : ForcePlatform, optional
        Left hand GRF data.
    right_hand_ground_reaction_force : ForcePlatform, optional
        Right hand GRF data.
    left_foot_ground_reaction_force : ForcePlatform, optional
        Left foot GRF data.
    right_foot_ground_reaction_force : ForcePlatform, optional
        Right foot GRF data.
    left_heel : Point3D, optional
        Left heel marker position.
    right_heel : Point3D, optional
        Right heel marker position.
    left_toe : Point3D, optional
        Left toe marker position.
    right_toe : Point3D, optional
        Right toe marker position.
    left_first_metatarsal_head : Point3D, optional
        Left first metatarsal head marker.
    left_fifth_metatarsal_head : Point3D, optional
        Left fifth metatarsal head marker.
    right_first_metatarsal_head : Point3D, optional
        Right first metatarsal head marker.
    right_fifth_metatarsal_head : Point3D, optional
        Right fifth metatarsal head marker.
    left_ankle_medial : Point3D, optional
        Left medial malleolus marker.
    left_ankle_lateral : Point3D, optional
        Left lateral malleolus marker.
    right_ankle_medial : Point3D, optional
        Right medial malleolus marker.
    right_ankle_lateral : Point3D, optional
        Right lateral malleolus marker.
    left_knee_medial : Point3D, optional
        Left medial femoral epicondyle marker.
    left_knee_lateral : Point3D, optional
        Left lateral femoral epicondyle marker.
    right_knee_medial : Point3D, optional
        Right medial femoral epicondyle marker.
    right_knee_lateral : Point3D, optional
        Right lateral femoral epicondyle marker.
    left_trochanter : Point3D, optional
        Left greater trochanter marker.
    right_trochanter : Point3D, optional
        Right greater trochanter marker.
    left_asis : Point3D, optional
        Left anterior superior iliac spine marker.
    right_asis : Point3D, optional
        Right anterior superior iliac spine marker.
    left_psis : Point3D, optional
        Left posterior superior iliac spine marker.
    right_psis : Point3D, optional
        Right posterior superior iliac spine marker.
    left_shoulder_anterior : Point3D, optional
        Left anterior shoulder marker.
    left_shoulder_posterior : Point3D, optional
        Left posterior shoulder marker.
    left_acromion : Point3D, optional
        Left acromion.
    right_shoulder_anterior : Point3D, optional
        Right anterior shoulder marker.
    right_shoulder_posterior : Point3D, optional
        Right posterior shoulder marker.
    right_acromion : Point3D, optional
        Right acromion.
    left_elbow_medial : Point3D, optional
        Left medial epicondyle marker.
    left_elbow_lateral : Point3D, optional
        Left lateral epicondyle marker.
    right_elbow_medial : Point3D, optional
        Right medial epicondyle marker.
    right_elbow_lateral : Point3D, optional
        Right lateral epicondyle marker.
    left_wrist_medial : Point3D, optional
        Left medial wrist marker.
    left_wrist_lateral : Point3D, optional
        Left lateral wrist marker.
    right_wrist_medial : Point3D, optional
        Right medial wrist marker.
    right_wrist_lateral : Point3D, optional
        Right lateral wrist marker.
    s2 : Point3D, optional
        Second sacral vertebra marker.
    l2 : Point3D, optional
        Second lumbar vertebra marker.
    c7 : Point3D, optional
        Seventh cervical vertebra marker.
    t5 : Point3D, optional
        Fifth thoracic vertebra marker.
    sc : Point3D, optional
        Sternoclavicular joint marker (midpoint between clavicles).
    head_anterior : Point3D, optional
        Anterior cranium marker (front of head).
    head_posterior : Point3D, optional
        Posterior cranium marker (back of head).
    head_left : Point3D, optional
        Left side cranium marker.
    head_right : Point3D, optional
        Right side cranium marker.
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, or ForcePlatform
        Additional signals to include in the record.

    Attributes
    ----------
    _angular_measures : list of str
        Auto-discovered list of all angular measure properties, populated
        by the class decorator based on naming conventions.

    Notes
    -----
    - Joint centers (ankle, knee, elbow, wrist) are calculated
      as midpoints between medial and lateral markers
    - Shoulder joint centers are calculated as midpoints between anterior and
      posterior markers (if available). Otherwise De Leva (1996) regression
      equations are used (if left and right acromion are available)
    - Hip joint centers use De Leva (1996) regression equations
    - Joint angles follow standard biomechanical conventions (flexion/extension,
      abduction/adduction, internal/external rotation)
    - Pelvis and trunk angles can be computed in global or local reference frames
    - All angle properties return Signal1D objects in degrees

    Examples
    --------
    >>> # Create from marker data
    >>> body = WholeBody(
    ...     left_heel=heel_marker,
    ...     right_heel=heel_marker,
    ...     left_ankle_medial=ankle_med,
    ...     left_ankle_lateral=ankle_lat,
    ...     # ... other markers
    ... )
    >>> # Access computed joint angles
    >>> knee_flexion = body.left_knee_flexionextension
    >>> hip_abd = body.left_hip_abductionadduction
    """

    def __init__(
        self,
        left_hand_ground_reaction_force: ForcePlatform | None = None,
        right_hand_ground_reaction_force: ForcePlatform | None = None,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        left_heel: Point3D | None = None,
        right_heel: Point3D | None = None,
        left_toe: Point3D | None = None,
        right_toe: Point3D | None = None,
        left_first_metatarsal_head: Point3D | None = None,
        left_fifth_metatarsal_head: Point3D | None = None,
        right_first_metatarsal_head: Point3D | None = None,
        right_fifth_metatarsal_head: Point3D | None = None,
        left_ankle_medial: Point3D | None = None,
        left_ankle_lateral: Point3D | None = None,
        right_ankle_medial: Point3D | None = None,
        right_ankle_lateral: Point3D | None = None,
        left_knee_medial: Point3D | None = None,
        left_knee_lateral: Point3D | None = None,
        right_knee_medial: Point3D | None = None,
        right_knee_lateral: Point3D | None = None,
        right_trochanter: Point3D | None = None,
        left_trochanter: Point3D | None = None,
        left_asis: Point3D | None = None,
        right_asis: Point3D | None = None,
        left_psis: Point3D | None = None,
        right_psis: Point3D | None = None,
        left_shoulder_anterior: Point3D | None = None,
        left_shoulder_posterior: Point3D | None = None,
        left_acromion: Point3D | None = None,
        right_shoulder_anterior: Point3D | None = None,
        right_shoulder_posterior: Point3D | None = None,
        right_acromion: Point3D | None = None,
        left_elbow_medial: Point3D | None = None,
        left_elbow_lateral: Point3D | None = None,
        right_elbow_medial: Point3D | None = None,
        right_elbow_lateral: Point3D | None = None,
        left_wrist_medial: Point3D | None = None,
        left_wrist_lateral: Point3D | None = None,
        right_wrist_medial: Point3D | None = None,
        right_wrist_lateral: Point3D | None = None,
        s2: Point3D | None = None,
        l2: Point3D | None = None,
        c7: Point3D | None = None,
        t5: Point3D | None = None,
        sc: Point3D | None = None,  # sternoclavicular joint
        head_anterior: Point3D | None = None,
        head_posterior: Point3D | None = None,
        head_left: Point3D | None = None,
        head_right: Point3D | None = None,
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        signals = {
            **extra_signals,
            **dict(
                left_hand_ground_reaction_force=left_hand_ground_reaction_force,
                right_hand_ground_reaction_force=right_hand_ground_reaction_force,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                left_heel=left_heel,
                right_heel=right_heel,
                left_toe=left_toe,
                right_toe=right_toe,
                left_first_metatarsal_head=left_first_metatarsal_head,
                left_fifth_metatarsal_head=left_fifth_metatarsal_head,
                right_first_metatarsal_head=right_first_metatarsal_head,
                right_fifth_metatarsal_head=right_fifth_metatarsal_head,
                left_ankle_medial=left_ankle_medial,
                left_ankle_lateral=left_ankle_lateral,
                right_ankle_medial=right_ankle_medial,
                right_ankle_lateral=right_ankle_lateral,
                left_knee_medial=left_knee_medial,
                left_knee_lateral=left_knee_lateral,
                right_knee_medial=right_knee_medial,
                right_knee_lateral=right_knee_lateral,
                left_trochanter=left_trochanter,
                right_trochanter=right_trochanter,
                left_asis=left_asis,
                right_asis=right_asis,
                left_psis=left_psis,
                right_psis=right_psis,
                left_shoulder_anterior=left_shoulder_anterior,
                left_shoulder_posterior=left_shoulder_posterior,
                left_acromion=left_acromion,
                right_shoulder_anterior=right_shoulder_anterior,
                right_shoulder_posterior=right_shoulder_posterior,
                right_acromion=right_acromion,
                left_elbow_medial=left_elbow_medial,
                left_elbow_lateral=left_elbow_lateral,
                right_elbow_medial=right_elbow_medial,
                right_elbow_lateral=right_elbow_lateral,
                left_wrist_medial=left_wrist_medial,
                left_wrist_lateral=left_wrist_lateral,
                right_wrist_medial=right_wrist_medial,
                right_wrist_lateral=right_wrist_lateral,
                s2=s2,
                c7=c7,
                t5=t5,
                sc=sc,
                l2=l2,
                head_anterior=head_anterior,
                head_posterior=head_posterior,
                head_left=head_left,
                head_right=head_right,
            ),
        }
        super().__init__(**{i: v for i, v in signals.items() if v is not None})

    @classmethod
    def from_tdf(
        cls,
        filename: str,
        ground_reaction_force: str | None = None,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        left_heel: str | None = None,
        right_heel: str | None = None,
        left_toe: str | None = None,
        right_toe: str | None = None,
        left_first_metatarsal_head: str | None = None,
        left_fifth_metatarsal_head: str | None = None,
        right_first_metatarsal_head: str | None = None,
        right_fifth_metatarsal_head: str | None = None,
        left_ankle_medial: str | None = None,
        left_ankle_lateral: str | None = None,
        right_ankle_medial: str | None = None,
        right_ankle_lateral: str | None = None,
        left_knee_medial: str | None = None,
        left_knee_lateral: str | None = None,
        right_knee_medial: str | None = None,
        right_knee_lateral: str | None = None,
        right_trochanter: str | None = None,
        left_trochanter: str | None = None,
        left_asis: str | None = None,
        right_asis: str | None = None,
        left_psis: str | None = None,
        right_psis: str | None = None,
        left_shoulder_anterior: str | None = None,
        left_shoulder_posterior: str | None = None,
        left_acromion: str | None = None,
        right_shoulder_anterior: str | None = None,
        right_shoulder_posterior: str | None = None,
        right_acromion: str | None = None,
        left_elbow_medial: str | None = None,
        left_elbow_lateral: str | None = None,
        right_elbow_medial: str | None = None,
        right_elbow_lateral: str | None = None,
        left_wrist_medial: str | None = None,
        left_wrist_lateral: str | None = None,
        right_wrist_medial: str | None = None,
        right_wrist_lateral: str | None = None,
        s2: str | None = None,
        l2: str | None = None,
        c7: str | None = None,
        t5: str | None = None,
        sc: str | None = None,
        head_anterior: str | None = None,
        head_posterior: str | None = None,
        head_left: str | None = None,
        head_right: str | None = None,
    ):

        # read the file
        tdf = TimeseriesRecord.from_tdf(filename)

        # check the inputs
        points = {
            "left_heel": left_heel,
            "right_heel": right_heel,
            "left_toe": left_toe,
            "right_toe": right_toe,
            "left_first_metatarsal_head": left_first_metatarsal_head,
            "left_fifth_metatarsal_head": left_fifth_metatarsal_head,
            "right_first_metatarsal_head": right_first_metatarsal_head,
            "right_fifth_metatarsal_head": right_fifth_metatarsal_head,
            "left_ankle_medial": left_ankle_medial,
            "left_ankle_lateral": left_ankle_lateral,
            "right_ankle_medial": right_ankle_medial,
            "right_ankle_lateral": right_ankle_lateral,
            "left_knee_medial": left_knee_medial,
            "left_knee_lateral": left_knee_lateral,
            "right_knee_medial": right_knee_medial,
            "right_knee_lateral": right_knee_lateral,
            "right_trochanter": right_trochanter,
            "left_trochanter": left_trochanter,
            "left_asis": left_asis,
            "right_asis": right_asis,
            "left_psis": left_psis,
            "right_psis": right_psis,
            "left_shoulder_anterior": left_shoulder_anterior,
            "left_shoulder_posterior": left_shoulder_posterior,
            "left_acromion": left_acromion,
            "right_shoulder_anterior": right_shoulder_anterior,
            "right_shoulder_posterior": right_shoulder_posterior,
            "right_acromion": right_acromion,
            "left_elbow_medial": left_elbow_medial,
            "left_elbow_lateral": left_elbow_lateral,
            "right_elbow_medial": right_elbow_medial,
            "right_elbow_lateral": right_elbow_lateral,
            "left_wrist_medial": left_wrist_medial,
            "left_wrist_lateral": left_wrist_lateral,
            "right_wrist_medial": right_wrist_medial,
            "right_wrist_lateral": right_wrist_lateral,
            "s2": s2,
            "c7": c7,
            "l2": l2,
            "t5": t5,
            "sc": sc,
            "head_anterior": head_anterior,
            "head_posterior": head_posterior,
            "head_left": head_left,
            "head_right": head_right,
        }
        forces = {
            "ground_reaction_force": ground_reaction_force,
            "left_hand_ground_reaction_force": left_hand_ground_reaction_force,
            "right_hand_ground_reaction_force": right_hand_ground_reaction_force,
            "left_foot_ground_reaction_force": left_foot_ground_reaction_force,
            "right_foot_ground_reaction_force": right_foot_ground_reaction_force,
        }
        keys = tdf.keys()
        mandatory = {}
        for key, lbl in forces.items():
            if lbl is not None:
                if lbl not in keys:
                    raise ValueError(f"{lbl} not found.")
                if not isinstance(tdf[lbl], ForcePlatform):
                    raise ValueError(f"{lbl} must be a ForcePlatform instance.")
                mandatory[key] = tdf[lbl]
                tdf.drop(lbl, True)
        for key, lbl in points.items():
            if lbl is not None:
                if lbl not in keys:
                    raise ValueError(f"{lbl} not found.")
                if not isinstance(tdf[lbl], Point3D):
                    raise ValueError(f"{lbl} must be a Point3D instance.")
                mandatory[key] = tdf[lbl]
                tdf.drop(lbl, inplace=True)
        extras = {i: v for i, v in tdf.items() if i not in list(mandatory.keys())}

        return cls(**mandatory, **extras)  # type: ignore

    def _get_point(self, label: str):
        element = self.get(label)
        if element is None:
            raise ValueError(f"{label} not found.")
        if not isinstance(element, Point3D):
            raise ValueError(f"{label} is not a Point3D.")
        return element

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

    @property
    def left_ankle_referenceframe(self):
        """
        Left ankle reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Left ankle center (use `self.left_ankle` property)

        Construction
        ------------
        1. lateral_axis: LEFT (left_ankle_lateral → left_ankle_medial)
        2. vertical_axis: UP (left_ankle → left_knee)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame object with origin at ankle center and orthonormalized axes.

        See Also
        --------
        left_ankle : Ankle center (origin of this frame)
        right_ankle_referenceframe : Right ankle reference frame
        left_ankle_flexionextension : Flexion angle using this frame
        left_ankle_inversioneversion : Inversion angle using this frame
        """
        ankle = self.left_ankle
        knee = self.left_knee

        try:
            ankle_lat = self._get_point("left_ankle_lateral")
            ankle_med = self._get_point("left_ankle_medial")
            # Construct lateral_axis: LEFT (lateral to medial)
            axis_x = (ankle_lat - ankle_med).to_numpy()
        except Exception:
            # Default LEFT: use lateral_axis property to determine which column
            lateral_idx = np.where(ankle.columns == ankle.lateral_axis)[0][0]
            axis_x = np.zeros((ankle.shape[0], 3))
            axis_x[:, lateral_idx] = -1  # LEFT (negative for left side)

        # Construct vertical_axis: UP (ankle to knee)
        axis_y = (knee - ankle).to_numpy()

        # Create ReferenceFrame
        return ReferenceFrame(origin=ankle, lateral_axis=axis_x, vertical_axis=axis_y)

    @property
    def right_ankle_referenceframe(self):
        """
        Right ankle reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Right ankle center (use `self.right_ankle` property)

        Construction
        ------------
        1. lateral_axis: RIGHT (right_ankle_lateral → right_ankle_medial)
        2. vertical_axis: UP (right_ankle → right_knee)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame object with origin at ankle center and orthonormalized axes.

        See Also
        --------
        right_ankle : Ankle center (origin of this frame)
        left_ankle_referenceframe : Left ankle reference frame
        right_ankle_flexionextension : Flexion angle using this frame
        right_ankle_inversioneversion : Inversion angle using this frame
        """
        ankle = self.right_ankle
        knee = self.right_knee

        try:
            ankle_lat = self._get_point("right_ankle_lateral")
            ankle_med = self._get_point("right_ankle_medial")
            # Construct lateral_axis: RIGHT (lateral to medial)
            axis_x = (ankle_lat - ankle_med).to_numpy()
        except Exception:
            # Default RIGHT: use lateral_axis property to determine which column
            lateral_idx = np.where(ankle.columns == ankle.lateral_axis)[0][0]
            axis_x = np.zeros((ankle.shape[0], 3))
            axis_x[:, lateral_idx] = 1  # RIGHT (positive for right side)

        # Construct vertical_axis: UP (ankle to knee)
        axis_y = (knee - ankle).to_numpy()

        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)

        # Create ReferenceFrame (left-handed: det(R) = -1)
        return ReferenceFrame(
            origin=ankle,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def left_knee_referenceframe(self):
        """
        Left knee reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Left knee center (use `self.left_knee` property)

        Construction
        ------------
        1. lateral_axis: LEFT (left_knee_lateral → left_knee_medial)
        2. vertical_axis: UP (left_knee → left_hip)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left knee center and orthonormal axes.

        See Also
        --------
        left_knee : Knee center (origin of this frame)
        right_knee_referenceframe : Right knee reference frame
        left_knee_flexionextension : Knee flexion angle using this frame
        left_knee_varusvalgus : Knee varus/valgus angle using this frame
        """
        knee = self.left_knee
        hip = self.left_hip

        try:
            knee_lat = self._get_point("left_knee_lateral")
            knee_med = self._get_point("left_knee_medial")
            # Construct lateral_axis: LEFT (lateral to medial)
            axis_x = (knee_lat - knee_med).to_numpy()
        except Exception:
            # Default LEFT: use lateral_axis property to determine which column
            lateral_idx = np.where(knee.columns == knee.lateral_axis)[0][0]
            axis_x = np.zeros((knee.shape[0], 3))
            axis_x[:, lateral_idx] = -1  # LEFT (negative for left side)

        # Construct vertical_axis: UP (knee to hip)
        axis_y = (hip - knee).to_numpy()

        return ReferenceFrame(origin=knee, lateral_axis=axis_x, vertical_axis=axis_y)

    @property
    def right_knee_referenceframe(self):
        """
        Right knee reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Right knee center (use `self.right_knee` property)

        Construction
        ------------
        1. lateral_axis: RIGHT (right_knee_lateral → right_knee_medial)
        2. vertical_axis: UP (right_knee → right_hip)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right knee center and orthonormal axes.

        See Also
        --------
        right_knee : Knee center (origin of this frame)
        left_knee_referenceframe : Left knee reference frame
        right_knee_flexionextension : Knee flexion angle using this frame
        right_knee_varusvalgus : Knee varus/valgus angle using this frame
        """
        knee = self.right_knee
        hip = self.right_hip

        try:
            knee_lat = self._get_point("right_knee_lateral")
            knee_med = self._get_point("right_knee_medial")
            # Construct lateral_axis: RIGHT (lateral to medial)
            axis_x = (knee_lat - knee_med).to_numpy()
        except Exception:
            # Default RIGHT: use lateral_axis property to determine which column
            lateral_idx = np.where(knee.columns == knee.lateral_axis)[0][0]
            axis_x = np.zeros((knee.shape[0], 3))
            axis_x[:, lateral_idx] = 1  # RIGHT (positive for right side)

        # Construct vertical_axis: UP (knee to hip)
        axis_y = (hip - knee).to_numpy()

        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)

        return ReferenceFrame(
            origin=knee,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def left_hip_referenceframe(self):
        """
        Left hip reference frame (based on pelvis frame with X pointing LEFT).

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Left hip joint center (use `self.left_hip` property, De Leva 1996)

        Construction
        ------------
        Uses pelvis reference frame directly (X already points LEFT for left side).

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left hip center and axes from pelvis frame.

        See Also
        --------
        left_hip : Hip joint center (origin of this frame)
        right_hip_referenceframe : Right hip reference frame
        pelvis_referenceframe : Base pelvis reference frame
        left_hip_flexionextension : Hip flexion angle using this frame
        left_hip_abductionadduction : Hip abduction angle using this frame
        left_hip_internalexternalrotation : Hip rotation angle using this frame
        """
        # For left hip, use pelvis frame axes with hip origin
        pelvis_rf = self.pelvis_referenceframe
        return ReferenceFrame(
            origin=self.left_hip,
            lateral_axis=pelvis_rf.lateral_axis,
            vertical_axis=pelvis_rf.vertical_axis,
            anteroposterior_axis=pelvis_rf.anteroposterior_axis,
        )

    @property
    def right_hip_referenceframe(self):
        """
        Right hip reference frame (mirrored pelvis frame with X pointing RIGHT).

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Right hip joint center (use `self.right_hip` property, De Leva 1996)

        Construction
        ------------
        Mirrors the pelvis reference frame:
        - X: negated pelvis X (points RIGHT instead of LEFT)
        - Y: same pelvis Y (points UP)
        - Z: same pelvis Z (points FORWARD, det(R) = -1, left-handed)

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right hip center, mirrored axes (left-handed).

        See Also
        --------
        right_hip : Hip joint center (origin of this frame)
        left_hip_referenceframe : Left hip reference frame
        pelvis_referenceframe : Base pelvis reference frame
        right_hip_flexionextension : Hip flexion angle using this frame
        right_hip_abductionadduction : Hip abduction angle using this frame
        right_hip_internalexternalrotation : Hip rotation angle using this frame
        """
        pelvis_rf = self.pelvis_referenceframe

        # For right hip: lateral axis points RIGHT, vertical UP, anteroposterior FORWARD
        # This creates a left-handed system (det(R) = -1) to maintain Z pointing forward
        axis_x = -pelvis_rf.lateral_axis  # Lateral: RIGHT
        axis_y = pelvis_rf.vertical_axis  # Vertical: UP
        axis_z = (
            pelvis_rf.anteroposterior_axis
        )  # Anteroposterior: FORWARD (same as pelvis)

        return ReferenceFrame(
            origin=self.right_hip,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def pelvis_referenceframe(self):
        """
        Pelvis reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (LEFT, from right to left ASIS-PSIS midpoints)
        - **vertical_axis**: Superior-inferior direction (UP, from hip_center to pelvis_center)
        - **anteroposterior_axis**: Anterior-posterior direction (FORWARD, Gram-Schmidt cross product)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Pelvis center (centroid of ASIS and PSIS markers)

        Construction
        ------------
        1. lateral_axis: LEFT (from right midpoint to left midpoint)
        2. vertical_axis: UP (hip_center → pelvis_center)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at pelvis center and orthonormal axes.

        See Also
        --------
        pelvis_center : Pelvis center (origin of this frame)
        hip_center : Hip center (used for vertical axis)
        trunk_lateralflexion : Trunk lateral flexion using this frame
        pelvis_anteroposterior_tilt_global : Pelvis anteroposterior tilt using this frame
        trunk_rotation : Trunk rotation using this frame
        """
        # Get pelvis points
        l_asis = self._get_point("left_asis")
        r_asis = self._get_point("right_asis")
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")

        # Calculate midpoints
        left_mid = (l_asis + l_psis) / 2
        right_mid = (r_asis + r_psis) / 2
        centroid = (l_asis + r_asis + l_psis + r_psis) / 4

        # Get hip_center for vertical_axis construction
        hip_center = self.hip_center

        # Construct lateral_axis: LEFT (right midpoint to left midpoint)
        axis_x = (left_mid - right_mid).to_numpy()
        axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)

        # Construct vertical_axis: UP (hip_center to pelvis_center)
        axis_y = (centroid - hip_center).to_numpy()
        axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)

        # Construct anteroposterior_axis: FORWARD (cross product)
        axis_z = np.cross(axis_x, axis_y)

        return ReferenceFrame(
            origin=centroid,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def left_shoulder_referenceframe(self):
        """
        Left shoulder reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Left shoulder joint center (use `self.left_shoulder` property, De Leva 1996)

        Construction
        ------------
        1. lateral_axis: LEFT (neck_base → left_shoulder)
        2. vertical_axis: UP (pelvis_center → neck_base)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left shoulder center and orthonormal axes.

        See Also
        --------
        left_shoulder : Shoulder joint center (origin of this frame)
        neck_base : Neck base (used for X-axis)
        pelvis_center : Pelvis center (used for Y-axis)
        right_shoulder_referenceframe : Right shoulder reference frame
        left_shoulder_flexionextension : Shoulder flexion angle using this frame
        left_shoulder_abductionadduction : Shoulder abduction angle using this frame
        left_shoulder_internalexternalrotation : Shoulder rotation angle using this frame
        """
        shoulder = self.left_shoulder
        neck_base = self.neck_base
        pelvis_center = self.pelvis_center

        # Construct lateral_axis: LEFT (neck_base to shoulder, points outward/LEFT for left shoulder)
        axis_x = (shoulder - neck_base).to_numpy()
        axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)

        # Construct vertical_axis: UP (pelvis_center to neck_base)
        axis_y = (neck_base - pelvis_center).to_numpy()
        axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)

        # Construct anteroposterior_axis: FORWARD (cross product)
        axis_z = np.cross(axis_x, axis_y)

        return ReferenceFrame(
            origin=shoulder,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def right_shoulder_referenceframe(self):
        """
        Right shoulder reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Right shoulder joint center (use `self.right_shoulder` property, De Leva 1996)

        Construction
        ------------
        1. lateral_axis: RIGHT (neck_base → right_shoulder)
        2. vertical_axis: UP (pelvis_center → neck_base)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right shoulder center and orthonormal axes.

        See Also
        --------
        right_shoulder : Shoulder joint center (origin of this frame)
        neck_base : Neck base (used for X-axis)
        pelvis_center : Pelvis center (used for Y-axis)
        left_shoulder_referenceframe : Left shoulder reference frame
        right_shoulder_flexionextension : Shoulder flexion angle using this frame
        right_shoulder_abductionadduction : Shoulder abduction angle using this frame
        right_shoulder_internalexternalrotation : Shoulder rotation angle using this frame
        """
        shoulder = self.right_shoulder
        neck_base = self.neck_base
        pelvis_center = self.pelvis_center

        # Construct lateral_axis: RIGHT (neck_base to shoulder, points outward/RIGHT for right shoulder)
        axis_x = (shoulder - neck_base).to_numpy()
        axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)

        # Construct vertical_axis: UP (pelvis_center to neck_base)
        axis_y = (neck_base - pelvis_center).to_numpy()
        axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)

        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)

        return ReferenceFrame(
            origin=shoulder,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def left_ankle(self):
        lat: Point3D = self._get_point("left_ankle_lateral")
        try:
            med: Point3D = self._get_point("left_ankle_medial")
            return Point3D(
                data=(lat._data + med._data) / 2,
                index=np.unique(np.concatenate([lat.index, med.index])),
                columns=lat.columns,
            )
        except Exception as e:
            return lat

    @property
    def right_ankle(self):
        lat: Point3D = self._get_point("right_ankle_lateral")
        try:
            med: Point3D = self._get_point("right_ankle_medial")
            return Point3D(
                data=(lat._data + med._data) / 2,
                index=np.unique(np.concatenate([lat.index, med.index])),
                columns=lat.columns,
            )
        except Exception as e:
            return lat

    @property
    def left_knee(self):
        lat: Point3D = self._get_point("left_knee_lateral")
        try:
            med: Point3D = self._get_point("left_knee_medial")
            return Point3D(
                data=(lat._data + med._data) / 2,
                index=np.unique(np.concatenate([lat.index, med.index])),
                columns=lat.columns,
            )
        except Exception as e:
            return lat

    @property
    def right_knee(self):
        lat: Point3D = self._get_point("right_knee_lateral")
        try:
            med: Point3D = self._get_point("right_knee_medial")
            return Point3D(
                data=(lat._data + med._data) / 2,
                index=np.unique(np.concatenate([lat.index, med.index])),
                columns=lat.columns,
            )
        except Exception as e:
            return lat

    @property
    def pelvis_center(self):
        """
        Pelvis center (centroid of 4 ASIS/PSIS markers).

        Returns
        -------
        Point3D
            Pelvis center point (average of ASIS and PSIS markers).
        """
        l_asis = self._get_point("left_asis")
        r_asis = self._get_point("right_asis")
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")
        return (l_asis + r_asis + l_psis + r_psis) / 4

    @property
    def head_center(self):
        """
        Calculate head center as centroid of 4 cranial markers.

        Returns
        -------
        Point3D
            Head center point (average of anterior, posterior, left, right markers).
        """
        h_ant = self._get_point("head_anterior")
        h_post = self._get_point("head_posterior")
        h_left = self._get_point("head_left")
        h_right = self._get_point("head_right")

        # Calculate centroid
        data = (
            h_ant.to_numpy()
            + h_post.to_numpy()
            + h_left.to_numpy()
            + h_right.to_numpy()
        ) / 4

        # Merge indices from all markers
        index = np.unique(
            np.concatenate([h_ant.index, h_post.index, h_left.index, h_right.index])
        ).tolist()

        return Point3D(
            data=data,
            index=index,
            columns=h_ant.columns,
        )

    @property
    def neck_base(self):
        """
        Calculate neck base as midpoint between sternoclavicular joint (sc) and C7.

        Returns
        -------
        Point3D
            Neck3D base point (average of sc and c7 markers).
        """
        sc = self._get_point("sc")
        c7 = self._get_point("c7")

        # Calculate midpoint
        data = (sc.to_numpy() + c7.to_numpy()) / 2

        # Merge indices
        index = np.unique(np.concatenate([sc.index, c7.index])).tolist()

        return Point3D(
            data=data,
            index=index,
            columns=sc.columns,
        )

    @property
    def neck_referenceframe(self):
        """
        Neck reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Neck base (use `self.neck_base` property)

        Construction
        ------------
        1. vertical_axis: UP (pelvis_center → neck_base)
        2. anteroposterior_axis: FORWARD (C7 → sc)
        3. lateral_axis: LEFT (cross product vertical × anteroposterior)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at neck base and orthonormal axes.

        See Also
        --------
        neck_base : Neck base (origin of this frame)
        pelvis_center : Pelvis center (used for Y-axis)
        neck_flexionextension : Neck flexion angle using this frame
        neck_lateralflexion : Neck lateral flexion using this frame
        pelvis_rotation : Pelvis rotation using this frame
        """
        neck_base = self.neck_base
        pelvis_center = self.pelvis_center

        # Construct vertical_axis: UP (pelvis_center to neck_base)
        axis_y = (neck_base - pelvis_center).to_numpy()
        axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)

        # Construct anteroposterior_axis: FORWARD (C7 to sternoclavicular junction)
        c7 = self._get_point("c7")
        sc = self._get_point("sc")
        axis_z = (sc - c7).to_numpy()
        axis_z = axis_z / np.linalg.norm(axis_z, axis=1, keepdims=True)

        # Construct lateral_axis: LEFT (cross product vertical × anteroposterior, to point left in right-handed system)
        axis_x = np.cross(axis_y, axis_z)

        return ReferenceFrame(
            origin=neck_base,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def _pelvis_plane(self):
        """
        Calculate pelvis plane using least squares method.

        This avoids circular dependency: pelvis_referenceframe needs hip_center,
        hip_center needs left/right_hip, which need pelvis_height, which needs
        _pelvis_plane. We use least squares plane fitting instead.
        """
        # get the pelvis references
        l_asis = self._get_point("left_asis")
        r_asis = self._get_point("right_asis")
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")

        # generate the pelvis plane by least squares
        plane = self._get_least_squares_plane_coefs(l_asis, r_asis, l_psis, r_psis)

        # return the timeseries
        return Timeseries(
            data=plane,
            index=l_asis.index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )

    @property
    def pelvis_width(self):
        l_asis: Point3D = self._get_point("left_asis")
        r_asis: Point3D = self._get_point("right_asis")
        index = [l_asis.index, r_asis.index]
        index = np.unique(np.concatenate(index)).tolist()
        data = np.asarray(l_asis - r_asis)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=l_asis.unit)

    @property
    def pelvis_height(self):
        pelvis_plane = self._pelvis_plane
        try:
            l_height = self._get_point_to_plane_distance(
                self._get_point("left_trochanter"),
                pelvis_plane,
            )
        except Exception as l:
            l_height = None
        try:
            r_height = self._get_point_to_plane_distance(
                self._get_point("right_trochanter"),
                pelvis_plane,
            )
        except Exception as r:
            r_height = None
        if l_height and r_height:
            return Signal1D(
                data=(l_height + r_height).to_numpy() / 2,
                index=pelvis_plane.index,
                unit=l_height.unit,
            )
        if l_height:
            return l_height
        if r_height:
            return r_height
        raise ValueError(
            "pelvis height could not be calculated neither from left nor right side."
        )

    @property
    def left_foot_height(self):
        """
        Calculate left foot height as perpendicular distance from ankle to foot plane.

        Returns
        -------
        Signal1D
            Distance in meters from ankle joint to foot plane.
        """
        ankle = self.left_ankle
        foot_plane = self._left_foot_plane
        return self._get_point_to_plane_distance(ankle, foot_plane)

    @property
    def right_foot_height(self):
        """
        Calculate right foot height as perpendicular distance from ankle to foot plane.

        Returns
        -------
        Signal1D
            Distance in meters from ankle joint to foot plane.
        """
        ankle = self.right_ankle
        foot_plane = self._right_foot_plane
        return self._get_point_to_plane_distance(ankle, foot_plane)

    @property
    def left_foot_length(self):
        """
        Calculate left foot length as distance from heel to toe.

        Returns
        -------
        Signal1D
            Distance in meters from heel to toe marker.
        """
        heel = self._get_point("left_heel")
        toe = self._get_point("left_toe")
        index = np.unique(np.concatenate([heel.index, toe.index])).tolist()
        data = np.asarray(toe - heel)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=heel.unit)

    @property
    def right_foot_length(self):
        """
        Calculate right foot length as distance from heel to toe.

        Returns
        -------
        Signal1D
            Distance in meters from heel to toe marker.
        """
        heel = self._get_point("right_heel")
        toe = self._get_point("right_toe")
        index = np.unique(np.concatenate([heel.index, toe.index])).tolist()
        data = np.asarray(toe - heel)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=heel.unit)

    @property
    def left_leg_length(self):
        """
        Calculate left leg length as distance from ankle to knee.

        Returns
        -------
        Signal1D
            Distance in meters from ankle to knee joint center.
        """
        ankle = self.left_ankle
        knee = self.left_knee
        index = np.unique(np.concatenate([ankle.index, knee.index])).tolist()
        data = np.asarray(knee - ankle)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=ankle.unit)

    @property
    def right_leg_length(self):
        """
        Calculate right leg length as distance from ankle to knee.

        Returns
        -------
        Signal1D
            Distance in meters from ankle to knee joint center.
        """
        ankle = self.right_ankle
        knee = self.right_knee
        index = np.unique(np.concatenate([ankle.index, knee.index])).tolist()
        data = np.asarray(knee - ankle)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=ankle.unit)

    @property
    def left_thigh_length(self):
        """
        Calculate left thigh length as distance from knee to hip.

        Returns
        -------
        Signal1D
            Distance in meters from knee to hip joint center.
        """
        knee = self.left_knee
        hip = self.left_hip
        index = np.unique(np.concatenate([knee.index, hip.index])).tolist()
        data = np.asarray(hip - knee)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=knee.unit)

    @property
    def right_thigh_length(self):
        """
        Calculate right thigh length as distance from knee to hip.

        Returns
        -------
        Signal1D
            Distance in meters from knee to hip joint center.
        """
        knee = self.right_knee
        hip = self.right_hip
        index = np.unique(np.concatenate([knee.index, hip.index])).tolist()
        data = np.asarray(hip - knee)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=knee.unit)

    @property
    def left_arm_length(self):
        """
        Calculate left arm length as distance from shoulder to elbow.

        Returns
        -------
        Signal1D
            Distance in meters from shoulder to elbow joint center.
        """
        shoulder = self.left_shoulder
        elbow = self.left_elbow
        index = np.unique(np.concatenate([shoulder.index, elbow.index])).tolist()
        data = np.asarray(elbow - shoulder)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=shoulder.unit)

    @property
    def right_arm_length(self):
        """
        Calculate right arm length as distance from shoulder to elbow.

        Returns
        -------
        Signal1D
            Distance in meters from shoulder to elbow joint center.
        """
        shoulder = self.right_shoulder
        elbow = self.right_elbow
        index = np.unique(np.concatenate([shoulder.index, elbow.index])).tolist()
        data = np.asarray(elbow - shoulder)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=shoulder.unit)

    @property
    def left_forearm_length(self):
        """
        Calculate left forearm length as distance from elbow to wrist.

        Returns
        -------
        Signal1D
            Distance in meters from elbow to wrist joint center.
        """
        elbow = self.left_elbow
        wrist = self.left_wrist
        index = np.unique(np.concatenate([elbow.index, wrist.index])).tolist()
        data = np.asarray(wrist - elbow)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=elbow.unit)

    @property
    def right_forearm_length(self):
        """
        Calculate right forearm length as distance from elbow to wrist.

        Returns
        -------
        Signal1D
            Distance in meters from elbow to wrist joint center.
        """
        elbow = self.right_elbow
        wrist = self.right_wrist
        index = np.unique(np.concatenate([elbow.index, wrist.index])).tolist()
        data = np.asarray(wrist - elbow)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=elbow.unit)

    @property
    def left_lower_limb_length(self):
        """
        Calculate total left lower limb length (thigh + leg).

        Returns the sum of thigh length (hip to knee) and leg length
        (knee to ankle). This represents the functional length of the
        entire lower limb.

        Returns
        -------
        Signal1D
            Distance in meters from hip to ankle joint center.

        See Also
        --------
        left_thigh_length : Length of thigh segment only
        left_leg_length : Length of leg segment only
        right_lower_limb_length : Right lower limb total length
        """
        return self.left_thigh_length + self.left_leg_length

    @property
    def right_lower_limb_length(self):
        """
        Calculate total right lower limb length (thigh + leg).

        Returns the sum of thigh length (hip to knee) and leg length
        (knee to ankle). This represents the functional length of the
        entire lower limb.

        Returns
        -------
        Signal1D
            Distance in meters from hip to ankle joint center.

        See Also
        --------
        right_thigh_length : Length of thigh segment only
        right_leg_length : Length of leg segment only
        left_lower_limb_length : Left lower limb total length
        """
        return self.right_thigh_length + self.right_leg_length

    @property
    def left_upper_limb_length(self):
        """
        Calculate total left upper limb length (arm + forearm).

        Returns the sum of arm length (shoulder to elbow) and forearm length
        (elbow to wrist). This represents the functional length of the
        entire upper limb.

        Returns
        -------
        Signal1D
            Distance in meters from shoulder to wrist joint center.

        See Also
        --------
        left_arm_length : Length of arm segment only
        left_forearm_length : Length of forearm segment only
        right_upper_limb_length : Right upper limb total length
        """
        return self.left_arm_length + self.left_forearm_length

    @property
    def right_upper_limb_length(self):
        """
        Calculate total right upper limb length (arm + forearm).

        Returns the sum of arm length (shoulder to elbow) and forearm length
        (elbow to wrist). This represents the functional length of the
        entire upper limb.

        Returns
        -------
        Signal1D
            Distance in meters from shoulder to wrist joint center.

        See Also
        --------
        right_arm_length : Length of arm segment only
        right_forearm_length : Length of forearm segment only
        left_upper_limb_length : Left upper limb total length
        """
        return self.right_arm_length + self.right_forearm_length

    @property
    def trunk_height(self):
        """
        Calculate trunk height as distance from pelvis center to neck base.

        Returns
        -------
        Signal1D
            Distance in meters from pelvis center to neck base.
        """
        pelvis = self.pelvis_center
        neck = self.neck_base
        index = np.unique(np.concatenate([pelvis.index, neck.index])).tolist()
        data = np.asarray(neck - pelvis)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=pelvis.unit)

    @property
    def shoulder_width(self):
        """
        Calculate shoulder width as distance between left and right shoulders.

        Returns
        -------
        Signal1D
            Distance in meters between shoulder joint centers.
        """
        l_shoulder = self.left_shoulder
        r_shoulder = self.right_shoulder
        index = np.unique(np.concatenate([l_shoulder.index, r_shoulder.index])).tolist()
        data = np.asarray(l_shoulder - r_shoulder)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=l_shoulder.unit)

    @property
    def hip_width(self):
        """
        Calculate hip width as distance between left and right trochanters.

        Returns
        -------
        Signal1D
            Distance in meters between greater trochanter markers.
        """
        l_troch = self._get_point("left_trochanter")
        r_troch = self._get_point("right_trochanter")
        index = np.unique(np.concatenate([l_troch.index, r_troch.index])).tolist()
        data = np.asarray(l_troch - r_troch)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=l_troch.unit)

    @property
    def trunk_length(self):
        """
        Calculate trunk length as vertical distance from pelvis center to neck base.

        The trunk length is computed as the Euclidean distance between the pelvis center
        (average of ASIS markers) and the neck base (midpoint between C7 and SC markers).

        Returns
        -------
        Signal1D
            Distance in meters from pelvis to neck base.

        Notes
        -----
        Requires pelvis markers (left_asis, right_asis) and neck markers (c7, sc).
        """
        # Get pelvis center from ASIS markers
        pelvis_center = self.pelvis_center

        # Get neck base from neck markers
        neck_base = self.neck_base

        # Calculate distance
        index = np.unique(
            np.concatenate([pelvis_center.index, neck_base.index])
        ).tolist()
        data = np.asarray(neck_base - pelvis_center)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=pelvis_center.unit)

    @property
    def left_foot_width(self):
        """
        Calculate left foot width as distance between first and fifth metatarsal heads.

        Returns
        -------
        Signal1D
            Distance in meters between first and fifth metatarsal heads.
        """
        first = self._get_point("left_first_metatarsal_head")
        fifth = self._get_point("left_fifth_metatarsal_head")
        index = np.unique(np.concatenate([first.index, fifth.index])).tolist()
        data = np.asarray(fifth - first)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=first.unit)

    @property
    def right_foot_width(self):
        """
        Calculate right foot width as distance between first and fifth metatarsal heads.

        Returns
        -------
        Signal1D
            Distance in meters between first and fifth metatarsal heads.
        """
        first = self._get_point("right_first_metatarsal_head")
        fifth = self._get_point("right_fifth_metatarsal_head")
        index = np.unique(np.concatenate([first.index, fifth.index])).tolist()
        data = np.asarray(fifth - first)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=first.unit)

    @property
    def left_ankle_width(self):
        """
        Calculate left ankle width as distance between medial and lateral ankle markers.

        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral ankle malleoli.
        """
        medial = self._get_point("left_ankle_medial")
        lateral = self._get_point("left_ankle_lateral")
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def right_ankle_width(self):
        """
        Calculate right ankle width as distance between medial and lateral ankle markers.

        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral ankle malleoli.
        """
        medial = self._get_point("right_ankle_medial")
        lateral = self._get_point("right_ankle_lateral")
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def left_knee_width(self):
        """
        Calculate left knee width as distance between medial and lateral knee markers.

        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral femoral epicondyles.
        """
        medial = self._get_point("left_knee_medial")
        lateral = self._get_point("left_knee_lateral")
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def right_knee_width(self):
        """
        Calculate right knee width as distance between medial and lateral knee markers.

        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral femoral epicondyles.
        """
        medial = self._get_point("right_knee_medial")
        lateral = self._get_point("right_knee_lateral")
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def left_elbow_width(self):
        """
        Calculate left elbow width as distance between medial and lateral elbow markers.

        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral elbow epicondyles.
        """
        medial = self._get_point("left_elbow_medial")
        lateral = self._get_point("left_elbow_lateral")
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def right_elbow_width(self):
        """
        Calculate right elbow width as distance between medial and lateral elbow markers.

        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral elbow epicondyles.
        """
        medial = self._get_point("right_elbow_medial")
        lateral = self._get_point("right_elbow_lateral")
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def left_hip(self):

        # get the De Leva (1996) approximations
        p0 = self.pelvis_center
        width = self.pelvis_width.to_numpy()
        height = self.pelvis_height.to_numpy()
        offset_ml = -0.19 * width
        offset_vt = -0.14 * height
        offset_ap = 0.36 * width
        offsets = Signal3D(
            data=np.hstack([offset_ml, offset_vt, offset_ap]),
            index=p0.index,
            columns=p0.columns,
            unit=p0.unit,
        )
        return self._get_translated_point_along_plane(
            self._get_point("left_trochanter"),
            offsets,
            self._pelvis_plane,
        )

    @property
    def right_hip(self):

        # get the De Leva (1996) approximations
        p0 = self.pelvis_center
        width = self.pelvis_width.to_numpy()
        height = self.pelvis_height.to_numpy()
        offset_ml = +0.19 * width
        offset_vt = -0.14 * height
        offset_ap = +0.36 * width
        offsets = Signal3D(
            data=np.hstack([offset_ml, offset_vt, offset_ap]),
            index=p0.index,
            columns=p0.columns,
            unit=p0.unit,
        )
        return self._get_translated_point_along_plane(
            self._get_point("right_trochanter"),
            offsets,
            self._pelvis_plane,
        )

    @property
    def hip_center(self):
        """
        Calculate hip center as midpoint between left and right hip joint centers.

        The hip joint centers are computed using De Leva (1996) regression from
        pelvis markers and trochanters.

        Returns
        -------
        Point3D
            Hip center position (midpoint between left_hip and right_hip)

        See Also
        --------
        left_hip : Left hip joint center (De Leva 1996)
        right_hip : Right hip joint center (De Leva 1996)
        pelvis_center : Pelvis center (centroid of ASIS/PSIS markers)
        """
        left = self.left_hip
        right = self.right_hip

        # Calculate midpoint
        data = (left.to_numpy() + right.to_numpy()) / 2

        # Use index from left_hip (should match right_hip)
        index = left.index

        return Point3D(
            data=data,
            index=index,
            columns=left.columns,
        )

    @property
    def left_shoulder(self):
        # Try primary method: midpoint of anterior and posterior markers
        try:
            ant: Point3D = self._get_point("left_shoulder_anterior")
            pos: Point3D = self._get_point("left_shoulder_posterior")
            return Point3D(
                (ant + pos).to_numpy() / 2,
                index=np.unique(np.concatenate([ant.index, pos.index])),
                columns=ant.columns,
            )
        except Exception:
            # Fallback: De Leva regression from acromion
            try:
                acr: Point3D = self._get_point("left_acromion")
                return self._calculate_shoulder_from_acromion(
                    acr,
                    side="left",
                )
            except Exception:
                raise ValueError("left_shoulder cannot be obtained")

    @property
    def right_shoulder(self):
        # Try primary method: midpoint of anterior and posterior markers
        try:
            ant: Point3D = self._get_point("right_shoulder_anterior")
            pos: Point3D = self._get_point("right_shoulder_posterior")
            return Point3D(
                (ant + pos).to_numpy() / 2,
                index=np.unique(np.concatenate([ant.index, pos.index])),
                columns=ant.columns,
            )
        except Exception:
            # Fallback: De Leva regression from acromion
            try:
                acr: Point3D = self._get_point("right_acromion")
                return self._calculate_shoulder_from_acromion(acr, side="right")
            except Exception:
                raise ValueError("right_shoulder cannot be obtained")

    @property
    def left_elbow(self):
        lat: Point3D = self._get_point("left_elbow_lateral")
        try:
            med: Point3D = self._get_point("left_elbow_medial")
            return Point3D(
                (lat + med).to_numpy() / 2,
                index=np.unique(np.concatenate([lat.index, med.index])).tolist(),
                columns=lat.columns,
            )
        except Exception as e:
            return lat

    @property
    def right_elbow(self):
        lat: Point3D = self._get_point("right_elbow_lateral")
        try:
            med: Point3D = self._get_point("right_elbow_medial")
            return Point3D(
                (lat + med).to_numpy() / 2,
                index=np.unique(np.concatenate([lat.index, med.index])),
                columns=lat.columns,
            )
        except Exception as e:
            return lat

    @property
    def left_wrist(self):
        lat: Point3D = self._get_point("left_wrist_lateral")
        try:
            med: Point3D = self._get_point("left_wrist_medial")
            return Point3D(
                (lat + med).to_numpy() / 2,
                index=np.unique(np.concatenate([lat.index, med.index])),
                columns=lat.columns,
            )
        except Exception as e:
            return lat

    @property
    def right_wrist(self):
        lat: Point3D = self._get_point("right_wrist_lateral")
        try:
            med: Point3D = self._get_point("right_wrist_medial")
            return Point3D(
                (lat + med).to_numpy() / 2,
                index=np.unique(np.concatenate([lat.index, med.index])),
                columns=lat.columns,
            )
        except Exception as e:
            return lat

    @property
    def left_elbow_referenceframe(self):
        """
        Left elbow reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Left elbow center (use `self.left_elbow` property)

        Construction
        ------------
        1. lateral_axis: LEFT (left_elbow_lateral → left_elbow_medial)
        2. vertical_axis: UP (left_elbow → left_shoulder)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left elbow center and orthonormal axes.

        See Also
        --------
        left_elbow : Elbow center (origin of this frame)
        right_elbow_referenceframe : Right elbow reference frame
        left_elbow_flexionextension : Elbow flexion angle using this frame
        """
        elbow = self.left_elbow
        shoulder = self.left_shoulder

        elbow_lat = self._get_point("left_elbow_lateral")
        elbow_med = self._get_point("left_elbow_medial")
        # Construct lateral_axis: LEFT (lateral to medial)
        axis_x = (elbow_lat - elbow_med).to_numpy()

        # Construct vertical_axis: UP (elbow to shoulder)
        axis_y = (shoulder - elbow).to_numpy()

        return ReferenceFrame(origin=elbow, lateral_axis=axis_x, vertical_axis=axis_y)

    @property
    def right_elbow_referenceframe(self):
        """
        Right elbow reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Right elbow center (use `self.right_elbow` property)

        Construction
        ------------
        1. lateral_axis: RIGHT (right_elbow_lateral → right_elbow_medial)
        2. vertical_axis: UP (right_elbow → right_shoulder)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right elbow center and orthonormal axes.

        See Also
        --------
        right_elbow : Elbow center (origin of this frame)
        left_elbow_referenceframe : Left elbow reference frame
        right_elbow_flexionextension : Elbow flexion angle using this frame
        """
        elbow = self.right_elbow
        shoulder = self.right_shoulder

        elbow_lat = self._get_point("right_elbow_lateral")
        elbow_med = self._get_point("right_elbow_medial")
        # Construct lateral_axis: RIGHT (lateral to medial)
        axis_x = (elbow_lat - elbow_med).to_numpy()

        # Construct vertical_axis: UP (elbow to shoulder)
        axis_y = (shoulder - elbow).to_numpy()

        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)

        return ReferenceFrame(
            origin=elbow,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def left_wrist_referenceframe(self):
        """
        Left wrist reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Left wrist center (use `self.left_wrist` property)

        Construction
        ------------
        1. lateral_axis: LEFT (left_wrist_lateral → left_wrist_medial)
        2. vertical_axis: UP (left_wrist → left_elbow)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left wrist center and orthonormal axes.

        See Also
        --------
        left_wrist : Wrist center (origin of this frame)
        right_wrist_referenceframe : Right wrist reference frame
        """
        wrist = self.left_wrist
        elbow = self.left_elbow

        try:
            wrist_lat = self._get_point("left_wrist_lateral")
            wrist_med = self._get_point("left_wrist_medial")
            # Construct lateral_axis: LEFT (lateral to medial)
            axis_x = (wrist_lat - wrist_med).to_numpy()
        except Exception:
            # Default LEFT: use lateral_axis property to determine which column
            lateral_idx = np.where(wrist.columns == wrist.lateral_axis)[0][0]
            axis_x = np.zeros((wrist.shape[0], 3))
            axis_x[:, lateral_idx] = -1  # LEFT (negative for left side)

        # Construct vertical_axis: UP (wrist to elbow)
        axis_y = (elbow - wrist).to_numpy()

        return ReferenceFrame(origin=wrist, lateral_axis=axis_x, vertical_axis=axis_y)

    @property
    def right_wrist_referenceframe(self):
        """
        Right wrist reference frame for angular measurements.

        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:

        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)

        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.

        Origin
        ------
        Right wrist center (use `self.right_wrist` property)

        Construction
        ------------
        1. lateral_axis: RIGHT (right_wrist_lateral → right_wrist_medial)
        2. vertical_axis: UP (right_wrist → right_elbow)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization

        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right wrist center and orthonormal axes.

        See Also
        --------
        right_wrist : Wrist center (origin of this frame)
        left_wrist_referenceframe : Left wrist reference frame
        """
        wrist = self.right_wrist
        elbow = self.right_elbow

        try:
            wrist_lat = self._get_point("right_wrist_lateral")
            wrist_med = self._get_point("right_wrist_medial")
            # Construct lateral_axis: RIGHT (lateral to medial)
            axis_x = (wrist_lat - wrist_med).to_numpy()
        except Exception:
            # Default RIGHT: use lateral_axis property to determine which column
            lateral_idx = np.where(wrist.columns == wrist.lateral_axis)[0][0]
            axis_x = np.zeros((wrist.shape[0], 3))
            axis_x[:, lateral_idx] = 1  # RIGHT (positive for right side)

        # Construct vertical_axis: UP (wrist to elbow)
        axis_y = (elbow - wrist).to_numpy()

        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)

        return ReferenceFrame(
            origin=wrist,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def _left_foot_plane(self):
        """
        Calculate left foot plane from toe, heel, and metatarsal markers.

        The plane is defined by least-squares fitting through four points:
        left_toe, left_heel, left_first_metatarsal_head, and left_fifth_metatarsal_head.

        Returns
        -------
        Timeseries
            Plane coefficients [a, b, c, d] defining the equation ax + by + cz + d = 0.
            Unit: dimensionless (a.u.).

        Notes
        -----
        The plane normal vector (a, b, c) is normalized to unit length.
        Used by left_foot_height and ankle angle calculations.
        """
        toe = self._get_point("left_toe")
        first_meta = self._get_point("left_first_metatarsal_head")
        fifth_meta = self._get_point("left_fifth_metatarsal_head")
        heel = self._get_point("left_heel")
        points = [toe, heel]
        if first_meta is not None:
            points.append(first_meta)
        if fifth_meta is not None:
            points.append(fifth_meta)
        if len(points) < 3:
            raise ValueError(
                "there are not enough Point3D to define the left foot plane."
            )
        return Timeseries(
            data=self._get_least_squares_plane_coefs(*points),
            index=toe.index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )

    @property
    def _right_foot_plane(self):
        """
        Calculate right foot plane from toe, heel, and metatarsal markers.

        The plane is defined by least-squares fitting through four points:
        right_toe, right_heel, right_first_metatarsal_head, and right_fifth_metatarsal_head.

        Returns
        -------
        Timeseries
            Plane coefficients [a, b, c, d] defining the equation ax + by + cz + d = 0.
            Unit: dimensionless (a.u.).

        Notes
        -----
        The plane normal vector (a, b, c) is normalized to unit length.
        Used by right_foot_height and ankle angle calculations.
        """
        toe = self._get_point("right_toe")
        first_meta = self._get_point("right_first_metatarsal_head")
        fifth_meta = self._get_point("right_fifth_metatarsal_head")
        heel = self._get_point("right_heel")
        points = [toe, heel]
        if first_meta is not None:
            points.append(first_meta)
        if fifth_meta is not None:
            points.append(fifth_meta)
        if len(points) < 3:
            raise ValueError(
                "there are not enough Point3D to define the right foot plane."
            )
        return Timeseries(
            data=self._get_least_squares_plane_coefs(*points),
            index=toe.index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )

    @property
    def left_ankle_flexionextension(self):
        """
        Calculate left ankle dorsiflexion/plantarflexion angle in sagittal plane.

        The angle represents the orientation of the foot relative to the shank,
        indicating dorsiflexion (toe up) or plantarflexion (toe down).

        Interpretation
        --------------
        - **Positive (+)**: Dorsiflexion (flessione dorsale)
          The foot is angled upward relative to the shin.
          Common in landing, deceleration, squatting.
        - **Negative (-)**: Plantarflexion (flessione plantare)
          The foot is angled downward relative to the shin.
          Common in toe-off, jumping, pointing.
        - **0°**: Neutral position (foot perpendicular to shin at 90°)

        Calculation Method
        ------------------
        Uses left ankle reference frame with:
        - Origin: Left ankle center (midpoint of lateral and medial ankle markers)
        - lateral_axis: LEFT (ankle_lateral → ankle_medial)
        - vertical_axis: UP (ankle → knee)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The foot plane (defined by heel, toe, and metatarsal markers) is projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is measured from the projected foot orientation to the vertical axis.

        The calculation uses self.anteroposterior_axis and self.vertical_axis to
        identify the appropriate rotation matrix components for the sagittal plane
        projection.

        Returns
        -------
        Signal1D
            Ankle flexion/extension angle in degrees.
            Positive = dorsiflexion (foot up)
            Negative = plantarflexion (foot down)

        See Also
        --------
        right_ankle_flexionextension : Right ankle dorsiflexion/plantarflexion
        left_ankle_inversioneversion : Left ankle frontal plane motion
        left_knee_flexionextension : Left knee flexion angle
        """
        # get points and reference frame
        ankle = self.left_ankle
        rmat = self.left_ankle_referenceframe.rotation_matrix
        proj = self._get_projection_point_on_plane(
            ankle,
            self._left_foot_plane,
        )
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            proj,
            ankle,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # adjust the signs
        return angle + 90

    @property
    def right_ankle_flexionextension(self):
        """
        Calculate right ankle dorsiflexion/plantarflexion angle in sagittal plane.

        The angle represents the orientation of the foot relative to the shank,
        indicating dorsiflexion (toe up) or plantarflexion (toe down).

        Interpretation
        --------------
        - **Positive (+)**: Dorsiflexion (flessione dorsale)
          The foot is angled upward relative to the shin.
          Common in landing, deceleration, squatting.
        - **Negative (-)**: Plantarflexion (flessione plantare)
          The foot is angled downward relative to the shin.
          Common in toe-off, jumping, pointing.
        - **0°**: Neutral position (foot perpendicular to shin at 90°)

        Calculation Method
        ------------------
        Uses right ankle reference frame with:
        - Origin: Right ankle center (midpoint of lateral and medial ankle markers)
        - lateral_axis: RIGHT (ankle_lateral → ankle_medial)
        - vertical_axis: UP (ankle → knee)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)

        The foot plane (defined by heel, toe, and metatarsal markers) is projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is measured from the projected foot orientation to the vertical axis.

        The calculation uses self.anteroposterior_axis and self.vertical_axis to
        identify the appropriate rotation matrix components for the sagittal plane
        projection.

        Returns
        -------
        Signal1D
            Ankle flexion/extension angle in degrees.
            Positive = dorsiflexion (foot up)
            Negative = plantarflexion (foot down)

        See Also
        --------
        left_ankle_flexionextension : Left ankle dorsiflexion/plantarflexion
        right_ankle_inversioneversion : Right ankle frontal plane motion
        right_knee_flexionextension : Right knee flexion angle
        """
        ankle = self.right_ankle
        rmat = self.right_ankle_referenceframe.rotation_matrix
        proj = self._get_projection_point_on_plane(
            ankle,
            self._right_foot_plane,
        )
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            proj,
            ankle,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return angle + 90

    @property
    def left_ankle_inversioneversion(self):
        """
        Calculate left ankle inversion/eversion angle in frontal plane.

        The angle represents the tilting of the foot relative to the shank
        in the frontal (coronal) plane, indicating inversion (sole inward)
        or eversion (sole outward).

        Interpretation
        --------------
        - **Positive (+)**: Eversion (eversione)
          The sole of the foot is tilted outward (away from midline).
          Common in overpronation, pes planus (flat feet).
        - **Negative (-)**: Inversion (inversione)
          The sole of the foot is tilted inward (toward midline).
          Common in supination, ankle sprains (lateral).
        - **0°**: Neutral position (foot aligned with shin in frontal plane)

        Calculation Method
        ------------------
        Uses left ankle reference frame with:
        - Origin: Left ankle center (midpoint of lateral and medial ankle markers)
        - lateral_axis: LEFT (ankle_lateral → ankle_medial)
        - vertical_axis: UP (ankle → knee)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The foot plane (defined by heel, toe, and metatarsal markers) is projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is measured from the projected foot orientation in the frontal plane.

        The calculation uses self.lateral_axis and self.vertical_axis to identify
        the appropriate rotation matrix components for the frontal plane projection.

        Returns
        -------
        Signal1D
            Ankle inversion/eversion angle in degrees.
            Positive = eversion (sole out)
            Negative = inversion (sole in)

        See Also
        --------
        right_ankle_inversioneversion : Right ankle frontal plane motion
        left_ankle_flexionextension : Left ankle sagittal plane motion
        left_knee_varusvalgus : Left knee frontal plane alignment
        """
        ankle = self.left_ankle
        rmat = self.left_ankle_referenceframe.rotation_matrix
        proj = self._get_projection_point_on_plane(
            ankle,
            self._left_foot_plane,
        )
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            proj,
            ankle,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return angle + 90

    @property
    def right_ankle_inversioneversion(self):
        """
        Calculate right ankle inversion/eversion angle in frontal plane.

        The angle represents the tilting of the foot relative to the shank
        in the frontal (coronal) plane, indicating inversion (sole inward)
        or eversion (sole outward).

        Interpretation
        --------------
        - **Positive (+)**: Eversion (eversione)
          The sole of the foot is tilted outward (away from midline).
          Common in overpronation, pes planus (flat feet).
        - **Negative (-)**: Inversion (inversione)
          The sole of the foot is tilted inward (toward midline).
          Common in supination, ankle sprains (lateral).
        - **0°**: Neutral position (foot aligned with shin in frontal plane)

        Calculation Method
        ------------------
        Uses right ankle reference frame with:
        - Origin: Right ankle center (midpoint of lateral and medial ankle markers)
        - lateral_axis: RIGHT (ankle_lateral → ankle_medial)
        - vertical_axis: UP (ankle → knee)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)

        The foot plane (defined by heel, toe, and metatarsal markers) is projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is measured from the projected foot orientation in the frontal plane.

        The calculation uses self.lateral_axis and self.vertical_axis to identify
        the appropriate rotation matrix components for the frontal plane projection.

        Returns
        -------
        Signal1D
            Ankle inversion/eversion angle in degrees.
            Positive = eversion (sole out)
            Negative = inversion (sole in)

        See Also
        --------
        left_ankle_inversioneversion : Left ankle frontal plane motion
        right_ankle_flexionextension : Right ankle sagittal plane motion
        right_knee_varusvalgus : Right knee frontal plane alignment
        """
        ankle = self.right_ankle
        rmat = self.right_ankle_referenceframe.rotation_matrix
        proj = self._get_projection_point_on_plane(
            ankle,
            self._right_foot_plane,
        )
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            proj,
            ankle,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return -1 * (90 + angle)

    @property
    def left_knee_flexionextension(self):
        """
        Calculate left knee flexion/extension angle in sagittal plane.

        The angle represents the bending or straightening of the knee joint,
        indicating flexion (bent knee) or extension (straight knee).

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The knee is bent, bringing the heel toward the buttock.
          Common in squatting, running, jumping preparation.
        - **Negative (-)**: Extension (estensione)
          The knee is straightened beyond neutral.
          Rare; indicates hyperextension.
        - **0°**: Neutral position (fully straight knee)

        Calculation Method
        ------------------
        Uses left knee reference frame with:
        - Origin: Left knee center (midpoint of lateral and medial knee markers)
        - lateral_axis: LEFT (knee_lateral → knee_medial)
        - vertical_axis: UP (knee → hip)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The ankle position is transformed to the knee reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is calculated using the anteroposterior and vertical components of
        the transformed ankle vector.

        The calculation uses rotation matrix indices that correspond to semantic axes:
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component

        Zero degrees corresponds to full knee extension (ankle directly below knee).

        Returns
        -------
        Signal1D
            Knee flexion/extension angle in degrees.
            Positive = flexion (bent knee)
            Range: [0°, 180°]

        See Also
        --------
        right_knee_flexionextension : Right knee flexion angle
        left_knee_varusvalgus : Left knee frontal plane alignment
        left_hip_flexionextension : Left hip flexion angle
        """
        knee = self.left_knee
        ankle = self.left_ankle
        rmat = self.left_knee_referenceframe.rotation_matrix

        # Ankle vector from knee origin
        ankle_vec = (ankle - knee).to_numpy()

        # Transform to knee reference frame
        ankle_rf = np.einsum("nij,nj->ni", rmat, ankle_vec)

        # Extract components in reference frame coordinates
        # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
        # Index [1] = vertical_axis component (by ReferenceFrame construction)
        # Knee flexion is measured from the extended position (ankle below knee, vertical negative)
        # arctan2 of anteroposterior and vertical components gives flexion/extension angle
        # At extended (ankle straight down): anteroposterior≈0, vertical<0 → angle ≈ 0°
        # At flexed (ankle forward): anteroposterior>0, vertical<0 → angle < 0°, so negate to get positive flexion
        flexion = -np.arctan2(ankle_rf[:, 2], -ankle_rf[:, 1]) * 180 / np.pi

        return Signal1D(data=flexion, index=knee.index, unit="°")

    @property
    def right_knee_flexionextension(self):
        """
        Calculate right knee flexion/extension angle in sagittal plane.

        The angle represents the bending or straightening of the knee joint,
        indicating flexion (bent knee) or extension (straight knee).

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The knee is bent, bringing the heel toward the buttock.
          Common in squatting, running, jumping preparation.
        - **Negative (-)**: Extension (estensione)
          The knee is straightened beyond neutral.
          Rare; indicates hyperextension.
        - **0°**: Neutral position (fully straight knee)

        Calculation Method
        ------------------
        Uses right knee reference frame with:
        - Origin: Right knee center (midpoint of lateral and medial knee markers)
        - lateral_axis: RIGHT (knee_lateral → knee_medial)
        - vertical_axis: UP (knee → hip)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)

        The ankle position is transformed to the knee reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is calculated using the anteroposterior and vertical components of
        the transformed ankle vector.

        The calculation uses rotation matrix indices that correspond to semantic axes:
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component

        Zero degrees corresponds to full knee extension (ankle directly below knee).

        Returns
        -------
        Signal1D
            Knee flexion/extension angle in degrees.
            Positive = flexion (bent knee)
            Negative = extension (hyperextension)

        See Also
        --------
        left_knee_flexionextension : Left knee flexion angle
        right_knee_varusvalgus : Right knee frontal plane alignment
        right_hip_flexionextension : Right hip flexion angle
        """
        knee = self.right_knee
        ankle = self.right_ankle
        rmat = self.right_knee_referenceframe.rotation_matrix

        # Ankle vector from knee origin
        ankle_vec = (ankle - knee).to_numpy()

        # Transform to knee reference frame
        ankle_rf = np.einsum("nij,nj->ni", rmat, ankle_vec)

        # Extract components in reference frame coordinates
        # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
        # Index [1] = vertical_axis component (by ReferenceFrame construction)
        # Knee flexion is measured from the extended position (ankle below knee, vertical negative)
        # For left-handed frame: anteroposterior_axis is negated to point forward
        # arctan2 of anteroposterior and vertical components gives flexion/extension angle
        # At extended (ankle straight down): anteroposterior≈0, vertical<0 → angle ≈ 0°
        # At flexed (ankle forward): anteroposterior<0 (negated in frame), vertical<0 → angle > 0°, but we need to negate result
        flexion = -np.arctan2(ankle_rf[:, 2], -ankle_rf[:, 1]) * 180 / np.pi

        return Signal1D(data=flexion, index=knee.index, unit="°")

    @property
    def left_knee_varusvalgus(self):
        """
        Calculate left knee varus/valgus angle in frontal plane.

        The angle represents the frontal plane alignment of the knee joint.

        Interpretation
        --------------
        - **Positive (+)**: Varus deformity (ginocchio varo, "a parentesi", bow-legged)
          The knee deviates laterally; the leg angle opens medially.
        - **Negative (-)**: Valgus deformity (ginocchio valgo, "a X", knock-knee)
          The knee deviates medially; the leg angle opens laterally.
        - **0°**: Neutral alignment (anca-ginocchio-caviglia collineari nel piano frontale)

        Calculation Method
        ------------------
        Uses left knee reference frame with:
        - Origin: Left knee center (midpoint of lateral and medial knee markers)
        - lateral_axis: LEFT (knee_lateral → knee_medial)
        - vertical_axis: UP (knee → hip)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The ankle position is transformed to the knee reference frame and projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is calculated using the lateral and vertical components of the
        transformed ankle vector.

        The calculation uses rotation matrix indices that correspond to semantic axes:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component

        Positive lateral deviation (ankle lateral to knee) indicates varus alignment.

        Returns
        -------
        Signal1D
            Knee varus/valgus angle in degrees.
            Positive = varus (ginocchio varo)
            Negative = valgus (ginocchio valgo)
        """
        knee = self.left_knee
        ankle = self.left_ankle
        rmat = self.left_knee_referenceframe.rotation_matrix

        # Ankle vector from knee origin
        ankle_vec = (ankle - knee).to_numpy()

        # Transform to knee reference frame
        ankle_rf = np.einsum("nij,nj->ni", rmat, ankle_vec)

        # Extract components in reference frame coordinates
        # Index [0] = lateral_axis component (by ReferenceFrame construction)
        # Index [1] = vertical_axis component (by ReferenceFrame construction)
        # Varus/valgus is measured in the frontal plane (lateral-vertical)
        # arctan2 of lateral and vertical components gives varus/valgus angle
        # Positive lateral with vertical negative (down) = varus (knee out, bow-legged)
        # Negative lateral (medial) with vertical negative (down) = valgus (knee in, knock-knee)
        angle = np.arctan2(ankle_rf[:, 0], -ankle_rf[:, 1]) * 180 / np.pi

        return Signal1D(data=angle, index=knee.index, unit="°")

    @property
    def right_knee_varusvalgus(self):
        """
        Calculate right knee varus/valgus angle in frontal plane.

        The angle represents the frontal plane alignment of the knee joint.

        Interpretation
        --------------
        - **Positive (+)**: Varus deformity (ginocchio varo, "a parentesi", bow-legged)
          The knee deviates laterally; the leg angle opens medially.
        - **Negative (-)**: Valgus deformity (ginocchio valgo, "a X", knock-knee)
          The knee deviates medially; the leg angle opens laterally.
        - **0°**: Neutral alignment (anca-ginocchio-caviglia collineari nel piano frontale)

        Calculation Method
        ------------------
        Uses right knee reference frame with:
        - Origin: Right knee center (midpoint of lateral and medial knee markers)
        - lateral_axis: RIGHT (knee_lateral → knee_medial)
        - vertical_axis: UP (knee → hip)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)

        The ankle position is transformed to the knee reference frame and projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is calculated using the lateral and vertical components of the
        transformed ankle vector.

        The calculation uses rotation matrix indices that correspond to semantic axes:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component

        Positive lateral deviation (ankle lateral to knee) indicates varus alignment.

        Returns
        -------
        Signal1D
            Knee varus/valgus angle in degrees.
            Positive = varus (ginocchio varo)
            Negative = valgus (ginocchio valgo)
        """
        knee = self.right_knee
        ankle = self.right_ankle
        rmat = self.right_knee_referenceframe.rotation_matrix

        # Ankle vector from knee origin
        ankle_vec = (ankle - knee).to_numpy()

        # Transform to knee reference frame
        ankle_rf = np.einsum("nij,nj->ni", rmat, ankle_vec)

        # Extract components in reference frame coordinates
        # Index [0] = lateral_axis component (by ReferenceFrame construction)
        # Index [1] = vertical_axis component (by ReferenceFrame construction)
        # Varus/valgus is measured in the frontal plane (lateral-vertical)
        # For left-handed frame: lateral_axis points RIGHT, already correct
        # arctan2 of lateral and vertical components gives varus/valgus angle
        # Positive lateral with vertical negative (down) = varus (knee out, bow-legged)
        # Negative lateral (medial) with vertical negative (down) = valgus (knee in, knock-knee)
        angle = np.arctan2(ankle_rf[:, 0], -ankle_rf[:, 1]) * 180 / np.pi

        return Signal1D(data=angle, index=knee.index, unit="°")

    @property
    def left_hip_flexionextension(self):
        """
        Calculate left hip flexion/extension angle in sagittal plane.

        The angle represents the forward (flexion) or backward (extension)
        movement of the thigh relative to the pelvis in the sagittal plane.

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The thigh is brought forward (anteriorly) toward the torso.
          Common in running, climbing, sitting.
        - **Negative (-)**: Extension (estensione)
          The thigh is moved backward (posteriorly) behind the body.
          Common in push-off phase of gait, sprinting.
        - **0°**: Neutral position (standing upright, thigh vertical)

        Calculation Method
        ------------------
        Uses left hip reference frame (based on pelvis frame with origin at left hip):
        - Origin: Left hip joint center (De Leva 1996 regression)
        - lateral_axis: LEFT (from pelvis right midpoint → left midpoint)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The knee position is transformed to the hip reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is calculated using the anteroposterior and vertical components
        of the transformed knee vector.

        Zero degrees corresponds to neutral position (thigh vertical, knee directly
        below hip). The result is normalized to [0°, 360°] range.

        Returns
        -------
        Signal1D
            Hip flexion/extension angle in degrees.
            Positive = flexion (thigh forward)
            Negative = extension (thigh backward)

        See Also
        --------
        right_hip_flexionextension : Right hip flexion angle
        left_hip_abductionadduction : Left hip frontal plane motion
        left_knee_flexionextension : Left knee flexion angle
        """
        # get points and reference frame
        hip = self.left_hip
        rmat = self.left_hip_referenceframe.rotation_matrix
        knee = self.left_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # Transform to anatomical convention: 0° = neutral (thigh vertical)
        # The reference frame vertical_axis points downward in the hip frame
        # (determined by the lateral_axis × vertical_axis orthonormalization)
        # For vertical thigh (knee below hip), arctan2 gives ~+90°
        # Subtract 90° to make neutral position = 0°, then normalize to [0°, 360°]
        result = (angle.to_numpy() - 90) % 360
        return Signal1D(data=result, index=angle.index, unit="°")

    @property
    def right_hip_flexionextension(self):
        """
        Calculate right hip flexion/extension angle in sagittal plane.

        The angle represents the forward (flexion) or backward (extension)
        movement of the thigh relative to the pelvis in the sagittal plane.

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The thigh is brought forward (anteriorly) toward the torso.
          Common in running, climbing, sitting.
        - **Negative (-)**: Extension (estensione)
          The thigh is moved backward (posteriorly) behind the body.
          Common in push-off phase of gait, sprinting.
        - **0°**: Neutral position (standing upright, thigh vertical)

        Calculation Method
        ------------------
        Uses right hip reference frame (mirrored pelvis frame with origin at right hip):
        - Origin: Right hip joint center (De Leva 1996 regression)
        - lateral_axis: RIGHT (negated pelvis lateral_axis)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (same as pelvis, creating left-handed frame)

        The knee position is transformed to the hip reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is calculated using the anteroposterior and vertical components
        of the transformed knee vector.

        Zero degrees corresponds to neutral position (thigh vertical, knee directly
        below hip). The result is normalized to [0°, 360°] range.

        Returns
        -------
        Signal1D
            Hip flexion/extension angle in degrees.
            Positive = flexion (thigh forward)
            Negative = extension (thigh backward)

        See Also
        --------
        left_hip_flexionextension : Left hip flexion angle
        right_hip_abductionadduction : Right hip frontal plane motion
        right_knee_flexionextension : Right knee flexion angle
        """
        # get points and reference frame
        hip = self.right_hip
        rmat = self.right_hip_referenceframe.rotation_matrix
        knee = self.right_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # Transform to anatomical convention: 0° = neutral (thigh vertical)
        # The reference frame vertical_axis points downward in the hip frame
        # (determined by the lateral_axis × vertical_axis orthonormalization)
        # For vertical thigh (knee below hip), arctan2 gives ~+90°
        # Subtract 90° to make neutral position = 0°, then normalize to [0°, 360°]
        result = (angle.to_numpy() - 90) % 360
        return Signal1D(data=result, index=angle.index, unit="°")

    @property
    def left_hip_abductionadduction(self):
        """
        Calculate left hip abduction/adduction angle in frontal plane.

        The angle represents the lateral (outward) or medial (inward)
        movement of the thigh relative to the pelvis in the frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Abduction (abduzione)
          The thigh is moved laterally away from the body midline.
          Common in side-stepping, lateral movements.
        - **Negative (-)**: Adduction (adduzione)
          The thigh is moved medially toward or across the body midline.
          Common in crossover movements, cutting.
        - **0°**: Neutral position (thigh vertical, aligned with hip)

        Calculation Method
        ------------------
        Uses left hip reference frame (based on pelvis frame with origin at left hip):
        - Origin: Left hip joint center (De Leva 1996 regression)
        - lateral_axis: LEFT (from pelvis right midpoint → left midpoint)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The knee position is transformed to the hip reference frame and projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is calculated using the lateral and vertical components of the
        transformed knee vector.

        Zero degrees corresponds to neutral position (thigh vertical, knee directly
        below hip). The result is normalized to [0°, 360°] range.

        Returns
        -------
        Signal1D
            Hip abduction/adduction angle in degrees.
            Positive = abduction (thigh outward)
            Negative = adduction (thigh inward)

        See Also
        --------
        right_hip_abductionadduction : Right hip frontal plane motion
        left_hip_flexionextension : Left hip sagittal plane motion
        left_knee_varusvalgus : Left knee frontal plane alignment
        """
        # get points and reference frame
        hip = self.left_hip
        rmat = self.left_hip_referenceframe.rotation_matrix
        knee = self.left_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # Transform to anatomical convention: 0° = neutral (thigh vertical)
        # The reference frame vertical_axis points downward in the hip frame
        # (determined by the lateral_axis × vertical_axis orthonormalization)
        # For vertical thigh (knee below hip), arctan2 gives ~+90°
        # Subtract 90° to make neutral position = 0°, then normalize to [0°, 360°]
        result = (angle.to_numpy() - 90) % 360
        return Signal1D(data=result, index=angle.index, unit="°")

    @property
    def right_hip_abductionadduction(self):
        """
        Calculate right hip abduction/adduction angle in frontal plane.

        The angle represents the lateral (outward) or medial (inward)
        movement of the thigh relative to the pelvis in the frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Abduction (abduzione)
          The thigh is moved laterally away from the body midline.
          Common in side-stepping, lateral movements.
        - **Negative (-)**: Adduction (adduzione)
          The thigh is moved medially toward or across the body midline.
          Common in crossover movements, cutting.
        - **0°**: Neutral position (thigh vertical, aligned with hip)

        Calculation Method
        ------------------
        Uses right hip reference frame (mirrored pelvis frame with origin at right hip):
        - Origin: Right hip joint center (De Leva 1996 regression)
        - lateral_axis: RIGHT (negated pelvis lateral_axis)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (same as pelvis, creating left-handed frame)

        The knee position is transformed to the hip reference frame and projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is calculated using the lateral and vertical components of the
        transformed knee vector.

        Zero degrees corresponds to neutral position (thigh vertical, knee directly
        below hip). The result is normalized to [0°, 360°] range.

        Returns
        -------
        Signal1D
            Hip abduction/adduction angle in degrees.
            Positive = abduction (thigh outward)
            Negative = adduction (thigh inward)

        See Also
        --------
        left_hip_abductionadduction : Left hip frontal plane motion
        right_hip_flexionextension : Right hip sagittal plane motion
        right_knee_varusvalgus : Right knee frontal plane alignment
        """
        hip = self.right_hip
        rmat = self.right_hip_referenceframe.rotation_matrix
        knee = self.right_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # Transform to anatomical convention: 0° = neutral (thigh vertical)
        # The reference frame vertical_axis points downward in the hip frame
        # (determined by the lateral_axis × vertical_axis orthonormalization)
        # For vertical thigh (knee below hip), arctan2 gives ~+90°
        # Subtract 90° to make neutral position = 0°, then normalize to [0°, 360°]
        result = (angle.to_numpy() - 90) % 360
        return Signal1D(data=result, index=angle.index, unit="°")

    @property
    def left_hip_internalexternalrotation(self):
        """
        Calculate left hip internal/external rotation angle in transverse plane.

        The angle represents the rotational orientation of the thigh around
        its longitudinal axis, indicating internal (medial) or external
        (lateral) rotation.

        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The thigh rotates medially, turning the knee and foot inward.
          Common in cutting maneuvers, toe-in gait.
        - **Negative (-)**: External rotation (rotazione esterna)
          The thigh rotates laterally, turning the knee and foot outward.
          Common in toe-out gait, dance movements.
        - **0°**: Neutral position (knee pointing straight ahead)

        Calculation Method
        ------------------
        Uses a custom thigh reference frame constructed from hip and knee positions:
        - lateral_axis: LEFT (left_hip → right_hip direction)
        - vertical_axis: thigh longitudinal axis (hip → knee direction)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The average of knee and ankle medial-lateral vectors is transformed to this
        thigh reference frame and projected onto the transverse plane (defined by
        lateral_axis and anteroposterior_axis).

        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 2] = anteroposterior_axis component

        The angle is calculated as arctan2(anteroposterior, lateral) and negated to
        match clinical convention: positive = internal rotation, negative = external.

        Returns
        -------
        Signal1D
            Hip rotation angle in degrees.
            Positive = internal rotation (knee/foot inward)
            Negative = external rotation (knee/foot outward)

        See Also
        --------
        right_hip_internalexternalrotation : Right hip rotation angle
        left_hip_flexionextension : Left hip sagittal plane motion
        left_knee_varusvalgus : Left knee frontal plane alignment
        """

        # Get necessary parameters
        knee_lat = self._get_point("left_knee_lateral")
        knee_med = self._get_point("left_knee_medial")
        ankle_lat = self._get_point("left_ankle_lateral")
        ankle_med = self._get_point("left_ankle_medial")

        # Compute average vector from medial-lateral markers
        v1 = (knee_lat - knee_med).to_numpy()
        v2 = (ankle_lat - ankle_med).to_numpy()
        va = (v1 + v2) / 2

        # Determine reference frame rotation matrix
        i = (self.left_hip - self.right_hip).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        k = (self.left_hip - self.left_knee).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))

        # Align vector to new reference frame
        vr = np.einsum("nij,nj->ni", rmat, va)

        # Extract components in reference frame coordinates
        # Index [0] = lateral_axis component (by reference frame construction)
        # Index [2] = anteroposterior_axis component (by reference frame construction)
        lateral_component = vr[:, 0]
        ap_component = vr[:, 2]

        # Calculate angle of vector with respect to transverse plane
        angle = np.degrees(np.arctan2(ap_component, lateral_component))

        # Invert sign to match clinical convention: positive = internal rotation, negative = external rotation
        # Original implementation had opposite convention
        angle = -angle

        # Return angle
        return Signal1D(data=angle, index=knee_lat.index, unit="°")

    @property
    def right_hip_internalexternalrotation(self):
        """
        Calculate right hip internal/external rotation angle in transverse plane.

        The angle represents the rotational orientation of the thigh around
        its longitudinal axis, indicating internal (medial) or external
        (lateral) rotation.

        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The thigh rotates medially, turning the knee and foot inward.
          Common in cutting maneuvers, toe-in gait.
        - **Negative (-)**: External rotation (rotazione esterna)
          The thigh rotates laterally, turning the knee and foot outward.
          Common in toe-out gait, dance movements.
        - **0°**: Neutral position (knee pointing straight ahead)

        Calculation Method
        ------------------
        Uses a custom thigh reference frame constructed from hip and knee positions:
        - lateral_axis: RIGHT (right_hip → left_hip direction)
        - vertical_axis: thigh longitudinal axis (hip → knee direction)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The average of knee and ankle medial-lateral vectors is transformed to this
        thigh reference frame and projected onto the transverse plane (defined by
        lateral_axis and anteroposterior_axis).

        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 2] = anteroposterior_axis component

        The angle is calculated as arctan2(anteroposterior, lateral) and negated to
        match clinical convention: positive = internal rotation, negative = external.

        Returns
        -------
        Signal1D
            Hip rotation angle in degrees.
            Positive = internal rotation (knee/foot inward)
            Negative = external rotation (knee/foot outward)

        See Also
        --------
        left_hip_internalexternalrotation : Left hip rotation angle
        right_hip_flexionextension : Right hip sagittal plane motion
        right_knee_varusvalgus : Right knee frontal plane alignment
        """
        # Get necessary parameters
        knee_lat = self._get_point("right_knee_lateral")
        knee_med = self._get_point("right_knee_medial")
        ankle_lat = self._get_point("right_ankle_lateral")
        ankle_med = self._get_point("right_ankle_medial")

        # Compute average vector from medial-lateral markers
        v1 = (knee_lat - knee_med).to_numpy()
        v2 = (ankle_lat - ankle_med).to_numpy()
        va = (v1 + v2) / 2

        # Determine reference frame rotation matrix
        # Fixed: Changed from left_hip - right_hip to right_hip - left_hip for correct symmetry
        i = (self.right_hip - self.left_hip).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        # Fixed: Changed from left_hip - left_knee to right_hip - right_knee (was copy-paste error)
        k = (self.right_hip - self.right_knee).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))

        # Align vector to new reference frame
        vr = np.einsum("nij,nj->ni", rmat, va)

        # Extract components in reference frame coordinates
        # Index [0] = lateral_axis component (by reference frame construction)
        # Index [2] = anteroposterior_axis component (by reference frame construction)
        lateral_component = vr[:, 0]
        ap_component = vr[:, 2]

        # Calculate angle of vector with respect to transverse plane
        angle = np.degrees(np.arctan2(ap_component, lateral_component))

        # NOTE: Removed compensatory transformations (180 - angle, conditional - 360)
        # These were only needed to compensate for the wrong reference frame above

        # Invert sign to match clinical convention: positive = internal rotation, negative = external rotation
        # Original implementation had opposite convention
        angle = -angle

        # Return angle
        return Signal1D(data=angle, index=knee_lat.index, unit="°")

    @property
    def pelvis_lateraltilt_global(self):
        """
        Calculate pelvis lateral tilt (roll) in frontal plane.

        The angle represents the left or right side tilting of the pelvis
        relative to the global horizontal, measured in the frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Left hip higher than right hip (inclinazione sinistra del bacino)
          The left hip joint center is elevated relative to the right.
          Common in left hip drop compensation or scoliosis.
        - **Negative (-)**: Right hip higher than left hip (inclinazione destra del bacino)
          The right hip joint center is elevated relative to the left.
          Common in right hip drop, Trendelenburg gait.
        - **0°**: Neutral position (hip joints level in frontal plane)

        Calculation Method
        ------------------
        Uses hip joint centers (computed using De Leva 1996 regression) to define
        the hip-to-hip vector.

        The vector from right hip to left hip is projected onto the global frontal
        plane (defined by lateral_axis and vertical_axis in the global coordinate system).

        The angle is calculated using the global coordinate components:
        - self.lateral_axis = global lateral direction
        - self.vertical_axis = global vertical direction

        The result is arctan2(vertical_component, lateral_component), giving the
        tilt angle relative to horizontal.

        Returns
        -------
        Signal1D
            Pelvis lateral tilt angle in degrees.
            Positive = left hip higher than right hip
            Negative = right hip higher than left hip

        See Also
        --------
        pelvis_anteroposterior_tilt_global : Pelvis sagittal plane tilt
        pelvis_rotation_global : Pelvis transverse plane rotation
        trunk_lateralflexion : Trunk frontal plane flexion
        """

        # Get hip joint centers
        left_hip = self.left_hip
        right_hip = self.right_hip

        # Calculate hip-to-hip vector (right to left)
        hip_vector = (left_hip - right_hip).to_numpy()

        # Project onto global frontal plane (lateral_axis, vertical_axis)
        cols = left_hip.columns
        lateral_idx = np.where(cols == self.lateral_axis)[0][0]
        vertical_idx = np.where(cols == self.vertical_axis)[0][0]

        lateral_comp = hip_vector[:, lateral_idx]
        vertical_comp = hip_vector[:, vertical_idx]

        # arctan2(vertical, lateral): positive when left hip is higher
        angle = np.degrees(np.arctan2(vertical_comp, lateral_comp))

        return Signal1D(data=angle, index=left_hip.index, unit="°")

    @property
    def pelvis_lateraltilt_local(self):
        """
        Calculate pelvis lateral tilt in local reference plane.

        The angle represents the left or right side tilting of the pelvis
        measured in a plane defined by the pelvis lateral axis and the
        vertical axis from pelvis center to C7.

        Interpretation
        --------------
        - **Positive (+)**: Left hip higher than right hip
          The left hip joint center is elevated relative to the right
          when measured in the local reference plane.
        - **Negative (-)**: Right hip higher than left hip
          The right hip joint center is elevated relative to the left
          when measured in the local reference plane.
        - **0°**: Neutral position (hip joints level in local plane)

        Calculation Method
        ------------------
        Uses hip joint centers (computed using De Leva 1996 regression) to define
        the hip-to-hip vector.

        The local reference plane is defined in the pelvis reference frame's
        frontal plane (lateral and vertical components):
        1. Transform pelvis_center and C7 to pelvis reference frame
        2. Extract frontal plane components (lateral and vertical)
        3. Define vertical axis from pelvis_center to C7 in this plane
        4. Orthonormalize lateral and vertical axes using Gram-Schmidt

        The hip-to-hip vector is projected onto this orthonormalized plane, and
        the angle is calculated as arctan2(vertical_component, lateral_component).

        Returns
        -------
        Signal1D
            Pelvis lateral tilt angle in degrees.
            Positive = left hip higher than right hip
            Negative = right hip higher than left hip

        See Also
        --------
        pelvis_lateraltilt_global : Pelvis lateral tilt in global frontal plane
        pelvis_rotation_local : Pelvis rotation in neck's transverse plane
        pelvis_referenceframe : Pelvis local reference frame
        """
        # Get the points
        lhip = self.left_hip
        rhip = self.right_hip
        pc = self.pelvis_center
        c7 = self._get_point("c7")

        # align to pelvis reference frame
        rf = self.pelvis_referenceframe
        lhip = rf.apply(lhip)
        rhip = rf.apply(rhip)
        pc = rf.apply(pc)
        c7 = rf.apply(c7)

        # define the vertical axis
        vax = c7 - pc.to_numpy()
        vax.loc[vax.index, vax.anteroposterior_axis] = 0
        vax = vax / np.linalg.norm(vax)

        # define the new reference frame
        lax = vax.copy()
        lax.loc[lax.index, lax.lateral_axis] = vax.loc[
            vax.index, vax.vertical_axis
        ].to_numpy()
        lax.loc[lax.index, lax.vertical_axis] = -vax.loc[
            vax.index, vax.lateral_axis
        ].to_numpy()
        ori = lax.copy() * 0
        aax = ori.copy()
        aax.loc[aax.index, aax.anteroposterior_axis] = 1
        rf2 = ReferenceFrame(
            ori,
            lax,
            vax,
            aax,
        )

        # rotate the left hip into this frame
        lh = rf2.apply(lhip)
        rh = rf2.apply(rhip)

        # Calculate hip-to-hip vector (right to left)
        hip_vector = lh - rh

        # Calculate angle: positive when left hip is higher
        vc = hip_vector[self.vertical_axis].to_numpy().flatten()
        lc = hip_vector[self.lateral_axis].to_numpy().flatten()
        angle = np.degrees(np.arctan2(vc, lc))

        return Signal1D(data=angle, index=lhip.index, unit="°")

    @property
    def pelvis_rotation_global(self):
        """
        Calculate pelvis axial rotation (yaw) in transverse plane.

        The angle represents the left or right rotational orientation of
        the pelvis relative to the global forward direction, measured in
        the transverse (horizontal) plane.

        Interpretation
        --------------
        - **Positive (+)**: Left hip forward relative to right hip (rotazione sinistra del bacino)
          The left hip joint center is more anterior than the right.
          The pelvis rotates counterclockwise (viewed from above).
        - **Negative (-)**: Right hip forward relative to left hip (rotazione destra del bacino)
          The right hip joint center is more anterior than the left.
          The pelvis rotates clockwise (viewed from above).
        - **0°**: Neutral position (hip joints aligned in transverse plane)

        Calculation Method
        ------------------
        Uses hip joint centers (computed using De Leva 1996 regression) to define
        the hip-to-hip vector.

        The vector from right hip to left hip is projected onto the global transverse
        plane (defined by lateral_axis and anteroposterior_axis in the global coordinate system).

        The angle is calculated using the global coordinate components:
        - self.lateral_axis = global lateral direction
        - self.anteroposterior_axis = global anteroposterior direction

        The result is arctan2(anteroposterior_component, lateral_component), giving
        the rotation angle in the transverse plane.

        Returns
        -------
        Signal1D
            Pelvis rotation angle in degrees.
            Positive = left hip forward relative to right hip
            Negative = right hip forward relative to left hip

        See Also
        --------
        pelvis_anteroposterior_tilt_global : Pelvis sagittal plane tilt
        pelvis_lateraltilt_global : Pelvis frontal plane tilt
        trunk_rotation : Trunk transverse plane rotation
        """

        # Get hip joint centers
        left_hip = self.left_hip
        right_hip = self.right_hip

        # Calculate hip-to-hip vector (right to left)
        hip_vector = (left_hip - right_hip).to_numpy()

        # Project onto global transverse plane (lateral_axis, anteroposterior_axis)
        cols = left_hip.columns
        lateral_idx = np.where(cols == self.lateral_axis)[0][0]
        ap_idx = np.where(cols == self.anteroposterior_axis)[0][0]

        lateral_comp = hip_vector[:, lateral_idx]
        ap_comp = hip_vector[:, ap_idx]

        # arctan2(anteroposterior, lateral): positive when left hip is forward
        angle = np.degrees(np.arctan2(ap_comp, lateral_comp))

        return Signal1D(data=angle, index=left_hip.index, unit="°")

    @property
    def pelvis_rotation_local(self):
        """
        Calculate pelvis rotation in neck's transverse plane.

        The angle represents the left or right rotational orientation of
        the pelvis measured in the transverse plane of the neck reference
        frame.

        Interpretation
        --------------
        - **Positive (+)**: Left hip forward relative to right hip
          The left hip joint center is more anterior than the right
          when measured in the neck's transverse plane.
        - **Negative (-)**: Right hip forward relative to left hip
          The right hip joint center is more anterior than the left
          when measured in the neck's transverse plane.
        - **0°**: Neutral position (hip joints aligned in neck's transverse plane)

        Calculation Method
        ------------------
        Uses hip joint centers (computed using De Leva 1996 regression) to define
        the hip-to-hip vector.

        The vector from right hip to left hip is transformed into the neck
        reference frame coordinate system. The transverse plane components
        (lateral and anteroposterior) are extracted, and the angle is
        calculated as arctan2(anteroposterior_component, lateral_component).

        The neck reference frame is defined with:
        - Origin at neck_base (midpoint between SC and C7)
        - Vertical axis: pelvis_center → neck_base (UP)
        - Anteroposterior axis: C7 → SC (FORWARD)
        - Lateral axis: cross product (LEFT)

        Returns
        -------
        Signal1D
            Pelvis rotation angle in degrees.
            Positive = left hip forward relative to right hip
            Negative = right hip forward relative to left hip

        See Also
        --------
        pelvis_rotation_global : Pelvis rotation in global transverse plane
        pelvis_lateraltilt_local : Pelvis lateral tilt in local reference plane
        neck_referenceframe : Neck local reference frame
        """
        # Get hip joint centers
        left_hip = self.left_hip
        right_hip = self.right_hip

        # Calculate hip-to-hip vector (right to left)
        hip_vector = (left_hip - right_hip).to_numpy()

        # Transform to neck reference frame
        neck_rframe = self.neck_referenceframe
        neck_rmat = neck_rframe.rotation_matrix
        hip_vector_rf = np.einsum("nij,nj->ni", neck_rmat, hip_vector)

        # Extract transverse plane components
        # Index [0] = lateral_axis, [2] = anteroposterior_axis
        lateral_comp = hip_vector_rf[:, 0]
        ap_comp = hip_vector_rf[:, 2]

        # Calculate angle: positive when left hip is forward
        angle = np.degrees(np.arctan2(ap_comp, lateral_comp))

        return Signal1D(data=angle, index=left_hip.index, unit="°")

    @property
    def pelvis_anteroposterior_tilt_global(self):
        """
        Calculate pelvis anteroposterior tilt (pitch) in global sagittal plane.

        The angle represents the forward or backward tilting of the pelvis
        relative to the global vertical, measured in the global sagittal plane.

        Interpretation
        --------------
        - **Positive (+)**: Anterior tilt (antiversione del bacino)
          ASIS markers are lower/more forward than PSIS markers relative to gravity.
          Lumbar lordosis typically increases.
          Common in hyperlordosis, tight hip flexors.
        - **Negative (-)**: Posterior tilt (retroversione del bacino)
          PSIS markers are lower/more forward than ASIS markers relative to gravity.
          Lumbar lordosis typically decreases.
          Common in hypolordosis, tight hamstrings.
        - **0°**: Neutral position (ASIS and PSIS at same height in global frame)

        Calculation Method
        ------------------
        Measures the angle of the pelvis plane (defined by ASIS-PSIS vector)
        relative to the global sagittal plane (vertical and anteroposterior axes).

        The tilt vector from PSIS midpoint to ASIS midpoint is projected onto
        the global sagittal plane:
        - self.anteroposterior_axis = global forward direction
        - self.vertical_axis = global vertical direction (gravity)

        The angle is arctan2(anteroposterior_component, vertical_component), where
        positive values indicate anterior tilt (ASIS forward/lower than PSIS).

        Returns
        -------
        Signal1D
            Pelvis anteroposterior tilt angle in degrees.
            Positive = anterior tilt (ASIS forward/lower)
            Negative = posterior tilt (PSIS forward/lower)

        See Also
        --------
        pelvis_lateraltilt_global : Pelvis frontal plane tilt
        pelvis_rotation_global : Pelvis transverse plane rotation
        lumbar_lordosis : Lumbar spine curvature
        """
        # Get pelvis markers
        l_asis = self._get_point("left_asis")
        r_asis = self._get_point("right_asis")
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")

        # Calculate tilt vector (PSIS midpoint to ASIS midpoint)
        psis_mid = (l_psis + r_psis) / 2
        asis_mid = (l_asis + r_asis) / 2
        tilt_vec = (asis_mid - psis_mid).to_numpy()

        # Project onto global sagittal plane (anteroposterior_axis, vertical_axis)
        cols = l_asis.columns
        ap_idx = np.where(cols == self.anteroposterior_axis)[0][0]
        vertical_idx = np.where(cols == self.vertical_axis)[0][0]

        ap_comp = tilt_vec[:, ap_idx]
        vertical_comp = tilt_vec[:, vertical_idx]

        # arctan2(-vertical, anteroposterior): positive when ASIS is lower (anterior tilt)
        # Negative vertical component means ASIS lower than PSIS (downward vector)
        angle = np.degrees(np.arctan2(-vertical_comp, ap_comp))

        return Signal1D(data=angle, index=l_asis.index, unit="°")

    @property
    def trunk_lateralflexion(self):
        """
        Calculate trunk lateral flexion in frontal plane of pelvis reference frame.

        The angle represents the lateral deviation of the trunk (neck_base) from
        vertical relative to the pelvis, measured in the pelvis reference frame.

        Interpretation
        --------------
        - **Positive (+)**: Right lateral flexion (flessione laterale destra)
          The trunk bends to the right; neck_base moves laterally right.
        - **Negative (-)**: Left lateral flexion (flessione laterale sinistra)
          The trunk bends to the left; neck_base moves laterally left.
        - **0°**: Neutral position (neck_base centered above pelvis_center)

        Note: Neutral position is at ~90° before zero correction. Angle < 90°
        indicates left lateral flexion, angle > 90° indicates right lateral flexion.

        Calculation Method
        ------------------
        Uses the trunk segment vector from pelvis_center to neck_base, transformed
        into the pelvis reference frame and projected onto the frontal plane
        (defined by lateral_axis and vertical_axis of pelvis reference frame).

        The angle is calculated as:
        - arctan2(vertical_component, lateral_component) - 90°
        - Neutral position (neck_base directly above pelvis_center) gives ~90° before
          correction, which becomes 0° after subtracting 90°.

        Returns
        -------
        Signal1D
            Trunk lateral flexion angle in degrees.
            Positive = right lateral flexion
            Negative = left lateral flexion

        See Also
        --------
        trunk_rotation : Trunk transverse plane rotation
        pelvis_anteroposterior_tilt_global : Pelvis sagittal plane tilt
        neck_lateralflexion : Neck frontal plane lateral flexion
        pelvis_referenceframe : Reference frame used for this calculation
        """
        neck_base = self.neck_base
        pelvis_center = self.pelvis_center
        rmat = self.pelvis_referenceframe.rotation_matrix

        # Calculate angle in pelvis reference frame's frontal plane
        # Frontal plane: lateral_axis (X) and vertical_axis (Y)
        # Using arctan2(y, x) convention: arctan2(vertical, lateral)
        angle, lateral_comp, vertical_comp = (
            self._get_angle_by_point_on_reference_frame_and_plane(
                neck_base,
                pelvis_center,
                rmat,
                self.lateral_axis,  # First arg (x in arctan2)
                self.vertical_axis,  # Second arg (y in arctan2)
            )
        )

        # The helper method returns arctan2(second_axis, first_axis)
        # So this gives arctan2(vertical, lateral)
        # At neutral (neck_base directly above): lateral≈0, vertical>0 → angle≈90°
        # Apply zero correction: subtract 90° so that vertical = 0°
        # Positive = right flexion, Negative = left flexion
        lateralflexion = angle.to_numpy() - 90

        return Signal1D(data=lateralflexion, index=neck_base.index, unit="°")

    @property
    def trunk_rotation(self):
        """
        Calculate trunk axial rotation (yaw) in transverse plane.

        The angle represents the rotational orientation of the shoulders
        relative to the pelvis, measured in the transverse (horizontal) plane.

        Interpretation
        --------------
        - **Positive (+)**: Left rotation (rotazione sinistra del tronco)
          The shoulders rotate counterclockwise (viewed from above).
          The right shoulder moves forward relative to the left shoulder.
          Common in left trunk rotation movements.
        - **Negative (-)**: Right rotation (rotazione destra del tronco)
          The shoulders rotate clockwise (viewed from above).
          The left shoulder moves forward relative to the right shoulder.
          Common in right trunk rotation movements.
        - **0°**: Neutral position (shoulders aligned with pelvis in transverse plane)

        Calculation Method
        ------------------
        Uses pelvis reference frame with:
        - Origin: Pelvis center (centroid of 4 ASIS/PSIS markers)
        - lateral_axis: LEFT (right midpoint → left midpoint)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The shoulder axis vector (C7 → sternoclavicular junction) is transformed
        to the pelvis reference frame and projected onto the transverse plane
        (defined by lateral_axis and anteroposterior_axis).

        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 2] = anteroposterior_axis component

        The angle is arctan2(lateral_component, anteroposterior_component), where
        positive values indicate left rotation (right shoulder forward).

        Returns
        -------
        Signal1D
            Trunk rotation angle in degrees.
            Positive = left rotation (right shoulder forward)
            Negative = right rotation (left shoulder forward)

        See Also
        --------
        trunk_lateralflexion : Trunk frontal plane flexion
        pelvis_rotation : Pelvis transverse plane rotation
        shoulder_lateral_tilt : Shoulder frontal plane tilt
        """
        # Get shoulder axis markers
        c7 = self._get_point("c7")
        sc = self._get_point("sc")

        # Get pelvis reference frame
        rmat = self.pelvis_referenceframe.rotation_matrix

        # Calculate shoulder axis vector (C7 to sternoclavicular junction)
        shoulder_axis = (sc - c7).to_numpy()

        # Transform to pelvis reference frame
        shoulder_axis_rf = np.einsum("nij,nj->ni", rmat, shoulder_axis)  # type: ignore

        # Extract components in reference frame coordinates
        # Index [0] = lateral_axis component (by ReferenceFrame construction)
        # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
        # arctan2 of lateral and anteroposterior components gives rotation angle
        # Positive lateral component (LEFT) = left rotation = positive angle
        angle = (
            np.arctan2(
                shoulder_axis_rf[:, 0],
                shoulder_axis_rf[:, 2],
            )
            * 180
            / np.pi
        )

        # Return angle
        return Signal1D(data=angle, index=c7.index, unit="°")

    @property
    def neck_lateralflexion(self):
        """
        Calculate neck lateral flexion in frontal plane of neck reference frame.

        The angle represents the lateral deviation of the head from vertical,
        measured relative to the neck reference frame.

        Interpretation
        --------------
        - **Positive (+)**: Right lateral flexion (flessione laterale destra)
          The head tilts toward the right shoulder.
        - **Negative (-)**: Left lateral flexion (flessione laterale sinistra)
          The head tilts toward the left shoulder.
        - **0°**: Neutral position (head centered above neck_base)

        Calculation Method
        ------------------
        Uses the head_center to neck_base vector transformed into the neck
        reference frame and projected onto the frontal plane (defined by
        lateral_axis and vertical_axis of the neck reference frame).

        The angle is calculated as:
        - arctan2(vertical_component, lateral_component) - 90°
        - Neutral position (head directly above neck_base) gives ~90° before
          correction, which becomes 0° after subtracting 90°.
        - Positive values indicate right lateral flexion.

        Returns
        -------
        Signal1D
            Neck lateral flexion angle in degrees.
            Positive = right lateral flexion (destra)
            Negative = left lateral flexion (sinistra)

        See Also
        --------
        neck_flexionextension : Neck sagittal plane flexion/extension
        neck_referenceframe : Reference frame used for this calculation
        """
        head = self.head_center
        neck_base = self.neck_base
        rmat = self.neck_referenceframe.rotation_matrix

        # Calculate angle in neck reference frame's frontal plane
        # Frontal plane: lateral_axis (X) and vertical_axis (Y)
        # Using arctan2(y, x) convention: arctan2(vertical, lateral)
        angle, lateral_comp, vertical_comp = (
            self._get_angle_by_point_on_reference_frame_and_plane(
                head,
                neck_base,
                rmat,
                self.lateral_axis,  # First arg (x in arctan2)
                self.vertical_axis,  # Second arg (y in arctan2)
            )
        )

        # The helper method returns arctan2(second_axis, first_axis)
        # So this gives arctan2(vertical, lateral)
        # At neutral (head directly above): lateral≈0, vertical>0 → angle≈90°
        # Apply zero correction: subtract 90° so that vertical = 0°
        # Positive = right flexion, Negative = left flexion
        lateralflexion = angle.to_numpy() - 90

        return Signal1D(data=lateralflexion, index=head.index, unit="°")

    @property
    def neck_flexionextension(self):
        """
        Calculate neck flexion/extension in sagittal plane of neck reference frame.

        The angle represents the forward or backward deviation of the head
        from vertical position, measured from neck_base to head_center.

        Interpretation
        --------------
        - **Positive (+)**: Forward flexion (flessione in avanti)
          The head moves forward; chin moves toward chest.
          Common in forward head posture, texting posture.
        - **Negative (-)**: Backward extension (estensione indietro)
          The head tilts backward; looking up.
          Common in looking up, backward head tilt.
        - **0°**: Neutral position (head vertical UP from neck_base, at 90° in sagittal plane)

        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)

        The head_center position is transformed to the neck reference frame and
        projected onto the sagittal plane (defined by anteroposterior_axis and
        vertical_axis).

        The angle is calculated as:
        - arctan2(vertical_component, anteroposterior_component) - 90°
        - Neutral position (head directly above neck_base) gives ~90° before
          correction, which becomes 0° after subtracting 90°.
        - Positive values indicate forward flexion (chin toward chest).

        Returns
        -------
        Signal1D
            Neck flexion/extension angle in degrees.
            Positive = forward flexion (chin to chest)
            Negative = backward extension (looking up)

        See Also
        --------
        neck_lateralflexion : Neck frontal plane lateral flexion
        trunk_rotation : Trunk transverse plane rotation
        """
        head = self.head_center
        neck_base = self.neck_base
        rmat = self.neck_referenceframe.rotation_matrix

        # Calculate angle in neck reference frame's sagittal plane
        # Sagittal plane: anteroposterior_axis (Z) and vertical_axis (Y)
        # Using arctan2(y, x) convention: arctan2(vertical, anteroposterior)
        angle, ap_comp, vertical_comp = (
            self._get_angle_by_point_on_reference_frame_and_plane(
                head,
                neck_base,
                rmat,
                self.anteroposterior_axis,  # First arg (x in arctan2)
                self.vertical_axis,  # Second arg (y in arctan2)
            )
        )

        # The helper method returns arctan2(second_axis, first_axis)
        # So this gives arctan2(vertical, anteroposterior)
        # At neutral (head directly above): anteroposterior≈0, vertical>0 → angle≈90°
        # Apply zero correction: subtract 90° so that vertical = 0°
        # Positive = flexion (forward), Negative = extension (backward)
        flexionextension = 90 - angle.to_numpy()

        return Signal1D(data=flexionextension, index=head.index, unit="°")

    @property
    def lumbar_lordosis(self):
        """
        Calculate lumbar lordosis (curvatura lordotica lombare).

        The angle quantifies the anterior curvature of the lumbar spine.

        Interpretation
        --------------
        The angle represents the degree of lumbar curvature:

        - **Larger angles (>160°)**: Indicates straighter spine or reduced lordosis
          (ipolordosi, "flat back")
        - **Smaller angles (<140°)**: Indicates increased lordotic curvature
          (iperlordosi, "sway back", excessive lumbar curve)
        - **Normal range**: Typically 140-160° (curvatura fisiologica)

        Note: This is the internal angle at L2. A more curved (lordotic) spine
        produces a smaller angle because the vertebrae form a tighter curve.

        Calculation Method
        ------------------
        Measured as the angle at L2 vertex formed by three points:
        1. Midpoint of PSIS (Posterior Superior Iliac Spine) markers
        2. L2 (Second Lumbar vertebra) - vertex
        3. T5 (Fifth Thoracic vertebra)

        This represents the transition from lumbar to thoracic curvature.

        Returns
        -------
        Signal1D
            Lumbar lordosis angle in degrees.
            Smaller angle = greater lordotic curvature (more curved)
            Larger angle = reduced lordosis (flatter)
        """
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")
        l2 = self._get_point("l2")
        t5 = self._get_point("t5")

        # Calculate PSIS midpoint (posterior pelvis reference)
        psis_mid_data = (l_psis.to_numpy() + r_psis.to_numpy()) / 2
        psis_mid_index = np.unique(
            np.concatenate([l_psis.index, r_psis.index])
        ).tolist()

        # Create a Point3D for psis_mid
        psis_mid = Point3D(
            data=psis_mid_data,
            index=psis_mid_index,
            columns=l_psis.columns,
        )

        # Calculate 3-point angle: T5 - L2 - PSIS_mid
        # Internal angle at L2 vertex (order: superior → vertex → inferior)
        angle = self._get_angle_between_three_points(t5, l2, psis_mid)

        return Signal1D(data=angle, index=l2.index, unit="°")

    @property
    def dorsal_kyphosis(self):
        """
        Calculate thoracic (dorsal) kyphosis (curvatura cifotica toracica).

        The angle quantifies the posterior curvature of the thoracic spine.

        Interpretation
        --------------
        The angle represents the degree of thoracic curvature:

        - **Larger angles (>160°)**: Indicates straighter spine or reduced kyphosis
          (ipocifosi, "flat upper back")
        - **Smaller angles (<140°)**: Indicates increased kyphotic curvature
          (ipercifosi, "rounded back", "hunchback", excessive thoracic curve)
        - **Normal range**: Typically 140-160° (curvatura fisiologica)

        Note: This is the internal angle at T5. A more curved (kyphotic) spine
        produces a smaller angle because the vertebrae form a tighter curve.

        Calculation Method
        ------------------
        Measured as the angle at T5 vertex formed by three points:
        1. L2 (Second Lumbar vertebra)
        2. T5 (Fifth Thoracic vertebra) - vertex
        3. C7 (Seventh Cervical vertebra)

        This spans the thoracic region from lower thoracic (near lumbar junction)
        to upper thoracic (near cervical junction).

        Returns
        -------
        Signal1D
            Thoracic kyphosis angle in degrees.
            Smaller angle = greater kyphotic curvature (more curved/rounded)
            Larger angle = reduced kyphosis (flatter upper back)
        """
        l2 = self._get_point("l2")
        t5 = self._get_point("t5")
        c7 = self._get_point("c7")

        # Calculate 3-point angle: C7 - T5 - L2
        # Internal angle at T5 vertex (order: superior → vertex → inferior)
        angle = self._get_angle_between_three_points(c7, t5, l2)

        return Signal1D(data=angle, index=t5.index, unit="°")

    @property
    def left_shoulder_flexionextension(self):
        """
        Calculate left shoulder flexion/extension angle in sagittal plane.

        The angle represents the forward (flexion) or backward (extension)
        movement of the arm relative to the shoulder in the sagittal plane.

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The arm is raised forward (anteriorly).
          Common in reaching forward, throwing preparation.
        - **Negative (-)**: Extension (estensione)
          The arm is moved backward (posteriorly).
          Common in backswing, reaching behind.
        - **0°**: Neutral position (arm hanging at side)

        Calculation Method
        ------------------
        Uses left shoulder reference frame with:
        - Origin: Left shoulder joint center
        - lateral_axis: LEFT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The elbow position is transformed to the shoulder reference frame and
        projected onto the sagittal plane (defined by anteroposterior_axis and
        vertical_axis).

        The calculation uses self.anteroposterior_axis and self.vertical_axis to
        identify rotation matrix components. The result is normalized to [0°, 360°]
        with zero corresponding to the arm hanging vertically downward.

        Returns
        -------
        Signal1D
            Shoulder flexion/extension angle in degrees.
            Positive = flexion (arm forward)
            Negative = extension (arm backward)

        See Also
        --------
        right_shoulder_flexionextension : Right shoulder flexion angle
        left_shoulder_abductionadduction : Left shoulder frontal plane motion
        left_elbow_flexionextension : Left elbow flexion angle
        """
        # ottengo i parametri necessari
        shoulder = self.left_shoulder
        rmat = self.left_shoulder_referenceframe.rotation_matrix
        elbow = self.left_elbow

        # calcolo l'orientamento del braccio rispetto al sistema di riferimento
        # della spalla
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow,
            shoulder,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # Transform to anatomical convention: 0° = neutral (arm hanging vertically)
        # The reference frame Y-axis points upward
        # For vertical arm (elbow below shoulder), arctan2 gives ~-90°
        # Add 90° to make neutral position = 0°, then normalize to [0°, 360°]
        angle_result = (angle.to_numpy() + 90) % 360

        return Signal1D(data=angle_result, index=angle.index, unit="°")

    @property
    def right_shoulder_flexionextension(self):
        """
        Calculate right shoulder flexion/extension angle in sagittal plane.

        The angle represents the forward (flexion) or backward (extension)
        movement of the arm relative to the shoulder in the sagittal plane.

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The arm is raised forward (anteriorly).
          Common in reaching forward, throwing preparation.
        - **Negative (-)**: Extension (estensione)
          The arm is moved backward (posteriorly).
          Common in backswing, reaching behind.
        - **0°**: Neutral position (arm hanging at side)

        Calculation Method
        ------------------
        Uses right shoulder reference frame with:
        - Origin: Right shoulder joint center
        - lateral_axis: RIGHT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)

        The elbow position is transformed to the shoulder reference frame and
        projected onto the sagittal plane (defined by anteroposterior_axis and
        vertical_axis).

        The calculation uses self.anteroposterior_axis and self.vertical_axis to
        identify rotation matrix components. The result is normalized to [0°, 360°]
        with zero corresponding to the arm hanging vertically downward.

        Returns
        -------
        Signal1D
            Shoulder flexion/extension angle in degrees.
            Positive = flexion (arm forward)
            Negative = extension (arm backward)

        See Also
        --------
        left_shoulder_flexionextension : Left shoulder flexion angle
        right_shoulder_abductionadduction : Right shoulder frontal plane motion
        right_elbow_flexionextension : Right elbow flexion angle
        """
        # ottengo i parametri necessari
        shoulder = self.right_shoulder
        rmat = self.right_shoulder_referenceframe.rotation_matrix
        elbow = self.right_elbow

        # calcolo l'orientamento del braccio rispetto al sistema di riferimento
        # della spalla
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow,
            shoulder,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # Transform to anatomical convention: 0° = neutral (arm hanging vertically)
        # The reference frame Y-axis points upward
        # For vertical arm (elbow below shoulder), arctan2 gives ~-90°
        # Add 90° to make neutral position = 0°, then normalize to [0°, 360°]
        angle_result = (angle.to_numpy() + 90) % 360

        return Signal1D(data=angle_result, index=angle.index, unit="°")

    @property
    def left_shoulder_abductionadduction(self):
        """
        Calculate left shoulder abduction/adduction angle in frontal plane.

        The angle represents the lateral (outward) or medial (inward)
        movement of the arm relative to the shoulder in the frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Abduction (abduzione)
          The arm is raised laterally away from the body.
          Common in lateral raises, overhead reaching.
        - **Negative (-)**: Adduction (adduzione)
          The arm is moved medially toward or across the body.
          Common in cross-body movements.
        - **0°**: Neutral position (arm hanging at side)

        Calculation Method
        ------------------
        Uses left shoulder reference frame with:
        - Origin: Left shoulder joint center
        - lateral_axis: LEFT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The elbow position is transformed to the shoulder reference frame and
        projected onto the frontal plane (defined by lateral_axis and vertical_axis).

        The calculation uses self.lateral_axis and self.vertical_axis to identify
        rotation matrix components. The result is normalized to [0°, 360°] with
        zero corresponding to the arm hanging vertically downward.

        Returns
        -------
        Signal1D
            Shoulder abduction/adduction angle in degrees.
            Positive = abduction (arm outward)
            Negative = adduction (arm inward)

        See Also
        --------
        right_shoulder_abductionadduction : Right shoulder frontal plane motion
        left_shoulder_flexionextension : Left shoulder sagittal plane motion
        left_scapular_protractionretraction : Left scapular position
        """
        # Get necessary parameters
        shoulder = self.left_shoulder
        rmat = self.left_shoulder_referenceframe.rotation_matrix
        elbow = self.left_elbow

        # Calculate arm orientation with respect to shoulder reference frame
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow,
            shoulder,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # Transform to anatomical convention: 0° = neutral (arm hanging vertically)
        # arctan2 gives ~-90° for vertical downward vector
        # Add 90° to make neutral = 0°, then normalize to [0°, 360°]
        result = (angle.to_numpy() + 90) % 360
        return Signal1D(data=result, index=angle.index, unit="°")

    @property
    def right_shoulder_abductionadduction(self):
        """
        Calculate right shoulder abduction/adduction angle in frontal plane.

        The angle represents the lateral (outward) or medial (inward)
        movement of the arm relative to the shoulder in the frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Abduction (abduzione)
          The arm is raised laterally away from the body.
          Common in lateral raises, overhead reaching.
        - **Negative (-)**: Adduction (adduzione)
          The arm is moved medially toward or across the body.
          Common in cross-body movements.
        - **0°**: Neutral position (arm hanging at side)

        Calculation Method
        ------------------
        Uses right shoulder reference frame with:
        - Origin: Right shoulder joint center
        - lateral_axis: RIGHT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)

        The elbow position is transformed to the shoulder reference frame and
        projected onto the frontal plane (defined by lateral_axis and vertical_axis).

        The calculation uses self.lateral_axis and self.vertical_axis to identify
        rotation matrix components. The result is normalized to [0°, 360°] with
        zero corresponding to the arm hanging vertically downward.

        Returns
        -------
        Signal1D
            Shoulder abduction/adduction angle in degrees.
            Positive = abduction (arm outward)
            Negative = adduction (arm inward)

        See Also
        --------
        left_shoulder_abductionadduction : Left shoulder frontal plane motion
        right_shoulder_flexionextension : Right shoulder sagittal plane motion
        right_scapular_protractionretraction : Right scapular position
        """
        # Get necessary parameters
        shoulder = self.right_shoulder
        rmat = self.right_shoulder_referenceframe.rotation_matrix
        elbow = self.right_elbow

        # Calculate arm orientation with respect to shoulder reference frame
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow,
            shoulder,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # Transform to anatomical convention: 0° = neutral (arm hanging vertically)
        # arctan2 gives ~-90° for vertical downward vector
        # Add 90° to make neutral = 0°, then normalize to [0°, 360°]
        result = (angle.to_numpy() + 90) % 360
        return Signal1D(data=result, index=angle.index, unit="°")

    @property
    def left_shoulder_internalexternalrotation(self):
        """
        Calculate left shoulder internal/external rotation angle in transverse plane.

        The angle represents the rotational orientation of the upper arm around
        its longitudinal axis, measured by the orientation of the elbow reference
        frame's anteroposterior axis relative to the shoulder reference frame.

        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The forearm rotates medially (toward the body).
          Common in throwing acceleration, reaching across body.
        - **Negative (-)**: External rotation (rotazione esterna)
          The forearm rotates laterally (away from the body).
          Common in throwing cocking phase, backhand motions.
        - **0°**: Neutral position (elbow anteroposterior axis parallel to shoulder anteroposterior axis)

        Calculation Method
        ------------------
        Uses left shoulder reference frame with:
        - Origin: Left shoulder joint center
        - lateral_axis: LEFT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The elbow reference frame's anteroposterior_axis (column 2 of its rotation
        matrix) is extracted and added to the elbow position to create a point in
        global space. This point is then transformed to the shoulder reference frame
        and projected onto the transverse plane (defined by lateral_axis and
        anteroposterior_axis).

        The calculation uses self.lateral_axis and self.anteroposterior_axis to
        identify rotation matrix components. The angle is calculated and adjusted
        so that positive values indicate internal rotation (forearm inward) and
        negative values indicate external rotation (forearm outward).

        Returns
        -------
        Signal1D
            Shoulder rotation angle in degrees.
            Positive = internal rotation (forearm inward)
            Negative = external rotation (forearm outward)

        See Also
        --------
        right_shoulder_internalexternalrotation : Right shoulder rotation angle
        left_shoulder_referenceframe : Shoulder reference frame used for calculation
        left_elbow_referenceframe : Elbow reference frame used for calculation
        left_shoulder_abductionadduction : Left shoulder frontal plane motion
        """
        # Get reference frames
        shoulder_rf = self.left_shoulder_referenceframe
        elbow_rf = self.left_elbow_referenceframe

        # Get elbow anteroposterior axis from rotation matrix (column 2)
        elbow_ap_axis = elbow_rf.rotation_matrix[:, :, 2]  # shape (N, 3)

        # Create a point at elbow origin + anteroposterior axis direction
        elbow = self.left_elbow
        elbow_ap_global = Point3D(
            data=elbow_rf.origin + elbow_ap_axis,
            index=elbow.index,
            columns=elbow.columns,
        )

        # Calculate the angle in shoulder reference frame's transverse plane
        shoulder = self.left_shoulder
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow_ap_global,
            shoulder,
            shoulder_rf.rotation_matrix,
            self.lateral_axis,
            self.anteroposterior_axis,
        )

        # Correct the angle (positive means internal rotation, negative external)
        angle = 90 - angle.to_numpy()

        return Signal1D(data=angle, index=elbow.index, unit="°")

    @property
    def right_shoulder_internalexternalrotation(self):
        """
        Calculate right shoulder internal/external rotation angle in transverse plane.

        The angle represents the rotational orientation of the upper arm around
        its longitudinal axis, measured by the orientation of the elbow reference
        frame's anteroposterior axis relative to the shoulder reference frame.

        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The forearm rotates medially (toward the body).
          Common in throwing acceleration, reaching across body.
        - **Negative (-)**: External rotation (rotazione esterna)
          The forearm rotates laterally (away from the body).
          Common in throwing cocking phase, backhand motions.
        - **0°**: Neutral position (elbow anteroposterior axis parallel to shoulder anteroposterior axis)

        Calculation Method
        ------------------
        Uses right shoulder reference frame with:
        - Origin: Right shoulder joint center
        - lateral_axis: RIGHT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)

        The elbow reference frame's anteroposterior_axis (column 2 of its rotation
        matrix) is extracted and added to the elbow position to create a point in
        global space. This point is then transformed to the shoulder reference frame
        and projected onto the transverse plane (defined by lateral_axis and
        anteroposterior_axis).

        The calculation uses self.lateral_axis and self.anteroposterior_axis to
        identify rotation matrix components. The angle is calculated and adjusted
        so that positive values indicate internal rotation (forearm inward) and
        negative values indicate external rotation (forearm outward).

        Returns
        -------
        Signal1D
            Shoulder rotation angle in degrees.
            Positive = internal rotation (forearm inward)
            Negative = external rotation (forearm outward)

        See Also
        --------
        left_shoulder_internalexternalrotation : Left shoulder rotation angle
        right_shoulder_referenceframe : Shoulder reference frame used for calculation
        right_elbow_referenceframe : Elbow reference frame used for calculation
        right_shoulder_abductionadduction : Right shoulder frontal plane motion
        """
        # Get reference frames
        shoulder_rf = self.right_shoulder_referenceframe
        elbow_rf = self.right_elbow_referenceframe

        # Get elbow anteroposterior axis from rotation matrix (column 2)
        elbow_ap_axis = elbow_rf.rotation_matrix[:, :, 2]  # shape (N, 3)

        # Create a point at elbow origin + anteroposterior axis direction
        elbow = self.right_elbow
        elbow_ap_global = Point3D(
            data=elbow_rf.origin + elbow_ap_axis,
            index=elbow.index,
            columns=elbow.columns,
        )

        # Calculate the angle in shoulder reference frame's transverse plane
        shoulder = self.right_shoulder
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow_ap_global,
            shoulder,
            shoulder_rf.rotation_matrix,
            self.lateral_axis,
            self.anteroposterior_axis,
        )

        # Correct the angle (positive means internal rotation, negative external)
        # Right shoulder RF has negated anteroposterior_axis, so negate the angle to match left side
        angle = angle.to_numpy() - 90

        return Signal1D(data=angle, index=elbow.index, unit="°")

    @property
    def shoulder_lateraltilt_global(self):
        """
        Calculate shoulder lateral tilt in global frontal plane.

        The angle represents the left or right side tilting of the shoulders
        relative to the global horizontal, measured in the frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Left shoulder higher than right shoulder
          The left shoulder joint center is elevated relative to the right.
        - **Negative (-)**: Right shoulder higher than left shoulder
          The right shoulder joint center is elevated relative to the left.
        - **0°**: Neutral position (shoulders level in frontal plane)

        Calculation Method
        ------------------
        Uses shoulder joint centers to define the shoulder-to-shoulder vector.

        The vector from right shoulder to left shoulder is projected onto the global frontal
        plane (defined by lateral_axis and vertical_axis in the global coordinate system).

        The angle is calculated using the global coordinate components:
        - self.lateral_axis = global lateral direction
        - self.vertical_axis = global vertical direction

        The result is arctan2(vertical_component, lateral_component), giving the
        tilt angle relative to horizontal.

        Returns
        -------
        Signal1D
            Shoulder lateral tilt angle in degrees.
            Positive = left shoulder higher than right shoulder
            Negative = right shoulder higher than left shoulder

        See Also
        --------
        shoulder_lateraltilt_local : Shoulder lateral tilt in neck reference plane
        pelvis_lateraltilt_global : Pelvis lateral tilt in global frontal plane
        """
        # Get shoulder joint centers
        left_shoulder = self.left_shoulder
        right_shoulder = self.right_shoulder

        # Calculate shoulder-to-shoulder vector (right to left)
        shoulder_vector = (left_shoulder - right_shoulder).to_numpy()

        # Project onto global frontal plane (lateral_axis, vertical_axis)
        cols = left_shoulder.columns
        lateral_idx = np.where(cols == self.lateral_axis)[0][0]
        vertical_idx = np.where(cols == self.vertical_axis)[0][0]

        lateral_comp = shoulder_vector[:, lateral_idx]
        vertical_comp = shoulder_vector[:, vertical_idx]

        # arctan2(vertical, lateral): positive when left shoulder is higher
        angle = np.degrees(np.arctan2(vertical_comp, lateral_comp))

        return Signal1D(data=angle, index=left_shoulder.index, unit="°")

    @property
    def shoulder_lateraltilt_local(self):
        """
        Calculate shoulder lateral tilt in local reference plane.

        The angle represents the left or right side tilting of the shoulders
        measured in a plane defined by the neck lateral axis and the
        vertical axis from neck_base to head (or C7 to head marker if available).

        Interpretation
        --------------
        - **Positive (+)**: Left shoulder higher than right shoulder
          The left shoulder joint center is elevated relative to the right
          when measured in the local reference plane.
        - **Negative (-)**: Right shoulder higher than left shoulder
          The right shoulder joint center is elevated relative to the left
          when measured in the local reference plane.
        - **0°**: Neutral position (shoulders level in local plane)

        Calculation Method
        ------------------
        Uses shoulder joint centers to define the shoulder-to-shoulder vector.

        The local reference plane is defined in the neck reference frame's
        frontal plane (lateral and vertical components):
        1. Transform neck_base and pelvis_center to neck reference frame
        2. Extract frontal plane components (lateral and vertical)
        3. Define vertical axis from pelvis_center to neck_base in this plane
        4. Rotate 90° to obtain lateral axis
        5. Create new reference frame with these orthonormalized axes

        The shoulder-to-shoulder vector is projected onto this plane, and
        the angle is calculated as arctan2(vertical_component, lateral_component).

        Returns
        -------
        Signal1D
            Shoulder lateral tilt angle in degrees.
            Positive = left shoulder higher than right shoulder
            Negative = right shoulder higher than left shoulder

        See Also
        --------
        shoulder_lateraltilt_global : Shoulder lateral tilt in global frontal plane
        pelvis_lateraltilt_local : Pelvis lateral tilt in local reference plane
        neck_referenceframe : Neck local reference frame
        """
        # Get the points
        lshoulder = self.left_shoulder
        rshoulder = self.right_shoulder
        neck_base = self.neck_base
        pc = self.pelvis_center

        # align to neck reference frame
        rf = self.neck_referenceframe
        lshoulder = rf.apply(lshoulder)
        rshoulder = rf.apply(rshoulder)
        neck_base = rf.apply(neck_base)
        pc = rf.apply(pc)

        # define the vertical axis (pelvis_center → neck_base projected on frontal plane)
        vax = neck_base - pc.to_numpy()
        vax.loc[vax.index, vax.anteroposterior_axis] = 0
        vax = vax / np.linalg.norm(vax)

        # define the new reference frame (rotate 90° to get lateral axis)
        lax = vax.copy()
        lax.loc[lax.index, lax.lateral_axis] = vax.loc[
            vax.index, vax.vertical_axis
        ].to_numpy()
        lax.loc[lax.index, lax.vertical_axis] = -vax.loc[
            vax.index, vax.lateral_axis
        ].to_numpy()
        ori = lax.copy() * 0
        aax = ori.copy()
        aax.loc[aax.index, aax.anteroposterior_axis] = 1
        rf2 = ReferenceFrame(
            ori,
            lax,
            vax,
            aax,
        )

        # rotate the shoulders into this frame
        ls = rf2.apply(lshoulder)
        rs = rf2.apply(rshoulder)

        # Calculate shoulder-to-shoulder vector (right to left)
        shoulder_vector = ls - rs

        # Calculate angle: positive when left shoulder is higher
        vc = shoulder_vector[self.vertical_axis].to_numpy().flatten()
        lc = shoulder_vector[self.lateral_axis].to_numpy().flatten()
        angle = np.degrees(np.arctan2(vc, lc))

        return Signal1D(data=angle, index=lshoulder.index, unit="°")

    @property
    def left_scapular_protractionretraction(self):
        """
        Calculate left scapular protraction/retraction angle in transverse plane.

        The angle represents the horizontal position of the left shoulder relative
        to neck_base (base of the neck), indicating scapular protraction or
        retraction.

        Interpretation
        --------------
        - **Positive (+)**: Scapular protraction (protrazione scapolare)
          The shoulder is positioned anteriorly (forward) relative to neck base.
          Common in rounded shoulder posture.
        - **Negative (-)**: Scapular retraction (retrazione scapolare)
          The shoulder is positioned posteriorly (backward) relative to neck base.
          Common in military/upright posture.
        - **0°**: Neutral position (shoulder aligned with neck base in transverse plane)

        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)

        The left shoulder position is transformed to the neck reference frame and
        projected onto the transverse plane (defined by lateral_axis and
        anteroposterior_axis).

        The calculation uses self.lateral_axis and self.anteroposterior_axis to
        identify rotation matrix components. Positive values indicate protraction
        (shoulder forward), negative values indicate retraction (shoulder backward).

        Returns
        -------
        Signal1D
            Scapular protraction/retraction angle in degrees.
            Positive = protraction (forward shoulder position)
            Negative = retraction (backward shoulder position)

        See Also
        --------
        right_scapular_protractionretraction : Right scapular protraction/retraction
        neck_referenceframe : Neck reference frame used for this calculation
        shoulder_lateraltilt_global : Shoulder elevation in frontal plane
        trunk_rotation_global : Trunk rotation that may affect shoulder position
        """
        shoulder = self.left_shoulder
        neck_base = self.neck_base
        rmat = self.neck_referenceframe.rotation_matrix

        # Calculate angle of left shoulder in neck reference frame's transverse plane
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            shoulder,
            neck_base,
            rmat,
            self.lateral_axis,
            self.anteroposterior_axis,
        )

        # In neck frame, left shoulder has:
        # - lateral_axis > 0 (shoulder is left, +lateral_axis points LEFT)
        # - anteroposterior_axis > 0 when protracted (shoulder is forward, +anteroposterior_axis points FORWARD)
        # arctan2 of anteroposterior and lateral components gives protraction/retraction angle
        # For symmetric sign convention: positive = protraction
        angle_result = angle.to_numpy()

        return Signal1D(data=angle_result, index=shoulder.index, unit="°")

    @property
    def right_scapular_protractionretraction(self):
        """
        Calculate right scapular protraction/retraction angle in transverse plane.

        The angle represents the horizontal position of the right shoulder relative
        to neck_base (base of the neck), indicating scapular protraction or
        retraction.

        Interpretation
        --------------
        - **Positive (+)**: Scapular protraction (protrazione scapolare)
          The shoulder is positioned anteriorly (forward) relative to neck base.
          Common in rounded shoulder posture.
        - **Negative (-)**: Scapular retraction (retrazione scapolare)
          The shoulder is positioned posteriorly (backward) relative to neck base.
          Common in military/upright posture.
        - **0°**: Neutral position (shoulder aligned with neck base in transverse plane)

        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)

        The right shoulder position is transformed to the neck reference frame and
        projected onto the transverse plane (defined by lateral_axis and
        anteroposterior_axis).

        The calculation uses self.lateral_axis and self.anteroposterior_axis to
        identify rotation matrix components. The lateral component is negated for
        the right side to maintain consistent sign convention: positive = protraction
        (shoulder forward), negative = retraction (shoulder backward).

        Returns
        -------
        Signal1D
            Scapular protraction/retraction angle in degrees.
            Positive = protraction (forward shoulder position)
            Negative = retraction (backward shoulder position)

        See Also
        --------
        left_scapular_protractionretraction : Left scapular protraction/retraction
        neck_referenceframe : Neck reference frame used for this calculation
        shoulder_lateraltilt_global : Shoulder elevation in frontal plane
        trunk_rotation_global : Trunk rotation that may affect shoulder position
        """
        shoulder = self.right_shoulder
        neck_base = self.neck_base
        rmat = self.neck_referenceframe.rotation_matrix

        # Calculate angle of right shoulder in neck reference frame's transverse plane
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            shoulder,
            neck_base,
            rmat,
            self.lateral_axis,
            self.anteroposterior_axis,
        )

        # In neck frame, right shoulder has:
        # - lateral_axis < 0 (shoulder is right, opposite to +lateral_axis which points LEFT)
        # - anteroposterior_axis > 0 when protracted (shoulder is forward, +anteroposterior_axis points FORWARD)
        # arctan2 with lateral < 0 gives angle in wrong quadrant
        # Recalculate angle using negated lateral component to match left side convention
        angle_result = np.degrees(np.arctan2(y, -x))

        return Signal1D(data=angle_result, index=shoulder.index, unit="°")

    @property
    def left_shoulder_elevationdepression(self):
        """
        Calculate left shoulder elevation/depression angle in frontal plane.

        The angle represents the vertical position of the left shoulder relative
        to the upper thoracic spine, indicating shoulder elevation (shrugging) or
        depression (dropping).

        Interpretation
        --------------
        - **Positive (+)**: Shoulder elevation (elevazione della spalla)
          The shoulder is elevated (shrugged upward).
          Common in upper trapezius tension, stress postures.
        - **Negative (-)**: Shoulder depression (depressione della spalla)
          The shoulder is depressed (pulled downward).
          Less common in static postures.
        - **0°**: Neutral position (shoulder aligned with thoracic reference)

        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)

        The left shoulder position is transformed to the neck reference frame and
        projected onto the frontal plane (defined by lateral_axis and vertical_axis).

        The calculation uses self.lateral_axis and self.vertical_axis to identify
        rotation matrix components. Positive values indicate elevation (shoulder up),
        negative values indicate depression (shoulder down).

        Returns
        -------
        Signal1D
            Shoulder elevation/depression angle in degrees.
            Positive = elevation (shoulder up)
            Negative = depression (shoulder down)

        See Also
        --------
        right_shoulder_elevationdepression : Right shoulder elevation/depression
        left_scapular_protractionretraction : Left scapular anterior/posterior position
        neck_referenceframe : Neck reference frame used for this calculation
        """
        shoulder = self.left_shoulder
        neck_base = self.neck_base
        rmat = self.neck_referenceframe.rotation_matrix

        # Calculate angle of left shoulder in neck reference frame's frontal plane
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            shoulder,
            neck_base,
            rmat,
            self.lateral_axis,
            self.vertical_axis,
        )

        # In neck frame, left shoulder has:
        # - lateral_axis > 0 (shoulder is left, +lateral_axis points LEFT)
        # - vertical_axis > 0 when elevated (shoulder is above, +vertical_axis points UP)
        # arctan2 of vertical and lateral components gives elevation/depression angle
        # For symmetric sign convention: positive = elevation
        angle_result = angle.to_numpy()

        return Signal1D(data=angle_result, index=shoulder.index, unit="°")

    @property
    def right_shoulder_elevationdepression(self):
        """
        Calculate right shoulder elevation/depression angle in frontal plane.

        The angle represents the vertical position of the right shoulder relative
        to the upper thoracic spine, indicating shoulder elevation (shrugging) or
        depression (dropping).

        Interpretation
        --------------
        - **Positive (+)**: Shoulder elevation (elevazione della spalla)
          The shoulder is elevated (shrugged upward).
          Common in upper trapezius tension, stress postures.
        - **Negative (-)**: Shoulder depression (depressione della spalla)
          The shoulder is depressed (pulled downward).
          Less common in static postures.
        - **0°**: Neutral position (shoulder aligned with thoracic reference)

        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)

        The right shoulder position is transformed to the neck reference frame and
        projected onto the frontal plane (defined by lateral_axis and vertical_axis).

        The calculation uses self.lateral_axis and self.vertical_axis to identify
        rotation matrix components. The lateral component is negated for the right
        side to maintain consistent sign convention: positive = elevation (shoulder up),
        negative = depression (shoulder down).

        Returns
        -------
        Signal1D
            Shoulder elevation/depression angle in degrees.
            Positive = elevation (shoulder up)
            Negative = depression (shoulder down)

        See Also
        --------
        left_shoulder_elevationdepression : Left shoulder elevation/depression
        right_scapular_protractionretraction : Right scapular anterior/posterior position
        neck_referenceframe : Neck reference frame used for this calculation
        """
        shoulder = self.right_shoulder
        neck_base = self.neck_base
        rmat = self.neck_referenceframe.rotation_matrix

        # Calculate angle of right shoulder in neck reference frame's frontal plane
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            shoulder,
            neck_base,
            rmat,
            self.lateral_axis,
            self.vertical_axis,
        )

        # In neck frame, right shoulder has:
        # - lateral_axis < 0 (shoulder is right, opposite to +lateral_axis which points LEFT)
        # - vertical_axis > 0 when elevated (shoulder is above, +vertical_axis points UP)
        # arctan2 with lateral < 0 gives angle in wrong quadrant
        # Recalculate angle using negated lateral component to match left side convention
        angle_result = np.degrees(np.arctan2(y, -x))

        return Signal1D(data=angle_result, index=shoulder.index, unit="°")

    @property
    def left_elbow_flexionextension(self):
        """
        Calculate left elbow flexion/extension angle in sagittal plane.

        The angle represents the bending or straightening of the elbow joint,
        indicating flexion (bent elbow) or extension (straight elbow).

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The elbow is bent, bringing the wrist toward the shoulder.
          Common in lifting, reaching, biceps curl.
        - **Negative (-)**: Extension (estensione)
          The elbow is straightened beyond neutral.
          Rare; indicates hyperextension.
        - **0°**: Neutral position (fully straight elbow)

        Calculation Method
        ------------------
        Uses left elbow reference frame with:
        - Origin: Left elbow center (midpoint of lateral and medial elbow markers)
        - lateral_axis: LEFT (elbow_lateral → elbow_medial)
        - vertical_axis: UP (elbow → shoulder)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)

        The wrist position is transformed to the elbow reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).

        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component

        Zero degrees corresponds to full elbow extension (wrist directly below elbow).
        Positive values indicate flexion (bent elbow).

        Returns
        -------
        Signal1D
            Elbow flexion/extension angle in degrees.
            Positive = flexion (bent elbow)
            Range: [0°, 180°]

        See Also
        --------
        right_elbow_flexionextension : Right elbow flexion angle
        left_shoulder_flexionextension : Left shoulder flexion angle
        """
        elbow = self.left_elbow
        wrist = self.left_wrist
        rmat = self.left_elbow_referenceframe.rotation_matrix

        # Wrist vector from elbow origin
        wrist_vec = (wrist - elbow).to_numpy()

        # Transform to elbow reference frame
        wrist_rf = np.einsum("nij,nj->ni", rmat, wrist_vec)

        # Extract components in reference frame coordinates
        # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
        # Index [1] = vertical_axis component (by ReferenceFrame construction)
        # Elbow flexion is measured from the extended position (wrist below elbow, vertical negative)
        # arctan2 of anteroposterior and vertical components gives flexion/extension angle
        # At extended (wrist straight down): anteroposterior≈0, vertical<0 → angle ≈ 0°
        # At flexed (wrist forward): anteroposterior>0, vertical<0 → angle < 0°, so negate to get positive flexion
        flexion = -np.arctan2(wrist_rf[:, 2], -wrist_rf[:, 1]) * 180 / np.pi

        return Signal1D(data=flexion, index=elbow.index, unit="°")

    @property
    def right_elbow_flexionextension(self):
        """
        Calculate right elbow flexion/extension angle in sagittal plane.

        The angle represents the bending or straightening of the elbow joint,
        indicating flexion (bent elbow) or extension (straight elbow).

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The elbow is bent, bringing the wrist toward the shoulder.
          Common in lifting, reaching, biceps curl.
        - **Negative (-)**: Extension (estensione)
          The elbow is straightened beyond neutral.
          Rare; indicates hyperextension.
        - **0°**: Neutral position (fully straight elbow)

        Calculation Method
        ------------------
        Uses right elbow reference frame with:
        - Origin: Right elbow center (midpoint of lateral and medial elbow markers)
        - lateral_axis: RIGHT (elbow_lateral → elbow_medial)
        - vertical_axis: UP (elbow → shoulder)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)

        The wrist position is transformed to the elbow reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).

        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component

        Zero degrees corresponds to full elbow extension (wrist directly below elbow).
        Positive values indicate flexion (bent elbow).

        Returns
        -------
        Signal1D
            Elbow flexion/extension angle in degrees.
            Positive = flexion (bent elbow)
            Negative = extension (hyperextension)

        See Also
        --------
        left_elbow_flexionextension : Left elbow flexion angle
        right_shoulder_flexionextension : Right shoulder flexion angle
        """
        elbow = self.right_elbow
        wrist = self.right_wrist
        rmat = self.right_elbow_referenceframe.rotation_matrix

        # Wrist vector from elbow origin
        wrist_vec = (wrist - elbow).to_numpy()

        # Transform to elbow reference frame
        wrist_rf = np.einsum("nij,nj->ni", rmat, wrist_vec)

        # Extract components in reference frame coordinates
        # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
        # Index [1] = vertical_axis component (by ReferenceFrame construction)
        # Elbow flexion is measured from the extended position (wrist below elbow, vertical negative)
        # For left-handed frame: anteroposterior_axis is negated to point forward
        # arctan2 of anteroposterior and vertical components gives flexion/extension angle
        # At extended (wrist straight down): anteroposterior≈0, vertical<0 → angle ≈ 0°
        # At flexed (wrist forward): anteroposterior<0 (negated), vertical<0 → angle > 0°, but we need to negate result
        flexion = -np.arctan2(wrist_rf[:, 2], -wrist_rf[:, 1]) * 180 / np.pi

        return Signal1D(data=flexion, index=elbow.index, unit="°")

    @property
    def segment_lengths(self) -> Timeseries:
        """
        All segment lengths and widths combined into a single Timeseries.

        Returns a Timeseries containing all computed segment dimensions including:
        - Foot heights, lengths, and widths (left/right)
        - Ankle widths (left/right)
        - Leg lengths (left/right)
        - Thigh lengths (left/right)
        - Knee widths (left/right)
        - Lower limb total lengths (left/right)
        - Arm lengths (left/right)
        - Forearm lengths (left/right)
        - Elbow widths (left/right)
        - Upper limb total lengths (left/right)
        - Body dimensions: shoulder_width, hip_width, trunk_length, pelvis_height

        Returns
        -------
        Timeseries
            Combined timeseries with all segment length and width measurements.
            Each column represents one segment dimension (e.g., 'left_foot_height',
            'right_thigh_length', 'left_ankle_width', 'shoulder_width').

        Notes
        -----
        Only includes properties that are computable from available markers.
        If markers are missing, the corresponding columns will not be present.

        Examples
        --------
        >>> body = WholeBody(left_heel=..., right_heel=..., ...)
        >>> lengths = body.segment_lengths
        >>> print(lengths.columns)
        ['left_foot_height', 'right_foot_height', 'left_leg_length', 'shoulder_width', ...]
        """
        length_properties = [
            "left_foot_height",
            "right_foot_height",
            "left_foot_length",
            "right_foot_length",
            "left_foot_width",
            "right_foot_width",
            "left_ankle_width",
            "right_ankle_width",
            "left_leg_length",
            "right_leg_length",
            "left_thigh_length",
            "right_thigh_length",
            "left_knee_width",
            "right_knee_width",
            "left_lower_limb_length",
            "right_lower_limb_length",
            "left_arm_length",
            "right_arm_length",
            "left_forearm_length",
            "right_forearm_length",
            "left_elbow_width",
            "right_elbow_width",
            "left_upper_limb_length",
            "right_upper_limb_length",
            "shoulder_width",
            "hip_width",
            "trunk_length",
            "pelvis_height",
        ]

        # Collect all available length measurements
        data_list = []
        column_names = []
        common_index = None
        common_unit = None

        for prop_name in length_properties:
            try:
                value = getattr(self, prop_name)
                if value is not None:
                    # Get the data as 1D array
                    data_1d = value.to_numpy().flatten()
                    data_list.append(data_1d)
                    column_names.append(prop_name)

                    # Use first property's index and unit as reference
                    if common_index is None:
                        common_index = value.index
                        common_unit = value.unit
            except (AttributeError, TypeError, ValueError):
                # Skip if property cannot be computed (missing markers)
                continue

        if not data_list:
            raise ValueError(
                "No segment lengths could be computed. Check that required markers are provided."
            )

        # Stack all columns into 2D array
        data_2d = np.column_stack(data_list)

        return Timeseries(
            data=data_2d,
            index=common_index,
            columns=column_names,
            unit=common_unit,
        )

    @property
    def joint_angles(self) -> Timeseries:
        """
        All joint angles combined into a single Timeseries.

        Returns a Timeseries containing all computed joint angles including:
        - Ankle angles: flexion/extension, inversion/eversion (left/right)
        - Knee angles: flexion/extension, varus/valgus (left/right)
        - Hip angles: flexion/extension, abduction/adduction, internal/external rotation (left/right)
        - Pelvis angles: lateral tilt, rotation, anteroposterior tilt (global frame)
        - Trunk angles: lateral flexion, rotation, flexion/extension (global and local frames)
        - Shoulder angles: flexion/extension, abduction/adduction, internal/external rotation,
          lateral tilt, elevation/depression (left/right, global and local frames)
        - Scapular angles: protraction/retraction (left/right)
        - Elbow angles: flexion/extension (left/right)
        - Neck angles: flexion/extension (local and global), lateral tilt
        - Spine curvature: lumbar lordosis, dorsal kyphosis

        Returns
        -------
        Timeseries
            Combined timeseries with all joint angle measurements in degrees.
            Each column represents one angular measurement (e.g., 'left_knee_flexionextension',
            'pelvis_rotation_global', 'lumbar_lordosis').

        Notes
        -----
        Only includes properties that are computable from available markers.
        If markers are missing, the corresponding columns will not be present.

        Sign conventions vary by joint - see individual angle property docstrings for details.

        Examples
        --------
        >>> body = WholeBody(left_asis=..., right_asis=..., ...)
        >>> angles = body.joint_angles
        >>> print(angles.columns)
        ['left_ankle_flexionextension', 'right_knee_flexionextension', ...]
        """
        angle_properties = [
            # Ankle angles
            "left_ankle_flexionextension",
            "right_ankle_flexionextension",
            "left_ankle_inversioneversion",
            "right_ankle_inversioneversion",
            # Knee angles
            "left_knee_flexionextension",
            "right_knee_flexionextension",
            "left_knee_varusvalgus",
            "right_knee_varusvalgus",
            # Hip angles
            "left_hip_flexionextension",
            "right_hip_flexionextension",
            "left_hip_abductionadduction",
            "right_hip_abductionadduction",
            "left_hip_internalexternalrotation",
            "right_hip_internalexternalrotation",
            # Pelvis angles
            "pelvis_anteroposterior_tilt_global",
            # Trunk angles
            "trunk_rotation",
            # Shoulder angles
            "left_shoulder_flexionextension",
            "right_shoulder_flexionextension",
            "left_shoulder_abductionadduction",
            "right_shoulder_abductionadduction",
            "left_shoulder_internalexternalrotation",
            "right_shoulder_internalexternalrotation",
            "left_shoulder_elevationdepression",
            "right_shoulder_elevationdepression",
            # Scapular angles
            "left_scapular_protractionretraction",
            "right_scapular_protractionretraction",
            # Elbow angles
            "left_elbow_flexionextension",
            "right_elbow_flexionextension",
            # Neck angles
            "neck_flexionextension",
            "neck_lateralflexion",
            # Spine curvature
            "lumbar_lordosis",
            "dorsal_kyphosis",
        ]

        # Collect all available angle measurements
        data_list = []
        column_names = []
        common_index = None
        common_unit = None

        for prop_name in angle_properties:
            try:
                value = getattr(self, prop_name)
                if value is not None:
                    # Get the data as 1D array
                    data_1d = value.to_numpy().flatten()
                    data_list.append(data_1d)
                    column_names.append(prop_name)

                    # Use first property's index and unit as reference
                    if common_index is None:
                        common_index = value.index
                        common_unit = value.unit
            except (AttributeError, TypeError, ValueError):
                # Skip if property cannot be computed (missing markers)
                continue

        if not data_list:
            raise ValueError(
                "No joint angles could be computed. Check that required markers are provided."
            )

        # Stack all columns into 2D array
        data_2d = np.column_stack(data_list)

        return Timeseries(
            data=data_2d,
            index=common_index,
            columns=column_names,
            unit=common_unit,
        )

    def copy(self):
        return WholeBody(**{i: v.copy() for i, v in self.items()})  # type: ignore
