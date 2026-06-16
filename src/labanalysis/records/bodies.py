"""full body module"""

from typing import Literal

import numpy as np
import pandas as pd

from ..signalprocessing import gram_schmidt
from .timeseries import *
from .records import *

__all__ = ["WholeBody"]


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
    left_throcanter : Point3D, optional
        Left greater trochanter marker.
    right_throcanter : Point3D, optional
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
        List of all available joint angle properties.

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

    _angular_measures = [
        "left_ankle_flexionextension",
        "right_ankle_flexionextension",
        "left_ankle_inversioneversion",
        "right_ankle_inversioneversion",
        "left_knee_flexionextension",
        "right_knee_flexionextension",
        "left_knee_varusvalgus",
        "right_knee_varusvalgus",
        "left_hip_flexionextension",
        "right_hip_flexionextension",
        "left_hip_abductionadduction",
        "right_hip_abductionadduction",
        "left_hip_internalexternalrotation",
        "right_hip_internalexternalrotation",
        "pelvis_anteroposteriortilt_global",
        "pelvis_lateraltilt_global",
        "pelvis_rotation_global",
        "trunk_flexionextension_global",
        "trunk_lateralflexion_global",
        "trunk_rotation_global",
        "trunk_rotation_local",
        "shoulder_lateraltilt_global",
        "shoulder_lateraltilt_local",
        "left_shoulder_abductionadduction",
        "right_shoulder_abductionadduction",
        "left_shoulder_flexionextension",
        "right_shoulder_flexionextension",
        "left_shoulder_internalexternalrotation",
        "right_shoulder_internalexternalrotation",
        "left_elbow_flexionextension",
        "right_elbow_flexionextension",
        "neck_lateral_tilt",
        "neck_flexionextension_local",
        "neck_flexionextension_global",
        "lumbar_lordosis",
        "dorsal_kyphosis",
    ]

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
        right_throcanter: Point3D | None = None,
        left_throcanter: Point3D | None = None,
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
                left_throcanter=left_throcanter,
                right_throcanter=right_throcanter,
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
        right_throcanter: str | None = None,
        left_throcanter: str | None = None,
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
            "right_throcanter": right_throcanter,
            "left_throcanter": left_throcanter,
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

    def to_dataframe(self):
        out = [super().to_dataframe()]
        for prop in self._angular_measures:
            try:
                df = getattr(self, prop).to_dataframe()
                df.columns = pd.Index([prop + "_" + i for i in df.columns])
                out += [df]
            except Exception as exc:
                continue

        return pd.concat(out, axis=1)

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

        # Calculate eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov)

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
        normals = plane.ix[:, :3].to_numpy()
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
        return the left ankle reference frame origin and rotation matrix.

        A point can be aligned to this reference frame by:
            new = np.einsum("nij,nj->ni", R, old - O)

        Where R is the rotation matrix (N, 3, 3) and O (N, 3) is the origin of
        the reference frame.
        """
        ankle_lat: Point3D = self._get_point("left_ankle_lateral")
        ankle = self.left_ankle
        knee = self.left_knee

        # get the rotation matrix
        try:
            ankle_med: Point3D = self._get_point("left_ankle_medial")
            new_x = (ankle_lat - ankle_med).to_numpy()
        except Exception as e:
            new_x = np.ones(ankle.shape) * np.array([1, 0, 0])
        new_y = (knee - ankle).to_numpy()
        rmat = gram_schmidt(new_x, new_y).transpose((0, 2, 1))
        rmat = np.asarray(rmat, float)

        return ankle, rmat

    @property
    def right_ankle_referenceframe(self):
        """
        return the right ankle reference frame origin and rotation matrix.

        A point can be aligned to this reference frame by:
            new = np.einsum("nij,nj->ni", R, old - O)

        Where R is the rotation matrix (N, 3, 3) and O (N, 3) is the origin of
        the reference frame.
        """
        ankle_lat: Point3D = self._get_point("right_ankle_lateral")
        ankle = self.right_ankle
        knee = self.right_knee

        # get the rotation matrix
        try:
            ankle_med: Point3D = self._get_point("right_ankle_medial")
            new_x = (ankle_med - ankle_lat).to_numpy()
        except Exception as e:
            new_x = np.ones(ankle.shape) * np.array([1, 0, 0])
        new_y = (knee - ankle).to_numpy()
        rmat = gram_schmidt(new_x, new_y).transpose((0, 2, 1))
        rmat = np.asarray(rmat, float)

        return ankle, rmat

    @property
    def left_hip_referenceframe(self):
        """
        return the left hip reference frame origin and rotation matrix.

        A point can be aligned to this reference frame by:
            new = np.einsum("nij,nj->ni", R, old - O)

        Where R is the rotation matrix (N, 3, 3) and O (N, 3) is the origin of
        the reference frame.
        """
        return self.left_hip, self.pelvis_referenceframe[1]

    @property
    def right_hip_referenceframe(self):
        """
        return the right hip reference frame origin and rotation matrix.

        A point can be aligned to this reference frame by:
            new = np.einsum("nij,nj->ni", R, old - O)

        Where R is the rotation matrix (N, 3, 3) and O (N, 3) is the origin of
        the reference frame.
        """
        return self.right_hip, self.pelvis_referenceframe[1]

    @property
    def pelvis_referenceframe(self):
        """
        return the pelvis reference frame origin and rotation matrix.

        A point can be aligned to this reference frame by:
            new = np.einsum("nij,nj->ni", R, old - O)

        Where R is the rotation matrix (N, 3, 3) and O (N, 3) is the origin of
        the reference frame.
        """

        # get the pelvis points projected into its least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # get the plane versors
        centroid = (l_asis + r_asis + l_psis + r_psis) / 4
        i = np.asarray((l_asis + l_psis) / 2 - centroid, float)
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        k = np.asarray((l_asis + r_asis) / 2 - centroid, float)
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)

        # obtain the rotation matrix
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))

        return centroid, rmat

    @property
    def left_shoulder_referenceframe(self):
        """
        Return the left shoulder reference frame origin and rotation matrix.

        A point can be aligned to this reference frame by:
            new = np.einsum("nij,nj->ni", R, old - O)

        Where R is the rotation matrix (N, 3, 3) and O (N, 3) is the origin of
        the reference frame.
        """
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()
        base = (r_psis + l_psis) / 2
        c7 = self._get_point("c7")
        pelvis_center = self.pelvis_center
        shoulder = self.left_shoulder

        # determino l'asse verticale
        j = c7 - base

        # calcolo le coordinate della proiezione della spalla sull'asse
        # verticale passante per il centro della pelvi
        proj = self._get_projection_point_onto_axis(
            point=shoulder,
            origin=pelvis_center,
            direction=j,
        )

        # calcolo l'asse laterale
        i = shoulder - proj

        # ottengo la matrice di rotazione
        rmat = gram_schmidt(i.to_numpy(), j.to_numpy()).transpose((0, 2, 1))

        return shoulder, rmat

    @property
    def right_shoulder_referenceframe(self):
        """
        Return the right shoulder reference frame origin and rotation matrix.

        A point can be aligned to this reference frame by:
            new = np.einsum("nij,nj->ni", R, old - O)

        Where R is the rotation matrix (N, 3, 3) and O (N, 3) is the origin of
        the reference frame.
        """
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()
        base = (r_psis + l_psis) / 2
        c7 = self._get_point("c7")
        pelvis_center = self.pelvis_center
        shoulder = self.right_shoulder

        # determino l'asse verticale
        j = c7 - base

        # calcolo le coordinate della proiezione della spalla sull'asse
        # verticale passante per il centro della pelvi
        proj = self._get_projection_point_onto_axis(
            point=shoulder,
            origin=pelvis_center,
            direction=j,
        )

        # calcolo l'asse laterale
        i = proj - shoulder

        # ottengo la matrice di rotazione
        rmat = gram_schmidt(i.to_numpy(), j.to_numpy()).transpose((0, 2, 1))

        return shoulder, rmat

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
        return self.pelvis_referenceframe[0]

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
        data = (h_ant.to_numpy() + h_post.to_numpy() +
                h_left.to_numpy() + h_right.to_numpy()) / 4

        # Merge indices from all markers
        index = np.unique(np.concatenate([
            h_ant.index, h_post.index, h_left.index, h_right.index
        ])).tolist()

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
            Neck base point (average of sc and c7 markers).
        """
        sc = self._get_point("sc")
        c7 = self._get_point("c7")

        # Calculate midpoint
        data = (sc.to_numpy() + c7.to_numpy()) / 2

        # Merge indices
        index = np.unique(np.concatenate([
            sc.index, c7.index
        ])).tolist()

        return Point3D(
            data=data,
            index=index,
            columns=sc.columns,
        )

    @property
    def pelvis_plane(self):

        # extract the normal as the vertical axis denoted by the rotation
        # matrix of the pelvis reference frame
        centroid, rmat = self.pelvis_referenceframe
        normal = rmat[:, 1, :]

        # get the "d" coefficient
        d = np.atleast_2d(-np.sum(normal * centroid.to_numpy(), axis=1)).T

        # return the timeseries
        return Timeseries(
            data=np.hstack([normal, d]),
            index=centroid.index,
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
        pelvis_plane = self.pelvis_plane
        try:
            l_height = self._get_point_to_plane_distance(
                self._get_point("left_throcanter"),
                pelvis_plane,
            )
        except Exception as l:
            l_height = None
        try:
            r_height = self._get_point_to_plane_distance(
                self._get_point("right_throcanter"),
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
            Distance in millimeters from ankle joint to foot plane.
        """
        ankle = self.left_ankle
        foot_plane = self.left_foot_plane
        return self._get_point_to_plane_distance(ankle, foot_plane)

    @property
    def right_foot_height(self):
        """
        Calculate right foot height as perpendicular distance from ankle to foot plane.

        Returns
        -------
        Signal1D
            Distance in millimeters from ankle joint to foot plane.
        """
        ankle = self.right_ankle
        foot_plane = self.right_foot_plane
        return self._get_point_to_plane_distance(ankle, foot_plane)

    @property
    def left_foot_length(self):
        """
        Calculate left foot length as distance from heel to toe.

        Returns
        -------
        Signal1D
            Distance in millimeters from heel to toe marker.
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
            Distance in millimeters from heel to toe marker.
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
            Distance in millimeters from ankle to knee joint center.
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
            Distance in millimeters from ankle to knee joint center.
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
            Distance in millimeters from knee to hip joint center.
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
            Distance in millimeters from knee to hip joint center.
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
            Distance in millimeters from shoulder to elbow joint center.
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
            Distance in millimeters from shoulder to elbow joint center.
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
            Distance in millimeters from elbow to wrist joint center.
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
            Distance in millimeters from elbow to wrist joint center.
        """
        elbow = self.right_elbow
        wrist = self.right_wrist
        index = np.unique(np.concatenate([elbow.index, wrist.index])).tolist()
        data = np.asarray(wrist - elbow)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=elbow.unit)

    @property
    def trunk_height(self):
        """
        Calculate trunk height as distance from pelvis center to neck base.

        Returns
        -------
        Signal1D
            Distance in millimeters from pelvis center to neck base.
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
            Distance in millimeters between shoulder joint centers.
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
            Distance in millimeters between greater trochanter markers.
        """
        l_troch = self._get_point("left_throcanter")
        r_troch = self._get_point("right_throcanter")
        index = np.unique(np.concatenate([l_troch.index, r_troch.index])).tolist()
        data = np.asarray(l_troch - r_troch)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=l_troch.unit)

    @property
    def left_foot_width(self):
        """
        Calculate left foot width as distance between first and fifth metatarsal heads.

        Returns
        -------
        Signal1D
            Distance in millimeters between first and fifth metatarsal heads.
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
            Distance in millimeters between first and fifth metatarsal heads.
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
            Distance in millimeters between medial and lateral ankle malleoli.
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
            Distance in millimeters between medial and lateral ankle malleoli.
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
            Distance in millimeters between medial and lateral femoral epicondyles.
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
            Distance in millimeters between medial and lateral femoral epicondyles.
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
            Distance in millimeters between medial and lateral elbow epicondyles.
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
            Distance in millimeters between medial and lateral elbow epicondyles.
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
            self._get_point("left_throcanter"),
            offsets,
            self.pelvis_plane,
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
            self._get_point("right_throcanter"),
            offsets,
            self.pelvis_plane,
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
    def left_foot_plane(self):
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
        return Timeseries(
            data=self._get_least_squares_plane_coefs(toe, first_meta, fifth_meta, heel),
            index=toe.index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )

    @property
    def right_foot_plane(self):
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
        return Timeseries(
            data=self._get_least_squares_plane_coefs(toe, first_meta, fifth_meta, heel),
            index=toe.index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )

    @property
    def left_ankle_flexionextension(self):
        """
        the the dorsal (positive) or plantar (negative) flexion of the
        left ankle with respect to he shin in degrees
        """
        # get points and reference frame
        ankle, rmat = self.left_ankle_referenceframe
        proj = self._get_projection_point_on_plane(
            ankle,
            self.left_foot_plane,
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
        the the dorsal (negative) or plantar (positive) flexion of the
        right ankle with respect to he shin in degrees
        """
        ankle, rmat = self.right_ankle_referenceframe
        proj = self._get_projection_point_on_plane(
            ankle,
            self.right_foot_plane,
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
        the inversion (negative) or eversion (positive) angle of the
        left ankle with respect to he shin in degrees
        """
        ankle, rmat = self.left_ankle_referenceframe
        proj = self._get_projection_point_on_plane(
            ankle,
            self.left_foot_plane,
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
        the inversion (negative) or eversion (positive) angle of the
        left ankle with respect to he shin in degrees
        """
        ankle, rmat = self.right_ankle_referenceframe
        proj = self._get_projection_point_on_plane(
            ankle,
            self.right_foot_plane,
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
        return the left knee flexion in degrees.
        Extension will be negative
        """
        p1 = self.left_hip
        p2 = self.left_knee
        p3 = self.left_ankle
        angle = 180 - self._get_angle_between_three_points(p1, p2, p3)
        return Signal1D(data=angle, index=p1.index, unit="°")

    @property
    def right_knee_flexionextension(self):
        """return the left knee flexion in degrees. Extension will be negative"""
        p1 = self.right_hip
        p2 = self.right_knee
        p3 = self.right_ankle
        angle = 180 - self._get_angle_between_three_points(p1, p2, p3)
        return Signal1D(data=angle, index=p1.index, unit="°")

    @property
    def left_knee_varusvalgus(self):
        """
        Calculate left knee varus/valgus angle in frontal plane.

        The angle represents the frontal plane alignment of the knee joint.

        Interpretation
        --------------
        - **Positive (+)**: Valgus deformity (ginocchio valgo, "a X", knock-knee)
          The knee deviates medially; the leg angle opens laterally.
        - **Negative (-)**: Varus deformity (ginocchio varo, "a parentesi", bow-legged)
          The knee deviates laterally; the leg angle opens medially.
        - **0°**: Neutral alignment (anca-ginocchio-caviglia collineari nel piano frontale)

        Calculation Method
        ------------------
        Measured as the difference between thigh and leg angles in the frontal plane:
        - Thigh angle: angle of hip-to-knee vector from vertical
        - Leg angle: angle of knee-to-ankle vector from vertical
        - Varus/Valgus = (Thigh angle) - (Leg angle)

        Returns
        -------
        Signal1D
            Knee varus/valgus angle in degrees.
            Positive = valgus (ginocchio valgo)
            Negative = varus (ginocchio varo)
        """
        hip = self.left_hip
        knee = self.left_knee
        ankle = self.left_ankle

        # Get vectors in frontal plane (lateral_axis, vertical_axis)
        v_thigh = (knee - hip).to_numpy()  # Vector from hip to knee
        v_leg = (ankle - knee).to_numpy()  # Vector from knee to ankle

        # Extract frontal plane components
        cols = knee.columns
        axes_labels = [self.lateral_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Project vectors onto frontal plane
        v_thigh_2d = v_thigh[:, col_map]
        v_leg_2d = v_leg[:, col_map]

        # Calculate angle between vectors using atan2
        # Angle from vertical (thigh angle)
        angle_thigh = np.degrees(np.arctan2(v_thigh_2d[:, 0], v_thigh_2d[:, 1]))
        # Angle from vertical (leg angle)
        angle_leg = np.degrees(np.arctan2(v_leg_2d[:, 0], v_leg_2d[:, 1]))

        # Varus/valgus is the difference
        # Positive when knee deviates medially (valgus)
        # Negative when knee deviates laterally (varus)
        angle = angle_thigh - angle_leg

        return Signal1D(data=angle, index=knee.index, unit="°")

    @property
    def right_knee_varusvalgus(self):
        """
        Calculate right knee varus/valgus angle in frontal plane.

        The angle represents the frontal plane alignment of the knee joint.

        Interpretation
        --------------
        - **Positive (+)**: Valgus deformity (ginocchio valgo, "a X", knock-knee)
          The knee deviates medially; the leg angle opens laterally.
        - **Negative (-)**: Varus deformity (ginocchio varo, "a parentesi", bow-legged)
          The knee deviates laterally; the leg angle opens medially.
        - **0°**: Neutral alignment (anca-ginocchio-caviglia collineari nel piano frontale)

        Calculation Method
        ------------------
        Measured as the difference between leg and thigh angles in the frontal plane.
        Sign is reversed compared to left side to maintain consistent interpretation
        (positive = valgus for both sides).

        Returns
        -------
        Signal1D
            Knee varus/valgus angle in degrees.
            Positive = valgus (ginocchio valgo)
            Negative = varus (ginocchio varo)
        """
        hip = self.right_hip
        knee = self.right_knee
        ankle = self.right_ankle

        # Get vectors in frontal plane (lateral_axis, vertical_axis)
        v_thigh = (knee - hip).to_numpy()  # Vector from hip to knee
        v_leg = (ankle - knee).to_numpy()  # Vector from knee to ankle

        # Extract frontal plane components
        cols = knee.columns
        axes_labels = [self.lateral_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Project vectors onto frontal plane
        v_thigh_2d = v_thigh[:, col_map]
        v_leg_2d = v_leg[:, col_map]

        # Calculate angle between vectors using atan2
        # Angle from vertical (thigh angle)
        angle_thigh = np.degrees(np.arctan2(v_thigh_2d[:, 0], v_thigh_2d[:, 1]))
        # Angle from vertical (leg angle)
        angle_leg = np.degrees(np.arctan2(v_leg_2d[:, 0], v_leg_2d[:, 1]))

        # Varus/valgus is the difference (sign reversed for right side)
        # This ensures positive = valgus for both left and right knees
        angle = angle_leg - angle_thigh

        return Signal1D(data=angle, index=knee.index, unit="°")

    @property
    def left_hip_flexionextension(self):
        """
        return the left hip flexion/extension in degrees.
        extension will be negative
        """
        # get points and reference frame
        hip, rmat = self.left_hip_referenceframe
        knee = self.left_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return 90 + angle

    @property
    def right_hip_flexionextension(self):
        """
        return the right hip flexion/extension in degrees.
        Extension will be negative"""
        # get points and reference frame
        hip, rmat = self.right_hip_referenceframe
        knee = self.right_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return 90 + angle

    @property
    def left_hip_abductionadduction(self):
        """
        Return the left hip abduction/adduction in degrees.
        Abduction will be positive, adduction negative.
        """
        # get points and reference frame
        hip, rmat = self.left_hip_referenceframe
        knee = self.left_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return angle + 90

    @property
    def right_hip_abductionadduction(self):
        """
        Return the right hip abduction/adduction in degrees.
        Abduction will be positive, adduction negative.
        """
        hip, rmat = self.right_hip_referenceframe
        knee = self.right_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # adjust the sign according to the knee position
        return -1 * (90 + angle)

    @property
    def left_hip_internalexternalrotation(self):
        """
        Return the left hip internal/external rotation in degrees.
        Internal rotation is positive, external rotation is negative.
        """

        # Get necessary parameters
        rmat = self.left_hip_referenceframe[1]
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

        # Consider transverse plane fixed to the generated reference frame
        cols = knee_lat.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)
        x, y = vr[:, col_map].T

        # Calculate angle of vector with respect to plane
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=angle, index=knee_lat.index, unit="°")

    @property
    def right_hip_internalexternalrotation(self):
        """
        Return the right hip internal/external rotation in degrees.
        Internal rotation is positive, external rotation is negative.
        """
        # Get necessary parameters
        rmat = self.right_hip_referenceframe[1]
        knee_lat = self._get_point("right_knee_lateral")
        knee_med = self._get_point("right_knee_medial")
        ankle_lat = self._get_point("right_ankle_lateral")
        ankle_med = self._get_point("right_ankle_medial")

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

        # Consider transverse plane fixed to the generated reference frame
        cols = knee_lat.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)
        x, y = vr[:, col_map].T

        # Calculate angle of vector with respect to plane
        angle = np.degrees(np.arctan2(y, x))

        # Correct angle sign
        angle = 180 - angle
        angle[y < 0] = angle[y < 0] - 360

        # Return angle
        return Signal1D(data=angle, index=knee_lat.index, unit="°")

    @property
    def pelvis_anteroposteriortilt_global(self):
        """
        Return the pelvis pitch (sagittal tilt) in degrees with respect to
        the global reference frame.
        Anterior tilt is negative, posterior tilt is positive.
        """

        # Get pelvis points projected into least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # Define vector determining anteroposterior axis of pelvis
        ap = ((l_asis + r_asis) / 2 - (l_psis + r_psis) / 2).to_numpy()

        # Consider only global sagittal plane
        cols = l_asis.columns
        axes_labels = [self.anteroposterior_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle
        x, y = ap[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=angle, index=l_asis.index, unit="°")

    @property
    def pelvis_lateraltilt_global(self):
        """
        Return the pelvis roll (frontal tilt) in degrees with respect to
        the global reference frame.
        Right tilt is positive, left tilt is negative.
        """

        # Get pelvis points projected into least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # Define vector determining lateral axis of pelvis
        ml = ((l_asis + l_psis) / 2 - (r_asis + r_psis) / 2).to_numpy()

        # Consider only global frontal plane
        cols = l_asis.columns
        axes_labels = [self.lateral_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle
        x, y = ml[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=angle, index=l_asis.index, unit="°")

    @property
    def pelvis_rotation_global(self):
        """
        Return the pelvis yaw (axial rotation) in degrees with respect
        to the global reference frame.
        Right rotation is positive, left rotation is negative.
        """

        # Get pelvis points projected into least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # Define vector determining lateral axis of pelvis
        ml = ((l_asis + l_psis) / 2 - (r_asis + r_psis) / 2).to_numpy()

        # Consider only global transverse plane
        cols = l_asis.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle
        x, y = ml[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=angle, index=l_asis.index, unit="°")

    @property
    def trunk_flexionextension_global(self):
        """
        Return the trunk flexion/extension in degrees with respect
        to the global reference frame.
        Flexion is positive, extension is negative.
        """

        # Get pelvis points projected into least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # Get vector defining spine (rachis)
        c7 = self._get_point("c7").to_numpy()
        base = ((l_psis + r_psis) / 2).to_numpy()
        vt = c7 - base

        # Consider only global sagittal plane
        cols = l_psis.columns
        axes_labels = [self.anteroposterior_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle
        x, y = vt[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=90 - angle, index=l_psis.index, unit="°")

    @property
    def trunk_lateralflexion_global(self):
        """
        Return the trunk lateral tilt (side bending) in degrees with respect
        to the global reference frame.
        Right tilt is negative, left tilt is positive.
        """

        # Get pelvis points projected into least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # Get vector defining spine (rachis)
        c7 = self._get_point("c7").to_numpy()
        base = ((l_psis + r_psis) / 2).to_numpy()
        vt = c7 - base

        # Consider only global frontal plane
        cols = l_psis.columns
        axes_labels = [self.lateral_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle
        x, y = vt[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=angle - 90, index=l_psis.index, unit="°")

    @property
    def trunk_rotation_global(self):
        """
        Return the trunk axial rotation in degrees with respect to the
        global reference frame.
        Right rotation is positive, left rotation is negative.
        """

        # Get vector defining shoulder axis
        vt = self.left_shoulder - self.right_shoulder

        # Consider only global transverse plane
        cols = vt.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle
        x, y = vt.copy().to_numpy()[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=angle, index=vt.index, unit="°")

    @property
    def trunk_rotation_local(self):
        """
        Return the shoulders axial rotation in degrees with respect to the
        pelvis local reference frame.
        Right rotation is positive, left rotation is negative.
        """
        return self.trunk_rotation_global - self.pelvis_rotation_global

    @property
    def neck_lateral_tilt(self):
        """
        Calculate neck lateral tilt (lateral flexion) in frontal plane.

        The angle represents the lateral deviation of the head from vertical.

        Interpretation
        --------------
        - **Positive (+)**: Right lateral tilt (inclinazione laterale destra)
          The head tilts toward the right shoulder.
        - **Negative (-)**: Left lateral tilt (inclinazione laterale sinistra)
          The head tilts toward the left shoulder.
        - **0°**: Neutral position (head centered, no lateral tilt)

        Calculation Method
        ------------------
        Measured as the angle between the head_center-to-neck_base vector
        and the vertical axis in the frontal (coronal) plane.

        Returns
        -------
        Signal1D
            Neck lateral tilt angle in degrees.
            Positive = right tilt (destra)
            Negative = left tilt (sinistra)
        """
        head = self.head_center
        neck = self.neck_base

        # Get neck vector (from neck base to head center)
        v_neck = (head - neck).to_numpy()

        # Extract frontal plane components (lateral_axis, vertical_axis)
        cols = head.columns
        axes_labels = [self.lateral_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle from vertical
        # x = lateral deviation, y = vertical component
        x, y = v_neck[:, col_map].T
        angle = np.degrees(np.arctan2(x, y))

        return Signal1D(data=angle, index=head.index, unit="°")

    @property
    def neck_flexionextension_local(self):
        """
        Calculate neck flexion/extension relative to upper trunk orientation.

        The angle represents neck movement relative to the upper thoracic spine.

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione cervicale)
          The head moves forward/anterior relative to the trunk.
          Chin approaches chest.
        - **Negative (-)**: Extension (estensione cervicale)
          The head moves backward/posterior relative to the trunk.
          Head tilts back, looking upward.
        - **0°**: Neutral position relative to trunk orientation

        Calculation Method
        ------------------
        Measured as the angular difference between:
        - Neck vector: head_center-to-neck_base
        - Trunk reference: C7-to-sc (sternoclavicular joint)
        Both projected onto the sagittal plane.

        This gives movement relative to trunk orientation, useful for
        analyzing neck posture independent of trunk lean.

        Returns
        -------
        Signal1D
            Neck flexion/extension angle in degrees relative to trunk.
            Positive = flexion (flessione, forward)
            Negative = extension (estensione, backward)
        """
        head = self.head_center
        neck = self.neck_base
        c7 = self._get_point("c7")
        sc = self._get_point("sc")

        # Get reference vector (C7 to sc) - upper trunk orientation
        v_ref = (c7 - sc).to_numpy()

        # Get neck vector (neck base to head center)
        v_neck = (head - neck).to_numpy()

        # Extract sagittal plane components (anteroposterior_axis, vertical_axis)
        cols = head.columns
        axes_labels = [self.anteroposterior_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angles from vertical in sagittal plane
        ref_angle = np.degrees(np.arctan2(v_ref[:, col_map[0]], v_ref[:, col_map[1]]))
        neck_angle = np.degrees(np.arctan2(v_neck[:, col_map[0]], v_neck[:, col_map[1]]))

        # Relative angle: positive when head is more anterior than trunk
        angle = neck_angle - ref_angle

        return Signal1D(data=angle, index=head.index, unit="°")

    @property
    def neck_flexionextension_global(self):
        """
        Calculate neck flexion/extension relative to global vertical axis.

        The angle represents absolute head/neck orientation in space.

        Interpretation
        --------------
        - **Positive (+)**: Forward head position (protrazione/flessione globale)
          The head is positioned anterior to vertical.
          Common in forward head posture.
        - **Negative (-)**: Backward head position (retrazione/estensione globale)
          The head is positioned posterior to vertical.
          Head tilted back.
        - **0°**: Neutral position (head directly above neck base, vertical alignment)

        Calculation Method
        ------------------
        Measured as the angle between the head_center-to-neck_base vector
        and the global vertical axis in the sagittal plane.

        This represents absolute neck orientation in space, independent of
        trunk position. Useful for postural analysis and ergonomic assessment.

        Note
        ----
        This differs from neck_flexionextension_local which measures neck
        position relative to trunk. A person leaning forward may have
        neck_flexionextension_global = +30° (head forward in space) but
        neck_flexionextension_local = 0° (neck aligned with trunk).

        Returns
        -------
        Signal1D
            Neck flexion/extension angle in degrees relative to vertical.
            Positive = forward/anterior position
            Negative = backward/posterior position
        """
        head = self.head_center
        neck = self.neck_base

        # Get neck vector (from neck base to head center)
        v_neck = (head - neck).to_numpy()

        # Extract sagittal plane components (anteroposterior_axis, vertical_axis)
        cols = head.columns
        axes_labels = [self.anteroposterior_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle from vertical
        # x = anteroposterior deviation, y = vertical component
        x, y = v_neck[:, col_map].T
        angle = np.degrees(np.arctan2(x, y))

        return Signal1D(data=angle, index=head.index, unit="°")

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

        Clinical Relevance
        ------------------
        - Hyperlordosis (angle < 140°): Associated with anterior pelvic tilt,
          weak abdominals, tight hip flexors
        - Hypolordosis (angle > 160°): Associated with posterior pelvic tilt,
          tight hamstrings, reduced shock absorption

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
        psis_mid_index = np.unique(np.concatenate([l_psis.index, r_psis.index])).tolist()

        # Create a Point3D for psis_mid
        psis_mid = Point3D(
            data=psis_mid_data,
            index=psis_mid_index,
            columns=l_psis.columns,
        )

        # Calculate 3-point angle: PSIS_mid - L2 - T5
        # Internal angle at L2 vertex
        angle = self._get_angle_between_three_points(psis_mid, l2, t5)

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

        Clinical Relevance
        ------------------
        - Hyperkyphosis (angle < 140°): Associated with:
          * Forward head posture
          * Rounded shoulders
          * Weak upper back extensors
          * Tight pectorals
          * Scheuermann's disease (in adolescents)
          * Postural kyphosis
        - Hypokyphosis (angle > 160°): Associated with:
          * Flat thoracic spine
          * Reduced shock absorption
          * Increased load on intervertebral discs

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

        # Calculate 3-point angle: L2 - T5 - C7
        # Internal angle at T5 vertex
        angle = self._get_angle_between_three_points(l2, t5, c7)

        return Signal1D(data=angle, index=t5.index, unit="°")

    @property
    def shoulder_lateraltilt_global(self):
        """
        Return the shoulders roll (frontal tilt) in degrees with respect to
        the global reference frame.
        Right tilt is positive, left tilt is negative.
        """

        # Define vector determining shoulder orientation
        ml = self.left_shoulder - self.right_shoulder

        # Consider only global frontal plane
        cols = ml.columns
        axes_labels = [self.lateral_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle
        x, y = ml.copy().to_numpy()[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=angle, index=ml.index, unit="°")

    @property
    def shoulder_lateraltilt_local(self):
        """
        Return the shoulders roll (frontal tilt) in degrees with respect to
        the spine lateral tilt.
        Right tilt is positive, left tilt is negative.
        """
        return self.shoulder_lateral_tilt_global - self.trunk_lateralflexion_global

    @property
    def left_shoulder_flexionextension(self):
        """
        Return the left shoulder flexion/extension in degrees with respect
        to the neck base.
        Positive values denote flexion, while negative values indicate
        extension.
        """
        # ottengo i parametri necessari
        shoulder, rmat = self.left_shoulder_referenceframe
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

        return 90 + angle

    @property
    def right_shoulder_flexionextension(self):
        """
        Return the right shoulder flexion/extension in degrees with respect
        to the neck base.
        Positive values denote flexion, while negative values indicate
        extension.
        """
        # ottengo i parametri necessari
        shoulder, rmat = self.right_shoulder_referenceframe
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

        return 90 + angle

    @property
    def left_shoulder_abductionadduction(self):
        """
        Return the left shoulder abduction/adduction in degrees.
        Abduction will be positive, adduction negative.
        """
        # Get necessary parameters
        shoulder, rmat = self.left_shoulder_referenceframe
        elbow = self.left_elbow

        # Calculate arm orientation with respect to shoulder reference frame
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow,
            shoulder,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return 90 + angle

    @property
    def right_shoulder_abductionadduction(self):
        """
        Return the right shoulder abduction/adduction in degrees.
        Abduction will be positive, adduction negative.
        """
        # Get necessary parameters
        shoulder, rmat = self.right_shoulder_referenceframe
        elbow = self.right_elbow

        # Calculate arm orientation with respect to shoulder reference frame
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow,
            shoulder,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # Correct angle sign
        # TODO [RMSIN-448]: Need to correct case of angles greater than 90 degrees
        return -1 * (90 + angle)

    @property
    def left_shoulder_internalexternalrotation(self):
        """
        Return the left shoulder internal/external rotation in degrees.
        Internal rotation is positive, external rotation is negative.
        """
        # Get necessary parameters
        elbow_lat = self._get_point("left_elbow_lateral")
        elbow_med = self._get_point("left_elbow_medial")
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()
        base = (r_psis + l_psis) / 2
        c7 = self._get_point("c7")
        pelvis_center = self.pelvis_center
        shoulder = self.left_shoulder
        proj = self._get_projection_point_onto_axis(
            shoulder,
            pelvis_center,
            c7 - base,
        )

        # Get reference vector
        va = (elbow_lat - elbow_med).to_numpy()

        # Determine rotation matrix for reference frame
        i = (shoulder - proj).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        k = (shoulder - self.left_elbow).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))

        # Align vector to new reference frame
        vr = np.einsum("nij,nj->ni", rmat, va)

        # Consider transverse plane fixed to the generated reference frame
        cols = elbow_lat.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)
        x, y = vr[:, col_map].T

        # Calculate angle of vector with respect to plane
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=angle, index=elbow_lat.index, unit="°")

    @property
    def right_shoulder_internalexternalrotation(self):
        """
        Return the right shoulder internal/external rotation in degrees.
        Internal rotation is positive, external rotation is negative.
        """
        # Get necessary parameters
        elbow_lat = self._get_point("right_elbow_lateral")
        elbow_med = self._get_point("right_elbow_medial")
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()
        base = (r_psis + l_psis) / 2
        c7 = self._get_point("c7")
        pelvis_center = self.pelvis_center
        shoulder = self.right_shoulder
        proj = self._get_projection_point_onto_axis(
            shoulder,
            pelvis_center,
            c7 - base,
        )

        # Get reference vector
        va = (elbow_med - elbow_lat).to_numpy()

        # Determine rotation matrix for reference frame
        i = (proj - shoulder).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        k = (shoulder - self.right_elbow).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))

        # Align vector to new reference frame
        vr = np.einsum("nij,nj->ni", rmat, va)

        # Consider transverse plane fixed to the generated reference frame
        cols = elbow_lat.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)
        x, y = vr[:, col_map].T

        # Calculate angle of vector with respect to plane
        angle = np.degrees(np.arctan2(y, x))

        # Return angle
        return Signal1D(data=-angle, index=elbow_lat.index, unit="°")

    @property
    def left_elbow_flexionextension(self):
        """return the left elbow flexion in degrees. Extension will be negative"""
        p1 = self.left_wrist
        p2 = self.left_elbow
        p3 = self.left_shoulder
        return Signal1D(
            data=180 - self._get_angle_between_three_points(p1, p2, p3),
            index=p1.index,
            unit="°",
        )

    @property
    def right_elbow_flexionextension(self):
        """return the right elbow flexion in degrees. Extension will be negative"""
        p1 = self.right_wrist
        p2 = self.right_elbow
        p3 = self.right_shoulder
        return Signal1D(
            data=180 - self._get_angle_between_three_points(p1, p2, p3),
            index=p1.index,
            unit="°",
        )

    def copy(self):
        return WholeBody(**{i: v.copy() for i, v in self.items()})  # type: ignore
