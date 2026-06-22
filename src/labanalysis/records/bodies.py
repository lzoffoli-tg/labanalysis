"""full body module"""

from typing import Literal

import numpy as np
import pandas as pd

from ..signalprocessing import gram_schmidt
from .records import *
from .timeseries import *

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
        "left_scapular_protractionretraction",
        "right_scapular_protractionretraction",
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
        Calculate left ankle anatomical reference frame.

        The reference frame is centered at the left ankle joint center with
        axes aligned to anatomical directions.

        Calculation Method
        ------------------
        - **Origin**: Left ankle joint center (midpoint of medial/lateral malleoli)
        - **X-axis**: Points laterally (from medial to lateral malleolus)
        - **Y-axis**: Points superiorly (from ankle to knee)
        - **Z-axis**: Points anteriorly (cross product of X and Y)

        The rotation matrix is computed using Gram-Schmidt orthonormalization
        to ensure orthogonal axes.

        Returns
        -------
        tuple[Point3D, np.ndarray]
            Origin point (N, 3) and rotation matrix (N, 3, 3).
            Transform points to this reference frame via:
            `new = np.einsum("nij,nj->ni", R, old - O)`

        See Also
        --------
        right_ankle_referenceframe : Right ankle reference frame
        left_ankle_flexionextension : Ankle angle calculated in this frame
        left_ankle_inversioneversion : Ankle angle calculated in this frame
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
        Calculate right ankle anatomical reference frame.

        The reference frame is centered at the right ankle joint center with
        axes aligned to anatomical directions.

        Calculation Method
        ------------------
        - **Origin**: Right ankle joint center (midpoint of medial/lateral malleoli)
        - **X-axis**: Points laterally (from lateral to medial malleolus)
        - **Y-axis**: Points superiorly (from ankle to knee)
        - **Z-axis**: Points anteriorly (cross product of X and Y)

        The rotation matrix is computed using Gram-Schmidt orthonormalization
        to ensure orthogonal axes.

        Returns
        -------
        tuple[Point3D, np.ndarray]
            Origin point (N, 3) and rotation matrix (N, 3, 3).
            Transform points to this reference frame via:
            `new = np.einsum("nij,nj->ni", R, old - O)`

        See Also
        --------
        left_ankle_referenceframe : Left ankle reference frame
        right_ankle_flexionextension : Ankle angle calculated in this frame
        right_ankle_inversioneversion : Ankle angle calculated in this frame
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
        Calculate left hip anatomical reference frame.

        The reference frame is centered at the left hip joint center with
        axes aligned to the pelvis reference frame.

        Calculation Method
        ------------------
        - **Origin**: Left hip joint center (estimated using De Leva 1996)
        - **Axes**: Aligned with pelvis reference frame (shared rotation matrix)

        Returns
        -------
        tuple[Point3D, np.ndarray]
            Origin point (N, 3) and rotation matrix (N, 3, 3).
            Transform points to this reference frame via:
            `new = np.einsum("nij,nj->ni", R, old - O)`

        See Also
        --------
        right_hip_referenceframe : Right hip reference frame
        pelvis_referenceframe : Pelvis reference frame (shared rotation)
        left_hip_flexionextension : Hip angle calculated in this frame
        """
        return self.left_hip, self.pelvis_referenceframe[1]

    @property
    def right_hip_referenceframe(self):
        """
        Calculate right hip anatomical reference frame.

        The reference frame is centered at the right hip joint center with
        axes aligned to the pelvis reference frame.

        Calculation Method
        ------------------
        - **Origin**: Right hip joint center (estimated using De Leva 1996)
        - **Axes**: Aligned with pelvis reference frame (shared rotation matrix)

        Returns
        -------
        tuple[Point3D, np.ndarray]
            Origin point (N, 3) and rotation matrix (N, 3, 3).
            Transform points to this reference frame via:
            `new = np.einsum("nij,nj->ni", R, old - O)`

        See Also
        --------
        left_hip_referenceframe : Left hip reference frame
        pelvis_referenceframe : Pelvis reference frame (shared rotation)
        right_hip_flexionextension : Hip angle calculated in this frame
        """
        return self.right_hip, self.pelvis_referenceframe[1]

    @property
    def pelvis_referenceframe(self):
        """
        Calculate pelvis anatomical reference frame.

        The reference frame is centered at the pelvis centroid with axes
        aligned to the pelvis anatomical landmarks.

        Calculation Method
        ------------------
        - **Origin**: Centroid of four pelvis landmarks (ASIS and PSIS, left and right)
        - **X-axis (i)**: Points laterally (from pelvis center to left side midpoint)
        - **Y-axis (j)**: Points superiorly (perpendicular to pelvis plane)
        - **Z-axis (k)**: Points anteriorly (from pelvis center to ASIS midpoint)

        All points are first projected onto the pelvis least-squares plane
        to ensure planar calculation. The rotation matrix is computed using
        Gram-Schmidt orthonormalization.

        Returns
        -------
        tuple[Point3D, np.ndarray]
            Origin point (N, 3) and rotation matrix (N, 3, 3).
            Transform points to this reference frame via:
            `new = np.einsum("nij,nj->ni", R, old - O)`

        See Also
        --------
        left_hip_referenceframe : Left hip reference frame (uses this rotation)
        right_hip_referenceframe : Right hip reference frame (uses this rotation)
        pelvis_anteroposteriortilt_global : Pelvis tilt calculated using this frame
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
        Calculate left shoulder anatomical reference frame.

        The reference frame is centered at the left shoulder joint center with
        axes aligned to trunk orientation.

        Calculation Method
        ------------------
        - **Origin**: Left shoulder joint center
        - **X-axis (i)**: Points laterally (from trunk axis to shoulder)
        - **Y-axis (j)**: Points superiorly (from pelvis base to C7)
        - **Z-axis**: Points anteriorly (cross product of X and Y)

        The shoulder is projected onto the vertical trunk axis (pelvis center
        to C7), and the lateral axis is defined as the vector from this
        projection to the shoulder. The rotation matrix is computed using
        Gram-Schmidt orthonormalization.

        Returns
        -------
        tuple[Point3D, np.ndarray]
            Origin point (N, 3) and rotation matrix (N, 3, 3).
            Transform points to this reference frame via:
            `new = np.einsum("nij,nj->ni", R, old - O)`

        See Also
        --------
        right_shoulder_referenceframe : Right shoulder reference frame
        left_shoulder_flexionextension : Shoulder angle calculated in this frame
        left_shoulder_abductionadduction : Shoulder angle calculated in this frame
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
        Calculate right shoulder anatomical reference frame.

        The reference frame is centered at the right shoulder joint center with
        axes aligned to trunk orientation.

        Calculation Method
        ------------------
        - **Origin**: Right shoulder joint center
        - **X-axis (i)**: Points laterally (from shoulder to trunk axis)
        - **Y-axis (j)**: Points superiorly (from pelvis base to C7)
        - **Z-axis**: Points anteriorly (cross product of X and Y)

        The shoulder is projected onto the vertical trunk axis (pelvis center
        to C7), and the lateral axis is defined as the vector from the shoulder
        to this projection. The rotation matrix is computed using Gram-Schmidt
        orthonormalization.

        Returns
        -------
        tuple[Point3D, np.ndarray]
            Origin point (N, 3) and rotation matrix (N, 3, 3).
            Transform points to this reference frame via:
            `new = np.einsum("nij,nj->ni", R, old - O)`

        See Also
        --------
        left_shoulder_referenceframe : Left shoulder reference frame
        right_shoulder_flexionextension : Shoulder angle calculated in this frame
        right_shoulder_abductionadduction : Shoulder angle calculated in this frame
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
            Neck base point (average of sc and c7 markers).
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
            Distance in meters from ankle joint to foot plane.
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
            Distance in meters from ankle joint to foot plane.
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
        Measured as the angle between the shank (knee-to-ankle) and foot
        (ankle-to-toe) vectors in the sagittal plane. The foot plane is
        defined by heel, toe, and metatarsal markers. The ankle reference
        frame is used to project the foot orientation onto the sagittal plane.

        Clinical Relevance
        ------------------
        - Limited dorsiflexion (< 10°): Associated with:
          * Tight gastrocnemius/soleus
          * Compensatory knee valgus
          * Increased fall risk in elderly
        - Excessive plantarflexion (> 50° at toe-off): Associated with:
          * Forefoot running pattern
          * Calf dominance
          * Achilles tendon stress

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
        Measured as the angle between the shank (knee-to-ankle) and foot
        (ankle-to-toe) vectors in the sagittal plane. The foot plane is
        defined by heel, toe, and metatarsal markers. The ankle reference
        frame is used to project the foot orientation onto the sagittal plane.

        Clinical Relevance
        ------------------
        - Limited dorsiflexion (< 10°): Associated with:
          * Tight gastrocnemius/soleus
          * Compensatory knee valgus
          * Increased fall risk in elderly
        - Excessive plantarflexion (> 50° at toe-off): Associated with:
          * Forefoot running pattern
          * Calf dominance
          * Achilles tendon stress

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
        Measured as the angle between the shank and foot vectors projected
        onto the frontal plane. The foot plane is defined by heel, toe, and
        metatarsal markers. The ankle reference frame is used to isolate
        frontal plane motion using the lateral and vertical axes.

        Clinical Relevance
        ------------------
        - Excessive inversion (< -10°): Associated with:
          * Lateral ankle instability
          * Increased lateral ankle sprain risk
          * Pes cavus (high arches)
        - Excessive eversion (> +10°): Associated with:
          * Medial tibial stress syndrome
          * Achilles tendinopathy
          * Pes planus (flat feet)

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
        Measured as the angle between the shank and foot vectors projected
        onto the frontal plane. The foot plane is defined by heel, toe, and
        metatarsal markers. The ankle reference frame is used to isolate
        frontal plane motion using the lateral and vertical axes.

        Clinical Relevance
        ------------------
        - Excessive inversion (< -10°): Associated with:
          * Lateral ankle instability
          * Increased lateral ankle sprain risk
          * Pes cavus (high arches)
        - Excessive eversion (> +10°): Associated with:
          * Medial tibial stress syndrome
          * Achilles tendinopathy
          * Pes planus (flat feet)

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
        Measured as the angle between the thigh (hip-to-knee) and shank
        (knee-to-ankle) vectors. Calculated as 180° minus the angle formed
        by the three points (hip, knee, ankle), so that flexion is positive
        and full extension is zero.

        Clinical Relevance
        ------------------
        - Limited flexion (< 120°): Associated with:
          * Knee joint stiffness
          * Quadriceps or hamstring tightness
          * Difficulty in squatting or climbing stairs
        - Hyperextension (negative values): Associated with:
          * Genu recurvatum
          * Posterior knee instability
          * Increased ACL injury risk

        Returns
        -------
        Signal1D
            Knee flexion/extension angle in degrees.
            Positive = flexion (bent knee)
            Negative = extension (hyperextension)

        See Also
        --------
        right_knee_flexionextension : Right knee flexion angle
        left_knee_varusvalgus : Left knee frontal plane alignment
        left_hip_flexionextension : Left hip flexion angle
        """
        p1 = self.left_hip
        p2 = self.left_knee
        p3 = self.left_ankle
        angle = 180 - self._get_angle_between_three_points(p1, p2, p3)
        return Signal1D(data=angle, index=p1.index, unit="°")

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
        Measured as the angle between the thigh (hip-to-knee) and shank
        (knee-to-ankle) vectors. Calculated as 180° minus the angle formed
        by the three points (hip, knee, ankle), so that flexion is positive
        and full extension is zero.

        Clinical Relevance
        ------------------
        - Limited flexion (< 120°): Associated with:
          * Knee joint stiffness
          * Quadriceps or hamstring tightness
          * Difficulty in squatting or climbing stairs
        - Hyperextension (negative values): Associated with:
          * Genu recurvatum
          * Posterior knee instability
          * Increased ACL injury risk

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
        - **Positive (+)**: Varus deformity (ginocchio varo, "a parentesi", bow-legged)
          The knee deviates laterally; the leg angle opens medially.
        - **Negative (-)**: Valgus deformity (ginocchio valgo, "a X", knock-knee)
          The knee deviates medially; the leg angle opens laterally.
        - **0°**: Neutral alignment (anca-ginocchio-caviglia collineari nel piano frontale)

        Calculation Method
        ------------------
        Measured as the difference between leg and thigh angles in the frontal plane:
        - Thigh angle: angle of hip-to-knee vector from vertical
        - Leg angle: angle of knee-to-ankle vector from vertical
        - Varus/Valgus = (Leg angle) - (Thigh angle)

        Returns
        -------
        Signal1D
            Knee varus/valgus angle in degrees.
            Positive = varus (ginocchio varo)
            Negative = valgus (ginocchio valgo)
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
        # Positive when knee deviates laterally (varus)
        # Negative when knee deviates medially (valgus)
        angle = angle_leg - angle_thigh

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
        Measured as the difference between leg and thigh angles in the frontal plane.
        Sign convention: positive = varus (lateral deviation) for both sides.

        Returns
        -------
        Signal1D
            Knee varus/valgus angle in degrees.
            Positive = varus (ginocchio varo)
            Negative = valgus (ginocchio valgo)
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

        # Varus/valgus is the difference
        # Positive when knee deviates laterally (varus)
        # Negative when knee deviates medially (valgus)
        angle = angle_thigh - angle_leg

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
        Measured as the angle between the thigh (hip-to-knee vector) and
        the vertical axis in the hip reference frame, projected onto the
        sagittal plane (anteroposterior and vertical axes).

        Clinical Relevance
        ------------------
        - Limited flexion (< 90°): Associated with:
          * Hip flexor weakness (iliopsoas)
          * Hip joint stiffness or arthritis
          * Difficulty in stair climbing
        - Limited extension (unable to reach negative values): Associated with:
          * Hip flexor tightness (iliopsoas, rectus femoris)
          * Anterior pelvic tilt compensation
          * Reduced stride length in gait

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
        hip, rmat = self.left_hip_referenceframe
        knee = self.left_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return 90 - angle

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
        Measured as the angle between the thigh (hip-to-knee vector) and
        the vertical axis in the hip reference frame, projected onto the
        sagittal plane (anteroposterior and vertical axes).

        Clinical Relevance
        ------------------
        - Limited flexion (< 90°): Associated with:
          * Hip flexor weakness (iliopsoas)
          * Hip joint stiffness or arthritis
          * Difficulty in stair climbing
        - Limited extension (unable to reach negative values): Associated with:
          * Hip flexor tightness (iliopsoas, rectus femoris)
          * Anterior pelvic tilt compensation
          * Reduced stride length in gait

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
        hip, rmat = self.right_hip_referenceframe
        knee = self.right_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.anteroposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return 90 - angle

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
        Measured as the angle between the thigh (hip-to-knee vector) and
        the vertical axis in the hip reference frame, projected onto the
        frontal plane (lateral and vertical axes).

        Clinical Relevance
        ------------------
        - Excessive abduction (> 15°): Associated with:
          * Trendelenburg gait pattern
          * Hip abductor weakness (gluteus medius)
          * Increased lateral hip stress
        - Excessive adduction (< -10°): Associated with:
          * Dynamic knee valgus (knee collapse)
          * Increased ACL injury risk
          * IT band syndrome

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
        hip, rmat = self.left_hip_referenceframe
        knee = self.left_knee
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            knee,
            hip,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        return 90 - angle

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
        Measured as the angle between the thigh (hip-to-knee vector) and
        the vertical axis in the hip reference frame, projected onto the
        frontal plane (lateral and vertical axes).

        Clinical Relevance
        ------------------
        - Excessive abduction (> 15°): Associated with:
          * Trendelenburg gait pattern
          * Hip abductor weakness (gluteus medius)
          * Increased lateral hip stress
        - Excessive adduction (< -10°): Associated with:
          * Dynamic knee valgus (knee collapse)
          * Increased ACL injury risk
          * IT band syndrome

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
        return (angle - 90)

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
        Measured using the orientation of the thigh's frontal plane
        (defined by medial-lateral knee and ankle markers) relative to
        the hip reference frame. The average vector from knee and ankle
        medial-lateral markers is projected onto the transverse plane
        fixed to the thigh reference frame.

        Clinical Relevance
        ------------------
        - Excessive internal rotation (> 10°): Associated with:
          * Femoral anteversion
          * Dynamic knee valgus
          * Patellofemoral pain syndrome
        - Excessive external rotation (< -15°): Associated with:
          * Femoral retroversion
          * Hip joint impingement
          * Piriformis syndrome

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
        Measured using the orientation of the thigh's frontal plane
        (defined by medial-lateral knee and ankle markers) relative to
        the hip reference frame. The average vector from knee and ankle
        medial-lateral markers is projected onto the transverse plane
        fixed to the thigh reference frame.

        Clinical Relevance
        ------------------
        - Excessive internal rotation (> 10°): Associated with:
          * Femoral anteversion
          * Dynamic knee valgus
          * Patellofemoral pain syndrome
        - Excessive external rotation (< -15°): Associated with:
          * Femoral retroversion
          * Hip joint impingement
          * Piriformis syndrome

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
        Calculate pelvis anterior/posterior tilt (pitch) in sagittal plane.

        The angle represents the forward (anterior) or backward (posterior)
        tilting of the pelvis relative to the global vertical, measured in
        the sagittal plane.

        Interpretation
        --------------
        - **Positive (+)**: Posterior tilt (retroversione del bacino)
          The pelvis rotates backward; ASIS moves up, PSIS moves down.
          Common in posterior pelvic tilt posture, flat back.
        - **Negative (-)**: Anterior tilt (anteroversione del bacino)
          The pelvis rotates forward; ASIS moves down, PSIS moves up.
          Common in hyperlordotic posture, arched back.
        - **0°**: Neutral position (ASIS and PSIS horizontally aligned)

        Calculation Method
        ------------------
        Measured as the angle between the pelvis anteroposterior axis
        (from PSIS midpoint to ASIS midpoint) and the horizontal plane.
        Uses projected pelvis points onto the pelvis least-squares plane
        to ensure planar calculation.

        Clinical Relevance
        ------------------
        - Excessive anterior tilt (< -15°): Associated with:
          * Lumbar hyperlordosis
          * Hip flexor tightness
          * Lower back pain
        - Excessive posterior tilt (> 15°): Associated with:
          * Lumbar hypolordosis (flat back)
          * Hamstring dominance
          * Hip extension limitation

        Returns
        -------
        Signal1D
            Pelvis tilt angle in degrees.
            Positive = posterior tilt
            Negative = anterior tilt

        See Also
        --------
        pelvis_lateraltilt_global : Pelvis frontal plane tilt
        pelvis_rotation_global : Pelvis transverse plane rotation
        trunk_flexionextension_global : Trunk sagittal plane flexion
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
        Calculate pelvis lateral tilt (roll) in frontal plane.

        The angle represents the left or right side tilting of the pelvis
        relative to the global horizontal, measured in the frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Right tilt (inclinazione destra del bacino)
          The right side of the pelvis drops lower than the left.
          Common in right hip drop, Trendelenburg gait.
        - **Negative (-)**: Left tilt (inclinazione sinistra del bacino)
          The left side of the pelvis drops lower than the right.
          Common in left hip drop.
        - **0°**: Neutral position (pelvis level, ASIS points horizontally aligned)

        Calculation Method
        ------------------
        Measured as the angle between the pelvis lateral axis (from right
        pelvis midpoint to left pelvis midpoint) and the horizontal plane.
        Uses projected pelvis points onto the pelvis least-squares plane.

        Clinical Relevance
        ------------------
        - Excessive lateral tilt (> 5-10°): Associated with:
          * Hip abductor weakness (gluteus medius)
          * Leg length discrepancy
          * Scoliosis compensation
          * Trendelenburg sign

        Returns
        -------
        Signal1D
            Pelvis lateral tilt angle in degrees.
            Positive = right tilt (right hip drop)
            Negative = left tilt (left hip drop)

        See Also
        --------
        pelvis_anteroposteriortilt_global : Pelvis sagittal plane tilt
        pelvis_rotation_global : Pelvis transverse plane rotation
        trunk_lateralflexion_global : Trunk frontal plane flexion
        """

        # Get pelvis points projected into least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # Define vector determining lateral axis of pelvis (left to right)
        ml = ((r_asis + r_psis) / 2 - (l_asis + l_psis) / 2).to_numpy()

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
        Calculate pelvis axial rotation (yaw) in transverse plane.

        The angle represents the left or right rotational orientation of
        the pelvis relative to the global forward direction, measured in
        the transverse (horizontal) plane.

        Interpretation
        --------------
        - **Positive (+)**: Right rotation (rotazione destra del bacino)
          The pelvis rotates clockwise (viewed from above).
          The left hip moves forward relative to the right hip.
        - **Negative (-)**: Left rotation (rotazione sinistra del bacino)
          The pelvis rotates counterclockwise (viewed from above).
          The right hip moves forward relative to the left hip.
        - **0°**: Neutral position (pelvis aligned with forward direction)

        Calculation Method
        ------------------
        Measured as the angle between the pelvis lateral axis (from right
        pelvis midpoint to left pelvis midpoint) and the global lateral
        axis, projected onto the transverse plane (horizontal).

        Clinical Relevance
        ------------------
        - Excessive rotation (> 10-15°): Associated with:
          * Asymmetric gait pattern
          * Hip mobility asymmetry
          * Compensatory trunk rotation

        Returns
        -------
        Signal1D
            Pelvis rotation angle in degrees.
            Positive = right rotation
            Negative = left rotation

        See Also
        --------
        pelvis_anteroposteriortilt_global : Pelvis sagittal plane tilt
        pelvis_lateraltilt_global : Pelvis frontal plane tilt
        trunk_rotation_global : Trunk transverse plane rotation
        """

        # Get pelvis points projected into least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # Define vector determining lateral axis of pelvis (left to right)
        ml = ((r_asis + r_psis) / 2 - (l_asis + l_psis) / 2).to_numpy()

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
        Calculate trunk flexion/extension angle in sagittal plane.

        The angle represents the forward (flexion) or backward (extension)
        bending of the trunk relative to the global vertical, measured from
        pelvis base to C7 (seventh cervical vertebra).

        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione del tronco)
          The trunk bends forward; C7 moves anteriorly.
          Common in bending, squatting, sitting.
        - **Negative (-)**: Extension (estensione del tronco)
          The trunk bends backward; C7 moves posteriorly.
          Common in arching, back bending.
        - **0°**: Neutral position (trunk vertical, C7 directly above pelvis base)

        Calculation Method
        ------------------
        Measured as the angle between the spine vector (from PSIS midpoint
        to C7) and the global vertical axis, projected onto the sagittal plane.

        Clinical Relevance
        ------------------
        - Excessive flexion (> 45° during activities): Associated with:
          * Increased lumbar disc pressure
          * Back extensor fatigue
          * Increased injury risk in lifting
        - Limited flexion (< 60° maximum): Associated with:
          * Thoracic or lumbar stiffness
          * Hamstring tightness
          * Reduced functional mobility

        Returns
        -------
        Signal1D
            Trunk flexion/extension angle in degrees.
            Positive = flexion (trunk forward)
            Negative = extension (trunk backward)

        See Also
        --------
        trunk_lateralflexion_global : Trunk frontal plane flexion
        trunk_rotation_global : Trunk transverse plane rotation
        pelvis_anteroposteriortilt_global : Pelvis sagittal plane tilt
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
        Calculate trunk lateral flexion (side bending) in frontal plane.

        The angle represents the left or right side bending of the trunk
        relative to the global vertical, measured from pelvis base to C7.

        Interpretation
        --------------
        - **Positive (+)**: Left lateral flexion (flessione laterale sinistra)
          The trunk bends to the left; C7 moves laterally left.
          Common in left side-bending movements.
        - **Negative (-)**: Right lateral flexion (flessione laterale destra)
          The trunk bends to the right; C7 moves laterally right.
          Common in right side-bending movements.
        - **0°**: Neutral position (trunk vertical, C7 centered over pelvis)

        Calculation Method
        ------------------
        Measured as the angle between the spine vector (from PSIS midpoint
        to C7) and the global vertical axis, projected onto the frontal plane.

        Clinical Relevance
        ------------------
        - Excessive lateral flexion (> 10° static): Associated with:
          * Scoliosis
          * Trunk muscle imbalance
          * Lateral pelvic tilt compensation
        - Asymmetric range of motion: Associated with:
          * Unilateral muscle tightness
          * Spinal mobility restrictions

        Returns
        -------
        Signal1D
            Trunk lateral flexion angle in degrees.
            Positive = left lateral flexion
            Negative = right lateral flexion

        See Also
        --------
        trunk_flexionextension_global : Trunk sagittal plane flexion
        trunk_rotation_global : Trunk transverse plane rotation
        pelvis_lateraltilt_global : Pelvis frontal plane tilt
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
        Calculate trunk axial rotation in transverse plane.

        The angle represents the left or right rotational orientation of
        the shoulders (upper trunk) relative to the global forward direction,
        measured in the transverse (horizontal) plane.

        Interpretation
        --------------
        - **Positive (+)**: Right rotation (rotazione destra del tronco)
          The trunk rotates clockwise (viewed from above).
          The left shoulder moves forward relative to the right shoulder.
        - **Negative (-)**: Left rotation (rotazione sinistra del tronco)
          The trunk rotates counterclockwise (viewed from above).
          The right shoulder moves forward relative to the left shoulder.
        - **0°**: Neutral position (shoulders aligned with forward direction)

        Calculation Method
        ------------------
        Measured as the angle between the shoulder axis (from right shoulder
        to left shoulder) and the global lateral axis, projected onto the
        transverse plane (horizontal).

        Clinical Relevance
        ------------------
        - Excessive rotation (> 15° static): Associated with:
          * Scoliosis
          * Asymmetric shoulder positioning
          * Trunk muscle imbalances

        Returns
        -------
        Signal1D
            Trunk rotation angle in degrees.
            Positive = right rotation
            Negative = left rotation

        See Also
        --------
        trunk_flexionextension_global : Trunk sagittal plane flexion
        trunk_lateralflexion_global : Trunk frontal plane flexion
        trunk_rotation_local : Trunk rotation relative to pelvis
        pelvis_rotation_global : Pelvis transverse plane rotation
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
        Calculate trunk axial rotation relative to pelvis in transverse plane.

        The angle represents the rotational separation between the shoulders
        (upper trunk) and the pelvis, indicating counter-rotation or
        co-rotation patterns.

        Interpretation
        --------------
        - **Positive (+)**: Right rotation of trunk relative to pelvis
          The shoulders rotate right more than (or left less than) the pelvis.
          Common in gait swing phase, throwing motions.
        - **Negative (-)**: Left rotation of trunk relative to pelvis
          The shoulders rotate left more than (or right less than) the pelvis.
        - **0°**: No relative rotation (trunk and pelvis aligned)

        Calculation Method
        ------------------
        Calculated as the difference between trunk_rotation_global and
        pelvis_rotation_global, isolating trunk rotation relative to
        the pelvis reference frame.

        Clinical Relevance
        ------------------
        - Limited relative rotation (< 20° in gait): Associated with:
          * Reduced counter-rotation
          * Trunk stiffness
          * Compensatory pelvic rotation
        - Excessive relative rotation (> 40°): Associated with:
          * Thoracic hypermobility
          * Lumbar rotation stress

        Returns
        -------
        Signal1D
            Trunk rotation relative to pelvis in degrees.
            Positive = right rotation relative to pelvis
            Negative = left rotation relative to pelvis

        See Also
        --------
        trunk_rotation_global : Trunk rotation in global frame
        pelvis_rotation_global : Pelvis rotation in global frame
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
        neck_angle = np.degrees(
            np.arctan2(v_neck[:, col_map[0]], v_neck[:, col_map[1]])
        )

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

        # Calculate 3-point angle: C7 - T5 - L2
        # Internal angle at T5 vertex (order: superior → vertex → inferior)
        angle = self._get_angle_between_three_points(c7, t5, l2)

        return Signal1D(data=angle, index=t5.index, unit="°")

    @property
    def shoulder_lateraltilt_global(self):
        """
        Calculate shoulders lateral tilt (roll) in frontal plane.

        The angle represents the left or right tilting of the shoulder girdle
        relative to the global horizontal, measured in the frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Right shoulder elevation
          The left shoulder is lower than the right.
          Common in right shoulder shrugging, right side lean.
        - **Negative (-)**: Left shoulder elevation
          The right shoulder is lower than the left.
          Common in left shoulder shrugging, left side lean.
        - **0°**: Neutral position (shoulders level)

        Calculation Method
        ------------------
        Measured as the angle between the shoulder axis (from right shoulder
        to left shoulder) and the global horizontal, projected onto the
        frontal plane (lateral and vertical axes).

        Clinical Relevance
        ------------------
        - Persistent asymmetry (> 5°): Associated with:
          * Scoliosis
          * Shoulder muscle imbalance
          * Scapular dyskinesis

        Returns
        -------
        Signal1D
            Shoulder tilt angle in degrees.
            Positive = right shoulder higher
            Negative = left shoulder higher

        See Also
        --------
        shoulder_lateraltilt_local : Shoulder tilt relative to trunk
        trunk_lateralflexion_global : Trunk lateral flexion
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
        Calculate shoulders lateral tilt relative to trunk in frontal plane.

        The angle represents shoulder tilt independent of trunk lean,
        isolating shoulder girdle orientation relative to the spine.

        Interpretation
        --------------
        - **Positive (+)**: Right shoulder elevated relative to trunk
        - **Negative (-)**: Left shoulder elevated relative to trunk
        - **0°**: Shoulders aligned with trunk lateral tilt

        Calculation Method
        ------------------
        Calculated as the difference between shoulder_lateraltilt_global and
        trunk_lateralflexion_global, removing trunk lean component.

        Returns
        -------
        Signal1D
            Shoulder tilt relative to trunk in degrees.
            Positive = right shoulder higher relative to trunk
            Negative = left shoulder higher relative to trunk

        See Also
        --------
        shoulder_lateraltilt_global : Shoulder tilt in global frame
        trunk_lateralflexion_global : Trunk lateral flexion
        """
        return self.shoulder_lateraltilt_global - self.trunk_lateralflexion_global

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
        Measured as the angle between the upper arm (shoulder-to-elbow vector)
        and the vertical axis in the shoulder reference frame, projected onto
        the sagittal plane (anteroposterior and vertical axes).

        Clinical Relevance
        ------------------
        - Limited flexion (< 150°): Associated with:
          * Shoulder joint stiffness
          * Rotator cuff pathology
          * Adhesive capsulitis (frozen shoulder)
        - Limited extension (unable to reach negative values): Associated with:
          * Anterior shoulder tightness
          * Pectoralis major tightness

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

        return -90 - angle

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
        Measured as the angle between the upper arm (shoulder-to-elbow vector)
        and the vertical axis in the shoulder reference frame, projected onto
        the sagittal plane (anteroposterior and vertical axes).

        Clinical Relevance
        ------------------
        - Limited flexion (< 150°): Associated with:
          * Shoulder joint stiffness
          * Rotator cuff pathology
          * Adhesive capsulitis (frozen shoulder)
        - Limited extension (unable to reach negative values): Associated with:
          * Anterior shoulder tightness
          * Pectoralis major tightness

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

        return -90 - angle

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
        Measured as the angle between the upper arm (shoulder-to-elbow vector)
        and the vertical axis in the shoulder reference frame, projected onto
        the frontal plane (lateral and vertical axes).

        Clinical Relevance
        ------------------
        - Limited abduction (< 150°): Associated with:
          * Rotator cuff pathology (supraspinatus)
          * Shoulder impingement syndrome
          * Adhesive capsulitis
        - Excessive adduction during dynamic tasks: Associated with:
          * Scapular dyskinesis
          * Shoulder instability

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
        Measured as the angle between the upper arm (shoulder-to-elbow vector)
        and the vertical axis in the shoulder reference frame, projected onto
        the frontal plane (lateral and vertical axes).

        Clinical Relevance
        ------------------
        - Limited abduction (< 150°): Associated with:
          * Rotator cuff pathology (supraspinatus)
          * Shoulder impingement syndrome
          * Adhesive capsulitis
        - Excessive adduction during dynamic tasks: Associated with:
          * Scapular dyskinesis
          * Shoulder instability

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
        Calculate left shoulder internal/external rotation angle in transverse plane.

        The angle represents the rotational orientation of the upper arm around
        its longitudinal axis, typically assessed with the elbow at 90° flexion.

        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The forearm rotates medially (toward the body).
          Common in throwing acceleration, reaching across body.
        - **Negative (-)**: External rotation (rotazione esterna)
          The forearm rotates laterally (away from the body).
          Common in throwing cocking phase, backhand motions.
        - **0°**: Neutral position (forearm pointing forward with elbow at 90°)

        Calculation Method
        ------------------
        Measured using the orientation of the elbow's frontal plane
        (defined by medial-lateral elbow markers) relative to the shoulder
        reference frame. The vector from elbow medial to lateral is projected
        onto the transverse plane fixed to the upper arm.

        Clinical Relevance
        ------------------
        - Limited internal rotation (< 60°): Associated with:
          * Posterior shoulder capsule tightness
          * GIRD (Glenohumeral Internal Rotation Deficit)
          * Overhead athlete adaptations
        - Limited external rotation (< 80°): Associated with:
          * Anterior shoulder tightness
          * Subscapularis tightness
          * Reduced throwing performance

        Returns
        -------
        Signal1D
            Shoulder rotation angle in degrees.
            Positive = internal rotation (forearm inward)
            Negative = external rotation (forearm outward)

        See Also
        --------
        right_shoulder_internalexternalrotation : Right shoulder rotation angle
        left_shoulder_abductionadduction : Left shoulder frontal plane motion
        left_elbow_flexionextension : Left elbow flexion angle
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
        Calculate right shoulder internal/external rotation angle in transverse plane.

        The angle represents the rotational orientation of the upper arm around
        its longitudinal axis, typically assessed with the elbow at 90° flexion.

        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The forearm rotates medially (toward the body).
          Common in throwing acceleration, reaching across body.
        - **Negative (-)**: External rotation (rotazione esterna)
          The forearm rotates laterally (away from the body).
          Common in throwing cocking phase, backhand motions.
        - **0°**: Neutral position (forearm pointing forward with elbow at 90°)

        Calculation Method
        ------------------
        Measured using the orientation of the elbow's frontal plane
        (defined by medial-lateral elbow markers) relative to the shoulder
        reference frame. The vector from elbow lateral to medial is projected
        onto the transverse plane fixed to the upper arm.

        Clinical Relevance
        ------------------
        - Limited internal rotation (< 60°): Associated with:
          * Posterior shoulder capsule tightness
          * GIRD (Glenohumeral Internal Rotation Deficit)
          * Overhead athlete adaptations
        - Limited external rotation (< 80°): Associated with:
          * Anterior shoulder tightness
          * Subscapularis tightness
          * Reduced throwing performance

        Returns
        -------
        Signal1D
            Shoulder rotation angle in degrees.
            Positive = internal rotation (forearm inward)
            Negative = external rotation (forearm outward)

        See Also
        --------
        left_shoulder_internalexternalrotation : Left shoulder rotation angle
        right_shoulder_abductionadduction : Right shoulder frontal plane motion
        right_elbow_flexionextension : Right elbow flexion angle
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
        Measured as the angle between the neck_base-to-left_shoulder vector and the
        lateral axis in the global transverse plane (horizontal plane).

        The angle quantifies the anterior-posterior displacement of the shoulder
        girdle, which is primarily driven by scapular movement on the thorax.

        Clinical Relevance
        ------------------
        - Excessive protraction (> +20°): Associated with:
          * Rounded shoulders
          * Forward head posture
          * Weak scapular retractors (rhomboids, mid-trapezius)
          * Tight pectorals
          * Upper crossed syndrome
        - Excessive retraction (< -10°): Less common, may indicate:
          * Overcorrection in posture training
          * Compensation for thoracic hyperkyphosis

        Returns
        -------
        Signal1D
            Scapular protraction/retraction angle in degrees.
            Positive = protraction (forward shoulder position)
            Negative = retraction (backward shoulder position)

        See Also
        --------
        right_scapular_protractionretraction : Right scapular protraction/retraction
        shoulder_lateraltilt_global : Shoulder elevation in frontal plane
        trunk_rotation_global : Trunk rotation that may affect shoulder position
        """
        neck_base = self.neck_base
        shoulder = self.left_shoulder

        # Get vector from neck_base to left shoulder
        v_shoulder = (shoulder - neck_base).to_numpy()

        # Extract transverse plane components (lateral_axis, anteroposterior_axis)
        cols = shoulder.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle from lateral axis in transverse plane
        # x = lateral component, y = anteroposterior component
        x, y = v_shoulder[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        return Signal1D(data=angle, index=shoulder.index, unit="°")

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
        Measured as the angle between the neck_base-to-right_shoulder vector and the
        lateral axis in the global transverse plane (horizontal plane).

        The angle quantifies the anterior-posterior displacement of the shoulder
        girdle, which is primarily driven by scapular movement on the thorax.

        Clinical Relevance
        ------------------
        - Excessive protraction (> +20°): Associated with:
          * Rounded shoulders
          * Forward head posture
          * Weak scapular retractors (rhomboids, mid-trapezius)
          * Tight pectorals
          * Upper crossed syndrome
        - Excessive retraction (< -10°): Less common, may indicate:
          * Overcorrection in posture training
          * Compensation for thoracic hyperkyphosis

        Returns
        -------
        Signal1D
            Scapular protraction/retraction angle in degrees.
            Positive = protraction (forward shoulder position)
            Negative = retraction (backward shoulder position)

        See Also
        --------
        left_scapular_protractionretraction : Left scapular protraction/retraction
        shoulder_lateraltilt_global : Shoulder elevation in frontal plane
        trunk_rotation_global : Trunk rotation that may affect shoulder position
        """
        neck_base = self.neck_base
        shoulder = self.right_shoulder

        # Get vector from neck_base to right shoulder
        v_shoulder = (shoulder - neck_base).to_numpy()

        # Extract transverse plane components (lateral_axis, anteroposterior_axis)
        cols = shoulder.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # Calculate angle from lateral axis in transverse plane
        # x = lateral component, y = anteroposterior component
        # Note: sign is reversed for right side to maintain consistent interpretation
        x, y = v_shoulder[:, col_map].T
        angle = np.degrees(np.arctan2(y, -x))  # Negate x for right side symmetry

        return Signal1D(data=angle, index=shoulder.index, unit="°")

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
        Measured as the angle at neck_base formed by three points in the frontal
        plane: shoulder - neck_base - T5 (fifth thoracic vertebra).

        This quantifies the vertical displacement of the shoulder girdle, which
        is primarily driven by upper trapezius (elevation) and lower trapezius/
        serratus anterior (depression) activity.

        Clinical Relevance
        ------------------
        - Excessive elevation (> +15°): Associated with:
          * Upper trapezius overactivity
          * Levator scapulae tightness
          * Upper crossed syndrome
          * Chronic neck tension
        - Asymmetric elevation (left vs right > 5°): Associated with:
          * Muscle imbalance
          * Scoliosis compensation
          * Unilateral overuse

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
        shoulder_lateraltilt_global : Shoulder tilt in frontal plane
        """
        shoulder = self.left_shoulder
        neck_base = self.neck_base
        t5 = self._get_point("t5")

        # Calculate 3-point angle in frontal plane: shoulder - neck_base - t5
        # This gives the elevation/depression relative to thoracic spine
        angle = self._get_angle_between_three_points(shoulder, neck_base, t5)

        # Adjust sign so positive = elevation (shoulder above neutral)
        # The angle is measured in frontal plane, so we need to determine
        # if shoulder is above or below the neck_base-t5 line
        return Signal1D(data=90 - angle, index=shoulder.index, unit="°")

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
        Measured as the angle at neck_base formed by three points in the frontal
        plane: shoulder - neck_base - T5 (fifth thoracic vertebra).

        This quantifies the vertical displacement of the shoulder girdle, which
        is primarily driven by upper trapezius (elevation) and lower trapezius/
        serratus anterior (depression) activity.

        Clinical Relevance
        ------------------
        - Excessive elevation (> +15°): Associated with:
          * Upper trapezius overactivity
          * Levator scapulae tightness
          * Upper crossed syndrome
          * Chronic neck tension
        - Asymmetric elevation (left vs right > 5°): Associated with:
          * Muscle imbalance
          * Scoliosis compensation
          * Unilateral overuse

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
        shoulder_lateraltilt_global : Shoulder tilt in frontal plane
        """
        shoulder = self.right_shoulder
        neck_base = self.neck_base
        t5 = self._get_point("t5")

        # Calculate 3-point angle in frontal plane: shoulder - neck_base - t5
        # This gives the elevation/depression relative to thoracic spine
        angle = self._get_angle_between_three_points(shoulder, neck_base, t5)

        # Adjust sign so positive = elevation (shoulder above neutral)
        return Signal1D(data=90 - angle, index=shoulder.index, unit="°")

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
        Measured as the angle between the upper arm (shoulder-to-elbow) and
        forearm (elbow-to-wrist) vectors. Calculated as 180° minus the angle
        formed by the three points (wrist, elbow, shoulder), so that flexion
        is positive and full extension is zero.

        Clinical Relevance
        ------------------
        - Limited flexion (< 135°): Associated with:
          * Elbow joint stiffness
          * Biceps or triceps contracture
          * Post-traumatic or post-surgical limitation
        - Hyperextension (negative values): Associated with:
          * Joint hypermobility
          * Ligamentous laxity
          * Common in women and gymnasts

        Returns
        -------
        Signal1D
            Elbow flexion/extension angle in degrees.
            Positive = flexion (bent elbow)
            Negative = extension (hyperextension)

        See Also
        --------
        right_elbow_flexionextension : Right elbow flexion angle
        left_shoulder_flexionextension : Left shoulder flexion angle
        """
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
        Measured as the angle between the upper arm (shoulder-to-elbow) and
        forearm (elbow-to-wrist) vectors. Calculated as 180° minus the angle
        formed by the three points (wrist, elbow, shoulder), so that flexion
        is positive and full extension is zero.

        Clinical Relevance
        ------------------
        - Limited flexion (< 135°): Associated with:
          * Elbow joint stiffness
          * Biceps or triceps contracture
          * Post-traumatic or post-surgical limitation
        - Hyperextension (negative values): Associated with:
          * Joint hypermobility
          * Ligamentous laxity
          * Common in women and gymnasts

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
