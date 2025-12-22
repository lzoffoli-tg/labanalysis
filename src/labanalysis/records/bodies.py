"""full body module"""

from typing import Literal

import numpy as np
import pandas as pd
from pint import UnitRegistry

from ..signalprocessing import gram_schmidt
from .timeseries import *
from .records import *

ureg = UnitRegistry()
ureg.define("au = [] = au")

__all__ = ["WholeBody"]


class WholeBody(TimeseriesRecord):

    _angular_measures = [
        "left_ankle_flexionextension",
        "right_ankle_flexionextension",
        "left_ankle_inversioneversion",
        "right_ankle_inversioneversion",
        "left_knee_flexionextension",
        "right_knee_flexionextension",
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
        left_metatarsal_head: Point3D | None = None,
        right_metatarsal_head: Point3D | None = None,
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
        right_shoulder_anterior: Point3D | None = None,
        right_shoulder_posterior: Point3D | None = None,
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
        sc: Point3D | None = None,  # sternoclavicular joint
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
                left_metatarsal_head=left_metatarsal_head,
                right_metatarsal_head=right_metatarsal_head,
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
                right_shoulder_anterior=right_shoulder_anterior,
                right_shoulder_posterior=right_shoulder_posterior,
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
                sc=sc,
                l2=l2,
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
        left_metatarsal_head: str | None = None,
        right_metatarsal_head: str | None = None,
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
        right_shoulder_anterior: str | None = None,
        right_shoulder_posterior: str | None = None,
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
        sc: str | None = None,
    ):

        # read the file
        tdf = TimeseriesRecord.from_tdf(filename)

        # check the inputs
        points = {
            "left_heel": left_heel,
            "right_heel": right_heel,
            "left_toe": left_toe,
            "right_toe": right_toe,
            "left_metatarsal_head": left_metatarsal_head,
            "right_metatarsal_head": right_metatarsal_head,
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
            "right_shoulder_anterior": right_shoulder_anterior,
            "right_shoulder_posterior": right_shoulder_posterior,
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
            "sc": sc,
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
        for key, lbl in points.items():
            if lbl is not None:
                if lbl not in keys:
                    raise ValueError(f"{lbl} not found.")
                if not isinstance(tdf[lbl], Point3D):
                    raise ValueError(f"{lbl} must be a Point3D instance.")
                mandatory[key] = tdf[lbl]
        extras = {i: v for i, v in tdf.items() if i not in list(mandatory.keys())}

        return cls(**mandatory, **extras)  # type: ignore

    def to_dataframe(self):
        out = [super().to_dataframe()]
        for prop in self._angular_measures:
            try:
                df = getattr(self, prop).to_dataframe()
                df.columns = pd.Index([prop + "_" + i] for i in df.columns)
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
        Calcola i coefficienti del piano (a, b, c, d) per ogni istante temporale
        dati almeno 3 punti 3D per ciascun campione.
        """
        for point in points:
            if not isinstance(point, Point3D):
                raise ValueError("all points must be Point3D instances.")

        # Calcola il baricentro per ogni campione
        mat = np.stack([i.to_numpy() for i in points], axis=1)
        centroid = np.mean(mat, axis=1, keepdims=True)

        # Centra i punti rispetto al baricentro
        centered = mat - centroid

        # Calcola la matrice di covarianza per ogni campione
        cov = np.einsum("nij,nik->njk", centered, centered) / mat.shape[1]

        # Calcola autovalori e autovettori
        eigvals, eigvecs = np.linalg.eigh(cov)

        # L'autovettore associato al minimo autovalore è la normale al piano
        normals = eigvecs[:, :, 0]

        # Coefficienti a, b, c
        a, b, c = normals[:, :3].T

        # Calcola d = - (a*x + b*y + c*z) usando il baricentro
        d = -np.sum(normals * centroid[:, 0, :], axis=1)

        # Stack dei coefficienti
        coefficients = np.stack([a, b, c, d], axis=1)

        return coefficients

    def _get_plane_coefs(
        self,
        onplane_a: Point3D,
        onplane_b: Point3D,
        ortogonal_to: Point3D,
    ):
        """
        Calcola i coefficienti (a, b, c, d) del piano passante per onplane_a e
        onplane_b, tale che la retta che congiunge ortogonal_to al piano sia
        perpendicolare al piano stesso.
        """
        # Estrai i dati: shape (T, 3)
        a_data = onplane_a.to_numpy()
        b_data = onplane_b.to_numpy()
        o_data = ortogonal_to.to_numpy()

        # Vettore nel piano: v = b - a
        v = b_data - a_data

        # Vettore dalla terza al piano: n_raw = a - ortogonal_to
        n_raw = a_data - o_data

        # Normalizzazione di v
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        v_unit = v / v_norm

        # Proiezione di n_raw su v
        proj = np.sum(n_raw * v_unit, axis=1, keepdims=True) * v_unit

        # Componente ortogonale a v: normale al piano
        normal = n_raw - proj

        # Coefficienti a, b, c
        a, b, c = normal[:, 0], normal[:, 1], normal[:, 2]

        # Calcolo di d usando il punto a_data
        d = -np.sum(normal * a_data, axis=1)

        # Stack dei coefficienti
        coefficients = np.stack([a, b, c, d], axis=1)

        return coefficients

    def _get_point_to_plane_distance(
        self,
        point: Point3D,
        planes: Timeseries,
    ):
        """
        Calcola la (minima) distanza di un punto da un piano.
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
        Trasla punti 3D lungo gli assi locali definiti da un piano.
        """
        # Estrai i coefficienti a, b, c
        normals = plane.ix[:, :3].to_numpy()
        norm_normals = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_unit = normals / norm_normals  # normalizzazione

        # Vettore arbitrario per costruire base ortonormale
        arbitrary = np.tile(np.array([1.0, 0.0, 0.0]), (len(normals_unit), 1))
        parallel_mask = np.isclose(
            np.abs(np.sum(normals_unit * arbitrary, axis=1)), 1.0
        )
        arbitrary[parallel_mask] = np.array([0.0, 1.0, 0.0])

        # Asse x locale
        x_local = np.cross(arbitrary, normals_unit)
        x_local /= np.linalg.norm(x_local, axis=1, keepdims=True)

        # Asse y locale
        y_local = np.cross(normals_unit, x_local)
        y_local /= np.linalg.norm(y_local, axis=1, keepdims=True)

        # Matrice di rotazione: colonne = [x_local, y_local, normale]
        R = np.stack([x_local, y_local, normals_unit], axis=2)  # shape (N, 3, 3)

        # Trasformazione delle traslazioni locali in globali
        l_trans = np.asarray(local_translations)
        T_global = np.einsum("nij,nj->ni", R, l_trans)

        # Applica la traslazione ai punti
        out: Point3D = point.copy() + T_global
        return out

    def _get_intersection_between_line_to_plane(
        self,
        p1: Point3D,
        p2: Point3D,
        plane: Timeseries,
    ):
        """
        Calcola il punto di intersezione tra una linea (p1 -> p2) e un
        piano definito da (a, b, c, d) per ogni istante temporale.
        """
        direction = np.asarray(p2 - p1)
        planearr = np.asarray(plane)
        a, b, c, d = planearr.T
        normal = np.stack([a, b, c], axis=1)

        numerator = -(np.sum(normal * p1, axis=1) + d)
        denominator = np.sum(normal * direction, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(denominator != 0, numerator / denominator, np.nan)

        intersection_points = p1 + direction * t[:, np.newaxis]
        return intersection_points._data

    def _get_projection_point_on_plane(
        self,
        point: Point3D,
        plane: Timeseries,
    ):
        """
        Calcola il punto sul piano (per ogni istante temporale) che minimizza
        la distanza da un punto 3D: la proiezione ortogonale.
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
        Calcola l'angolo (in gradi) formato da tre punti 3D: A, B, C
        dove B è il vertice.
        """

        # get the segments length
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

        # Vettori dal punto all'origine
        v = point_np - origin_np  # shape (n, 3)

        # Lunghezza della proiezione scalare su ciascun asse
        proj_length = np.sum(v * direction_np, axis=1)
        proj_length /= np.sum(direction_np**2, axis=1)  # shape (n,)

        # Calcolo dei punti proiettati
        proj_np = origin_np + (
            proj_length[:, np.newaxis] * direction_np
        )  # shape (n, 3)

        # Restituisce un nuovo Point3D con le stesse proprietà del punto originale
        return Point3D(
            proj_np,
            point.index,
            point.unit,
            point.columns,
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
        ankle_med: Point3D = self._get_point("left_ankle_medial")
        ankle = self.left_ankle
        knee = self.left_knee

        # get the rotation matrix
        new_x = (ankle_lat - ankle_med).to_numpy()
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
        ankle_med: Point3D = self._get_point("right_ankle_medial")
        ankle = self.right_ankle
        knee = self.right_knee

        # get the rotation matrix
        new_x = (ankle_med - ankle_lat).to_numpy()
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
        med: Point3D = self._get_point("left_ankle_medial")
        return Point3D(
            data=(lat._data + med._data) / 2,
            index=np.unique(np.concatenate([lat.index, med.index])),
            columns=lat.columns,
        )

    @property
    def right_ankle(self):
        lat: Point3D = self._get_point("right_ankle_lateral")
        med: Point3D = self._get_point("right_ankle_medial")
        return Point3D(
            data=(lat._data + med._data) / 2,
            index=np.unique(np.concatenate([lat.index, med.index])),
            columns=lat.columns,
        )

    @property
    def left_knee(self):
        lat: Point3D = self._get_point("left_knee_lateral")
        med: Point3D = self._get_point("left_knee_medial")
        return Point3D(
            data=(lat._data + med._data) / 2,
            index=np.unique(np.concatenate([lat.index, med.index])),
            columns=lat.columns,
        )

    @property
    def right_knee(self):
        lat: Point3D = self._get_point("right_knee_lateral")
        med: Point3D = self._get_point("right_knee_medial")
        return Point3D(
            data=(lat._data + med._data) / 2,
            index=np.unique(np.concatenate([lat.index, med.index])),
            columns=lat.columns,
        )

    @property
    def pelvis_center(self):
        return self.pelvis_referenceframe[0]

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
        l_troc = self._get_point("left_throcanter")
        r_troc = self._get_point("right_throcanter")
        l_dist = self._get_point_to_plane_distance(l_troc, pelvis_plane)
        r_dist = self._get_point_to_plane_distance(r_troc, pelvis_plane)
        return Signal1D(
            data=(l_dist + r_dist).to_numpy() / 2,
            index=pelvis_plane.index,
            unit=l_dist.unit,
        )

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
        ant: Point3D = self._get_point("left_shoulder_anterior")
        pos: Point3D = self._get_point("left_shoulder_posterior")
        return Point3D(
            (ant + pos).to_numpy() / 2,
            index=np.unique(np.concatenate([ant.index, pos.index])),
            columns=ant.columns,
        )

    @property
    def right_shoulder(self):
        ant: Point3D = self._get_point("right_shoulder_anterior")
        pos: Point3D = self._get_point("right_shoulder_posterior")
        return Point3D(
            (ant + pos).to_numpy() / 2,
            index=np.unique(np.concatenate([ant.index, pos.index])),
            columns=ant.columns,
        )

    @property
    def left_elbow(self):
        lat: Point3D = self._get_point("left_elbow_lateral")
        med: Point3D = self._get_point("left_elbow_medial")
        return Point3D(
            (lat + med).to_numpy() / 2,
            index=np.unique(np.concatenate([lat.index, med.index])).tolist(),
            columns=lat.columns,
        )

    @property
    def right_elbow(self):
        lat: Point3D = self._get_point("right_elbow_lateral")
        med: Point3D = self._get_point("right_elbow_medial")
        return Point3D(
            (lat + med).to_numpy() / 2,
            index=np.unique(np.concatenate([lat.index, med.index])),
            columns=lat.columns,
        )

    @property
    def left_wrist(self):
        lat: Point3D = self._get_point("left_wrist_lateral")
        med: Point3D = self._get_point("left_wrist_medial")
        return Point3D(
            (lat + med).to_numpy() / 2,
            index=np.unique(np.concatenate([lat.index, med.index])),
            columns=lat.columns,
        )

    @property
    def right_wrist(self):
        lat: Point3D = self._get_point("right_wrist_lateral")
        med: Point3D = self._get_point("right_wrist_medial")
        return Point3D(
            (lat + med).to_numpy() / 2,
            index=np.unique(np.concatenate([lat.index, med.index])),
            columns=lat.columns,
        )

    @property
    def left_foot_plane(self):
        toe = self._get_point("left_toe")
        meta = self._get_point("left_metatarsal_head")
        heel = self._get_point("left_heel")
        return Timeseries(
            data=self._get_least_squares_plane_coefs(toe, meta, heel),
            index=toe.index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )

    @property
    def right_foot_plane(self):
        toe = self._get_point("right_toe")
        meta = self._get_point("right_metatarsal_head")
        heel = self._get_point("right_heel")
        return Timeseries(
            data=self._get_least_squares_plane_coefs(toe, meta, heel),
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

        # Ottengo i parametri necessari
        rmat = self.left_hip_referenceframe[1]
        knee_lat = self._get_point("left_knee_lateral")
        knee_med = self._get_point("left_knee_medial")
        ankle_lat = self._get_point("left_ankle_lateral")
        ankle_med = self._get_point("left_ankle_medial")

        # ottengo il vettore medio
        v1 = (knee_lat - knee_med).to_numpy()
        v2 = (ankle_lat - ankle_med).to_numpy()
        va = (v1 + v2) / 2

        # determio la matrice di rotazione del sistema di riferimento
        i = (self.left_hip - self.right_hip).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        k = (self.left_hip - self.left_knee).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))

        # allineo il vettore al nuovo sistema di riferimento
        vr = np.einsum("nij,nj->ni", rmat, va)

        # considero il piano trasverso solidale al sistema di riferimento
        # generato
        cols = knee_lat.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)
        x, y = vr[:, col_map].T

        # calcolo l'angolo del vettore rispetto al piano
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l'angolo
        return Signal1D(data=angle, index=knee_lat.index, unit="°")

    @property
    def right_hip_internalexternalrotation(self):
        """
        Return the right hip internal/external rotation in degrees.
        Internal rotation is positive, external rotation is negative.
        """
        # Ottengo i parametri necessari
        rmat = self.right_hip_referenceframe[1]
        knee_lat = self._get_point("right_knee_lateral")
        knee_med = self._get_point("right_knee_medial")
        ankle_lat = self._get_point("right_ankle_lateral")
        ankle_med = self._get_point("right_ankle_medial")

        # ottengo il vettore medio
        v1 = (knee_lat - knee_med).to_numpy()
        v2 = (ankle_lat - ankle_med).to_numpy()
        va = (v1 + v2) / 2

        # determio la matrice di rotazione del sistema di riferimento
        i = (self.left_hip - self.right_hip).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        k = (self.left_hip - self.left_knee).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))

        # allineo il vettore al nuovo sistema di riferimento
        vr = np.einsum("nij,nj->ni", rmat, va)

        # considero il piano trasverso solidale al sistema di riferimento
        # generato
        cols = knee_lat.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)
        x, y = vr[:, col_map].T

        # calcolo l'angolo del vettore rispetto al piano
        angle = np.degrees(np.arctan2(y, x))

        # correggo il segno dell'angolo
        angle = 180 - angle
        angle[y < 0] = angle[y < 0] - 360

        # Ritorno l'angolo
        return Signal1D(data=angle, index=knee_lat.index, unit="°")

    @property
    def pelvis_anteroposteriortilt_global(self):
        """
        Return the pelvis pitch (sagittal tilt) in degrees with respect to
        the global reference frame.
        Anterior tilt is negative, posterior tilt is positive.
        """

        # get the pelvis points projected into its least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # definisco il vettore che determina l'asse anteroposteriore del bacino
        ap = ((l_asis + r_asis) / 2 - (l_psis + r_psis) / 2).to_numpy()

        # considero solo il piano sagittale globale
        cols = l_asis.columns
        axes_labels = [self.anteroposterior_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # calcolo l'angolo
        x, y = ap[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l’angolo
        return Signal1D(data=angle, index=l_asis.index, unit="°")

    @property
    def pelvis_lateraltilt_global(self):
        """
        Return the pelvis roll (frontal tilt) in degrees with respect to
        the global reference frame.
        Right tilt is positive, left tilt is negative.
        """

        # get the pelvis points projected into its least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # definisco il vettore che determina l'asse laterale del bacino
        ml = ((l_asis + l_psis) / 2 - (r_asis + r_psis) / 2).to_numpy()

        # considero solo il piano frontale globale
        cols = l_asis.columns
        axes_labels = [self.lateral_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # calcolo l'angolo
        x, y = ml[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l’angolo
        return Signal1D(data=angle, index=l_asis.index, unit="°")

    @property
    def pelvis_rotation_global(self):
        """
        Return the pelvis yaw (axial rotation) in degrees with respect
        to the global reference frame.
        Right rotation is positive, left rotation is negative.
        """

        # get the pelvis points projected into its least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # definisco il vettore che determina l'asse laterale del bacino
        ml = ((l_asis + l_psis) / 2 - (r_asis + r_psis) / 2).to_numpy()

        # considero solo il piano trasverso globale
        cols = l_asis.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # calcolo l'angolo
        x, y = ml[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l’angolo
        return Signal1D(data=angle, index=l_asis.index, unit="°")

    @property
    def trunk_flexionextension_global(self):
        """
        Return the trunk flexion/extension in degrees with respect
        to the global reference frame.
        Flexion is positive, extension is negative.
        """

        # get the pelvis points projected into its least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # ottengo il vettore che definisce il rachide
        c7 = self._get_point("c7").to_numpy()
        base = ((l_psis + r_psis) / 2).to_numpy()
        vt = c7 - base

        # considero solo il piano sagittale globale
        cols = l_psis.columns
        axes_labels = [self.anteroposterior_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # calcolo l'angolo
        x, y = vt[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l’angolo
        return Signal1D(data=90 - angle, index=l_psis.index, unit="°")

    @property
    def trunk_lateralflexion_global(self):
        """
        Return the trunk lateral tilt (side bending) in degrees with respect
        to the global reference frame.
        Right tilt is negative, left tilt is positive.
        """

        # get the pelvis points projected into its least squares plane
        l_asis, r_asis, l_psis, r_psis = self._get_projected_pelvis_points()

        # ottengo il vettore che definisce il rachide
        c7 = self._get_point("c7").to_numpy()
        base = ((l_psis + r_psis) / 2).to_numpy()
        vt = c7 - base

        # considero solo il piano frontale globale
        cols = l_psis.columns
        axes_labels = [self.lateral_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # calcolo l'angolo
        x, y = vt[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l’angolo
        return Signal1D(data=angle - 90, index=l_psis.index, unit="°")

    @property
    def trunk_rotation_global(self):
        """
        Return the trunk axial rotation in degrees with respect to the
        global reference frame.
        Right rotation is positive, left rotation is negative.
        """

        # ottengo il vettore che definisce l'asse delle spalle
        vt = self.left_shoulder - self.right_shoulder

        # considero solo il piano trasverso globale
        cols = vt.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # calcolo l'angolo
        x, y = vt.copy().to_numpy()[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l’angolo
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
    def shoulder_lateraltilt_global(self):
        """
        Return the shoulders roll (frontal tilt) in degrees with respect to
        the global reference frame.
        Right tilt is positive, left tilt is negative.
        """

        # definisco il vettore che determina l'orientamento delle spalle
        ml = self.left_shoulder - self.right_shoulder

        # considero solo il piano frontale globale
        cols = ml.columns
        axes_labels = [self.lateral_axis, self.vertical_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)

        # calcolo l'angolo
        x, y = ml.copy().to_numpy()[:, col_map].T
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l’angolo
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
        # ottengo i parametri necessari
        shoulder, rmat = self.left_shoulder_referenceframe
        elbow = self.left_elbow

        # calcolo l'orientamento del braccio rispetto al sistema di riferimento
        # della spalla
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
        # ottengo i parametri necessari
        shoulder, rmat = self.right_shoulder_referenceframe
        elbow = self.right_elbow

        # calcolo l'orientamento del braccio rispetto al sistema di riferimento
        # della spalla
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow,
            shoulder,
            rmat,
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

        # correggo il segno dell'angolo
        # TODO [RMSIN-448] rimane da correggere il caso di gradi maggiori di 90 gradi
        return -1 * (90 + angle)

    @property
    def left_shoulder_internalexternalrotation(self):
        """
        Return the left shoulder internal/external rotation in degrees.
        Internal rotation is positive, external rotation is negative.
        """
        # Ottengo i parametri necessari
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

        # ottengo il vettore di riferimento
        va = (elbow_lat - elbow_med).to_numpy()

        # determio la matrice di rotazione del sistema di riferimento
        i = (shoulder - proj).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        k = (shoulder - self.left_elbow).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))

        # allineo il vettore al nuovo sistema di riferimento
        vr = np.einsum("nij,nj->ni", rmat, va)

        # considero il piano trasverso solidale al sistema di riferimento
        # generato
        cols = elbow_lat.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)
        x, y = vr[:, col_map].T

        # calcolo l'angolo del vettore rispetto al piano
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l'angolo
        return Signal1D(data=angle, index=elbow_lat.index, unit="°")

    @property
    def right_shoulder_internalexternalrotation(self):
        """
        Return the right shoulder internal/external rotation in degrees.
        Internal rotation is positive, external rotation is negative.
        """
        # Ottengo i parametri necessari
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

        # ottengo il vettore di riferimento
        va = (elbow_med - elbow_lat).to_numpy()

        # determio la matrice di rotazione del sistema di riferimento
        i = (proj - shoulder).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        k = (shoulder - self.right_elbow).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))

        # allineo il vettore al nuovo sistema di riferimento
        vr = np.einsum("nij,nj->ni", rmat, va)

        # considero il piano trasverso solidale al sistema di riferimento
        # generato
        cols = elbow_lat.columns
        axes_labels = [self.lateral_axis, self.anteroposterior_axis]
        col_map = [np.where(cols == i)[0][0] for i in axes_labels]
        col_map = np.array(col_map)
        x, y = vr[:, col_map].T

        # calcolo l'angolo del vettore rispetto al piano
        angle = np.degrees(np.arctan2(y, x))

        # Ritorno l'angolo
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
