"""right shoulder joint module"""

import numpy as np
from ....timeseries import Point3D
from .pelvis import Pelvis
from .joint import Joint
from .utils import estimate_rigid_joint_center

__all__ = ["RightShoulder"]


class RightShoulder(Joint):
    """RightShoulder Joint."""

    def __init__(
        self,
        c7: Point3D,
        sc: Point3D,
        pelvis: Pelvis,
        right_acromion: Point3D,
        right_elbow_lateral: Point3D,
        right_elbow_medial: Point3D,
    ):

        Ag = right_acromion.copy()
        N = (c7 + sc) / 2
        AP = sc - c7
        AP = AP / np.atleast_2d(np.linalg.norm(AP.to_numpy(), axis=1)).T
        VT = N - pelvis.center
        VT = VT / np.atleast_2d(np.linalg.norm(VT.to_numpy(), axis=1)).T
        J = Joint(
            N,  # type: ignore
            None,
            AP,  # type: ignore
            VT,  # type: ignore
        )
        W = float((((N.to_numpy() - Ag.to_numpy()) ** 2).sum(axis=1) ** 0.5).mean())

        # estimate the joint center
        Al = J.apply(Ag)
        Sl: Point3D = Al.copy()  # type: ignore
        Sl.loc[Sl.index, [Sl.lateral_axis]] += 0.33 * W
        Sl.loc[Sl.index, [Sl.vertical_axis]] -= 0.30 * W
        Sl.loc[Sl.index, [Sl.anteroposterior_axis]] -= 0.19 * W
        Sg: Point3D = J.apply_inverse(Sl)  # type: ignore
        """
        # extrapolate the left shoulder joint center
        Eg = (right_elbow_lateral + right_elbow_medial) / 2
        S = estimate_rigid_joint_center(
            J,
            Eg,  # type: ignore
            Ag,
            Sg,
            # bounds=[(-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)],
        )
        """

        # generate the object
        VT = (-1) * VT
        ML = Ag - N  # type: ignore
        ML = ML / np.atleast_2d(np.linalg.norm(ML.to_numpy(), axis=1)).T
        super().__init__(Sg, ML, VT)  # type: ignore

        # elbow
        self["elbow_medial"] = right_elbow_medial
        self["elbow_lateral"] = right_elbow_lateral

    @property
    def _elbow_lateral(self):
        """return the elbow lateral marker in the local joint reference frame"""
        out: Point3D = self.apply(self["elbow_lateral"])  # type: ignore
        return out

    @property
    def _elbow_medial(self):
        """return the elbow medial marker in the local joint reference frame"""
        out: Point3D = self.apply(self["knee_medial"])  # type: ignore
        return out

    @property
    def _elbow_center(self):
        """return the right elbow in the local joint reference frame"""
        out: Point3D = 0.5 * (self._elbow_lateral + self._elbow_medial)  # type: ignore
        return out

    @property
    def flexionextension(self):
        """
        Calculate shoulder flexion-extension angle.

        Interpretation
        --------------
        Positive = flexion
        Negative = extension

        Returns
        -------
        Signal1D
            shoulder flexion/extension angle in degrees.
        """
        return self.get_angle_by_point(
            self.apply(self._elbow_center),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.anteriorposterior_axis,  # type: ignore
        )

    @property
    def adductionabduction(self):
        """
        Calculate shoulder adduction/abduction

        Interpretation
        --------------
        Positive = abduction
        Negative = adductor

        Returns
        -------
        Signal1D
            shoulder adduction/abduction angle in degrees.
        """
        out = self.get_angle_by_point(
            self.apply(self._elbow_center),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.lateral_axis,  # type: ignore
        )
        return out

    @property
    def internalexternalrotation(self):
        """
        Calculate shoulder internal-external rotation

        Interpretation
        --------------
        Positive = external rotation
        Negative = internal rotation

        Returns
        -------
        Signal1D
            shoulder internal/external rotation angle in degrees.
        """

        # generate a knee vector aligned with the shoulder reference frame
        vec = self["elbow_lateral"] - self["elbow_medial"]  # type: ignore
        vec = self.apply(vec)

        # obtain the angle in the transverse plane
        out = self.get_angle_by_point(
            vec,  # type: ignore
            self.lateral_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )
        out *= -1
        return out
