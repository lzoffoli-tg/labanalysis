"""right shoulder joint module"""

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
        pelvis_plane: Pelvis,
        right_acromion: Point3D,
        right_elbow_lateral: Point3D,
        right_elbow_medial: Point3D,
    ):

        # setup the object reference frame and extract the versors
        N = (c7 + sc) / 2
        VT = pelvis_plane.vertical_versor - N
        ML = right_acromion - N
        J = Joint(N, ML, None, AP)  # type: ignore

        Ag = right_acromion.copy()
        Al = J.apply(Ag)

        # estimate the joint center
        W = ((N.to_numpy() - Ag.to_numpy()) ** 2).sum(axis=1).mean()
        Sl: Point3D = Al.copy()  # type: ignore
        Sl.loc[Sl.index, Sl.lateral_axis] -= 0.33 * W
        Sl.loc[Sl.index, Sl.vertical_axis] -= 0.30 * W
        Sl.loc[Sl.index, Sl.anteroposterior_axis] -= 0.19 * W

        # return the estimated point in the global frame
        Sg: Point3D = J.apply_inverse(Sl)  # type: ignore

        # extrapolate the right shoulder joint center
        elbow = (right_elbow_lateral + right_elbow_medial) / 2
        S = estimate_rigid_joint_center(
            J,
            elbow,  # type: ignore
            right_acromion,
            Sg,
        )

        # generate the object
        ML = N - S  # type: ignore
        super().__init__(N, ML, VT)  # type: ignore

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
