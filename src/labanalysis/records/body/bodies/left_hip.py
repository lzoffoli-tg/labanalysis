"""left hip joint module"""

from ....timeseries import Point3D
from .pelvis import Pelvis
from .joint import Joint
from .utils import estimate_rigid_joint_center

__all__ = ["LeftHip"]


class LeftHip(Joint):
    """LeftHip Joint."""

    def __init__(
        self,
        s2: Point3D,
        pelvis: Pelvis,
        left_knee_lateral: Point3D,
        left_knee_medial: Point3D,
        left_trochanter: Point3D | None = None,
    ):
        """
        knee = (left_knee_lateral + left_knee_medial) / 2

        # extrapolate the left hip joint center
        Hg = estimate_rigid_joint_center(
            pelvis,
            knee,  # type: ignore
            left_trochanter,
            pelvis.left_hip_approximated,
        )
        """

        # get the lateral and vertical axes of the Hip reference frame
        ML = pelvis.right_midpoint - pelvis.left_midpoint
        VT = pelvis.psis_midpoint - s2

        # build the class
        super().__init__(
            pelvis.left_hip_approximated,
            ML,  # type: ignore
            VT,  # type: ignore
            None,
        )  # type: ignore

        # knee
        self["knee_medial"] = left_knee_medial
        self["knee_lateral"] = left_knee_lateral

    @property
    def _knee_lateral(self):
        """return the knee lateral marker in the global reference frame"""
        out: Point3D = self["knee_lateral"]  # type: ignore
        return out

    @property
    def _knee_medial(self):
        """return the knee medial marker in the global reference frame"""
        out: Point3D = self["knee_medial"]  # type: ignore
        return out

    @property
    def _knee_center(self):
        """return the left knee in the global reference frame"""
        out: Point3D = 0.5 * (self._knee_lateral + self._knee_medial)  # type: ignore
        return out

    @property
    def flexionextension(self):
        """
        Calculate hip flexion-extension angle.

        Interpretation
        --------------
        Positive = flexion
        Negative = extension

        Returns
        -------
        Signal1D
            hip flexion/extension angle in degrees.
        """
        return self.get_angle_by_point(
            self.apply(self._knee_center),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )

    @property
    def adductionabduction(self):
        """
        Calculate hip adduction/abduction

        Interpretation
        --------------
        Positive = abduction
        Negative = adductor

        Returns
        -------
        Signal1D
            hip adduction/abduction angle in degrees.
        """
        return self.get_angle_by_point(
            self.apply(self._knee_center),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.lateral_axis,  # type: ignore
        )

    @property
    def internalexternalrotation(self):
        """
        Calculate hip internal-external rotation

        Interpretation
        --------------
        Positive = external rotation
        Negative = internal rotation

        Returns
        -------
        Signal1D
            hip internal/external rotation angle in degrees.
        """

        # generate a knee vector aligned with the hip reference frame
        vec = self.left_knee_medial - self.left_knee_lateral
        vec = self.apply(vec)

        # obtain the angle in the transverse plane
        return self.get_angle_by_point(
            vec,  # type: ignore
            self.lateral_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )
