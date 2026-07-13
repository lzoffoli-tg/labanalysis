"""left knee joint module"""

from ....timeseries.point3d import Point3D
from .joint import Joint
from .left_ankle import LeftAnkle
from .left_hip import LeftHip

__all__ = ["LeftKnee"]


class LeftKnee(Joint):
    """LeftKnee Joint."""

    def __init__(
        self,
        left_hip: LeftHip,
        left_ankle: LeftAnkle,
        left_knee_lateral: Point3D,
        left_knee_medial: Point3D,
    ):

        # object with reference frame
        lax = left_knee_lateral - left_knee_medial
        kne = (left_knee_medial + left_knee_lateral) / 2
        vrt = kne - left_hip.center
        super().__init__(center=kne, lateral_vector=lax, vertical_vector=vrt, anteroposterior_vector=None)  # type: ignore

        # ankle
        self["ankle"] = left_ankle.center

    @property
    def _ankle(self):
        """return the left ankle in the knee reference frame"""
        out: Point3D = self["ankle"]  # type: ignore
        return out

    @property
    def flexionextension(self):
        """
        Calculate left knee flexion-extension angle in sagittal plane.

        Returns
        -------
        Signal1D
            knee flexion/extension angle in degrees.
            Positive = flexion
            Negative = (hyper)extension
        """
        return self.get_angle_by_point(
            self.apply(self._ankle),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )

    @property
    def varusvalgus(self):
        """
        Calculate left knee varus-valgus angle in frontal plane.

        Returns
        -------
        Signal1D
            knee varus/valgus angle in degrees.
            Negative = (X-shaped knee valgus)
            Positive = (O-shaped knee varus)
        """
        angle = self.get_angle_by_point(
            self.apply(self._ankle),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.lateral_axis,  # type: ignore
        )
        angle *= -1
        return angle
