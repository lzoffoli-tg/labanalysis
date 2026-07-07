"""right knee joint module"""

from ....timeseries.point3d import Point3D
from .joint import Joint
from .right_ankle import RightAnkle
from .right_hip import RightHip

__all__ = ["RightKnee"]


class RightKnee(Joint):
    """RightKnee Joint."""

    def __init__(
        self,
        right_hip: RightHip,
        right_ankle: RightAnkle,
        right_knee_lateral: Point3D,
        right_knee_medial: Point3D,
    ):

        # object with reference frame
        lax = right_knee_medial - right_knee_lateral
        kne = (right_knee_medial + right_knee_lateral) / 2
        vrt = kne - right_hip
        super().__init__(origin=kne, lateral_vector=lax, vertical_vector=vrt)  # type: ignore

        # ankle
        self["ankle"] = right_ankle

    @property
    def _ankle(self):
        """return the right ankle in the knee reference frame"""
        out: Point3D = self["ankle"]  # type: ignore
        return out

    @property
    def flexionextension(self):
        """
        Calculate right knee flexion-extension angle in sagittal plane.

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
