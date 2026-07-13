"""right elbow joint module"""

from ....timeseries.point3d import Point3D
from .joint import Joint
from .right_shoulder import RightShoulder

__all__ = ["RightElbow"]


class RightElbow(Joint):
    """RightElbow Joint."""

    def __init__(
        self,
        right_shoulder: RightShoulder,
        right_elbow_lateral: Point3D,
        right_elbow_medial: Point3D,
        right_wrist_lateral: Point3D,
        right_wrist_medial: Point3D,
    ):
        lax = right_elbow_lateral - right_elbow_medial
        elb = (right_elbow_lateral + right_elbow_medial) / 2
        vrt = elb - right_shoulder.center
        super().__init__(center=elb, lateral_vector=lax, vertical_vector=vrt, anteroposterior_vector=None)  # type: ignore
        self["wrist_medial"] = right_wrist_medial
        self["wrist_lateral"] = right_wrist_lateral

    @property
    def _wrist_medial(self):
        """return the right wrist medial in the global reference frame"""
        out: Point3D = self["wrist_medial"]  # type: ignore
        return out

    @property
    def _wrist_lateral(self):
        """return the right wrist lateral in the global reference frame"""
        out: Point3D = self["wrist_lateral"]  # type: ignore
        return out

    @property
    def _wrist_center(self):
        """return the right wrist centre in the elbow reference frame"""
        out: Point3D = (self["wrist_medial"] + self["wrist_lateral"]) / 2  # type: ignore
        return out

    @property
    def flexionextension(self):
        """
        Calculate right elbow flexion-extension angle in sagittal plane.

        Returns
        -------
        Signal1D
            elbow flexion/extension angle in degrees.
            Positive = flexion
            Negative = (hyper)extension
        """
        return self.get_angle_by_point(
            self.apply(self._wrist_center),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )
