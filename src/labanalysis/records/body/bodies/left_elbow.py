"""left elbow joint module"""

from .left_shoulder import LeftShoulder
from ....timeseries.point3d import Point3D
from .joint import Joint

__all__ = ["LeftElbow"]


class LeftElbow(Joint):
    """LeftElbow Joint."""

    def __init__(
        self,
        left_shoulder: LeftShoulder,
        left_elbow_lateral: Point3D,
        left_elbow_medial: Point3D,
        left_wrist_lateral: Point3D,
        left_wrist_medial: Point3D,
    ):

        lax = left_elbow_lateral - left_elbow_medial
        elb = (left_elbow_lateral + left_elbow_medial) / 2
        vrt = elb - left_shoulder.center
        super().__init__(origin=elb, lateral_vector=lax, vertical_vector=vrt)  # type: ignore

        self["wrist_medial"] = left_wrist_medial
        self["wrist_lateral"] = left_wrist_lateral

    @property
    def _wrist_medial(self):
        """return the left wrist medial in the global reference frame"""
        out: Point3D = self["wrist_medial"]  # type: ignore
        return out

    @property
    def _wrist_lateral(self):
        """return the left wrist lateral in the global reference frame"""
        out: Point3D = self["wrist_lateral"]  # type: ignore
        return out

    @property
    def _wrist_center(self):
        """return the left wrist centre in the elbow reference frame"""
        out: Point3D = (self["wrist_medial"] + self["wrist_lateral"]) / 2  # type: ignore
        return out

    @property
    def flexionextension(self):
        """
        Calculate left elbow flexion-extension angle in sagittal plane.

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
