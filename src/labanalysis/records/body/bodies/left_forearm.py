"""left forearm segment"""

from .joint import Joint
from .segment import Segment
from ....timeseries.point3d import Point3D


class LeftForearm(Segment):
    """left forearm segment class"""

    def __init__(
        self,
        left_elbow: Joint | Point3D | None,
        left_wrist: Joint | Point3D | None,
    ):
        super().__init__(left_elbow=left_elbow, left_wrist=left_wrist)

    @property
    def length(self):
        """return the length of the left forearm segment"""
        return self._get_distance(self.left_elbow, self.left_wrist)
