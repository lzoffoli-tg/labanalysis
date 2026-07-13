"""right forearm segment"""

from .joint import Joint
from ....timeseries.point3d import Point3D
from .segment import Segment


class RightForearm(Segment):
    """right forearm segment class"""

    def __init__(
        self,
        right_elbow: Joint | Point3D | None,
        right_wrist: Joint | Point3D | None,
    ):
        super().__init__(right_elbow=right_elbow, right_wrist=right_wrist)

    @property
    def length(self):
        """return the length of the right forearm segment"""
        return self._get_distance(self.right_elbow, self.right_wrist)
