"""left foot segment"""

from ....timeseries.point3d import Point3D
from .body_plane import BodyPlane
from .segment import Segment

__all__ = ["LeftFoot"]


class LeftFoot(Segment, BodyPlane):
    """left foot segment class"""

    def __init__(
        self,
        left_toe: Point3D | None,
        left_heel: Point3D | None,
        left_fifth_metatarsal_head: Point3D | None,
        left_first_metatarsal_head: Point3D | None,
    ):
        super().__init__(
            left_toe=left_toe,
            left_heel=left_heel,
            left_fifth_metatarsal_head=left_fifth_metatarsal_head,
            left_first_metatarsal_head=left_first_metatarsal_head,
        )

    @property
    def length(self):
        """return the length of the left foot segment"""
        return self._get_distance(self.left_toe, self.left_heel)

    @property
    def width(self):
        """return the width of the left foot segment"""
        return self._get_distance(
            self.left_first_metatarsal_head,
            self.left_fifth_metatarsal_head,
        )
