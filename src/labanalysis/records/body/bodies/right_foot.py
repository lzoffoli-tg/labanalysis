"""right foot segment"""

from ....timeseries.point3d import Point3D
from .body_plane import BodyPlane
from .segment import Segment


class RightFoot(Segment, BodyPlane):
    """right foot segment class"""

    def __init__(
        self,
        right_toe: Point3D | None,
        right_heel: Point3D | None,
        right_fifth_metatarsal_head: Point3D | None,
        right_first_metatarsal_head: Point3D | None,
    ):
        super().__init__(
            right_toe=right_toe,
            right_heel=right_heel,
            right_fifth_metatarsal_head=right_fifth_metatarsal_head,
            right_first_metatarsal_head=right_first_metatarsal_head,
        )

    @property
    def length(self):
        """return the length of the right foot segment"""
        return self._get_distance(self.right_toe, self.right_heel)

    @property
    def width(self):
        """return the width of the right foot segment"""
        return self._get_distance(
            self.right_first_metatarsal_head, self.right_fifth_metatarsal_head
        )
