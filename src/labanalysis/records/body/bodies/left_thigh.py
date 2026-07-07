"""left foot segment"""

from .joint import Joint
from .segment import Segment


class LeftThigh(Segment):
    """left thigh segment class"""

    def __init__(
        self,
        left_hip: Joint | None,
        left_knee: Joint | None,
    ):
        super().__init__(left_hip=left_hip, left_knee=left_knee)

    @property
    def length(self):
        """return the length of the left thigh segment"""
        return self._get_distance(self.left_hip, self.left_knee)
