"""right foot segment"""

from .joint import Joint
from .segment import Segment


class RightThigh(Segment):
    """right thigh segment class"""

    def __init__(
        self,
        right_hip: Joint | None,
        right_knee: Joint | None,
    ):
        super().__init__(right_hip=right_hip, right_knee=right_knee)

    @property
    def length(self):
        """return the length of the right thigh segment"""
        return self._get_distance(self.right_hip, self.right_knee)
