"""right arm segment"""

from .joint import Joint
from .segment import Segment


class RightArm(Segment):
    """right arm segment class"""

    def __init__(
        self,
        right_shoulder: Joint | None,
        right_elbow: Joint | None,
    ):
        super().__init__(right_shoulder=right_shoulder, right_elbow=right_elbow)

    @property
    def length(self):
        """return the length of the right arm segment"""
        return self._get_distance(self.right_shoulder, self.right_elbow)
