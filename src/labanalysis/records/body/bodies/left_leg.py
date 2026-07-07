"""left foot segment"""

from .joint import Joint
from .segment import Segment


class LeftLeg(Segment):
    """left leg segment class"""

    def __init__(
        self,
        left_ankle: Joint | None,
        left_knee: Joint | None,
    ):
        super().__init__(left_ankle=left_ankle, left_knee=left_knee)

    @property
    def length(self):
        """return the length of the left foot segment"""
        return self._get_distance(self.left_ankle, self.left_knee)
