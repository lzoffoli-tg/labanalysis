"""right foot segment"""

from .joint import Joint
from .segment import Segment


class RightLeg(Segment):
    """right leg segment class"""

    def __init__(
        self,
        right_ankle: Joint | None,
        right_knee: Joint | None,
    ):
        super().__init__(right_ankle=right_ankle, right_knee=right_knee)

    @property
    def length(self):
        """return the length of the right foot segment"""
        return self._get_distance(self.right_ankle, self.right_knee)
