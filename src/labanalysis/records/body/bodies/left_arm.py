"""left arm segment"""

from .joint import Joint
from .segment import Segment


class LeftArm(Segment):
    """left arm segment class"""

    def __init__(
        self,
        left_shoulder: Joint | None,
        left_elbow: Joint | None,
    ):
        super().__init__(left_shoulder=left_shoulder, left_elbow=left_elbow)

    @property
    def length(self):
        """return the length of the left arm segment"""
        return self._get_distance(self.left_shoulder, self.left_elbow)
