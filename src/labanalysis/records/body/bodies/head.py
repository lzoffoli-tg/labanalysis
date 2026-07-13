"""left hip joint module"""

from .segment import Segment
from .body_plane import BodyPlane
from ....timeseries import Point3D
from ....events.signal import Signal

__all__ = ["Head"]


class Head(BodyPlane, Segment):
    """Head Plane."""

    def __init__(
        self,
        front: Point3D | None,
        back: Point3D | None,
        left: Point3D | None,
        right: Point3D | None,
    ):
        points = {}
        if front is not None:
            points["front"] = front
        if back is not None:
            points["back"] = back
        if left is not None:
            points["left"] = left
        if right is not None:
            points["right"] = right
        super().__init__(**points)
