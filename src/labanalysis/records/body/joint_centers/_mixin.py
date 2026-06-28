"""Composite mixin for all joint center properties."""

from .ankle import AnkleJointsMixin
from .knee import KneeJointsMixin
from .hip import HipJointsMixin
from .shoulder import ShoulderJointsMixin
from .elbow import ElbowJointsMixin
from .wrist import WristJointsMixin
from .axial import AxialJointsMixin

__all__ = ["JointCentersMixin"]


class JointCentersMixin(
    AnkleJointsMixin,
    KneeJointsMixin,
    HipJointsMixin,
    ShoulderJointsMixin,
    ElbowJointsMixin,
    WristJointsMixin,
    AxialJointsMixin,
):
    """
    Composite mixin for all joint center and reference frame properties.

    Combines anatomical joint center mixins in proper MRO order:
    - Ankle joints (lateral/medial markers)
    - Knee joints (lateral/medial markers)
    - Hip joints (ASIS/trochanter/pelvis center)
    - Shoulder joints (acromion/anterior/posterior markers)
    - Elbow joints (lateral/medial markers)
    - Wrist joints (radial/ulnar markers)
    - Axial joints (head, neck base)
    """
    pass
