"""Composite mixin for all angular measurement properties."""

from ._helpers import AnglesHelpersMixin
from .ankle import AnkleAnglesMixin
from .knee import KneeAnglesMixin
from .hip import HipAnglesMixin
from .pelvis import PelvisAnglesMixin
from .trunk import TrunkAnglesMixin
from .neck import NeckAnglesMixin
from .shoulder import ShoulderAnglesMixin
from .scapular import ScapularAnglesMixin
from .elbow import ElbowAnglesMixin
from .spine import SpineAnglesMixin

__all__ = ["AngularMeasuresMixin"]


class AngularMeasuresMixin(
    AnglesHelpersMixin,
    AnkleAnglesMixin,
    KneeAnglesMixin,
    HipAnglesMixin,
    PelvisAnglesMixin,
    TrunkAnglesMixin,
    NeckAnglesMixin,
    ShoulderAnglesMixin,
    ScapularAnglesMixin,
    ElbowAnglesMixin,
    SpineAnglesMixin,
):
    """
    Composite mixin for all angular measurement properties.

    Combines angle mixins in proper MRO order:
    - Helpers (foot planes, pelvis plane - used by other angle properties)
    - Ankle angles (flexion/extension, inversion/eversion)
    - Knee angles (flexion/extension, varus/valgus)
    - Hip angles (flexion/extension, abduction/adduction, rotation)
    - Pelvis angles (reference frame, tilt, rotation global/local)
    - Trunk angles (lateral flexion, rotation, flexion/extension)
    - Neck angles (flexion/extension, lateral flexion)
    - Shoulder angles (flexion/extension, abduction/adduction, rotation, elevation)
    - Scapular angles (protraction/retraction, elevation/depression)
    - Elbow angles (flexion/extension)
    - Spine angles (lumbar lordosis, dorsal kyphosis)
    """
    pass
