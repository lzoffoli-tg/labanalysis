"""Composite mixin for all anthropometric measurement properties."""

from .pelvis import PelvisMeasuresMixin
from .foot import FootMeasuresMixin
from .lower_limb import LowerLimbMeasuresMixin
from .upper_limb import UpperLimbMeasuresMixin
from .trunk import TrunkMeasuresMixin

__all__ = ["AnthropometryMixin"]


class AnthropometryMixin(
    PelvisMeasuresMixin,
    FootMeasuresMixin,
    LowerLimbMeasuresMixin,
    UpperLimbMeasuresMixin,
    TrunkMeasuresMixin,
):
    """
    Composite mixin for all anthropometric measurement properties.

    Combines measurement mixins in proper MRO order:
    - Pelvis measurements (width, height)
    - Foot measurements (height, length, width for left/right)
    - Lower limb measurements (ankle width, leg/thigh lengths, knee width)
    - Upper limb measurements (elbow width, arm/forearm lengths)
    - Trunk measurements (shoulder width, hip width, trunk length, reference frame)
    """
    pass
