"""
Helper functions for jump tests to work around bodymass_kg issue.

There is currently a bug in SingleJump/DropJump where bodymass_kg
is a required parameter but WholeBodyBase doesn't handle it properly.
This module provides workarounds for testing.
"""

from labanalysis.exercises import DropJump, SingleJump
from labanalysis.records import ForcePlatform


def create_singlejump_bypassing_init(
    left_foot_grf: ForcePlatform | None = None,
    right_foot_grf: ForcePlatform | None = None,
    bodymass_kg: float = 75.0,
) -> SingleJump:
    """
    Create a SingleJump instance bypassing the __init__ validation.

    This is a workaround for the bodymass_kg bug where SingleJump requires
    bodymass_kg but WholeBodyBase doesn't accept it.

    Parameters
    ----------
    left_foot_grf : ForcePlatform, optional
        Left foot ground reaction force platform
    right_foot_grf : ForcePlatform, optional
        Right foot ground reaction force platform
    bodymass_kg : float
        Body mass in kilograms

    Returns
    -------
    SingleJump
        Jump instance with bodymass_kg set directly as attribute
    """
    # Create instance using __new__ to bypass __init__
    jump = SingleJump.__new__(SingleJump)

    # Initialize parent class manually WITHOUT bodymass_kg
    from labanalysis.records.body import WholeBody
    WholeBody.__init__(
        jump,
        left_foot_ground_reaction_force=left_foot_grf,
        right_foot_ground_reaction_force=right_foot_grf,
    )

    # Set bodymass_kg directly in __dict__ to bypass __setattr__
    object.__setattr__(jump, 'bodymass_kg', bodymass_kg)

    # Set the force platforms using the proper methods
    jump.set_left_foot_ground_reaction_force(left_foot_grf)
    jump.set_right_foot_ground_reaction_force(right_foot_grf)

    return jump


def create_dropjump_bypassing_init(
    left_foot_grf: ForcePlatform | None = None,
    right_foot_grf: ForcePlatform | None = None,
    bodymass_kg: float = 75.0,
    box_height_cm: int = 40,
) -> DropJump:
    """
    Create a DropJump instance bypassing the __init__ validation.

    This is a workaround for the bodymass_kg bug.

    Parameters
    ----------
    left_foot_grf : ForcePlatform, optional
        Left foot ground reaction force platform
    right_foot_grf : ForcePlatform, optional
        Right foot ground reaction force platform
    bodymass_kg : float
        Body mass in kilograms
    box_height_cm : int
        Drop height in centimeters

    Returns
    -------
    DropJump
        Drop jump instance with bodymass_kg and box_height_cm set
    """
    # Create instance using __new__ to bypass __init__
    jump = DropJump.__new__(DropJump)

    # Initialize parent class manually WITHOUT bodymass_kg
    from labanalysis.records.body import WholeBody
    WholeBody.__init__(
        jump,
        left_foot_ground_reaction_force=left_foot_grf,
        right_foot_ground_reaction_force=right_foot_grf,
    )

    # Set bodymass_kg directly in __dict__ to bypass __setattr__
    object.__setattr__(jump, 'bodymass_kg', bodymass_kg)

    # Set the force platforms and box height using proper methods
    jump.set_left_foot_ground_reaction_force(left_foot_grf)
    jump.set_right_foot_ground_reaction_force(right_foot_grf)
    jump.set_box_height_cm(box_height_cm)

    return jump


__all__ = [
    'create_singlejump_bypassing_init',
    'create_dropjump_bypassing_init',
]
