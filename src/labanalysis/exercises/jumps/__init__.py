"""
Vertical jump analysis module.

This module provides classes for comprehensive vertical jump performance
assessment from force platform and motion capture data.

Classes
-------
SquatJump
    Concentric-only jump from static semi-squat position.
CounterMovementJump
    Jump with eccentric pre-stretch movement (countermovement).
SingleJump
    Single vertical jump (extends CounterMovementJump).
DropJump
    Plyometric jump from elevated surface with landing phase.
RepeatedJumps
    Continuous jumping sequence for fatigue assessment.

Notes
-----
Class hierarchy:
- SquatJump extends WholeBody
- CounterMovementJump extends SquatJump
- SingleJump extends CounterMovementJump
- DropJump extends CounterMovementJump
- RepeatedJumps extends WholeBody

All jump classes support:
- Force platform data processing
- Motion capture marker tracking
- Phase detection (contact/flight)
- Kinetic and kinematic analysis
- Multiple jump height estimation methods
"""

from .squat_jump import *
from .counter_movement_jump import *
from .single_jump import *
from .repeated_jumps import *
from .drop_jump import *
