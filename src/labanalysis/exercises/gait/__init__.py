"""Gait analysis module.

This module provides classes and utilities for gait analysis, including
GaitObject, GaitCycle, and GaitExercise, which support kinematic and kinetic
cycle detection, event extraction, and biofeedback summary generation.
"""

from ._base import GaitObject
from ._cycle import GaitCycle
from ._exercise import GaitExercise
from .running_step import RunningStep
from .running_exercise import RunningExercise
from .walking_stride import WalkingStride
from .walking_exercise import WalkingExercise


__all__ = [
    "GaitExercise",
    "GaitCycle",
    "GaitObject",
    "RunningExercise",
    "WalkingExercise",
    "RunningStep",
    "WalkingStride",
]
