"""Gait analysis module.

This module provides classes and utilities for gait analysis, including
GaitObject, GaitCycle, and GaitExercise, which support kinematic and kinetic
cycle detection, event extraction, and biofeedback summary generation.
"""

from ._base import GaitObject
from ._cycle import GaitCycle
from ._exercise import GaitExercise
from .running import RunningStep, RunningExercise
from .walking import WalkingStride, WalkingExercise


__all__ = [
    "GaitExercise",
    "GaitCycle",
    "GaitObject",
    "RunningExercise",
    "WalkingExercise",
    "RunningStep",
    "WalkingStride",
]
