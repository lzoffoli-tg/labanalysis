"""Metabolic and strength prediction equations."""

from .cardio import Run, Bike
from .strength import Brzycki1RM

__all__ = ["Run", "Bike", "Brzycki1RM"]
