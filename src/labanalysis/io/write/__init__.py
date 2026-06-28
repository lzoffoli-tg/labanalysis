"""Writing data to various biomechanical file formats."""

from .opensim import *

__all__ = [
    "write_mot",
    "write_trc",
]
