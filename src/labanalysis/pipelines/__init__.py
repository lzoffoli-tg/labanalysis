"""
Processing pipelines for biomechanical data.

Provides configurable processing pipelines for various signal types.
"""

from ._base import ProcessingPipeline
from ._defaults import (
    get_default_processing_pipeline,
    get_default_emgsignal_processing_func,
    get_default_signal1d_processing_func,
    get_default_signal3d_processing_func,
    get_default_point3d_processing_func,
    get_default_forceplatform_processing_func,
    get_default_metabolicrecord_processing_func,
)

__all__ = [
    "ProcessingPipeline",
    "get_default_processing_pipeline",
    "get_default_emgsignal_processing_func",
    "get_default_signal1d_processing_func",
    "get_default_signal3d_processing_func",
    "get_default_point3d_processing_func",
    "get_default_forceplatform_processing_func",
    "get_default_metabolicrecord_processing_func",
]
