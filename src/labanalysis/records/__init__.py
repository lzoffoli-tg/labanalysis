"""Records module for biomechanical data containers."""

# Core records classes
from ._base import *
from ._loc_indexer import *
from ._iloc_indexer import *
from .forceplatform import *
from .metabolicrecord import *
from .timeseriesrecord import *

# Submodules
from .body import *

__all__ = [
    # Core records
    "Record",
    "RecordLocIndexer",
    "RecordILocIndexer",
    "ForcePlatform",
    "MetabolicRecord",
    "TimeseriesRecord",
]
