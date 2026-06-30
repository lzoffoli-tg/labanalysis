"""
labanalysis - Biomechanical and physiological data analysis library.

Main modules:
- records: Data containers for biomechanical measurements
- timeseries: Time-indexed signal processing
- protocols: Test protocols and normative comparisons
- modelling: Regression and machine learning models
- io: File I/O for various formats
- equations: Metabolic and strength prediction equations
"""

# Version
__version__ = "221"

# Wildcard imports from all main submodules and subpackages
from .equations import *
from .exercises import *
from .io import *
from .modelling import *
from .pipelines import *
from .plotting import *
from .protocols import *
from .records import *
from .referenceframes import *
from .timeseries import *
from .utils import *
from .signalprocessing import *
from .constants import *
from .messages import *

# Common usage:
# import labanalysis as la
# body = la.WholeBody(...)
# signal = la.Signal1D(...)
# rf = la.ReferenceFrame(...)
# participant = la.Participant(...)
# jump = la.SingleJump.from_tdf(...)
