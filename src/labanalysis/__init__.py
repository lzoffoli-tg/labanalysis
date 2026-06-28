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
__version__ = "211"

# Sub-modules are accessible via qualified imports:
# from labanalysis.records import Record, ForcePlatform
# from labanalysis.timeseries import Signal1D, EMGSignal
# from labanalysis.protocols import UprightBalanceTest
# from labanalysis.modelling.ols import Ellipse
# from labanalysis.io import read_trc, write_mot

__all__ = []  # Prefer qualified imports for clarity
