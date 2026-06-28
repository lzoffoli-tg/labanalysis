"""Testing protocols module for biomechanical assessments."""

# Base protocols classes
from .participant import *
from .test_results import *
from .test_protocol import *

# Test modules
from .balancetests import *
from .strengthtests import *
from .jumptests import *
from .agilitytests import *
from .locomotiontests import *
from .vo2max import *

# Normative data
from .normativedata import *

__all__ = [
    # Base protocols classes
    "Participant",
    "TestResults",
    "TestProtocol",
    # Balance tests
    "UprightBalanceTest",
    "UprightBalanceTestResults",
    "PlankBalanceTest",
    "PlankBalanceTestResults",
    # Strength tests
    "Isokinetic1RMTest",
    "Isokinetic1RMTestResults",
    "IsometricTest",
    "IsometricTestResults",
    # Jump tests
    "JumpTest",
    "JumpTestResults",
    # Agility tests
    "ShuttleTest",
    "ShuttleTestResults",
    # Locomotion tests
    "RunningTest",
    "WalkingTest",
    # VO2max tests
    "SubmaximalVO2MaxTest",
    "SubmaximalVO2MaxTestResults",
]
