# Code Style Guide

Python coding standards and conventions for labanalysis.

## Overview

Consistent code style improves readability, maintainability, and collaboration. This guide covers:

- **PEP 8 compliance** for Python code
- **NumPy docstring format** for documentation
- **Type hints** for function signatures
- **Naming conventions** for classes, functions, variables
- **Import organization**

## Quick Reference

```python
"""Module docstring describing purpose."""

import standard_library
import third_party
import labanalysis as laban

from labanalysis.records import Signal1D


def calculate_velocity(position: Signal1D, method: str = 'winter') -> Signal1D:
    """
    Calculate velocity from position signal.
    
    Parameters
    ----------
    position : Signal1D
        Position signal in meters
    method : str, optional
        Differentiation method (default: 'winter')
    
    Returns
    -------
    Signal1D
        Velocity signal in m/s
    
    Examples
    --------
    >>> velocity = calculate_velocity(ankle_position)
    """
    # Implementation here
    pass
```

## PEP 8 Compliance

Follow [PEP 8](https://peps.python.org/pep-0008/) standards:

### Line Length

```python
# Maximum 88 characters (Black default)
def long_function_name(parameter1, parameter2, parameter3, parameter4, parameter5):
    """Keep lines under 88 characters."""
    pass

# Break long lines
result = some_long_function_name(
    first_parameter,
    second_parameter,
    third_parameter,
)
```

### Indentation

```python
# 4 spaces per indentation level
def function():
    if condition:
        do_something()
    else:
        do_something_else()

# Align multi-line constructs
data = {
    'key1': value1,
    'key2': value2,
    'key3': value3,
}
```

### Blank Lines

```python
# 2 blank lines around top-level functions and classes
def function1():
    pass


def function2():
    pass


class MyClass:
    """Class docstring."""
    
    # 1 blank line between methods
    def method1(self):
        pass
    
    def method2(self):
        pass
```

### Whitespace

```python
# ✅ GOOD: Spaces around operators
result = value1 + value2
x = y * 2

# ❌ BAD: No spaces
result=value1+value2

# ✅ GOOD: No space before comma, space after
items = [1, 2, 3, 4]

# ❌ BAD
items = [1,2,3,4]

# ✅ GOOD: Function calls
function(arg1, arg2, kwarg1=value1)

# ❌ BAD
function(arg1,arg2,kwarg1 = value1)
```

## Naming Conventions

### Functions and Variables

```python
# snake_case for functions and variables
def calculate_jump_height(flight_time):
    peak_velocity = 9.81 * flight_time / 2
    return peak_velocity**2 / (2 * 9.81)

# ❌ BAD: camelCase
def calculateJumpHeight(flightTime):
    pass
```

### Classes

```python
# PascalCase for classes
class JumpTest:
    """Jump test protocol."""
    pass

class SingleJump:
    """Single jump analysis."""
    pass

# ❌ BAD: snake_case
class jump_test:
    pass
```

### Constants

```python
# UPPER_CASE for constants
GRAVITY = 9.81  # m/s²
MAX_JUMP_HEIGHT = 2.0  # meters
DEFAULT_FILTER_FREQUENCY = 10  # Hz

# Use in module-level or class-level scope
class PhysicalConstants:
    GRAVITY = 9.81
    STANDARD_PRESSURE = 101325  # Pa
```

### Private Members

```python
class MyClass:
    """Example class."""
    
    def __init__(self):
        self.public_attribute = 10
        self._internal_attribute = 20  # Internal (not part of public API)
        self.__private_attribute = 30  # Name mangled (avoid unless needed)
    
    def public_method(self):
        """Public method."""
        pass
    
    def _internal_method(self):
        """Internal method (not documented in public API)."""
        pass
```

## Type Hints

Use type hints for all function signatures:

```python
from typing import Optional, List, Tuple, Union
import numpy as np
from labanalysis.records import Signal1D, Signal3D

def butterworth_filter(
    signal: Union[Signal1D, Signal3D],
    frequency: float,
    order: int = 4
) -> Union[Signal1D, Signal3D]:
    """
    Apply Butterworth filter.
    
    Parameters
    ----------
    signal : Signal1D or Signal3D
        Input signal
    frequency : float
        Cutoff frequency in Hz
    order : int, optional
        Filter order (default: 4)
    
    Returns
    -------
    Signal1D or Signal3D
        Filtered signal (same type as input)
    """
    pass


def find_peaks(
    signal: Signal1D,
    threshold: Optional[float] = None,
    distance: int = 1
) -> Tuple[np.ndarray, dict]:
    """
    Find peaks in signal.
    
    Parameters
    ----------
    signal : Signal1D
        Input signal
    threshold : float, optional
        Minimum peak height (default: None)
    distance : int, optional
        Minimum distance between peaks (default: 1)
    
    Returns
    -------
    peaks : np.ndarray
        Indices of peaks
    properties : dict
        Peak properties
    """
    pass
```

### Common Type Hints

```python
# Built-in types
def process_data(value: int, name: str, enabled: bool) -> float:
    pass

# Optional (can be None)
def get_marker(label: str) -> Optional[Signal3D]:
    pass

# Union (multiple types)
def load_signal(data: Union[np.ndarray, List[float]]) -> Signal1D:
    pass

# Collections
def batch_process(files: List[str]) -> List[dict]:
    pass

def get_statistics(signal: Signal1D) -> dict[str, float]:
    pass

# NumPy arrays
def calculate_mean(data: np.ndarray) -> float:
    pass
```

## Docstring Format

Use NumPy docstring style:

### Function Docstrings

```python
def calculate_rfd(force_signal: Signal1D, window: float = 0.05) -> float:
    """
    Calculate rate of force development.
    
    Computes the maximum rate of force development (RFD) over a sliding
    window. RFD is calculated as the slope of the force-time curve.
    
    Parameters
    ----------
    force_signal : Signal1D
        Force signal in Newtons
    window : float, optional
        Time window for RFD calculation in seconds (default: 0.05)
    
    Returns
    -------
    float
        Maximum RFD in N/s
    
    Raises
    ------
    ValueError
        If window is negative or larger than signal duration
    
    See Also
    --------
    calculate_impulse : Calculate force impulse
    find_peaks : Find force peaks
    
    Notes
    -----
    RFD is calculated using linear regression over the specified window.
    A 50ms window (0.05s) is commonly used in the literature [1]_.
    
    References
    ----------
    .. [1] Maffiuletti et al. (2016). Rate of force development: 
           physiological and methodological considerations. Eur J Appl
           Physiol, 116(6), 1091-1116.
    
    Examples
    --------
    >>> force = load_force_signal('jump.tdf')
    >>> rfd = calculate_rfd(force, window=0.05)
    >>> print(f"Max RFD: {rfd:.0f} N/s")
    Max RFD: 8500 N/s
    """
    pass
```

### Class Docstrings

```python
class SingleJump:
    """
    Single countermovement jump analysis.
    
    Analyzes countermovement jump (CMJ) performance from force platform
    data, extracting jump height, peak force, peak power, and phase
    characteristics.
    
    Parameters
    ----------
    force_platform : ForcePlatform
        Force platform data containing vertical GRF
    body_mass : float
        Athlete body mass in kg
    
    Attributes
    ----------
    grf_z : Signal1D
        Vertical ground reaction force
    body_weight : float
        Body weight in Newtons (mass × g)
    
    Methods
    -------
    process()
        Process jump and extract metrics
    detect_phases()
        Identify jump phases (unweighting, braking, propulsion)
    
    Examples
    --------
    >>> jump = SingleJump.from_tdf_file('athlete_cmj.tdf')
    >>> results = jump.process()
    >>> print(f"Jump height: {results.jump_height:.2f} m")
    Jump height: 0.42 m
    
    See Also
    --------
    DropJump : Drop jump analysis
    RepeatedJumps : Repeated jump analysis
    
    Notes
    -----
    Jump height is calculated from flight time using the equation:
    
    .. math:: h = \\frac{g \\cdot t_{flight}^2}{8}
    
    where g is gravitational acceleration (9.81 m/s²).
    """
    
    def __init__(self, force_platform, body_mass):
        """Initialize jump analysis."""
        pass
    
    def process(self) -> 'JumpResults':
        """
        Process jump and extract metrics.
        
        Returns
        -------
        JumpResults
            Results container with all jump metrics
        """
        pass
```

### Property Docstrings

```python
class WholeBody:
    """Full-body kinematic model."""
    
    @property
    def left_ankle(self) -> Point3D:
        """
        Left ankle joint center.
        
        Computed as the midpoint between left ankle medial and lateral
        malleolus markers.
        
        Returns
        -------
        Point3D
            3D position of left ankle center in meters
        
        See Also
        --------
        right_ankle : Right ankle joint center
        left_knee : Left knee joint center
        """
        pass
```

## Import Organization

Organize imports in three groups:

```python
"""Module for signal processing functions."""

# 1. Standard library
import sys
from pathlib import Path
from typing import Optional, Union

# 2. Third-party packages
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# 3. Local imports
import labanalysis as laban
from labanalysis.records import Signal1D, Signal3D
from labanalysis.utils import validate_signal
```

**Import guidelines**:
- Absolute imports preferred over relative
- One import per line (except `from X import a, b`)
- Alphabetical within groups
- Avoid wildcard imports (`from module import *`)

## Code Organization

### Function Length

Keep functions focused and concise:

```python
# ✅ GOOD: Single responsibility
def detect_takeoff(force_signal: Signal1D, threshold: float) -> int:
    """Detect takeoff instant from force signal."""
    above_threshold = force_signal.to_numpy() > threshold
    crossings = np.diff(above_threshold.astype(int))
    takeoff_idx = np.where(crossings == -1)[0][0]
    return takeoff_idx

# ❌ BAD: Too many responsibilities
def analyze_jump(force_signal):
    """Analyze jump (too much in one function)."""
    # Detect takeoff
    # Calculate flight time
    # Calculate jump height
    # Calculate peak force
    # Calculate peak power
    # ... 100+ lines
    pass
```

### Class Design

Follow Single Responsibility Principle:

```python
# ✅ GOOD: Focused classes
class JumpPhaseDetector:
    """Detects phases in jump tests."""
    
    def detect_unweighting(self):
        pass
    
    def detect_braking(self):
        pass
    
    def detect_propulsion(self):
        pass


class JumpMetricsCalculator:
    """Calculates jump metrics."""
    
    def calculate_jump_height(self):
        pass
    
    def calculate_peak_power(self):
        pass

# ❌ BAD: God class (too many responsibilities)
class JumpAnalyzer:
    """Does everything."""
    
    def load_data(self):
        pass
    
    def filter_data(self):
        pass
    
    def detect_phases(self):
        pass
    
    def calculate_metrics(self):
        pass
    
    def generate_report(self):
        pass
    
    def export_to_excel(self):
        pass
```

## Error Handling

### Raise Informative Exceptions

```python
# ✅ GOOD: Clear error message
def butterworth_filter(signal, frequency, order=4):
    """Apply Butterworth filter."""
    if frequency <= 0:
        raise ValueError(
            f"Frequency must be positive, got {frequency} Hz"
        )
    
    if order < 1:
        raise ValueError(
            f"Filter order must be >= 1, got {order}"
        )
    
    # Implementation
    pass

# ❌ BAD: Vague error
def butterworth_filter(signal, frequency, order=4):
    if frequency <= 0:
        raise ValueError("Invalid frequency")
```

### Use Appropriate Exception Types

```python
# ValueError for invalid values
if value < 0:
    raise ValueError("Value must be non-negative")

# TypeError for wrong types
if not isinstance(signal, Signal1D):
    raise TypeError("Expected Signal1D")

# FileNotFoundError for missing files
if not filepath.exists():
    raise FileNotFoundError(f"File not found: {filepath}")

# KeyError for missing keys
if label not in self.markers:
    raise KeyError(f"Marker '{label}' not found")
```

## Comments

### When to Comment

```python
# ✅ GOOD: Explain WHY, not WHAT
# Use Winter's method for derivatives to minimize noise amplification
velocity = derivative(position, method='winter')

# De Leva 1996 anthropometric parameters for center of mass estimation
com_fraction = 0.433  # Fraction of segment length

# ❌ BAD: Stating the obvious
# Calculate velocity
velocity = derivative(position)

# Set variable to 10
threshold = 10
```

### Complex Algorithms

```python
def calculate_force_velocity_profile(forces, velocities):
    """Calculate F-V profile from jump data."""
    
    # Fit 2nd-degree polynomial: F = a*v² + b*v + c
    # Using least squares to minimize residuals
    coeffs = np.polyfit(velocities, forces, deg=2)
    
    # Extract F0 (force at v=0) and V0 (velocity at F=0)
    # F0 = c (y-intercept)
    # V0 = larger root of quadratic equation
    F0 = coeffs[2]
    roots = np.roots(coeffs)
    V0 = np.max(roots)  # Take positive root
    
    return F0, V0
```

## Automated Formatting

### Black

Use [Black](https://black.readthedocs.io/) for automatic code formatting:

```bash
# Format file
black src/labanalysis/signalprocessing.py

# Format directory
black src/labanalysis/

# Check without modifying
black --check src/
```

**Black configuration** (pyproject.toml):
```toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
```

### Flake8

Use [Flake8](https://flake8.pycqa.org/) for style checking:

```bash
# Check file
flake8 src/labanalysis/signalprocessing.py

# Check directory
flake8 src/labanalysis/
```

**Flake8 configuration** (.flake8):
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist
```

### isort

Use [isort](https://pycqa.github.io/isort/) for import sorting:

```bash
# Sort imports
isort src/labanalysis/

# Check without modifying
isort --check-only src/
```

**isort configuration** (pyproject.toml):
```toml
[tool.isort]
profile = "black"
line_length = 88
```

## Type Checking

Use [mypy](http://mypy-lang.org/) for static type checking:

```bash
# Check types
mypy src/labanalysis/
```

**mypy configuration** (mypy.ini):
```ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
ignore_missing_imports = True
```

## Pre-commit Hooks

Automate style checks with [pre-commit](https://pre-commit.com/):

**.pre-commit-config.yaml**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
```

Install hooks:
```bash
pip install pre-commit
pre-commit install
```

## Summary Checklist

Before committing code:

- [ ] Follows PEP 8 conventions
- [ ] Uses type hints for function signatures
- [ ] Has NumPy-style docstrings
- [ ] Formatted with Black
- [ ] Passes Flake8 checks
- [ ] Imports sorted with isort
- [ ] Functions are focused and concise
- [ ] Clear variable/function names
- [ ] Appropriate error handling
- [ ] Comments explain WHY, not WHAT

## See Also

- [Testing Guide](testing.md) - Writing tests
- [Contributing Guide](contributing.md) - Contribution workflow
- [Architecture Guide](architecture.md) - Library design

---

**Follow PEP 8, use type hints, and write NumPy docstrings.** Use Black, Flake8, and isort to automate formatting and style checking.
