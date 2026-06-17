# Library Architecture

Design principles and module organization of labanalysis.

## Overview

The labanalysis library follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────┐
│     User-Facing Protocols Layer    │  High-level test protocols
├─────────────────────────────────────┤
│      Records and Bodies Layer       │  Data structures (WholeBody, ForcePlatform)
├─────────────────────────────────────┤
│   Signal Processing & Modeling      │  Core algorithms
├─────────────────────────────────────┤
│          I/O and Utils              │  File reading, utilities
└─────────────────────────────────────┘
```

## Module Structure

```
labanalysis/
├── __init__.py                  # Public API exports
├── records/                     # Data structures
│   ├── timeseries.py           # Signal1D, Signal3D, Point3D, EMGSignal
│   ├── records.py              # Record, TimeseriesRecord, ForcePlatform
│   ├── bodies.py               # WholeBody (88 properties)
│   ├── jumping.py              # SingleJump, DropJump, RepeatedJumps
│   ├── locomotion.py           # RunningExercise, WalkingExercise
│   ├── posture.py              # UprightPosture, PronePosture
│   └── agility.py              # ChangeOfDirectionExercise
├── protocols/                   # Test protocols
│   ├── protocols.py            # TestProtocol, TestResults base classes
│   ├── jump_protocols.py       # JumpTest, JumpTestResults
│   ├── balance_protocols.py    # UprightBalanceTest, PlankBalanceTest
│   ├── strength_protocols.py  # Isokinetic1RMTest
│   ├── locomotion_protocols.py # RunningTest, WalkingTest
│   ├── agility_protocols.py    # ShuttleTest
│   └── vo2max_protocols.py     # SubmaximalVO2MaxTest
├── signalprocessing/            # Signal processing algorithms
│   ├── filters.py              # Butterworth, median, running mean
│   ├── peaks.py                # Peak detection
│   ├── derivatives.py          # Winter's method, gradients
│   ├── frequency.py            # FFT, PSD
│   └── transformations.py      # Coordinate transforms, rotations
├── modelling/                   # Machine learning
│   ├── ols.py                  # OLS, polynomial, power regression
│   └── pytorch/                # PyTorch utilities
│       ├── trainer.py          # TorchTrainer
│       ├── datasets.py         # CustomDataset
│       └── models.py           # Standard architectures
├── equations/                   # Biomechanical equations
│   ├── strength.py             # Brzycki1RM, Epley
│   └── cardio.py               # VO2max equations (Run, Bike)
├── io/                          # File I/O
│   ├── read/                   # Reading functions
│   │   ├── bts.py              # read_tdf()
│   │   ├── opensim.py          # read_opensim()
│   │   ├── biostrength.py      # read_biostrength()
│   │   └── ircam.py            # read_ircam()
│   └── write/                  # Writing functions
│       └── opensim.py          # write_opensim()
├── plotting/                    # Visualization
│   └── plotly.py               # Plotly wrappers
├── utils/                       # Utilities
│   ├── validation.py           # Input validation
│   └── constants.py            # Physical constants
└── messages/                    # User messages
    └── messages.py             # Error, warning messages
```

## Design Principles

### 1. Layered Architecture

**Bottom-up layers**:

1. **I/O Layer** - File reading/writing
2. **Core Data Structures** - Timeseries, Signal, Point3D
3. **Signal Processing** - Filters, derivatives, peaks
4. **Biomechanical Records** - WholeBody, ForcePlatform
5. **Test Protocols** - High-level analysis workflows

Each layer depends only on layers below it.

### 2. Separation of Concerns

```python
# ✅ GOOD: Separate responsibilities

# Data structure (records/timeseries.py)
class Signal1D:
    """Time series data container."""
    pass

# Algorithm (signalprocessing/filters.py)
def butterworth_filter(signal: Signal1D) -> Signal1D:
    """Filter signal."""
    pass

# Protocol (protocols/jump_protocols.py)
class JumpTest:
    """High-level jump analysis."""
    def process(self):
        # Uses Signal1D + butterworth_filter
        pass
```

### 3. Composition Over Inheritance

```python
# ✅ GOOD: Composition
class SingleJump:
    """Jump analysis using composition."""
    
    def __init__(self, force_platform, body_mass):
        self.force_platform = force_platform  # Has-a relationship
        self.body_mass = body_mass
    
    def process(self):
        # Uses force_platform methods
        grf = self.force_platform["FORCE", "Z"]
        # ...

# ❌ AVOID: Deep inheritance chains
class JumpTest(TestProtocol):
    pass

class CountermovementJump(JumpTest):
    pass

class BiomechanicalCountermovementJump(CountermovementJump):
    pass  # Too deep
```

### 4. Dependency Injection

```python
# ✅ GOOD: Inject dependencies
def analyze_jump(force_platform: ForcePlatform, body_mass: float):
    """Analyze jump with injected dependencies."""
    test = SingleJump(force_platform, body_mass)
    return test.process()

# Usage
fp = ForcePlatform.from_tdf_file("jump.tdf")
results = analyze_jump(fp, body_mass=75)

# ❌ BAD: Hidden dependencies
def analyze_jump(filepath: str):
    """Hidden file I/O dependency."""
    fp = ForcePlatform.from_tdf_file(filepath)  # Hardcoded
    # ...
```

### 5. Immutability When Possible

```python
# ✅ GOOD: Immutable operations
def butterworth_filter(signal: Signal1D) -> Signal1D:
    """Return new filtered signal (original unchanged)."""
    filtered_data = apply_filter(signal.to_numpy())
    
    return Signal1D(
        data=filtered_data,
        index=signal.index,
        columns=signal.columns,
        unit=signal.unit
    )

# Original signal unchanged
filtered = butterworth_filter(original_signal)

# ❌ BAD: Mutating operations
def butterworth_filter(signal: Signal1D) -> None:
    """Modify signal in-place."""
    signal._data = apply_filter(signal._data)  # Mutates
```

## Core Components

### Timeseries Hierarchy

```
Timeseries (Abstract)
    ├── Signal1D (1D time series)
    │   └── EMGSignal (EMG-specific)
    ├── Signal3D (3D time series)
    └── Point3D (3D position)
```

**Design rationale**:
- `Signal1D`: Generic 1D signal (force, angle, etc.)
- `Signal3D`: Multi-channel signal (3 axes)
- `Point3D`: Specialized 3D signal with spatial operations
- `EMGSignal`: Specialized 1D with EMG-specific methods

### Record Pattern

```python
class TimeseriesRecord:
    """
    Container for multiple timeseries.
    
    Uses hierarchical column indexing for organization.
    """
    
    def __init__(self):
        self._data = {}  # Internal storage
    
    def set(self, key, value):
        """Add timeseries to record."""
        self._data[key] = value
    
    def get(self, key):
        """Retrieve timeseries from record."""
        return self._data[key]
    
    def __getitem__(self, key):
        """Dictionary-like access."""
        return self._data[key]
```

**Examples**:
- `ForcePlatform`: Contains FORCE and MOMENT channels
- `WholeBody`: Contains marker positions and computed properties

### Protocol Pattern

```python
class TestProtocol(ABC):
    """Abstract base for test protocols."""
    
    @classmethod
    @abstractmethod
    def from_files(cls, *args, **kwargs):
        """Load test from file(s)."""
        pass
    
    @abstractmethod
    def process(self) -> 'TestResults':
        """Process test and return results."""
        pass


class TestResults(ABC):
    """Abstract base for test results."""
    
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Export results to DataFrame."""
        pass
```

**Benefits**:
- Consistent API across all test types
- Easy to add new protocols
- Clear separation: data loading → processing → results

## Data Flow

### Typical Analysis Workflow

```python
# 1. Load data (I/O layer)
from labanalysis.io import read_tdf
markers = read_tdf("gait.tdf", labels=["left_ankle", "left_knee"])

# 2. Create records (Records layer)
from labanalysis.records import WholeBody
body = WholeBody.from_tdf_file("gait.tdf")

# 3. Extract signals (Data structures)
ankle = body.left_ankle  # Point3D

# 4. Process signals (Signal processing layer)
from labanalysis.signalprocessing import butterworth_filter
ankle_filtered = butterworth_filter(ankle, frequency=6, order=4)

# 5. Compute metrics (Biomechanics layer)
ankle_angle = body.left_ankle_flexionextension  # Signal1D

# 6. High-level analysis (Protocols layer)
from labanalysis.protocols import WalkingTest
test = WalkingTest.from_tdf_file("gait.tdf")
results = test.process()

# 7. Export (I/O layer)
results.to_dataframe().to_csv("results.csv")
```

## Extension Points

### Adding New Test Protocols

1. **Inherit from TestProtocol**
2. **Implement required methods**
3. **Create results class**

```python
# protocols/custom_protocol.py

from labanalysis.protocols import TestProtocol, TestResults

class CustomTest(TestProtocol):
    """Custom test protocol."""
    
    @classmethod
    def from_tdf_file(cls, filepath):
        # Load data
        pass
    
    def process(self):
        # Analyze
        return CustomResults(...)


class CustomResults(TestResults):
    """Custom test results."""
    
    def to_dataframe(self):
        # Export
        pass
```

### Adding Signal Processing Functions

1. **Create function in signalprocessing/**
2. **Accept Signal1D or Signal3D**
3. **Return same type as input**

```python
# signalprocessing/custom_filter.py

from labanalysis.records import Signal1D, Signal3D
from typing import Union

def custom_filter(
    signal: Union[Signal1D, Signal3D],
    param: float
) -> Union[Signal1D, Signal3D]:
    """
    Custom filter implementation.
    
    Parameters
    ----------
    signal : Signal1D or Signal3D
        Input signal
    param : float
        Filter parameter
    
    Returns
    -------
    Signal1D or Signal3D
        Filtered signal (same type as input)
    """
    # Process
    filtered_data = apply_custom_filter(signal.to_numpy(), param)
    
    # Return same type as input
    if isinstance(signal, Signal1D):
        return Signal1D(
            data=filtered_data,
            index=signal.index,
            columns=signal.columns,
            unit=signal.unit
        )
    else:  # Signal3D
        return Signal3D(
            data=filtered_data,
            index=signal.index,
            columns=signal.columns,
            unit=signal.unit
        )
```

### Adding New Body Properties

1. **Add property to WholeBody class**
2. **Use existing markers as input**
3. **Return Signal1D or Point3D**

```python
# records/bodies.py

class WholeBody(TimeseriesRecord):
    """Full-body kinematic model."""
    
    @property
    def custom_angle(self) -> Signal1D:
        """
        Custom joint angle.
        
        Returns
        -------
        Signal1D
            Angle in degrees
        """
        # Get required markers
        marker1 = self.get_point("marker1")
        marker2 = self.get_point("marker2")
        
        # Calculate angle
        angle = calculate_angle(marker1, marker2)
        
        return Signal1D(
            data=angle,
            index=marker1.index,
            columns=['angle'],
            unit='deg'
        )
```

## Error Handling Strategy

### Validation at Boundaries

```python
# ✅ GOOD: Validate inputs at API boundaries
def butterworth_filter(signal, frequency, order=4):
    """Filter with input validation."""
    # Validate at entry point
    if frequency <= 0:
        raise ValueError(f"Frequency must be positive, got {frequency}")
    
    if order < 1:
        raise ValueError(f"Order must be >= 1, got {order}")
    
    # Internal functions assume valid inputs
    return _apply_butterworth(signal, frequency, order)

def _apply_butterworth(signal, frequency, order):
    """Internal implementation (no validation)."""
    # Assume inputs are valid
    pass
```

### Fail Fast

```python
# ✅ GOOD: Fail immediately with clear error
def load_marker(filepath, label):
    """Load marker from file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = read_file(filepath)
    
    if label not in data:
        raise KeyError(f"Marker '{label}' not found in file")
    
    return data[label]

# ❌ BAD: Silent failure or late error
def load_marker(filepath, label):
    """Load marker (fails late)."""
    data = read_file(filepath)  # May return None
    return data.get(label)  # Returns None if missing (error not caught)
```

## Performance Considerations

### Lazy Evaluation

```python
class WholeBody:
    """Lazy property evaluation."""
    
    def __init__(self):
        self._ankle_cache = None
    
    @property
    def left_ankle(self):
        """Compute only when accessed."""
        if self._ankle_cache is None:
            # Expensive computation
            self._ankle_cache = self._compute_ankle()
        
        return self._ankle_cache
```

### Vectorization

```python
# ✅ GOOD: Vectorized operations
def calculate_distances(markers1: np.ndarray, markers2: np.ndarray) -> np.ndarray:
    """Vectorized distance calculation."""
    diff = markers1 - markers2
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return distances

# ❌ BAD: Loop-based
def calculate_distances(markers1, markers2):
    """Slow loop-based calculation."""
    distances = []
    for i in range(len(markers1)):
        dist = np.sqrt(np.sum((markers1[i] - markers2[i])**2))
        distances.append(dist)
    return np.array(distances)
```

## Testing Architecture

```
test/
├── conftest.py              # Shared fixtures
├── test_signalprocessing.py # Unit tests for algorithms
├── test_records.py          # Unit tests for data structures
└── test_protocols/          # Integration tests for protocols
    └── test_jump_protocols.py
```

**Testing strategy**:
- Unit tests for individual functions
- Integration tests for protocols
- Fixtures for reusable test data
- Parametrized tests for multiple cases

## Documentation Architecture

```
docs/
├── getting-started/         # Quickstart guides
├── user-guide/              # Feature documentation
├── api-reference/           # Auto-generated API docs
├── tutorials/               # Step-by-step guides
├── examples/                # Executable examples
├── advanced/                # Advanced topics
├── development/             # Contributing guides
└── references/              # Scientific references
```

## See Also

- [Contributing Guide](contributing.md) - How to contribute
- [Testing Guide](testing.md) - Writing tests
- [Code Style Guide](code-style.md) - Coding standards

---

**labanalysis follows layered architecture** with clear separation of concerns. Extend via protocols, signal processing functions, or WholeBody properties.
