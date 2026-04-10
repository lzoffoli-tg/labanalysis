# labanalysis

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

**labanalysis** is a comprehensive Python package for laboratory data analysis, with a focus on biomechanics, exercise science, and human movement research. It provides a unified, extensible framework for reading, processing, analyzing, and visualizing data from laboratory equipment and protocols.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [Usage Examples](#usage-examples)
- [Module Reference](#module-reference)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)
- [Contact](#contact)

---

## Features

✨ **Comprehensive Data Analysis**
- Read data from multiple laboratory equipment formats (BTS Bioengineering, Cosmed, OpenSim, IRCAM, Biostrength)
- Process and analyze biomechanical signals with advanced signal processing tools
- Standardized test protocols for balance, jumping, locomotion, strength, agility, and VO2max

📊 **Advanced Signal Processing**
- Filtering (Butterworth, FIR, moving average, median, RMS)
- Peak detection and derivatives (Winter 2009 method)
- Cross-correlation, power spectral density, and residual analysis
- Missing data interpolation with multiple strategies

🧮 **Modelling & Regression**
- Ordinary Least Squares (OLS) and polynomial regression
- Geometric modeling and crossover analysis
- PyTorch integration for deep learning models
- ONNX model support for deployment

📈 **Visualization & Reporting**
- Interactive Plotly-based visualizations
- Automated reporting for test protocols
- Bland-Altman plots, regression plots, and time series visualization

🏃 **Standardized Test Protocols**
- Balance tests (static and dynamic)
- Jump tests (CMJ, SJ, DJ, etc.)
- Locomotion analysis (walking, running, gait parameters)
- Strength assessment (isometric, isokinetic, free weights)
- Agility tests (change of direction, reactive agility)
- VO2max protocols

---

## Installation

### Prerequisites
- Python >= 3.12
- pip or conda package manager

### Install from Git Repository

```bash
pip install git+https://github.com/lzoffoli-tg/labanalysis.git
```

### Install in Development Mode

```bash
git clone https://github.com/lzoffoli-tg/labanalysis.git
cd labanalysis
pip install -e .
```

### Dependencies

Core dependencies are automatically installed:
- `scikit-learn` - Machine learning and statistical modeling
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation and analysis
- `openpyxl` - Excel file support
- `pint` - Physical units handling
- `torch` - Deep learning framework
- `onnxmodels` - ONNX model utilities

---

## Quick Start

### Basic Example: Load and Process Data

```python
import labanalysis as laban

# Load a timeseries record from a TDF file (BTS Bioengineering format)
record = laban.TimeseriesRecord.from_tdf("path/to/file.tdf")

# Access signals in the record
force_plate_data = record['FP1']  # Force platform data
marker_data = record['MKRS']      # Marker trajectories

# Apply signal processing
filtered_signal = laban.butterworth_filt(
    signal=force_plate_data.data['Fz'],
    freq=1000,
    cut=10,
    order=4
)

# Convert to pandas DataFrame for analysis
df = record.to_dataframe()
```

### Example: Analyze a Jump Test

```python
import labanalysis as laban

# Create a participant
participant = laban.Participant(
    name="John",
    surname="Doe",
    height=1.80,  # meters
    weight=75,    # kg
    age=25
)

# Load and analyze a countermovement jump
cmj_test = laban.CounterMovementJump.from_tdf(
    "path/to/jump.tdf",
    participant=participant
)

# Get results
results = cmj_test.results
print(f"Jump Height: {results['jump_height']:.2f} cm")
print(f"Peak Power: {results['peak_power']:.2f} W")
print(f"RSI: {results['rsi']:.3f}")

# Generate interactive plot
fig = cmj_test.plot()
fig.show()
```

### Example: Gait Analysis

```python
import labanalysis as laban

# Create participant
participant = laban.Participant(
    name="Jane",
    surname="Smith",
    height=1.65,
    weight=60
)

# Analyze walking test
walking_test = laban.WalkingTest.from_tdf(
    "path/to/walking.tdf",
    participant=participant
)

# Extract gait parameters
results = walking_test.results
print(f"Stride Length: {results['stride_length']:.2f} m")
print(f"Cadence: {results['cadence']:.1f} steps/min")
print(f"Walking Speed: {results['speed']:.2f} m/s")

# Get cycle-by-cycle analysis
gait_cycles = walking_test.gait_cycles
```

---

## Package Structure

```
labanalysis/
├── constants.py          # Physical and physiological constants
├── messages.py           # Standardized messages and warnings
├── signalprocessing.py   # Signal processing utilities
├── utils.py              # General utility functions
├── docs/                 # 📚 Documentation and guides
│   ├── README.md         # Documentation index
│   └── CPU_OPTIMIZATION_GUIDE.md  # TorchTrainer CPU optimizations
├── equations/            # Predictive equations
│   ├── strength.py       # Strength prediction models
│   └── cardio.py         # Cardiovascular prediction models
├── io/                   # Data import/export
│   ├── read/             # File readers
│   │   ├── btsbioengineering.py
│   │   ├── opensim.py
│   │   ├── biostrength.py
│   │   └── ircam.py
│   └── write/            # File writers
│       └── opensim.py
├── records/              # Data structure classes
│   ├── records.py        # Base record classes
│   ├── timeseries.py     # Time series data structures
│   ├── bodies.py         # Participant and body segment classes
│   ├── jumping.py        # Jump-specific records
│   ├── locomotion.py     # Locomotion-specific records
│   ├── strength/         # Strength assessment records
│   ├── posture.py        # Posture and balance records
│   ├── agility.py        # Agility test records
│   └── pipelines.py      # Data processing pipelines
├── modelling/            # Regression and ML models
│   ├── ols/              # Ordinary Least Squares
│   │   ├── regression.py # OLS and polynomial regression
│   │   └── geometry.py   # Geometric modeling
│   └── pytorch/          # PyTorch utilities
│       ├── modules.py    # Neural network modules
│       └── utils.py      # Training and evaluation utilities
├── plotting/             # Visualization tools
│   └── plotly.py         # Plotly-based plots
└── protocols/            # Standardized test protocols
    ├── balancetests.py
    ├── jumptests.py
    ├── locomotiontests.py
    ├── strengthtests.py
    ├── agilitytests.py
    ├── vo2max.py
    ├── normativedata.py
    └── protocols.py
```

---

## Usage Examples

### Signal Processing

```python
import labanalysis as laban
import numpy as np

# Generate sample signal
time = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * time) + 0.5 * np.random.randn(1000)

# Apply Butterworth filter
filtered = laban.butterworth_filt(
    signal=signal,
    freq=100,        # Sampling frequency (Hz)
    cut=5,           # Cut-off frequency (Hz)
    order=4,         # Filter order
    filt_type='low'  # Low-pass filter
)

# Find peaks
peaks = laban.find_peaks(
    signal=filtered,
    height=0.5,
    distance=50
)

# Calculate derivative (Winter 2009 method)
velocity = laban.winter_derivative1(signal, freq=100)
acceleration = laban.winter_derivative2(signal, freq=100)

# Fill missing data with interpolation (cubic spline)
from labanalysis import Signal1D
signal_with_gaps = Signal1D(signal.copy(), time, 'V')
signal_with_gaps._data[100:120] = np.nan
filled_signal = signal_with_gaps.fillna()  # Default: cubic spline interpolation

# Or fill with constant value
filled_constant = signal_with_gaps.fillna(value=0.0)

# Or fill with regression using regressors
regressors = np.column_stack([time, time**2])  # Example regressors
filled_regression = signal_with_gaps.fillna(regressors=regressors)
```

### Working with Records

```python
import labanalysis as laban

# Create a custom record
from labanalysis import Signal1D, Signal3D, Record

# Create individual signals
force_z = Signal1D(
    data=np.array([0, 100, 200, 300]),
    index=np.array([0.0, 0.01, 0.02, 0.03]),
    label='Fz',
    unit='N'
)

cop_position = Signal3D(
    data=np.array([[0, 0, 0], [0.1, 0.2, 0], [0.15, 0.25, 0], [0.2, 0.3, 0]]),
    index=np.array([0.0, 0.01, 0.02, 0.03]),
    labels=['COPx', 'COPy', 'COPz'],
    units=['m', 'm', 'm']
)

# Create a record
record = Record(force=force_z, cop=cop_position)

# Access data
print(record.force.data)
print(record.cop.data)

# Convert to DataFrame
df = record.to_dataframe()

# Apply processing pipeline
processed_record = record.apply(
    lambda x: laban.butterworth_filt(x, freq=1000, cut=10),
    axis=0
)
```

### Regression and Modelling

```python
import labanalysis as laban
import numpy as np

# Polynomial regression
x = np.linspace(0, 10, 100)
y = 2 * x**2 + 3 * x + 1 + np.random.randn(100) * 5

poly_model = laban.PolynomialRegression(degree=2)
poly_model.fit(x, y)

# Predictions
y_pred = poly_model.predict(x)

# Get coefficients and R²
print(f"Coefficients: {poly_model.coef_}")
print(f"R²: {poly_model.score(x, y):.3f}")

# Crossover analysis (find intersection points)
crossovers_x = laban.crossovers(x, y, n_segments=2)
print(f"Crossover at x = {crossovers_x}")
```

### Predictive Equations

```python
import labanalysis as laban

# Estimate 1RM from submaximal load
one_rm = laban.estimate_1rm(
    load=100,      # kg
    reps=8,        # repetitions
    method='epley' # or 'brzycki', 'lander', etc.
)

# Estimate VO2max from submaximal test
vo2max = laban.estimate_vo2max(
    age=25,
    gender='M',
    heart_rate=150,
    workload=150,  # watts
    method='astrand'
)
```

### Visualization

```python
import labanalysis as laban

# Bland-Altman plot
fig = laban.bland_altman_plot(
    y_true=[100, 110, 120, 130, 140],
    y_pred=[98, 112, 118, 132, 139],
    title="Method Comparison"
)
fig.show()

# Regression plot with confidence intervals
fig = laban.regression_plot(
    x=x_data,
    y=y_data,
    model=poly_model,
    show_ci=True,
    title="Dose-Response Relationship"
)
fig.show()
```

---

## Module Reference

### Core Modules

#### `labanalysis.constants`
Physical and physiological constants used throughout the package.

**Constants:**
- `G`: Gravitational acceleration (9.81 m/s²)
- Additional domain-specific constants

#### `labanalysis.signalprocessing`
Comprehensive signal processing utilities.

**Key Functions:**
- `butterworth_filt()`: Butterworth filtering
- `fir_filt()`: FIR filtering
- `mean_filt()`, `median_filt()`, `rms_filt()`: Statistical filters
- `find_peaks()`: Peak detection
- `winter_derivative1()`, `winter_derivative2()`: Derivatives (Winter 2009)
- `residual_analysis()`: Optimal cut-off frequency determination
- `crossings()`: Zero-crossing detection
- `xcorr()`: Cross-correlation
- `psd()`: Power spectral density
- `to_reference_frame()`: 3D coordinate transformations

#### `labanalysis.utils`
General-purpose utility functions and unit handling.

**Features:**
- `ureg`: Unit registry (via `pint`) for physical quantities
- Helper functions for data manipulation

### Data Structures

#### `labanalysis.records`
Core data structure classes for representing laboratory measurements.

**Main Classes:**
- `Record`: Dictionary-like container for timeseries data
- `TimeseriesRecord`: Specialized record for time-indexed data
- `ForcePlatform`: Force platform data representation
- `Participant`: Subject/participant information
- `Signal1D`, `Signal3D`: 1D and 3D signal representations
- `EMGSignal`: Electromyography signal class
- `Point3D`: 3D point/marker data

**Key Methods:**
- `.from_tdf()`: Load from BTS TDF files
- `.to_dataframe()`: Convert to pandas DataFrame
- `.apply()`: Apply functions to all signals
- `.fillna(value=None, regressors=None, inplace=False)`: Handle missing data via constant value, cubic spline interpolation, or multiple linear regression
- `.strip()`, `.reset_time()`: Data cleaning

### I/O Operations

#### `labanalysis.io.read`
Import data from various laboratory equipment formats.

**Supported Formats:**
- **BTS Bioengineering**: `.tdf` files (force platforms, motion capture)
- **OpenSim**: `.mot`, `.sto` files
- **Cosmed**: Metabolic data
- **IRCAM**: Audio/motion data
- **Biostrength**: Strength testing equipment

#### `labanalysis.io.write`
Export data to standard formats.

**Export Formats:**
- OpenSim motion files (`.mot`)
- Custom formats for analysis pipelines

### Modelling

#### `labanalysis.modelling.ols`
Ordinary Least Squares regression and geometric modeling.

**Classes:**
- `PolynomialRegression`: Polynomial regression models
- `LinearRegression`: Standard linear regression
- Geometric fitting utilities

#### `labanalysis.modelling.pytorch`
PyTorch-based deep learning utilities with CPU optimizations.

**Features:**
- Custom neural network modules
- `TorchTrainer`: Optimized trainer with 2-3x CPU speedup ([see guide](docs/CPU_OPTIMIZATION_GUIDE.md))
- Training and evaluation helpers
- Model validation and visualization
- Integration with ONNX for model deployment

### Test Protocols

#### `labanalysis.protocols`
Standardized test protocols with automated analysis.

**Available Protocols:**
- **Balance Tests**: `StaticBalanceTest`, `DynamicBalanceTest`
- **Jump Tests**: `CounterMovementJump`, `SquatJump`, `DropJump`
- **Locomotion**: `WalkingTest`, `RunningTest`, `GaitAnalysis`
- **Strength**: `IsometricTest`, `IsokineticTest`, `FreeWeightTest`
- **Agility**: `ChangeOfDirectionTest`, `ReactiveAgilityTest`
- **VO2max**: `GradedExerciseTest`, `SubmaximalTest`

Each protocol class provides:
- `.from_tdf()`: Load data
- `.results`: Dictionary of computed outcomes
- `.plot()`: Interactive visualization
- `.report()`: Formatted report generation

### Equations

#### `labanalysis.equations.strength`
Predictive equations for strength assessment.

**Functions:**
- `estimate_1rm()`: One-repetition maximum estimation
- Various load-velocity relationships

#### `labanalysis.equations.cardio`
Cardiovascular and metabolic prediction equations.

**Functions:**
- `estimate_vo2max()`: VO2max prediction from submaximal tests
- Heart rate and energy expenditure equations

---

## Documentation

Additional documentation and guides are available in the [`docs/`](docs/) directory:

📚 **Available Guides:**
- **[CPU Optimization Guide](docs/CPU_OPTIMIZATION_GUIDE.md)** - Performance optimization for `TorchTrainer` on CPU
  - 11 implemented optimizations for 2-3x speedup
  - Best practices for batch size and multiprocessing
  - Configuration examples and profiling tips

For a complete list of available documentation, see [`docs/README.md`](docs/README.md).

---

## Development

### Contributing

This is a proprietary package maintained by Technogym Scientific Research. Internal contributions should follow these guidelines:

1. **Code Style**: Follow PEP 8 conventions
2. **Type Hints**: Use type annotations for all functions
3. **Documentation**: Include docstrings (NumPy format)
4. **Testing**: Add tests for new features

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=labanalysis tests/
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs
make html
```

---

## License

**Proprietary Software**

This software is proprietary and may not be copied, distributed, or used without the explicit written approval of the author.

For permission requests, please contact: **Luca Zoffoli** – [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)

© 2024 Technogym S.p.A. All rights reserved.

---

## Contact

**Author**: Luca Zoffoli  
**Email**: [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)  
**Organization**: Technogym S.p.A. - Scientific Research Department  
**Repository**: [https://github.com/lzoffoli-tg/labanalysis](https://github.com/lzoffoli-tg/labanalysis)

---

## Acknowledgments

This package integrates and builds upon several excellent open-source libraries:
- **NumPy** and **Pandas** for numerical computing and data analysis
- **SciPy** for scientific computing and signal processing
- **Plotly** for interactive visualizations
- **PyTorch** for deep learning capabilities
- **scikit-learn** for machine learning tools

---

**Version**: 187  
**Python**: ≥ 3.12  
**Last Updated**: April 2026
