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

[📖 Detailed installation guide →](docs/getting-started/installation.md)

### Prerequisites
- Python >= 3.12
- pip or conda package manager

### Install from Git Repository

```bash
pip install git+https://github.com/lzoffoli-tg/labanalysis.git
```

### Install in Development Mode (with Tests)

For development work or running tests, install with the `[dev]` extra:

```bash
git clone https://github.com/lzoffoli-tg/labanalysis.git
cd labanalysis
pip install -e ".[dev]"
```

This installs additional dependencies for testing:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `pytest-timeout` - Test timeouts

**Note**: The `test/` directory is excluded from standard package installations and is only available when cloning the repository.

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

[⚡ 5-minute quick start →](docs/getting-started/quick-start.md) | [🎯 Your first complete analysis →](docs/getting-started/first-analysis.md)

### Basic Example: Load and Process Data

[→ Complete data loading guide](docs/user-guide/data-loading/README.md)

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

[→ Complete jump analysis tutorial](docs/tutorials/01-jump-analysis.md) | [→ Jump tests guide](docs/user-guide/test-protocols/jump-tests.md)

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

[→ Complete gait analysis tutorial](docs/tutorials/02-gait-analysis.md) | [→ Gait analysis guide](docs/user-guide/test-protocols/gait-analysis.md)

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

### Example: Full Body Biomechanical Analysis

[→ Complete WholeBody tutorial](docs/tutorials/03-full-body-kinematics.md) | [→ WholeBody model guide](docs/user-guide/biomechanics/whole-body-model.md)

```python
import labanalysis as laban

# Load motion capture data with anatomical markers
body = laban.WholeBody.from_tdf(
    "path/to/mocap.tdf",
    # Foot markers (new: first/fifth metatarsal for accurate foot plane)
    left_first_metatarsal_head="LFM1",
    left_fifth_metatarsal_head="LFM5",
    right_first_metatarsal_head="RFM1",
    right_fifth_metatarsal_head="RFM5",
    left_heel="LHEE",
    right_heel="RHEE",
    # Spine markers (new: T5 for enhanced thoracic modeling)
    c7="C7",
    t5="T5",
    sc="SC",
    # Head markers (new: 4 cranial markers for head center calculation)
    head_anterior="HANT",
    head_posterior="HPOST",
    head_left="HLEFT",
    head_right="HRIGHT",
    # ... additional markers
)

# Access computed properties
head_center = body.head_center          # Centroid of 4 cranial markers
neck_base = body.neck_base              # Midpoint between SC and C7
left_foot_plane = body.left_foot_plane  # Plane from 4 foot markers

# Access joint angles (36 available)
knee_flexion = body.left_knee_flexionextension
neck_tilt = body.neck_lateral_tilt
neck_flex = body.neck_flexionextension_global

# Access all available angular measures
all_angles = body._angular_measures
print(f"Available angles: {len(all_angles)}")  # 36 joint angles
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
│   ├── bodies.py         # Full body biomechanical model with 42+ anatomical markers
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

[📚 Complete API Reference →](docs/api-reference/README.md)

### Core Modules

#### `labanalysis.constants` [→ API Reference](docs/api-reference/constants.md)
Physical and physiological constants used throughout the package.

**Constants:**
- `G`: Gravitational acceleration (9.81 m/s²)
- Additional domain-specific constants

#### `labanalysis.signalprocessing` [→ API Reference](docs/api-reference/signalprocessing.md)
Comprehensive signal processing utilities.

**Key Functions:**
- [`butterworth_filt()`](docs/api-reference/signalprocessing.md#butterworth_filt) - Butterworth filtering
- [`fir_filt()`](docs/api-reference/signalprocessing.md#fir_filt) - FIR filtering
- [`mean_filt()`](docs/api-reference/signalprocessing.md#mean_filt), [`median_filt()`](docs/api-reference/signalprocessing.md#median_filt), [`rms_filt()`](docs/api-reference/signalprocessing.md#rms_filt) - Statistical filters
- [`find_peaks()`](docs/api-reference/signalprocessing.md#find_peaks) - Peak detection
- [`winter_derivative1()`](docs/api-reference/signalprocessing.md#winter_derivative1), [`winter_derivative2()`](docs/api-reference/signalprocessing.md#winter_derivative2) - Derivatives (Winter 2009)
- [`residual_analysis()`](docs/api-reference/signalprocessing.md#residual_analysis) - Optimal cut-off frequency determination
- [`crossings()`](docs/api-reference/signalprocessing.md#crossings) - Zero-crossing detection
- [`xcorr()`](docs/api-reference/signalprocessing.md#xcorr) - Cross-correlation
- [`psd()`](docs/api-reference/signalprocessing.md#psd) - Power spectral density
- [`to_reference_frame()`](docs/api-reference/signalprocessing.md#to_reference_frame) - 3D coordinate transformations

[→ Complete signal processing guide](docs/user-guide/signal-processing/README.md)

#### `labanalysis.utils` [→ API Reference](docs/api-reference/utils.md)
General-purpose utility functions and unit handling.

**Features:**
- `ureg`: Unit registry (via `pint`) for physical quantities
- Helper functions for data manipulation

### Data Structures

#### `labanalysis.records` [→ API Reference](docs/api-reference/records/README.md)
Core data structure classes for representing laboratory measurements.

**Main Classes:**
- [`Record`](docs/api-reference/records/records.md#record) - Dictionary-like container for timeseries data
- [`TimeseriesRecord`](docs/api-reference/records/records.md#timeseriesrecord) - Specialized record for time-indexed data
- [`WholeBody`](docs/api-reference/records/bodies.md) - Full body biomechanical model with 42+ anatomical landmarks
  - Supports 4 metatarsal markers (first/fifth bilateral) for precise foot plane calculation
  - Includes thoracic vertebra T5 marker for enhanced spine modeling
  - Features 4 cranial markers (anterior/posterior/left/right) for head center and neck analysis
  - Computes 38 joint angular measures, reference frames, and derived properties
  - [→ Complete WholeBody guide](docs/user-guide/biomechanics/whole-body-model.md)
- [`ForcePlatform`](docs/api-reference/records/records.md#forceplatform) - Force platform data representation
- [`Participant`](docs/api-reference/protocols/protocols.md#participant) - Subject/participant information
- [`Signal1D`](docs/api-reference/records/timeseries.md#signal1d), [`Signal3D`](docs/api-reference/records/timeseries.md#signal3d) - 1D and 3D signal representations
- [`EMGSignal`](docs/api-reference/records/timeseries.md#emgsignal) - Electromyography signal class
- [`Point3D`](docs/api-reference/records/timeseries.md#point3d) - 3D point/marker data

**Key Methods:**
- `.from_tdf()`: Load from BTS TDF files
- `.to_dataframe()`: Convert to pandas DataFrame
- `.apply()`: Apply functions to all signals
- `.fillna(value=None, regressors=None, inplace=False)`: Handle missing data via constant value, cubic spline interpolation, or multiple linear regression
- `.strip()`, `.reset_time()`: Data cleaning

### I/O Operations

#### `labanalysis.io.read` [→ API Reference](docs/api-reference/io/read.md)
Import data from various laboratory equipment formats.

**Supported Formats:**
- **BTS Bioengineering**: `.tdf` files (force platforms, motion capture) - [Guide](docs/user-guide/data-loading/bts-bioengineering.md)
- **OpenSim**: `.mot`, `.sto` files - [Guide](docs/user-guide/data-loading/opensim.md)
- **Cosmed**: Metabolic data - [Guide](docs/user-guide/data-loading/cosmed.md)
- **IRCAM**: Audio/motion data - [Guide](docs/user-guide/data-loading/ircam.md)
- **Biostrength**: Strength testing equipment - [Guide](docs/user-guide/data-loading/biostrength.md)

[→ Complete data loading guide](docs/user-guide/data-loading/README.md)

#### `labanalysis.io.write` [→ API Reference](docs/api-reference/io/write.md)
Export data to standard formats.

**Export Formats:**
- OpenSim motion files (`.mot`)
- Custom formats for analysis pipelines

[→ Data export guide](docs/user-guide/data-export/README.md)

### Modelling

#### `labanalysis.modelling.ols` [→ API Reference](docs/api-reference/modelling/ols.md)
Ordinary Least Squares regression and geometric modeling.

**Classes:**
- `PolynomialRegression`: Polynomial regression models
- `LinearRegression`: Standard linear regression
- Geometric fitting utilities

[→ Regression guide](docs/user-guide/modeling/regression.md)

#### `labanalysis.modelling.pytorch` [→ API Reference](docs/api-reference/modelling/pytorch.md)
PyTorch-based deep learning utilities with CPU optimizations.

**Features:**
- Custom neural network modules
- `TorchTrainer`: Optimized trainer with 2-3x CPU speedup ([→ CPU optimization guide](docs/advanced/CPU_OPTIMIZATION_GUIDE.md))
- Training and evaluation helpers
- Model validation and visualization
- Integration with ONNX for model deployment

[→ PyTorch guide](docs/user-guide/modeling/pytorch-basics.md) | [→ TorchTrainer guide](docs/user-guide/modeling/torch-trainer.md)

### Test Protocols

#### `labanalysis.protocols` [→ API Reference](docs/api-reference/protocols/README.md)
Standardized test protocols with automated analysis.

**Available Protocols:**
- **Balance Tests**: `StaticBalanceTest`, `DynamicBalanceTest` - [Guide](docs/user-guide/test-protocols/balance-tests.md)
- **Jump Tests**: `CounterMovementJump`, `SquatJump`, `DropJump` - [Guide](docs/user-guide/test-protocols/jump-tests.md)
- **Locomotion**: `WalkingTest`, `RunningTest`, `GaitAnalysis` - [Guide](docs/user-guide/test-protocols/gait-analysis.md)
- **Strength**: `IsometricTest`, `IsokineticTest`, `FreeWeightTest` - [Guide](docs/user-guide/test-protocols/strength-tests.md)
- **Agility**: `ChangeOfDirectionTest`, `ReactiveAgilityTest` - [Guide](docs/user-guide/test-protocols/agility-tests.md)
- **VO2max**: `GradedExerciseTest`, `SubmaximalTest` - [Guide](docs/user-guide/test-protocols/vo2max-tests.md)

Each protocol class provides:
- `.from_tdf()`: Load data
- `.results`: Dictionary of computed outcomes
- `.plot()`: Interactive visualization
- `.report()`: Formatted report generation

[→ Complete test protocols guide](docs/user-guide/test-protocols/README.md)

### Equations

#### `labanalysis.equations.strength` [→ API Reference](docs/api-reference/equations/strength.md)
Predictive equations for strength assessment.

**Functions:**
- `estimate_1rm()`: One-repetition maximum estimation
- Various load-velocity relationships

#### `labanalysis.equations.cardio` [→ API Reference](docs/api-reference/equations/cardio.md)
Cardiovascular and metabolic prediction equations.

**Functions:**
- `estimate_vo2max()`: VO2max prediction from submaximal tests
- Heart rate and energy expenditure equations

---

## Documentation

📚 **[Complete Documentation Hub →](docs/README.md)** - Start here for all guides, tutorials, and API reference

### Quick Links

**Getting Started:**
- 🚀 [Installation Guide](docs/getting-started/installation.md) - Install from pip or git
- ⚡ [Quick Start (5 min)](docs/getting-started/quick-start.md) - Your first analysis
- 💡 [Core Concepts](docs/getting-started/core-concepts.md) - Understand Record, Signal, Protocol patterns
- 🎯 [Your First Analysis](docs/getting-started/first-analysis.md) - Complete walkthrough

**User Guides (by task):**
- 📂 [Data Loading](docs/user-guide/data-loading/README.md) - Load from BTS, OpenSim, Biostrength, IRCAM, Cosmed
- 📊 [Signal Processing](docs/user-guide/signal-processing/README.md) - Filtering, peaks, derivatives, frequency analysis
- 🧍 [Biomechanics](docs/user-guide/biomechanics/README.md) - WholeBody model (104+ properties, 38 angular measures), force platforms, joint angles
- 🧪 [Test Protocols](docs/user-guide/test-protocols/README.md) - Jump, gait, balance, strength, agility, VO2max
- 🤖 [Modeling](docs/user-guide/modeling/README.md) - Regression, PyTorch, ONNX deployment
- 📈 [Visualization](docs/user-guide/visualization/README.md) - Plotly-based interactive plots
- 💾 [Data Export](docs/user-guide/data-export/README.md) - Export to Excel, OpenSim, reports

**Tutorials (complete workflows):**
1. [Jump Analysis](docs/tutorials/01-jump-analysis.md) - Complete CMJ analysis from TDF to report
2. [Gait Analysis](docs/tutorials/02-gait-analysis.md) - Walking/running analysis workflow
3. [Full Body Kinematics](docs/tutorials/03-full-body-kinematics.md) - WholeBody model workflow
4. [Strength Assessment](docs/tutorials/04-strength-assessment.md) - Isokinetic/isometric testing
5. [Signal Processing Pipeline](docs/tutorials/05-signal-processing.md) - Complete processing workflow
6. [Custom Protocol](docs/tutorials/06-custom-protocol.md) - Extend TestProtocol
7. [Batch Processing](docs/tutorials/07-batch-processing.md) - Process multiple files
8. [ML Modeling](docs/tutorials/08-ml-modeling.md) - PyTorch training workflow

[View all 8 tutorials →](docs/tutorials/README.md)

**API Reference:**
- 📚 [Complete API Documentation](docs/api-reference/README.md) - All classes and functions
- [Records Module](docs/api-reference/records/README.md) - Timeseries, Signal, Record, WholeBody
- [Protocols Module](docs/api-reference/protocols/README.md) - Participant, TestProtocol, TestResults
- [Signal Processing](docs/api-reference/signalprocessing.md) - 30+ processing functions
- [I/O Module](docs/api-reference/io/README.md) - Data readers and writers
- [Modeling Module](docs/api-reference/modelling/README.md) - OLS and PyTorch models

**Advanced Topics:**
- ⚡ [CPU Optimization Guide](docs/advanced/CPU_OPTIMIZATION_GUIDE.md) - Achieve 2-3x speedup for TorchTrainer
  - 11 implemented optimizations
  - Best practices for batch size and multiprocessing
  - Configuration examples and profiling tips
- 🔧 [Custom Protocols](docs/advanced/extending-protocols.md) - Create custom TestProtocol implementations
- 🎨 [Custom Signals](docs/advanced/custom-signals.md) - Extend Signal1D and Signal3D
- ⚙️ [Unit Handling](docs/advanced/unit-handling.md) - Deep dive into Pint integration
- 📈 [Performance Tips](docs/advanced/performance-tips.md) - Optimization best practices

**Examples:**
- [Basic Examples](docs/examples/basic/) - Loading, filtering, exporting (3 scripts)
- [Biomechanics Examples](docs/examples/biomechanics/) - Joint angles, GRF, markers (3 scripts)
- [Protocol Examples](docs/examples/protocols/) - CMJ, running, balance tests (3 scripts)
- [Modeling Examples](docs/examples/modeling/) - Polynomial fit, PyTorch, 1RM (3 scripts)

[View all examples →](docs/examples/README.md)

**Support:**
- 🐛 [Troubleshooting](docs/troubleshooting/README.md) - Common errors and solutions
- 💻 [Developer Guide](docs/development/README.md) - Contributing and development setup

---

## Development

### Contributing

This is a proprietary package maintained by Technogym Scientific Research. Internal contributions should follow these guidelines:

1. **Code Style**: Follow PEP 8 conventions
2. **Type Hints**: Use type annotations for all functions
3. **Documentation**: Include docstrings (NumPy format)
4. **Testing**: Add tests for new features

### Running Tests

The test suite is comprehensive but **only available in development mode**. Tests are not included when installing the package with `pip install`.

```bash
# Install in development mode first
pip install -e ".[dev]"

# Run all tests
pytest test/

# Run specific test module
pytest test/test_runningexercise.py -v

# Run tests in parallel (faster)
pytest test/ -n auto

# Run with coverage report
pytest test/ --cov=labanalysis --cov-report=html
```

For detailed information about the test suite, see [`test/README.md`](test/README.md).

**Key Test Modules:**
- `test_bodies.py` - WholeBody biomechanical model tests (15 test cases)
  - New anatomical markers (metatarsals, T5, cranial markers)
  - Computed properties (head_center, neck_base, foot_plane)
  - Derived angles (neck angles, foot height, ankle angles)
- `test_runningexercise.py` - Comprehensive RunningExercise tests (30+ test cases)
- `test_jumps.py` - Jump test protocols
- `test_balance.py` - Balance and posture tests
- `test_strengthtests.py` - Strength assessment tests
- And many more...

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

**Python**: ≥ 3.12  
**Last Updated**: June 2026
