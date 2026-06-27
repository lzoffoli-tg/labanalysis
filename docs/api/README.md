# API Reference

Complete API documentation for all labanalysis classes, functions, and modules.

## Core Modules

### Records Module

Data structures for time-series and biomechanical data:

- **[Timeseries](records/timeseries.md)** - Time-series data classes
  - `Timeseries` - Base time-indexed data
  - `Signal1D` - Single-column signals
  - `Signal3D` - Three-component signals
  - `EMGSignal` - Electromyography signals
  - `Point3D` - 3D marker positions

- **[Records](records/records.md)** - Container classes
  - `Record` - Basic signal container
  - `TimeseriesRecord` - Extended container
  - `ForcePlatform` - Force platform data
  - `MetabolicRecord` - Metabolic measurements

- **[WholeBody](records/bodies.md)** - Full body biomechanical model
  - 42 anatomical markers
  - 36 joint angles
  - 8 computed properties

- **[ReferenceFrame](records/referenceframes.md)** - Anatomical reference frames
  - `ReferenceFrame` - Coordinate system transformations
  - Semantic axis naming (lateral, vertical, anteroposterior)
  - Point3D/Signal3D input support

- **[Jumping](records/jumping.md)** - Jump analysis classes
  - `SingleJump` - Single jump movement
  - `DropJump` - Drop jump specialization
  - `RepeatedJumps` - Repeated jumps

- **[Locomotion](records/locomotion.md)** - Gait analysis classes
  - `RunningExercise` - Running analysis
  - `WalkingExercise` - Walking analysis
  - `GaitCycle`, `GaitObject` - Base classes

- **[Posture](records/posture.md)** - Posture analysis
  - `UprightPosture` - Standing posture
  - `PronePosture` - Prone posture

- **[Agility](records/agility.md)** - Agility movements
  - `ChangeOfDirectionExercise` - COD analysis

- **[Pipelines](records/pipelines.md)** - Processing workflows
  - `ProcessingPipeline` - Signal processing pipelines

### Protocols Module

Test protocols and participant information:

- **[Base Protocols](protocols/protocols.md)** - Core protocol classes
  - `Participant` - Test participant data
  - `TestProtocol` - Protocol interface
  - `TestResults` - Results interface

- **[Jump Tests](protocols/jump-tests.md)** - Jump testing
  - `JumpTest` - Jump test protocol
  - `JumpTestResults` - Jump results

- **[Locomotion Tests](protocols/locomotion-tests.md)** - Gait testing
  - `RunningTest` - Running protocol
  - `WalkingTest` - Walking protocol

- **[Balance Tests](protocols/balance-tests.md)** - Balance assessment
  - `UprightBalanceTest` - Standing balance
  - `PlankBalanceTest` - Plank balance

- **[Strength Tests](protocols/strength-tests.md)** - Strength assessment
  - `Isokinetic1RMTest` - Isokinetic 1RM
  - `IsometricTest` - Isometric strength

- **[Agility Tests](protocols/agility-tests.md)** - Agility assessment
  - `ShuttleTest` - Shuttle run protocol

- **[VO2max](protocols/vo2max.md)** - Aerobic capacity
  - `SubmaximalVO2MaxTest` - Submaximal protocol

### Signal Processing

Signal processing functions (30+ functions):

- **[Signal Processing Functions](signalprocessing.md)** - Complete reference
  - Filtering: `butterworth_filt()`, `fir_filt()`, `mean_filt()`, `median_filt()`, `rms_filt()`
  - Peaks: `find_peaks()`
  - Derivatives: `winter_derivative1()`, `winter_derivative2()`
  - Interpolation: `cubicspline_interp()`, `fillna()`
  - Frequency: `psd()`, `residual_analysis()`, `xcorr()`
  - Transformations: `to_reference_frame()`, `gram_schmidt()`, `tkeo()`
  - Utilities: `crossings()`, `normalize()`, `threshold()`

### I/O Module

Data readers and writers:

- **[Read Functions](io/read.md)** - Data import
  - `read_tdf()` - BTS Bioengineering TDF files
  - `read_opensim()` - OpenSim MOT/STO files
  - `read_ircam()` - IRCAM pressure mat
  - `BiostrengthProduct` - Biostrength readers

- **[Write Functions](io/write.md)** - Data export
  - `write_opensim()` - Export to OpenSim format

### Modeling Module

Regression and machine learning:

- **[OLS Models](modelling/ols.md)** - Ordinary least squares
  - `BaseRegression` - Base regression class
  - `PolynomialRegression` - Polynomial fitting
  - `PowerRegression` - Power law regression
  - `ExponentialRegression` - Exponential regression
  - `MultiSegmentRegression` - Piecewise regression
  - `Line2D`, `Line3D`, `Circle`, `Ellipse` - Geometric fitting

- **[PyTorch Modules](modelling/pytorch.md)** - Deep learning
  - `TorchTrainer` - Training workflow manager
  - `FeaturesGenerator` - Feature engineering
  - `BoxCoxTransform`, `PCA`, `Lasso` - Transform modules
  - Loss functions: `PinballLoss`, `StandardizedMSELoss`, `QuantilicRangeLoss`

### Equations Module

Predictive equations:

- **[Strength Equations](equations/strength.md)** - Strength prediction
  - `Brzycki1RM` - 1RM prediction from submaximal loads

- **[Cardio Equations](equations/cardio.md)** - Metabolic prediction
  - `Run` - ACSM running/walking equations
  - `Bike` - Cycling equations

### Plotting Module

Visualization utilities:

- **[Plotting Functions](plotting.md)** - Plotly-based visualization
  - `plot_comparisons()` - Regression, Bland-Altman, error plots
  - `bars_with_normative_bands()` - Bar charts with reference bands

### Utilities

General utilities and constants:

- **[Utils](utils.md)** - Utility functions
  - `ureg` - Unit registry (Pint)
  - `bpm_quantity()`, `au_quantity()`, `Q_()` - Quantity helpers
  - `magnitude()` - Order of magnitude
  - `get_files()` - File discovery
  - `split_data()` - Data splitting
  - Type annotations: `FloatArray1D`, `FloatArray2D`, `IntArray1D`, `TextArray1D`

- **[Constants](constants.md)** - Physical constants
  - `MINIMUM_CONTACT_FORCE_N` - Force threshold
  - `G` - Gravitational acceleration

- **[Messages](messages.md)** - User interaction
  - `askyesno()`, `askyesnocancel()` - Dialog functions

## Quick Navigation

### By Task

**Loading Data:**
- [I/O Read Functions](io/read.md)
- [TimeseriesRecord.from_tdf()](records/records.md#timeseriesrecord)

**Processing Signals:**
- [Signal Processing Functions](signalprocessing.md)
- [Signal1D Methods](records/timeseries.md#signal1d)

**Biomechanical Analysis:**
- [WholeBody](records/bodies.md)
- [ForcePlatform](records/records.md#forceplatform)

**Test Protocols:**
- [Test Protocol Classes](protocols/protocols.md)
- [Jump Tests](protocols/jump-tests.md)

**Modeling:**
- [OLS Regression](modelling/ols.md)
- [PyTorch Training](modelling/pytorch.md)

### By Type

**Classes:**
- [Records Module](records/README.md) - All data structure classes
- [Protocols Module](protocols/README.md) - All test protocol classes
- [Modeling Module](modelling/README.md) - All model classes

**Functions:**
- [Signal Processing](signalprocessing.md) - All processing functions
- [I/O Functions](io/README.md) - All read/write functions
- [Equations](equations/README.md) - All predictive equations

## Using This Reference

### Find a Class

Use the module navigation above to find the module, then browse the class documentation.

Example: To find `Signal1D`:
1. Go to [Records Module](records/README.md)
2. Click [Timeseries](records/timeseries.md)
3. Find `Signal1D` section

### Find a Function

Check the appropriate module:
- Signal processing → [signalprocessing.md](signalprocessing.md)
- Data loading → [io/read.md](io/read.md)
- Plotting → [plotting.md](plotting.md)

### Find a Method

Methods are documented in their class documentation.

Example: To find `.fillna()`:
1. It's a method on signals
2. Go to [Timeseries](records/timeseries.md)
3. Find the Methods section

## Documentation Conventions

### Code Examples

All API documentation includes runnable examples:

```python
import labanalysis as laban
import numpy as np

# Example usage
signal = laban.Signal1D(data=np.random.randn(100), index=np.arange(100), label='test', unit='V')
```

### Parameters

Parameters are documented with:
- **Name and type** - `data (ndarray)`
- **Description** - What it does
- **Default** - If applicable
- **Units** - For physical quantities

### Returns

Return values specify:
- **Type** - What is returned
- **Description** - What it contains

### Examples

Each class/function includes at least one complete, runnable example.

## See Also

- **[User Guide](../guides/README.md)** - Task-oriented guides
- **[Tutorials](../tutorials/README.md)** - Complete workflows
- **[Examples](../examples/README.md)** - Code examples

---

**Found an error?** Please report to [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
