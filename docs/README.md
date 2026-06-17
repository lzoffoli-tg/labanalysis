# labanalysis Documentation

Welcome to the comprehensive documentation for **labanalysis**, a Python package for biomechanical and laboratory data analysis developed by Technogym Scientific Research.

## 🚀 New to labanalysis?

Start here to get up and running in 30 minutes:

1. **[Installation Guide](getting-started/installation.md)** - Install from pip or git repository
2. **[Quick Start (5 min)](getting-started/quick-start.md)** - Your first analysis in minutes
3. **[Core Concepts](getting-started/core-concepts.md)** - Understand Record, Signal, and Protocol patterns
4. **[Your First Analysis](getting-started/first-analysis.md)** - Complete walkthrough with real data

## 📖 User Guide (task-oriented)

Task-oriented guides to help you accomplish specific goals:

### Data Loading
- **[Overview](user-guide/data-loading/README.md)** - Supported formats and general workflow
- [BTS Bioengineering](user-guide/data-loading/bts-bioengineering.md) - Load TDF files from BTS systems
- [OpenSim](user-guide/data-loading/opensim.md) - Load C3D, MOT, and STO files
- [Biostrength](user-guide/data-loading/biostrength.md) - Load strength machine data
- [IRCAM](user-guide/data-loading/ircam.md) - Load pressure mat data
- [Cosmed](user-guide/data-loading/cosmed.md) - Load metabolic data

### Signal Processing
- **[Overview](user-guide/signal-processing/README.md)** - Complete signal processing toolkit
- [Filtering](user-guide/signal-processing/filtering.md) - Butterworth, FIR, moving average, median, RMS
- [Peak Detection](user-guide/signal-processing/peak-detection.md) - Find peaks in signals
- [Derivatives](user-guide/signal-processing/derivatives.md) - Winter 2009 derivative methods
- [Missing Data](user-guide/signal-processing/missing-data.md) - Handle gaps with fillna() strategies
- [Frequency Analysis](user-guide/signal-processing/frequency-analysis.md) - PSD and residual analysis
- [Transformations](user-guide/signal-processing/transformations.md) - Reference frame transformations

### Biomechanics
- **[Overview](user-guide/biomechanics/README.md)** - Biomechanical analysis tools
- [WholeBody Model](user-guide/biomechanics/whole-body-model.md) - 86 properties (42 markers, 36 angles, 8 computed)
- [Force Platforms](user-guide/biomechanics/force-platforms.md) - Ground reaction force analysis
- [Joint Angles](user-guide/biomechanics/joint-angles.md) - Calculate 36 joint angles
- [Coordinate Systems](user-guide/biomechanics/coordinate-systems.md) - Reference frames and transformations
- [EMG Signals](user-guide/biomechanics/emg-signals.md) - Electromyography signal analysis

### Test Protocols
- **[Overview](user-guide/test-protocols/README.md)** - Standardized test protocols
- [Jump Tests](user-guide/test-protocols/jump-tests.md) - CMJ, SJ, DJ analysis
- [Gait Analysis](user-guide/test-protocols/gait-analysis.md) - Walking and running analysis
- [Balance Tests](user-guide/test-protocols/balance-tests.md) - Upright and plank balance
- [Strength Tests](user-guide/test-protocols/strength-tests.md) - Isokinetic and isometric testing
- [Agility Tests](user-guide/test-protocols/agility-tests.md) - Shuttle and change of direction
- [VO2max Tests](user-guide/test-protocols/vo2max-tests.md) - Submaximal VO2max protocols

### Modeling
- **[Overview](user-guide/modeling/README.md)** - Statistical and machine learning models
- [Regression](user-guide/modeling/regression.md) - OLS, Polynomial, Power, Exponential models
- [PyTorch Basics](user-guide/modeling/pytorch-basics.md) - Introduction to PyTorch integration
- [TorchTrainer](user-guide/modeling/torch-trainer.md) - Training workflows and optimization
- [Custom Models](user-guide/modeling/custom-models.md) - Extend existing models
- [ONNX Deployment](user-guide/modeling/onnx-deployment.md) - Export models for deployment

### Visualization
- **[Overview](user-guide/visualization/README.md)** - Interactive visualizations with Plotly
- [Plotly Basics](user-guide/visualization/plotly-basics.md) - Plotly integration fundamentals
- [Comparison Plots](user-guide/visualization/comparison-plots.md) - Bland-Altman, regression plots
- [Protocol Reports](user-guide/visualization/protocol-reports.md) - Automated test reports
- [Custom Figures](user-guide/visualization/custom-figures.md) - Customize your visualizations

### Data Export
- **[Overview](user-guide/data-export/README.md)** - Export workflows and formats
- [OpenSim Export](user-guide/data-export/opensim-export.md) - Export to OpenSim format
- [DataFrames](user-guide/data-export/dataframes.md) - Convert to pandas DataFrames
- [Reports](user-guide/data-export/reports.md) - Generate analysis reports

## 🎓 Tutorials (complete workflows)

End-to-end tutorials showing real analysis workflows from raw data to results:

1. **[Jump Analysis](tutorials/01-jump-analysis.md)** - Complete CMJ analysis from TDF file to report
2. **[Gait Analysis](tutorials/02-gait-analysis.md)** - Walking and running analysis workflow
3. **[Full Body Kinematics](tutorials/03-full-body-kinematics.md)** - WholeBody model complete workflow
4. **[Strength Assessment](tutorials/04-strength-assessment.md)** - Isokinetic and isometric testing
5. **[Signal Processing Pipeline](tutorials/05-signal-processing.md)** - Complete processing workflow
6. **[Custom Protocol](tutorials/06-custom-protocol.md)** - Create and extend TestProtocol
7. **[Batch Processing](tutorials/07-batch-processing.md)** - Process multiple files efficiently
8. **[ML Modeling](tutorials/08-ml-modeling.md)** - PyTorch training and deployment workflow

## 📚 API Reference

Complete API documentation for all classes and functions:

### Core Modules
- **[Records Module](api-reference/records/README.md)** - Data structures (Timeseries, Signal, Record, WholeBody)
  - [Timeseries](api-reference/records/timeseries.md) - Time-series data structures
  - [Records](api-reference/records/records.md) - Record containers
  - [WholeBody](api-reference/records/bodies.md) - Full body biomechanical model (86 properties)
  - [Jumping](api-reference/records/jumping.md) - Jump-specific classes
  - [Locomotion](api-reference/records/locomotion.md) - Gait and running classes
  - [Posture](api-reference/records/posture.md) - Posture analysis classes
  - [Agility](api-reference/records/agility.md) - Agility movement classes
  - [Pipelines](api-reference/records/pipelines.md) - Processing pipelines

- **[Protocols Module](api-reference/protocols/README.md)** - Test protocols and participants
  - [Base Protocols](api-reference/protocols/protocols.md) - Participant, TestProtocol, TestResults
  - [Jump Tests](api-reference/protocols/jump-tests.md) - Jump testing protocols
  - [Locomotion Tests](api-reference/protocols/locomotion-tests.md) - Running and walking tests
  - [Balance Tests](api-reference/protocols/balance-tests.md) - Balance testing protocols
  - [Strength Tests](api-reference/protocols/strength-tests.md) - Strength assessment protocols
  - [Agility Tests](api-reference/protocols/agility-tests.md) - Agility testing protocols
  - [VO2max](api-reference/protocols/vo2max.md) - VO2max testing protocols

- **[Signal Processing](api-reference/signalprocessing.md)** - 30+ signal processing functions
- **[I/O Module](api-reference/io/README.md)** - Data readers and writers
  - [Read Functions](api-reference/io/read.md) - read_tdf(), read_opensim(), etc.
  - [Write Functions](api-reference/io/write.md) - write_opensim()
  
- **[Modeling Module](api-reference/modelling/README.md)** - Regression and ML models
  - [OLS Models](api-reference/modelling/ols.md) - Ordinary least squares regression
  - [PyTorch Modules](api-reference/modelling/pytorch.md) - TorchTrainer, loss functions

- **[Equations Module](api-reference/equations/README.md)** - Predictive equations
  - [Strength Equations](api-reference/equations/strength.md) - Brzycki 1RM and more
  - [Cardio Equations](api-reference/equations/cardio.md) - VO2 prediction equations

- **[Plotting](api-reference/plotting.md)** - Visualization utilities
- **[Utils](api-reference/utils.md)** - General utilities and unit handling
- **[Constants](api-reference/constants.md)** - Physical and physiological constants
- **[Messages](api-reference/messages.md)** - User interaction utilities

[View complete API index →](api-reference/README.md)

## 🔬 Advanced Topics

Deep dives into advanced features and optimizations:

- **[CPU Optimization Guide](advanced/CPU_OPTIMIZATION_GUIDE.md)** - Achieve 2-3x speedup for TorchTrainer
- [Custom Signals](advanced/custom-signals.md) - Extend Signal1D and Signal3D classes
- [Extending Protocols](advanced/extending-protocols.md) - Create custom TestProtocol implementations
- [Parallel Processing](advanced/parallel-processing.md) - Batch processing patterns and strategies
- [Unit Handling](advanced/unit-handling.md) - Deep dive into Pint integration
- [Performance Tips](advanced/performance-tips.md) - Best practices for performance

## 💡 Examples

Runnable Python scripts demonstrating common tasks:

- **[Basic Examples](examples/basic/README.md)** - Loading, filtering, exporting (3 scripts)
  - [Load and Plot](examples/basic/load-and-plot.py)
  - [Filter Signal](examples/basic/filter-signal.py)
  - [Export to Excel](examples/basic/export-to-excel.py)

- **[Biomechanics Examples](examples/biomechanics/README.md)** - Joint angles, GRF, markers (3 scripts)
  - [Joint Angles](examples/biomechanics/joint-angles.py)
  - [GRF Analysis](examples/biomechanics/grf-analysis.py)
  - [Marker Tracking](examples/biomechanics/marker-tracking.py)

- **[Protocol Examples](examples/protocols/README.md)** - CMJ, running, balance tests (3 scripts)
  - [CMJ Test](examples/protocols/cmj-test.py)
  - [Running Test](examples/protocols/running-test.py)
  - [Balance Test](examples/protocols/balance-test.py)

- **[Modeling Examples](examples/modeling/README.md)** - Polynomial fit, PyTorch, 1RM (3 scripts)
  - [Polynomial Fit](examples/modeling/polynomial-fit.py)
  - [PyTorch Training](examples/modeling/pytorch-training.py)
  - [1RM Prediction](examples/modeling/1rm-prediction.py)

[View all examples →](examples/README.md)

## 🐛 Troubleshooting

Solutions to common issues and problems:

- **[Common Errors](troubleshooting/common-errors.md)** - Frequent errors and their solutions
- [Installation Issues](troubleshooting/installation-issues.md) - Resolve installation problems
- [Data Loading Issues](troubleshooting/data-loading-issues.md) - Fix file format issues
- [Performance Issues](troubleshooting/performance-issues.md) - Optimize analysis speed

## 🛠️ Development

Contributing to labanalysis development:

- **[Developer Guide](development/README.md)** - Development overview and setup
- [Contributing](development/contributing.md) - How to contribute to the project
- [Testing](development/testing.md) - Test suite documentation ([see test/README.md](../test/README.md))
- [Code Style](development/code-style.md) - PEP 8 and NumPy docstring guidelines
- [Architecture](development/architecture.md) - Library design and architecture

## 📖 References

Scientific and technical references:

- [Biomechanics References](references/biomechanics.md) - Winter 2009, De Leva 1996, and more
- [Equations References](references/equations.md) - ACSM, Brzycki, and other equations
- [File Format Specifications](references/file-formats.md) - TDF, C3D technical specifications

---

**Need help?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)

**Version**: 202 | **Python**: ≥ 3.12 | **Last Updated**: June 2026
