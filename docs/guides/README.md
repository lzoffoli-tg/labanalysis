# labanalysis Guides

Practical guides and examples for using labanalysis in biomechanical analysis and sports science.

## Quick Start

### Installation

```bash
pip install git+https://github.com/lzoffoli-tg/labanalysis.git
```

### Basic Workflow

```python
import labanalysis as laban

# Load data from TDF file
record = laban.TimeseriesRecord.from_tdf("data.tdf")

# Access force platform data
fp = record['FP1']
force_z = fp.force['Fz']

# Filter the signal
filtered = laban.butterworth_filt(
    signal=force_z.data,
    freq=force_z.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# Detect peaks
peaks = laban.find_peaks(filtered, height=500, distance=50)
print(f"Found {len(peaks['peak_heights'])} peaks")
```

---

## Data Loading

Load data from various biomechanical systems and file formats.

- [**Overview**](data-loading/overview.md) - Introduction to data loading patterns
- [BTS Bioengineering](data-loading/bts.md) - Load TDF files from BTS systems
- [OpenSim](data-loading/opensim.md) - Import OpenSim motion and model files
- [Biostrength](data-loading/biostrength.md) - Load isokinetic dynamometry data
- [IRCAM](data-loading/ircam.md) - Load IRCAM motion capture data

Related: [File Format Reference](../references/file-formats.md)

---

## Signal Processing

Filter, analyze, and transform time-series signals.

- [**Overview**](signal-processing/overview.md) - Introduction to signal processing
- [Filtering](signal-processing/filtering.md) - Butterworth, moving average, and other filters
- [Peak Detection](signal-processing/peaks.md) - Find peaks, valleys, and events
- [Derivatives](signal-processing/derivatives.md) - Calculate velocity and acceleration
- [Missing Data](signal-processing/missing-data.md) - Handle gaps and interpolation
- [Frequency Analysis](signal-processing/frequency.md) - FFT and spectral analysis
- [Transformations](signal-processing/transforms.md) - Coordinate transformations

---

## Biomechanics

Analyze human movement using marker-based kinematics and kinetics.

- [**Overview**](biomechanics/overview.md) - Introduction to biomechanical analysis
- [WholeBody Model](biomechanics/wholebody.md) - 42-marker full-body kinematics
- [Force Platforms](biomechanics/force-platforms.md) - Ground reaction force analysis
- [Joint Angles](biomechanics/joint-angles.md) - Calculate joint kinematics
- [Coordinate Systems](biomechanics/coordinates.md) - Reference frames
- [EMG Signals](biomechanics/emg.md) - Electromyography processing
- [Quick Reference](biomechanics/quick-ref.md) - Common formulas and patterns

Related: [Scientific References](../references/biomechanics.md)

---

## Test Protocols

Standardized protocols for functional tests and assessments.

- [**Overview**](test-protocols/overview.md) - Introduction to test protocol system
- [Jump Tests](test-protocols/jump-tests.md) - CMJ, SJ, drop jump analysis
- [Gait Analysis](test-protocols/gait.md) - Walking and running biomechanics
- [Balance Tests](test-protocols/balance.md) - Postural stability
- [Strength Assessment](test-protocols/strength.md) - Isokinetic and isometric testing
- [Agility Tests](test-protocols/agility.md) - Change of direction tests
- [VO2max Tests](test-protocols/vo2max.md) - Cardiorespiratory fitness

Related: [Equation References](../references/equations.md)

---

## Modeling & Analysis

Statistical modeling and machine learning for biomechanical data.

- [Regression](modeling/regression.md) - Ordinary least squares and polynomial fitting
- [PyTorch Basics](modeling/pytorch-basics.md) - Introduction to neural networks
- [TorchTrainer](modeling/trainer.md) - Training workflow and optimization
- [Custom Models](modeling/custom.md) - Build domain-specific architectures
- [ONNX Deployment](modeling/onnx.md) - Export models for production

---

## Data Export

Export analysis results to various formats.

- [OpenSim Export](export/opensim.md) - Write motion files for OpenSim
- [DataFrames](export/dataframes.md) - Convert to pandas
- [Reports](export/reports.md) - Generate automated reports

---

## Visualization

Create interactive plots and figures.

- [Plotly Basics](visualization/plotly.md) - Interactive plotting introduction
- [Comparison Plots](visualization/comparisons.md) - Compare trials and participants
- [Protocol Reports](visualization/reports.md) - Automated visualization
- [Custom Figures](visualization/custom.md) - Publication-ready figures

---

## Advanced Topics

Extend labanalysis for specialized use cases.

- [Custom Signals](advanced/custom-signals.md) - Subclass Signal1D/Signal3D
- [Extending Protocols](advanced/protocols.md) - Create custom TestProtocol classes
- [Parallel Processing](advanced/parallel.md) - Batch processing patterns
- [Unit Handling](advanced/units.md) - Pint integration deep dive
- [Performance Tips](advanced/performance.md) - Optimization best practices

---

## Troubleshooting

- [**Troubleshooting Overview**](troubleshooting/overview.md) - Common errors and solutions

---

## Contributing

- [**Contributing Guide**](contributing.md) - How to contribute
- [**Testing Guide**](testing.md) - Running and writing tests

---

## Additional Resources

- **[API Reference](../api/README.md)** - Complete API documentation
- **[Scientific References](../references/README.md)** - Peer-reviewed literature and technical specs
- **Contact:** [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
