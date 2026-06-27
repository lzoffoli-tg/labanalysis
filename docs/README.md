# labanalysis Documentation

Welcome to the comprehensive documentation for **labanalysis**, a Python package for biomechanical and laboratory data analysis developed by Technogym Scientific Research.

## Quick Navigation

### 📖 [Guides](guides/README.md)

Practical guides and examples for using labanalysis - from data loading to advanced modeling.

**Start here if**: You want to learn how to use labanalysis for your biomechanical analysis tasks.

**Topics covered**:
- Data loading (BTS, OpenSim, Biostrength, IRCAM)
- Signal processing (filtering, peak detection, derivatives, FFT)
- Biomechanics (WholeBody model, force platforms, joint angles, EMG)
- Test protocols (jump, gait, balance, strength, agility, VO2max)
- Modeling & analysis (regression, PyTorch, ONNX deployment)
- Data export and visualization
- Advanced topics and troubleshooting

### 🔧 [API Reference](api/README.md)

Complete API documentation for all classes, functions, and modules.

**Start here if**: You need detailed reference documentation for specific functions or classes.

**Modules covered**:
- Records (Signal1D, Signal3D, Record, WholeBody, EMGSignal)
- Protocols (TestProtocol, Participant, jump/gait/balance/strength tests)
- Signal Processing (30+ functions for filtering, peaks, derivatives, FFT)
- I/O (read/write TDF, OpenSim, and other formats)
- Modeling (OLS regression, PyTorch integration, TorchTrainer)
- Equations (Brzycki 1RM, VO2 prediction, ACSM formulas)
- Plotting and visualization utilities

### 📚 [Scientific References](references/README.md)

Peer-reviewed literature and technical specifications supporting labanalysis methodology.

**Start here if**: You need scientific citations, DOIs, or want to understand the theoretical foundations.

**References included**:
- Biomechanics (Winter 2009, De Leva 1996, ISB standards, Wu et al. 2002/2005)
- Predictive equations (Brzycki 1RM, ACSM VO2max, Bosco jump formulas)
- File formats (TDF, C3D, OpenSim technical specifications)
- Signal processing (Butterworth filtering, Winter derivatives)
- Gait analysis (Perry & Burnfield, spatiotemporal parameters)
- EMG processing (SENIAM recommendations, De Luca 2010)

---

## Quick Start

### Installation

```bash
pip install git+https://github.com/lzoffoli-tg/labanalysis.git
```

### Basic Example

```python
import labanalysis as laban

# Load data from TDF file
record = laban.TimeseriesRecord.from_tdf("data.tdf")

# Access force platform data
fp = record['FP1']
force = fp.force['Fz']

# Filter the signal
filtered = laban.butterworth_filt(
    signal=force.data,
    freq=force.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# Detect peaks
peaks = laban.find_peaks(filtered, height=500, distance=50)
print(f"Found {len(peaks['peak_heights'])} peaks")
```

**Next steps**: See the [Guides](guides/README.md) for detailed tutorials.

---

## Documentation Structure

```
docs/
├── guides/              Task-oriented guides organized by topic
│   ├── data-loading/    Loading data from various systems
│   ├── signal-processing/  Filtering, peaks, derivatives, FFT
│   ├── biomechanics/    Kinematics, kinetics, joint angles
│   ├── test-protocols/  Jump, gait, balance, strength tests
│   ├── modeling/        Regression and machine learning
│   ├── export/          Data export workflows
│   ├── visualization/   Interactive plotting
│   ├── advanced/        Custom extensions and optimization
│   └── troubleshooting/ Common errors and solutions
│
├── api/       Complete API documentation by module
│   ├── records/         Data structures (Signal, Record, WholeBody)
│   ├── protocols/       Test protocols and participants
│   ├── io/              Data readers and writers
│   ├── modelling/       Statistical and ML models
│   └── equations/       Predictive equations
│
└── references/          Scientific literature and technical specs
    ├── biomechanics.md  Peer-reviewed biomechanics papers
    ├── equations.md     Equation sources and validation
    └── file-formats.md  Technical format specifications
```

---

## Getting Help

- Check [Common Errors](guides/troubleshooting/overview.md) for quick solutions
- Review the [Guides](guides/README.md) for task-specific help
- Consult the [API Reference](api/README.md) for function documentation
- See [Scientific References](references/README.md) for methodology citations
- **Contact**: [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)

---

## Contributing

Interested in contributing to labanalysis? See the [Contributing Guide](guides/contributing.md) and [Testing Guide](guides/testing.md).

---

**Version**: 210 | **Python**: ≥ 3.12 | **Last Updated**: June 2026
