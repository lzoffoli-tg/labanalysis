# labanalysis

**labanalysis** is a comprehensive Python package for laboratory data analysis, with a focus on biomechanics, exercise science, and human movement research. It provides a unified, extensible framework for reading, processing, analyzing, and visualizing data from laboratory equipment and protocols.

---

## Package Overview

The package is organized into several submodules, each targeting a specific aspect of laboratory data analysis:

### Core Modules

- **labanalysis.constants**
  Provides physical and physiological constants (e.g., gravity).

- **labanalysis.messages**
  Contains standardized messages and warnings for consistent user feedback.

- **labanalysis.signalprocessing**
  Utilities for filtering, peak detection, derivatives, and other signal processing tasks.

- **labanalysis.utils**
  General-purpose utility functions used throughout the package.

### Data Structures

- **labanalysis.records**
  Classes for representing participants, timeseries, force platforms, and other core data types.

### Data Import/Export

- **labanalysis.io**
  Tools for reading and writing data in various laboratory formats.
  - **io.read**: Importers for BTS Bioengineering, Cosmed, OpenSim, and more.
  - **io.write**: Exporters for supported formats.

### Modelling & Regression

- **labanalysis.modelling**
  Tools for regression, geometric modeling, and machine learning.
  - **modelling.ols**: Ordinary least squares and geometric regression.
  - **modelling.pytorch**: Utilities for PyTorch-based deep learning.

### Equations

- **labanalysis.equations**
  Predictive equations for strength, cardiovascular, and other physiological parameters.

### Pipelines

- **labanalysis.pipelines**
  Data processing pipelines for reproducible and automated workflows.

### Plotting

- **labanalysis.plotting**
  Visualization tools, primarily using Plotly for interactive plots.

### Test Protocols

- **labanalysis.testprotocols**
  Standardized protocols for laboratory tests, including:
  - **balancetests**: Balance assessment protocols.
  - **jumptests**: Jump and plyometric test protocols.
  - **locomotiontests**: Gait and locomotion analysis.
  - **strengthtests**: Strength and force assessment.
  - **normativedata**: Reference normative datasets.

---

## Typical Usage

```python
import labanalysis as laban

# Access constants
g = laban.G

# Load a timeseries record from a .tdf file
ts = laban.TimeseriesRecord.from_tdf("path/to/file.tdf")

# Create a participant
pr = laban.Participant(name="John", surname="Doe", height=1.80, weight=75)

# Run a gait analysis test
gait_test = laban.WalkingTest.from_tdf("path/to/file.tdf", participant=pr)
results = gait_test.results
```

## License
This software is proprietary and may not be copied, distributed, or used without the explicit written approval of the author.
For permission requests, please contact: Luca Zoffoli – [lzoffoli@technogym.com](lzoffoli@technogym.com)
