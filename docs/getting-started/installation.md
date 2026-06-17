# Installation Guide

Complete guide to installing labanalysis on your system.

## Prerequisites

Before installing labanalysis, ensure you have:

- **Python**: Version 3.12 or higher
- **pip** or **conda**: Package manager for Python
- **Git** (optional): For installing from source

### Check Python Version

```bash
python --version
# Output: Python 3.12.x or higher
```

If you need to upgrade Python, visit [python.org/downloads](https://www.python.org/downloads/).

## Installation Methods

### Method 1: Install from Git Repository (Recommended)

Install directly from the GitHub repository using pip:

```bash
pip install git+https://github.com/lzoffoli-tg/labanalysis.git
```

This installs the latest stable version with all required dependencies.

**Advantages:**
- Always get the latest version
- Simple one-command installation
- Automatic dependency resolution

### Method 2: Development Installation

For development work, contributing to the project, or running tests, install in development mode:

```bash
# Clone the repository
git clone https://github.com/lzoffoli-tg/labanalysis.git
cd labanalysis

# Install in development mode with test dependencies
pip install -e ".[dev]"
```

**What this includes:**
- Editable installation (changes to source code are immediately available)
- All test dependencies:
  - `pytest` - Testing framework
  - `pytest-cov` - Coverage reporting
  - `pytest-xdist` - Parallel test execution
  - `pytest-timeout` - Test timeouts

**Use this method when you:**
- Want to modify the library code
- Need to run the test suite
- Are contributing to the project

### Method 3: Conda Environment (Recommended for Isolation)

Create a dedicated conda environment for labanalysis:

```bash
# Create new environment with Python 3.12
conda create -n labanalysis python=3.12

# Activate the environment
conda activate labanalysis

# Install labanalysis
pip install git+https://github.com/lzoffoli-tg/labanalysis.git
```

**Advantages:**
- Isolated environment
- No conflicts with other Python projects
- Easy to recreate or remove

## Dependencies

labanalysis automatically installs these core dependencies:

### Scientific Computing
- **numpy** - Numerical computing and arrays
- **scipy** - Scientific computing and signal processing
- **pandas** - Data manipulation and analysis

### Machine Learning
- **scikit-learn** - Machine learning and statistical modeling
- **torch** (PyTorch) - Deep learning framework

### Visualization
- **plotly** - Interactive visualizations

### Physical Units
- **pint** - Physical units handling and conversions

### File I/O
- **openpyxl** - Excel file support

### Model Deployment
- **onnxmodels** - ONNX model utilities

All dependencies are installed automatically. No manual installation needed.

## Verify Installation

Test that labanalysis is correctly installed:

```python
import labanalysis as laban

# Check version
print(f"labanalysis version: {laban.__version__}")
# Output: labanalysis version: 202

# Test basic functionality
import numpy as np
signal = laban.Signal1D(
    data=np.random.randn(100),
    index=np.linspace(0, 1, 100),
    label='test_signal',
    unit='V'
)
print(f"Signal created with {len(signal)} samples")
# Output: Signal created with 100 samples
```

If this runs without errors, installation was successful.

## Common Installation Issues

### Issue: ImportError after installation

**Symptom:**
```
ImportError: No module named 'labanalysis'
```

**Solution:**
Ensure you're using the correct Python environment:
```bash
# Check which Python is active
which python  # Linux/Mac
where python  # Windows

# Verify labanalysis is installed
pip list | grep labanalysis
```

### Issue: Version conflicts

**Symptom:**
```
ERROR: package-name has requirement dependency<version, but you have dependency>version
```

**Solution:**
Create a fresh virtual environment:
```bash
python -m venv labanalysis_env
source labanalysis_env/bin/activate  # Linux/Mac
labanalysis_env\Scripts\activate     # Windows
pip install git+https://github.com/lzoffoli-tg/labanalysis.git
```

### Issue: Git not found

**Symptom:**
```
'git' is not recognized as an internal or external command
```

**Solution:**
1. Install Git from [git-scm.com](https://git-scm.com/)
2. Or use conda: `conda install git`
3. Or download and install manually from GitHub

[→ See all troubleshooting →](../troubleshooting/installation-issues.md)

## Upgrading labanalysis

To upgrade to the latest version:

```bash
pip install --upgrade git+https://github.com/lzoffoli-tg/labanalysis.git
```

For development installations:
```bash
cd labanalysis
git pull origin main
pip install -e ".[dev]"
```

## Uninstalling

To remove labanalysis:

```bash
pip uninstall labanalysis
```

For conda environments, you can also remove the entire environment:
```bash
conda deactivate
conda remove -n labanalysis --all
```

## Next Steps

Once installed, proceed to:

1. **[Quick Start (5 min)](quick-start.md)** - Your first analysis
2. **[Core Concepts](core-concepts.md)** - Understand the library structure
3. **[Your First Analysis](first-analysis.md)** - Complete walkthrough

## Additional Resources

- [System Requirements](../troubleshooting/installation-issues.md#system-requirements)
- [Development Setup](../development/README.md)
- [Test Suite](../development/testing.md)

---

**Need help?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
