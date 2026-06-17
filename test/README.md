# labanalysis Test Suite

Comprehensive test suite for the labanalysis biomechanical analysis library.

## Overview

This test suite validates the core functionality of labanalysis, covering signal processing, biomechanical data structures, and modeling capabilities.

**Current Status:** 518 passing tests, 3 skipped

## 📦 Installation

The test suite is **not included** in normal installations of labanalysis. To access the tests, you need to install the package in **development mode**.

### Standard Installation (No Tests)
```bash
pip install labanalysis
```

### Development Installation (With Tests)
```bash
# Clone the repository
git clone https://github.com/lzoffoli-tg/labanalysis.git
cd labanalysis

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

The `[dev]` extra installs additional dependencies needed for testing:
- pytest
- pytest-cov
- pytest-xdist
- pytest-timeout

## 🧪 Quick Start

```bash
# Run all tests
pytest test/

# Run specific module tests
pytest test/signalprocessing/
pytest test/records/
pytest test/modelling/

# Run with coverage
pytest test/ --cov=labanalysis --cov-report=html
```

## 📁 Directory Structure

```
test/
├── README.md               # This file
│
├── signalprocessing/       # Signal processing tests (300 tests)
│   └── test_signalprocessing.py
│
├── records/                # Data structures tests (52 tests)
│   ├── test_bodies.py
│   ├── test_item_access.py
│   └── test_records.py
│
└── modelling/              # Modelling and ML tests (166 tests)
    ├── test_pytorch_modules.py
    ├── test_pytorch_utils.py
    └── test_regression.py
```

**Total:** 518 passing tests

## 🏗️ Test Coverage

The test suite covers three main areas of labanalysis:

1. **Signal Processing** (`test/signalprocessing/`)
   - Filters (Butterworth, moving average, Savitzky-Golay)
   - Derivatives (Winter's method, finite differences)
   - Spectral analysis (FFT, PSD, autocorrelation)
   - Utilities (peak detection, crossings, interpolation)

2. **Records & Data Structures** (`test/records/`)
   - Timeseries, Signal1D, Signal3D, Point3D
   - EMGSignal, Record, ForcePlatform
   - WholeBody (full-body marker model with 38 angular measures)
     * Joint centers with missing medial markers
     * All 38 angular measures accessibility
     * New marker support (cranial, foot, spine)
     * Reference frame calculations
   - Attribute/item interchangeable access

3. **Modelling** (`test/modelling/`)
   - OLS regression models (polynomial, power, exponential)
   - PyTorch utilities and modules
   - Multi-segment regression

## 📊 Test Characteristics

All tests are **unit tests** with these characteristics:
- Fast execution (test suite completes in ~75 seconds)
- No external file dependencies
- Test individual functions/classes in isolation
- Generate test data inline using numpy

**Example:**
```python
def test_butterworth_filter():
    """Test Butterworth lowpass filter."""
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    
    # Apply filter
    from labanalysis.signalprocessing import butterworth
    filtered = butterworth(signal, cutoff=50.0, order=4, fs=1000)
    
    assert filtered.shape == signal.shape
```

### Important: Source Code Testing

Tests in `test/records/test_bodies.py` import directly from the `src/` directory rather than 
the installed package. This ensures tests validate the current codebase state:

```python
import sys
from pathlib import Path

# Add src directory to path to import from source code
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import labanalysis as laban  # Now imports from src/
```

This approach:
- ✅ Tests current development code, not installed version
- ✅ Catches bugs before package installation
- ✅ Enables testing without reinstalling after each edit

## ⏩ Running Tests

**Run all tests:**
```bash
pytest test/
# 518 passed, 3 skipped in ~75s
```

**Run specific modules:**
```bash
pytest test/signalprocessing/  # 300 tests
pytest test/records/           # 52 tests  
pytest test/modelling/         # 166 tests
```

**Run with verbose output:**
```bash
pytest test/ -v
```

**Run specific test:**
```bash
pytest test/signalprocessing/test_signalprocessing.py::TestButterworth::test_lowpass_basic
```

## 🔧 Test Data Generation

Tests generate data inline using numpy, scipy, and labanalysis constructors:

```python
import numpy as np
import labanalysis as laban

def test_signal_filtering():
    """Test signal filtering with synthetic data."""
    # Generate test signal inline
    time = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * time)  # 10 Hz sine
    
    # Create Signal1D object
    sig = laban.Signal1D(data=signal, index=time, unit='V')
    
    # Test filtering
    filtered = laban.signalprocessing.butterworth(sig, cutoff=5.0)
    assert filtered.shape == sig.shape
```

**Benefits:**
- ✅ No external dependencies
- ✅ Deterministic and reproducible
- ✅ Fast execution
- ✅ Clear test intent



## 🎯 Writing New Tests

### Guidelines

1. **Generate data inline** using numpy/scipy
2. **Test one thing** per test function
3. **Use descriptive names** that explain what's being tested
4. **Keep tests fast** (avoid long computations)
5. **Document complex tests** with docstrings

### Example Structure

```python
import numpy as np
import pytest
import labanalysis as laban

class TestButterworth:
    """Tests for Butterworth filter."""
    
    def test_lowpass_basic(self):
        """Test basic lowpass filtering removes high frequencies."""
        # Arrange: Create test signal
        fs = 1000
        t = np.linspace(0, 1, fs)
        signal_10hz = np.sin(2 * np.pi * 10 * t)
        signal_100hz = np.sin(2 * np.pi * 100 * t)
        signal = signal_10hz + signal_100hz
        
        # Act: Apply 50 Hz lowpass filter
        filtered = laban.signalprocessing.butterworth(
            signal, cutoff=50.0, order=4, fs=fs
        )
        
        # Assert: High frequency attenuated
        assert filtered.shape == signal.shape
        assert np.max(np.abs(filtered - signal_10hz)) < 0.1
```

### Test Naming Convention
- Test files: `test_<module>.py`
- Test classes: `Test<FeatureName>`
- Test functions: `test_<specific_behavior>`

## 📈 Coverage

**Check current coverage:**
```bash
pytest test/ --cov=labanalysis --cov-report=html
open htmlcov/index.html  # View detailed coverage report
```

**Current coverage:**
- `signalprocessing`: Excellent (300 tests)
- `records`: Excellent (52 tests)
- `modelling`: Good (166 tests)

## 💡 Common Patterns

### Testing Signal Processing

```python
def test_filter_removes_high_frequency():
    """Test that lowpass filter removes high-frequency noise."""
    # Generate signal with known frequency components
    time = np.arange(0, 1.0, 0.001)
    signal_low = np.sin(2 * np.pi * 10 * time)   # 10 Hz
    signal_high = np.sin(2 * np.pi * 100 * time) # 100 Hz
    noisy = signal_low + signal_high
    
    sig = laban.Signal1D(data=noisy, index=time, unit='V')
    
    # Apply 50 Hz lowpass filter
    filtered = laban.signalprocessing.butterworth(sig, cutoff=50.0, order=4)
    
    # High frequency should be attenuated
    fft_orig = np.fft.rfft(sig.data)
    fft_filt = np.fft.rfft(filtered.data)
    freqs = np.fft.rfftfreq(len(time), 0.001)
    
    idx_100hz = np.argmin(np.abs(freqs - 100))
    attenuation = np.abs(fft_filt[idx_100hz]) / np.abs(fft_orig[idx_100hz])
    assert attenuation < 0.1  # >90% attenuation at 100 Hz
```

### Testing Protocols

```python
@pytest.fixture
def participant():
    """Standard test participant."""
    return laban.Participant(
        name="Test", surname="Subject",
        gender="Male", height=181, weight=75,
        birthdate=date(2000, 1, 1),
        recordingdate=date(2025, 1, 1)
    )

def test_jump_protocol(participant, cmj_force_platform):
    """Test complete jump analysis workflow."""
    jump = laban.SingleJump(
        participant=participant,
        force_platform=cmj_force_platform,
        jump_type="CMJ"
    )
    
    # Validate core metrics
    assert jump.jump_height is not None
    assert_realistic_jump_height(jump.jump_height)
    
    # Validate phase detection
    assert hasattr(jump, 'unweighting_phase')
    assert hasattr(jump, 'propulsion_phase')
    assert hasattr(jump, 'flight_phase')
    
    # Validate performance metrics
    assert jump.rsi is not None
    assert jump.peak_power is not None
```

### Testing Data I/O

```python
import tempfile
from pathlib import Path

def test_tdf_roundtrip():
    """Test that data survives TDF write/read cycle."""
    # Generate synthetic data
    fp_orig = generate_force_platform_data(duration_s=5.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write to TDF
        filepath = Path(tmpdir) / "test.tdf"
        laban.io.write.tdf(filepath, force_platform=fp_orig)
        
        # Read back
        fp_read = laban.io.read.tdf(filepath)
        
        # Validate
        np.testing.assert_allclose(
            fp_orig.force.data,
            fp_read.force.data,
            rtol=1e-6
        )
```

## 🐛 Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'labanalysis'`

**Solution:** Install package in development mode:
```bash
pip install -e .
```

### Test Discovery

**Problem:** Pytest doesn't find tests

**Solution:** Run from project root:
```bash
cd /path/to/labanalysis
pytest test/
```

### Slow Tests

The full test suite takes ~75 seconds. To run faster:
```bash
# Run only one module
pytest test/records/  # ~7 seconds

# Parallel execution (if pytest-xdist installed)
pytest test/ -n auto
```

## 🚀 CI/CD Integration

**Recommended GitHub Actions workflow:**

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest -m "not slow and not pytorch" --cov=labanalysis
      - run: pytest -m slow --timeout=600  # Slow tests with longer timeout
```

## 📝 Contributing

When adding new features to labanalysis:

1. **Write tests first** (TDD approach preferred)
2. **Use synthetic data** (avoid .tdf file dependencies)
3. **Mirror library structure** (test location matches source location)
4. **Apply appropriate markers** (`@pytest.mark.unit` or `@pytest.mark.integration`)
5. **Document test purpose** (clear docstrings)
6. **Validate coverage** (`pytest --cov`)

## 📚 Resources

- [pytest documentation](https://docs.pytest.org/)
- [labanalysis source code](../src/labanalysis/)
- [TDF file format specification](https://www.btsbioengineering.com/)
- [ACSM metabolic equations](https://www.acsm.org/)

---

**Maintainer:** Technogym Research Team  
**Last Updated:** 2026-06-17
