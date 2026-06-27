# Testing Guide

Comprehensive guide for writing and running tests in labanalysis.

## Overview

The labanalysis test suite ensures code quality, prevents regressions, and validates new features. This guide covers:

- **Test structure and organization**
- **Writing effective unit tests**
- **Testing with real biomechanical data**
- **Running and debugging tests**
- **Coverage analysis**

## Test Structure

```
test/
├── assets/                    # Test data files
│   ├── balance_data/
│   ├── jump_data/
│   └── motion_data/
├── test_bodies.py            # WholeBody class tests
├── test_jumping.py           # Jump protocol tests
├── test_signalprocessing.py  # Signal processing tests
├── test_protocols/           # Protocol-specific tests
│   ├── test_jump_protocols.py
│   ├── test_balance_protocols.py
│   └── test_strength_protocols.py
└── conftest.py               # Pytest fixtures and configuration
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific file
pytest test/test_jumping.py

# Run specific test
pytest test/test_jumping.py::test_jump_height_calculation

# Run tests matching pattern
pytest -k "jump"

# Stop at first failure
pytest -x

# Show local variables on failure
pytest -l
```

### Coverage Analysis

```bash
# Run with coverage
pytest --cov=labanalysis

# Generate HTML report
pytest --cov=labanalysis --cov-report=html

# View report
open htmlcov/index.html  # Mac/Linux
start htmlcov/index.html  # Windows

# Show missing lines
pytest --cov=labanalysis --cov-report=term-missing
```

**Coverage goals**:
- Overall: >80%
- Core modules (signalprocessing, records): >90%
- Protocols: >75%

## Writing Tests

### Basic Test Structure

```python
# test/test_jumping.py

import pytest
import numpy as np
import labanalysis as laban

def test_jump_height_calculation():
    """Test jump height calculation from flight time."""
    # Arrange - Set up test data
    flight_time = 0.5  # seconds
    expected_height = 0.306  # meters (calculated manually)
    
    # Act - Perform operation
    height = laban.SingleJump._calculate_jump_height_from_flight_time(flight_time)
    
    # Assert - Check result
    assert abs(height - expected_height) < 0.001
```

**Test naming**:
- Test files: `test_*.py`
- Test functions: `test_*`
- Descriptive names: `test_butterworth_filter_removes_noise`

### Using Fixtures

Fixtures provide reusable test data:

```python
# conftest.py

import pytest
import numpy as np
import labanalysis as laban
from pathlib import Path

@pytest.fixture
def sample_force_data():
    """Generate sample force platform data."""
    time = np.linspace(0, 2, 200)
    force_z = 1500 * np.sin(2 * np.pi * time) + 800
    
    fp = laban.ForcePlatform(
        data=np.column_stack([
            np.zeros_like(force_z),  # Fx
            np.zeros_like(force_z),  # Fy
            force_z,                 # Fz
            np.zeros_like(force_z),  # Mx
            np.zeros_like(force_z),  # My
            np.zeros_like(force_z),  # Mz
        ]),
        index=time,
        columns=[
            ("FORCE", "X"),
            ("FORCE", "Y"),
            ("FORCE", "Z"),
            ("MOMENT", "X"),
            ("MOMENT", "Y"),
            ("MOMENT", "Z"),
        ],
        unit='N'
    )
    
    return fp

@pytest.fixture
def jump_test_file():
    """Path to real jump test file."""
    return Path("test/assets/jump_data/athlete1_cmj.tdf")

# Usage in tests
def test_force_peak_detection(sample_force_data):
    """Test peak detection on force signal."""
    from labanalysis.signalprocessing import find_peaks
    
    peaks, props = find_peaks(sample_force_data["FORCE", "Z"], threshold=1000)
    
    assert len(peaks) > 0
    assert all(sample_force_data["FORCE", "Z"].to_numpy()[peaks] > 1000)
```

### Parametrized Tests

Test multiple cases with one function:

```python
import pytest

@pytest.mark.parametrize("flight_time,expected_height", [
    (0.3, 0.110),
    (0.5, 0.306),
    (0.7, 0.600),
    (1.0, 1.226),
])
def test_jump_heights(flight_time, expected_height):
    """Test jump height calculation for multiple flight times."""
    import labanalysis as laban
    
    height = laban.SingleJump._calculate_jump_height_from_flight_time(flight_time)
    
    assert abs(height - expected_height) < 0.001
```

### Testing with Real Data

```python
import pytest
from pathlib import Path
import labanalysis as laban

@pytest.mark.slow  # Mark as slow test
def test_cmj_protocol_with_real_data():
    """Test CMJ protocol on real athlete data."""
    # Load real test file
    test_file = Path("test/assets/jump_data/athlete1_cmj.tdf")
    
    if not test_file.exists():
        pytest.skip("Test data not available")
    
    # Process test
    test = laban.SingleJump.from_tdf_file(test_file)
    results = test.process()
    
    # Validate results are within reasonable ranges
    assert 0.1 < results.jump_height < 1.0  # 10cm to 1m
    assert 500 < results.peak_force < 5000  # 500N to 5000N
    assert 0.1 < results.flight_time < 1.0  # 0.1s to 1s
```

### Testing Exceptions

```python
import pytest
import labanalysis as laban

def test_invalid_filter_frequency_raises_error():
    """Test that invalid filter frequency raises ValueError."""
    from labanalysis.signalprocessing import butterworth_filter
    
    signal = laban.Signal1D(
        data=[1, 2, 3, 4],
        index=[0, 1, 2, 3],
        columns=['value'],
        unit='m'
    )
    
    # Should raise ValueError for negative frequency
    with pytest.raises(ValueError, match="Frequency must be positive"):
        butterworth_filter(signal, frequency=-10, order=4)
```

### Testing Approximate Values

```python
import numpy as np
import pytest

def test_numerical_calculation():
    """Test calculation with floating point tolerance."""
    result = complex_calculation()
    expected = 3.14159
    
    # Use pytest.approx for floating point comparison
    assert result == pytest.approx(expected, rel=1e-5)
    
    # Or use numpy for arrays
    result_array = np.array([1.0, 2.0, 3.0])
    expected_array = np.array([1.0, 2.0, 3.0])
    
    np.testing.assert_allclose(result_array, expected_array, rtol=1e-7)
```

## Test Categories

### Unit Tests

Test individual functions in isolation:

```python
def test_butterworth_filter_signal1d():
    """Test Butterworth filter on 1D signal."""
    from labanalysis.signalprocessing import butterworth_filter
    import labanalysis as laban
    
    # Create noisy signal
    time = np.linspace(0, 1, 100)
    clean = np.sin(2 * np.pi * time)
    noisy = clean + 0.5 * np.random.randn(100)
    
    signal = laban.Signal1D(data=noisy, index=time, columns=['value'], unit='m')
    
    # Filter
    filtered = butterworth_filter(signal, frequency=2, order=4)
    
    # Check noise reduction
    noise_original = np.std(noisy - clean)
    noise_filtered = np.std(filtered.to_numpy().flatten() - clean)
    
    assert noise_filtered < noise_original
```

### Integration Tests

Test multiple components working together:

```python
def test_jump_protocol_end_to_end():
    """Test complete jump analysis workflow."""
    import labanalysis as laban
    from pathlib import Path
    
    # Load data
    test_file = Path("test/assets/jump_data/athlete1_cmj.tdf")
    test = laban.SingleJump.from_tdf_file(test_file)
    
    # Process
    results = test.process()
    
    # Export
    df = results.to_dataframe()
    
    # Validate complete workflow
    assert 'jump_height' in df.columns
    assert len(df) > 0
    assert results.jump_height > 0
```

### Regression Tests

Prevent known bugs from reappearing:

```python
def test_issue_123_force_platform_unit_conversion():
    """
    Regression test for issue #123.
    
    Force platform unit conversion was incorrectly
    applied to moment channels.
    """
    import labanalysis as laban
    
    fp = laban.ForcePlatform(
        data=sample_data,
        index=sample_index,
        columns=sample_columns,
        unit='N'
    )
    
    fp_kN = fp.to_unit('kN')
    
    # Bug: Moment should NOT be converted with force
    # Moment has different dimensions (N·m)
    
    # Verify fix
    assert fp_kN["MOMENT", "Z"].unit == 'N·m'  # Not 'kN·m'
```

## Test Best Practices

### 1. Arrange-Act-Assert Pattern

```python
def test_peak_detection():
    """Test peak detection in force signal."""
    # Arrange - Set up test data
    time = np.linspace(0, 2, 200)
    force = 1000 * np.sin(2 * np.pi * time) + 500
    signal = create_signal(force, time)
    
    # Act - Perform operation
    peaks, props = find_peaks(signal, threshold=1000)
    
    # Assert - Verify results
    assert len(peaks) == 2
    assert all(signal[peaks] > 1000)
```

### 2. One Assertion Per Concept

```python
# ❌ BAD: Multiple unrelated assertions
def test_jump_results():
    assert results.jump_height > 0
    assert results.peak_force > 0
    assert results.flight_time > 0

# ✅ GOOD: Separate tests
def test_jump_height_is_positive():
    assert results.jump_height > 0

def test_peak_force_is_positive():
    assert results.peak_force > 0

def test_flight_time_is_positive():
    assert results.flight_time > 0
```

### 3. Clear Test Names

```python
# ❌ BAD: Vague name
def test_filter():
    pass

# ✅ GOOD: Descriptive name
def test_butterworth_filter_removes_high_frequency_noise():
    pass
```

### 4. Use Fixtures for Setup

```python
# ❌ BAD: Repeated setup
def test_force_peak():
    fp = create_force_platform()
    peak = fp.peak_force
    assert peak > 0

def test_force_impulse():
    fp = create_force_platform()  # Repeated
    impulse = fp.impulse
    assert impulse > 0

# ✅ GOOD: Fixture
@pytest.fixture
def force_platform():
    return create_force_platform()

def test_force_peak(force_platform):
    peak = force_platform.peak_force
    assert peak > 0

def test_force_impulse(force_platform):
    impulse = force_platform.impulse
    assert impulse > 0
```

### 5. Test Edge Cases

```python
def test_filter_with_empty_signal():
    """Test filter handles empty signal."""
    empty_signal = laban.Signal1D(data=[], index=[], columns=['value'], unit='m')
    
    with pytest.raises(ValueError):
        butterworth_filter(empty_signal, frequency=10, order=4)

def test_filter_with_single_sample():
    """Test filter handles single sample."""
    signal = laban.Signal1D(data=[1.0], index=[0.0], columns=['value'], unit='m')
    
    with pytest.raises(ValueError):
        butterworth_filter(signal, frequency=10, order=4)
```

## Debugging Failed Tests

### View Failure Details

```bash
# Show local variables
pytest -l

# Show full traceback
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Print statements (use -s to see output)
pytest -s
```

### Using pytest-pdb

```python
def test_complex_calculation():
    result = complex_function()
    
    # Set breakpoint for debugging
    import pdb; pdb.set_trace()
    
    assert result == expected
```

### Isolate Failing Test

```bash
# Run only failing test
pytest test/test_jumping.py::test_specific_failure -v

# Run with print output
pytest test/test_jumping.py::test_specific_failure -s
```

## Continuous Integration

Tests run automatically on GitHub Actions for every commit:

```yaml
# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=labanalysis --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Test Data Management

### Organizing Test Assets

```
test/assets/
├── jump_data/
│   ├── athlete1_cmj.tdf
│   ├── athlete2_cmj.tdf
│   └── athlete1_dropjump.tdf
├── balance_data/
│   ├── eyes_open.tdf
│   └── eyes_closed.tdf
└── motion_data/
    └── gait_trial.tdf
```

### Generating Synthetic Data

```python
# conftest.py

import numpy as np
import labanalysis as laban

def generate_jump_data(jump_height=0.3, body_mass=75):
    """
    Generate synthetic jump data.
    
    Useful for testing without real data files.
    """
    # Calculate flight time from jump height
    g = 9.81
    flight_time = np.sqrt(8 * jump_height / g)
    
    # Generate force-time curve
    total_time = 2.0  # seconds
    fs = 1000  # Hz
    time = np.arange(0, total_time, 1/fs)
    
    # Simulate jump phases
    force = np.zeros_like(time)
    
    # Standing phase (body weight)
    force[:500] = body_mass * g
    
    # Propulsion phase (accelerating)
    # ... implementation ...
    
    return force, time
```

## See Also

- [Contributing Guide](contributing.md) - How to contribute tests
- [Code Style Guide](code-style.md) - Python style guidelines
- [Architecture Guide](architecture.md) - Library structure

---

**Write tests for all new code.** Use fixtures for reusable setup, parametrize for multiple cases, and test edge cases. Aim for >80% coverage.
