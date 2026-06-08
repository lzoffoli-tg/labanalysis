# Labanalysis Test Suite

This directory contains the comprehensive test suite for the labanalysis package.

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

## 🧪 Running Tests

### Run All Tests
```bash
pytest test/
```

### Run Specific Test Module
```bash
# Run only RunningExercise tests
pytest test/test_runningexercise.py -v

# Run only jump tests
pytest test/test_jumps.py -v
```

### Run Specific Test Class or Function
```bash
# Run a specific test class
pytest test/test_runningexercise.py::TestRunningExerciseKinematics -v

# Run a specific test function
pytest test/test_runningexercise.py::TestRunningExerciseKinematics::test_basic_cycle_detection -v
```

### Run Tests in Parallel
```bash
# Use multiple CPU cores
pytest test/ -n auto
```

### Run Tests with Coverage
```bash
# Generate coverage report
pytest test/ --cov=labanalysis --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

## 📁 Test Structure

```
test/
├── README.md                    # This file
├── conftest.py                  # Shared fixtures and utilities
├── assets/                      # Test data files
│   ├── running_test/           # Running exercise test data
│   ├── jumptest_data/          # Jump test data
│   ├── balance_data/           # Balance test data
│   └── ...
├── test_runningexercise.py     # RunningExercise comprehensive tests
├── test_jumps.py               # Jump test protocols
├── test_balance.py             # Balance tests
├── test_strengthtests.py       # Strength tests
└── ...
```

## 🔬 Test Coverage

### RunningExercise Tests (`test_runningexercise.py`)

Comprehensive test protocol with 30+ test cases:

- **Kinematics Algorithm**: Marker-based cycle detection
- **Kinetics Algorithm**: Force platform-based cycle detection
- **Edge Cases**: Error handling, boundary conditions
- **Integration**: Real TDF file testing
- **Parametric Tests**: Various configurations

**Fixtures:**
- `mock_running_markers`: Synthetic Point3D markers (500 Hz)
- `mock_force_platform`: Synthetic ForcePlatform data (500 Hz)
- `running_tdf_file`: Path to real running test TDF
- `marker_mapping`: Standard marker set mapping

**Test Classes:**
- `TestRunningExerciseKinematics` (9 tests)
- `TestRunningExerciseKinetics` (7 tests)
- `TestRunningExerciseEdgeCases` (3 tests)
- `TestRunningStepProperties` (5 tests)
- `TestRunningExerciseIntegration` (4 tests)
- `TestRunningExerciseParametric` (2 tests)

## 🎯 Writing New Tests

### Test Naming Convention
- Test files: `test_<module>.py`
- Test classes: `Test<FeatureName>`
- Test functions: `test_<specific_behavior>`

### Example Test Structure
```python
import pytest
import src.labanalysis as laban

class TestMyFeature:
    """Tests for MyFeature functionality."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic feature behavior."""
        # Arrange
        obj = laban.MyClass(param=value)
        
        # Act
        result = obj.method()
        
        # Assert
        assert result == expected
        assert isinstance(result, ExpectedType)
```

### Using Fixtures
```python
@pytest.fixture
def my_fixture():
    """Provide test data or objects."""
    data = create_test_data()
    return data

def test_with_fixture(my_fixture):
    """Test using the fixture."""
    result = process(my_fixture)
    assert result is not None
```

### Parametric Tests
```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_doubling(input, expected):
    """Test doubling function with multiple inputs."""
    assert double(input) == expected
```

## 📊 Test Data

Test data files are located in `test/assets/` and organized by test type:

- `running_test/` - Running biomechanics data (TDF files)
- `jumptest_data/` - Jump test protocols
- `balance_data/` - Balance and posture tests
- `shuttle_test_data/` - Agility tests
- `cosmed_data/` - Metabolic data

**Note**: Large test data files (>10MB) are included in the repository but excluded from the built package distribution.

## 🐛 Troubleshooting

### Test Discovery Issues
If pytest doesn't find your tests:
```bash
# Make sure you're in the project root
cd /path/to/labanalysis

# Check pytest can find the tests
pytest --collect-only test/
```

### Import Errors
If you get import errors:
```bash
# Reinstall in editable mode
pip install -e ".[dev]"

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%cd%\src  # Windows
```

### Slow Tests
Some tests (especially integration tests with real TDF files) can be slow:
```bash
# Skip slow tests
pytest test/ -m "not slow"

# Run only fast unit tests
pytest test/ -k "not Integration"

# Set timeout for hanging tests
pytest test/ --timeout=60
```

## 📝 Contributing

When adding new tests:

1. **Follow project conventions** (see existing tests)
2. **Add docstrings** explaining what is being tested
3. **Use meaningful assertions** with clear failure messages
4. **Keep tests isolated** (no shared state between tests)
5. **Use fixtures** for common setup
6. **Document complex test logic** with comments

## 📚 References

- [pytest documentation](https://docs.pytest.org/)
- [labanalysis documentation](https://github.com/lzoffoli-tg/labanalysis)
- [Testing best practices](https://docs.pytest.org/en/stable/goodpractices.html)
