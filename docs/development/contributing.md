# Contributing to labanalysis

Guidelines for contributing to the labanalysis project.

## Overview

We welcome contributions from the biomechanics and sports science community! Whether you're fixing bugs, adding new test protocols, improving documentation, or proposing new features, your help is appreciated.

**Types of contributions**:
- **Bug reports and fixes**
- **New test protocols** (agility tests, sport-specific assessments)
- **Documentation improvements** (examples, tutorials, API docs)
- **Performance optimizations**
- **New features** (signal processing methods, analysis techniques)

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/labanalysis.git
cd labanalysis

# Add upstream remote
git remote add upstream https://github.com/technogym/labanalysis.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### 3. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

**Branch naming conventions**:
- `feature/` - New features (e.g., `feature/add-agility-test`)
- `fix/` - Bug fixes (e.g., `fix/force-platform-unit-conversion`)
- `docs/` - Documentation changes (e.g., `docs/improve-jump-tutorial`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-signal-class`)

## Development Workflow

### 1. Make Changes

Follow the [code style guide](code-style.md):

```python
# Use type hints
def calculate_jump_height(flight_time: float) -> float:
    """
    Calculate jump height from flight time.
    
    Parameters
    ----------
    flight_time : float
        Flight time in seconds
    
    Returns
    -------
    float
        Jump height in meters
    """
    g = 9.81  # m/s²
    return (g * flight_time**2) / 8
```

### 2. Write Tests

Every new feature or bug fix should include tests (see [testing guide](testing.md)):

```python
# test/test_jumping.py

def test_jump_height_calculation():
    """Test jump height calculation from flight time."""
    # Arrange
    flight_time = 0.5  # seconds
    expected_height = 0.306  # meters
    
    # Act
    from labanalysis.records.jumping import SingleJump
    height = SingleJump._calculate_jump_height_from_flight_time(flight_time)
    
    # Assert
    assert abs(height - expected_height) < 0.001
```

### 3. Run Tests Locally

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=labanalysis --cov-report=html

# Run specific test file
pytest test/test_jumping.py

# Run specific test
pytest test/test_jumping.py::test_jump_height_calculation
```

### 4. Format Code

```bash
# Format with black
black src/labanalysis/

# Check style with flake8
flake8 src/labanalysis/

# Type check with mypy
mypy src/labanalysis/
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add agility test protocol for T-test

- Implement TTestProtocol class
- Add turn detection from pelvis velocity
- Include segment time analysis
- Add tests and documentation"
```

**Commit message format**:
```
Short summary (50 chars or less)

More detailed explanatory text if needed. Wrap at 72 characters.
Explain what changes were made and why.

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

### 6. Push and Create Pull Request

```bash
# Push branch to your fork
git push origin feature/your-feature-name
```

Then create a pull request on GitHub:

1. Go to your fork on GitHub
2. Click "Pull Request"
3. Select your branch
4. Fill in PR template (see below)
5. Submit for review

## Pull Request Guidelines

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for this change
- [ ] Updated existing tests

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code sections
- [ ] Updated documentation
- [ ] No new warnings generated
```

### What Makes a Good PR

**Good**:
- Focused on single feature or fix
- Well-tested with clear test cases
- Includes documentation updates
- Clear commit messages
- No unrelated changes

**Avoid**:
- Mixing multiple unrelated changes
- Large refactors without discussion
- Breaking existing API without deprecation
- Missing tests or documentation
- Formatting-only changes mixed with logic changes

## Adding New Features

### New Test Protocol

When adding a new test protocol:

1. **Create protocol class** in appropriate module (e.g., `src/labanalysis/protocols/agility.py`)
2. **Inherit from TestProtocol** base class
3. **Implement required methods**: `from_files()`, `process()`
4. **Create results class** inheriting from `TestResults`
5. **Add tests** in `test/test_protocols/`
6. **Document** in `docs/user-guide/test-protocols/`
7. **Add tutorial** in `docs/tutorials/`

Example structure:

```python
# src/labanalysis/protocols/agility.py

from labanalysis.protocols.protocols import TestProtocol, TestResults

class TTestProtocol(TestProtocol):
    """T-Test agility assessment."""
    
    @classmethod
    def from_tdf_file(cls, filepath, **kwargs):
        """Load from BTS file."""
        # Implementation
        pass
    
    def process(self):
        """Process test and return results."""
        # Implementation
        return TTestResults(...)

class TTestResults(TestResults):
    """Results container for T-Test."""
    
    def to_dataframe(self):
        """Export to DataFrame."""
        pass
```

### New Signal Processing Method

1. **Add function** to `src/labanalysis/signalprocessing/`
2. **Use NumPy docstring format**
3. **Support both Signal1D and Signal3D**
4. **Preserve units** through operations
5. **Add tests** with known inputs/outputs
6. **Document** in user guide

Example:

```python
# src/labanalysis/signalprocessing/filters.py

def custom_filter(signal, param1, param2=10):
    """
    Apply custom filter to signal.
    
    Parameters
    ----------
    signal : Signal1D or Signal3D
        Input signal
    param1 : float
        First parameter
    param2 : int, optional
        Second parameter (default: 10)
    
    Returns
    -------
    Signal1D or Signal3D
        Filtered signal (same type as input)
    
    Examples
    --------
    >>> filtered = custom_filter(signal, param1=5.0)
    """
    # Implementation
    pass
```

## Code Review Process

### As Author

1. **Self-review** before requesting review
2. **Respond to feedback** constructively
3. **Update PR** based on comments
4. **Mark conversations as resolved** when addressed

### As Reviewer

1. **Be constructive** and specific
2. **Ask questions** rather than demanding changes
3. **Suggest improvements** with examples
4. **Approve** when ready to merge

**Review checklist**:
- [ ] Code is readable and well-documented
- [ ] Tests cover new functionality
- [ ] No obvious bugs or edge cases missed
- [ ] Performance considerations addressed
- [ ] Breaking changes justified and documented

## Reporting Bugs

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug.

## To Reproduce
Steps to reproduce:
1. Load file '...'
2. Call method '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.10.5]
- labanalysis version: [e.g., 1.2.3]

## Minimal Reproducible Example
```python
import labanalysis as laban

# Minimal code that reproduces the bug
test = laban.SingleJump.from_tdf_file("example.tdf")
# Error occurs here
```

## Additional Context
Any other relevant information.
```

## Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature.

## Use Case
Why this feature is needed and how it would be used.

## Proposed API
Example of how the API would look:
```python
# Example usage
result = new_feature(input_data, param1=value)
```

## Alternatives Considered
Other approaches you've considered.

## Additional Context
Any other relevant information.
```

## Documentation

### Documentation Changes

- Update relevant user guide sections
- Add examples in `docs/examples/`
- Update API reference if needed
- Add tutorial for complex features

See [Writing Documentation](#writing-documentation) for style guidelines.

## Community

### Getting Help

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and ideas
- **Email** - [research@technogym.com](mailto:research@technogym.com)

### Code of Conduct

Be respectful and constructive:

- Use welcoming and inclusive language
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

## Release Process

Maintainers handle releases:

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and upload to PyPI
5. Create GitHub release

## Questions?

If you have questions about contributing:

- Check existing issues and PRs
- Read the documentation
- Ask in GitHub Discussions
- Contact maintainers

## Quick Reference

```bash
# Setup
git clone https://github.com/YOUR_USERNAME/labanalysis.git
cd labanalysis
python -m venv venv
venv\Scripts\activate  # Windows
pip install -e .
pip install pytest pytest-cov black flake8

# Development
git checkout -b feature/my-feature
# Make changes
pytest
black src/
flake8 src/
git commit -m "Description"
git push origin feature/my-feature

# Update from upstream
git fetch upstream
git rebase upstream/main
```

---

**Thank you for contributing to labanalysis!** Your contributions help advance biomechanics research and sports science.
