# labanalysis.protocols.protocols

Base protocol classes for test organization and participant management.

**Source**: `src/labanalysis/protocols/protocols.py`

## Overview

The `protocols.protocols` module provides the foundational classes for organizing biomechanical tests:

- **Participant**: Stores participant demographic and anthropometric data
- **TestProtocol**: Base class for all test protocols (interface)
- **TestResults**: Base class for test results (interface)

All specific test protocols (jump tests, gait tests, balance tests, etc.) inherit from these base classes.

## Classes

### Participant

Stores participant information for lab tests.

```python
class Participant:
    """
    Represents a participant in a lab test.
    
    Stores demographic and anthropometric data with automatic BMI calculation,
    age computation, and theoretical HR max estimation.
    
    Parameters
    ----------
    surname : str, optional
        Participant's surname
    name : str, optional
        Participant's first name
    gender : str, optional
        Gender ('M', 'F', or custom)
    height : int or float, optional
        Height in centimeters (converted to meters internally)
    weight : int or float, optional
        Weight in kilograms
    age : int or float, optional
        Age in years (alternative to birthdate)
    birthdate : datetime.date, optional
        Birth date (used for age calculation)
    recordingdate : datetime.date, optional
        Test recording date (defaults to current date)
    
    Properties
    ----------
    surname : str or None
        Participant's surname
    name : str or None
        Participant's first name
    fullname : str
        Full name (surname + name)
    gender : str or None
        Gender
    height : float or None
        Height in meters
    weight : float or None
        Weight in kilograms
    bmi : float or None
        Body Mass Index (kg/m²)
    birthdate : datetime.date or None
        Birth date
    recordingdate : datetime.date or None
        Test recording date
    age : int or None
        Age in years (calculated from birthdate and recordingdate)
    hrmax : float or None
        Maximum theoretical heart rate (Gellish formula: 207 - 0.7 * age)
    dict : dict
        Dictionary representation
    series : pd.Series
        Pandas Series representation
    dataframe : pd.DataFrame
        Pandas DataFrame representation (single row)
    units : dict
        Units for each attribute
    
    Methods
    -------
    set_surname(surname)
        Set surname
    set_name(name)
        Set first name
    set_gender(gender)
        Set gender
    set_height(height)
        Set height (in meters)
    set_weight(weight)
        Set weight (in kg)
    set_age(age)
        Set age (in years)
    set_birthdate(birthdate)
        Set birth date
    set_recordingdate(recordingdate)
        Set recording date
    copy()
        Return a copy of the participant
    
    Examples
    --------
    >>> from labanalysis.protocols import Participant
    >>> from datetime import date
    >>> 
    >>> # Create participant with basic info
    >>> p = Participant(
    ...     surname='Rossi',
    ...     name='Mario',
    ...     gender='M',
    ...     height=175,  # cm (converted to 1.75 m)
    ...     weight=75,   # kg
    ...     birthdate=date(1990, 5, 15)
    ... )
    >>> 
    >>> # Access properties
    >>> print(f"Name: {p.fullname}")
    >>> print(f"Age: {p.age} years")
    >>> print(f"BMI: {p.bmi:.1f} kg/m²")
    >>> print(f"HR max: {p.hrmax:.0f} bpm")
    >>> 
    >>> # Get as DataFrame
    >>> df = p.dataframe
    >>> print(df)
    """
```

**Example - Complete Participant:**

```python
from labanalysis.protocols import Participant
from datetime import date

# Create participant
participant = Participant(
    surname='Smith',
    name='John',
    gender='M',
    height=180,  # cm
    weight=80,   # kg
    birthdate=date(1985, 3, 20),
    recordingdate=date(2024, 6, 15)
)

# Access computed properties
print(f"Full name: {participant.fullname}")
# Output: Full name: Smith John

print(f"Height: {participant.height:.2f} m")
# Output: Height: 1.80 m

print(f"Age: {participant.age} years")
# Output: Age: 39 years

print(f"BMI: {participant.bmi:.1f} kg/m²")
# Output: BMI: 24.7 kg/m²

print(f"HR max (Gellish): {participant.hrmax:.0f} bpm")
# Output: HR max (Gellish): 180 bpm

# Export to DataFrame
df = participant.dataframe
df.to_excel("participant_info.xlsx", index=False)
```

**Example - BMI Categories:**

```python
from labanalysis.protocols import Participant

# Create participants with different BMI
participants = [
    Participant(surname='A', height=170, weight=55),   # Underweight
    Participant(surname='B', height=175, weight=75),   # Normal
    Participant(surname='C', height=180, weight=95),   # Overweight
    Participant(surname='D', height=175, weight=105),  # Obese
]

# Categorize BMI
for p in participants:
    bmi = p.bmi
    if bmi < 18.5:
        category = 'Underweight'
    elif 18.5 <= bmi < 25:
        category = 'Normal'
    elif 25 <= bmi < 30:
        category = 'Overweight'
    else:
        category = 'Obese'
    
    print(f"{p.surname}: BMI {bmi:.1f} - {category}")
```

**Example - Age Calculation:**

```python
from labanalysis.protocols import Participant
from datetime import date

# Use birthdate for automatic age calculation
p1 = Participant(
    surname='Young',
    birthdate=date(2005, 1, 1),
    recordingdate=date(2024, 6, 1)
)
print(f"{p1.surname}: {p1.age} years old")
# Output: Young: 19 years old

# Or provide age directly
p2 = Participant(
    surname='Elder',
    age=65
)
print(f"{p2.surname}: {p2.age} years old")
# Output: Elder: 65 years old
```

---

### TestProtocol

Base interface for all test protocols.

```python
@runtime_checkable
class TestProtocol(Protocol):
    """
    Protocol (interface) for test classes.
    
    All specific test protocols (JumpTest, RunningTest, etc.) must implement
    this interface to ensure consistent API across different test types.
    
    Required Attributes
    -------------------
    participant : Participant
        The participant performing the test
    
    Required Methods
    ----------------
    process() -> TestResults
        Process raw data and return test results
    save(file_path: str) -> None
        Save test protocol to file
    load(file_path: str) -> Self
        Load test protocol from file (class method)
    
    Notes
    -----
    This is a Protocol (duck typing) - classes don't need to explicitly inherit
    from TestProtocol, they just need to implement the required interface.
    
    Use `isinstance(obj, TestProtocol)` to check if an object implements the
    protocol.
    
    Examples
    --------
    >>> from labanalysis.protocols import TestProtocol, Participant
    >>> from labanalysis.protocols import JumpTest
    >>> 
    >>> # JumpTest implements TestProtocol
    >>> participant = Participant(surname='Rossi', weight=75)
    >>> test = JumpTest(participant=participant, test_type='cmj')
    >>> 
    >>> # Check protocol compliance
    >>> assert isinstance(test, TestProtocol)
    >>> 
    >>> # Process test
    >>> results = test.process()
    >>> 
    >>> # Save/load
    >>> test.save("jump_test.pkl")
    >>> loaded_test = JumpTest.load("jump_test.pkl")
    """
```

**Protocol Requirements:**

Any class implementing `TestProtocol` must have:

1. **Attribute**: `participant` (Participant object)
2. **Method**: `process()` → returns TestResults
3. **Method**: `save(file_path: str)` → saves to file
4. **Class Method**: `load(file_path: str)` → loads from file

**Example Implementation Pattern:**

```python
from labanalysis.protocols import TestProtocol, Participant, TestResults

class MyCustomTest:
    """Custom test implementing TestProtocol."""
    
    def __init__(self, participant: Participant, **kwargs):
        self.participant = participant
        # ... other initialization
    
    def process(self) -> TestResults:
        """Process test data and return results."""
        # ... processing logic
        return MyCustomTestResults(...)
    
    def save(self, file_path: str) -> None:
        """Save test to file."""
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_path: str) -> 'MyCustomTest':
        """Load test from file."""
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)

# Verify protocol compliance
from labanalysis.protocols import TestProtocol
assert isinstance(MyCustomTest(participant=Participant()), TestProtocol)
```

---

### TestResults

Base interface for test results.

```python
@runtime_checkable
class TestResults(Protocol):
    """
    Protocol (interface) for test results classes.
    
    All specific test results (JumpTestResults, RunningTestResults, etc.) must
    implement this interface to ensure consistent API.
    
    Required Attributes
    -------------------
    participant : Participant
        The participant who performed the test
    
    Required Methods
    ----------------
    save(file_path: str) -> None
        Save results to file
    load(file_path: str) -> Self
        Load results from file (class method)
    plot() -> go.Figure
        Generate visualization of results
    
    Optional Methods
    ----------------
    to_dataframe() -> pd.DataFrame
        Export results to DataFrame
    summary() -> dict
        Get summary metrics
    
    Notes
    -----
    This is a Protocol (duck typing) - classes don't need to explicitly inherit
    from TestResults, they just need to implement the required interface.
    
    Examples
    --------
    >>> from labanalysis.protocols import TestResults
    >>> from labanalysis.protocols import JumpTestResults
    >>> 
    >>> # JumpTestResults implements TestResults
    >>> results = jump_test.process()
    >>> 
    >>> # Check protocol compliance
    >>> assert isinstance(results, TestResults)
    >>> 
    >>> # Use standard interface
    >>> results.save("results.pkl")
    >>> fig = results.plot()
    >>> df = results.to_dataframe()
    """
```

**Protocol Requirements:**

Any class implementing `TestResults` must have:

1. **Attribute**: `participant` (Participant object)
2. **Method**: `save(file_path: str)` → saves to file
3. **Class Method**: `load(file_path: str)` → loads from file
4. **Method**: `plot()` → returns plotly Figure

**Example Implementation Pattern:**

```python
from labanalysis.protocols import TestResults, Participant
import plotly.graph_objects as go
import pandas as pd

class MyCustomTestResults:
    """Custom test results implementing TestResults."""
    
    def __init__(self, participant: Participant, metrics: dict):
        self.participant = participant
        self.metrics = metrics
    
    def save(self, file_path: str) -> None:
        """Save results to file."""
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_path: str) -> 'MyCustomTestResults':
        """Load results from file."""
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def plot(self) -> go.Figure:
        """Generate visualization."""
        fig = go.Figure()
        # ... add traces
        return fig
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export to DataFrame."""
        return pd.DataFrame([self.metrics])

# Verify protocol compliance
from labanalysis.protocols import TestResults
assert isinstance(MyCustomTestResults(Participant(), {}), TestResults)
```

---

## Common Workflows

### 1. Create and Process Test

```python
from labanalysis.protocols import Participant, JumpTest

# Create participant
participant = Participant(
    surname='Athlete',
    name='Pro',
    gender='M',
    height=180,
    weight=75,
    birthdate=date(1995, 1, 1)
)

# Create test protocol
test = JumpTest(
    participant=participant,
    test_type='cmj',
    num_trials=3
)

# Process test
results = test.process()

# Save results
results.save("athlete_cmj_results.pkl")

# Visualize
fig = results.plot()
fig.show()
```

### 2. Load and Compare Previous Tests

```python
from labanalysis.protocols import JumpTestResults

# Load previous tests
baseline = JumpTestResults.load("baseline.pkl")
post_training = JumpTestResults.load("post_training.pkl")

# Compare
baseline_height = baseline.to_dataframe()['jump_height_m'].values[0]
post_height = post_training.to_dataframe()['jump_height_m'].values[0]

improvement = ((post_height - baseline_height) / baseline_height) * 100
print(f"Jump height improvement: {improvement:.1f}%")
```

### 3. Batch Process Multiple Participants

```python
from labanalysis.protocols import Participant, JumpTest
import pandas as pd

# Define participants
participants = [
    Participant(surname='A', height=170, weight=65),
    Participant(surname='B', height=175, weight=70),
    Participant(surname='C', height=180, weight=75),
]

# Process all
all_results = []
for p in participants:
    test = JumpTest(participant=p, test_type='cmj')
    results = test.process()
    df = results.to_dataframe()
    df['participant'] = p.surname
    all_results.append(df)

# Combine
team_results = pd.concat(all_results, ignore_index=True)
team_results.to_excel("team_jump_results.xlsx", index=False)
```

---

## Design Patterns

### Protocol Pattern (Duck Typing)

The Protocol pattern allows for flexible test implementations without strict inheritance:

```python
from labanalysis.protocols import TestProtocol

# Any class implementing the interface works
def process_any_test(test: TestProtocol):
    """Process any test that implements TestProtocol."""
    results = test.process()
    results.save(f"{test.participant.surname}_results.pkl")
    return results

# Works with any test type
process_any_test(JumpTest(...))
process_any_test(RunningTest(...))
process_any_test(MyCustomTest(...))  # Your own implementation
```

### Save/Load Pattern

All protocols support save/load for persistence:

```python
# Save test before processing
test.save("test_protocol.pkl")

# Load later
loaded_test = JumpTest.load("test_protocol.pkl")
results = loaded_test.process()

# Save results
results.save("test_results.pkl")

# Load results later
loaded_results = JumpTestResults.load("test_results.pkl")
fig = loaded_results.plot()
```

---

## See Also

- [Jump Tests](jump-tests.md) - Jump test protocols
- [Balance Tests](balance-tests.md) - Balance test protocols
- [Strength Tests](strength-tests.md) - Strength test protocols
- [Locomotion Tests](locomotion-tests.md) - Gait test protocols
- [VO2max Tests](vo2max.md) - Aerobic capacity tests
- [Agility Tests](agility-tests.md) - Agility test protocols

---

**Base classes for organizing biomechanical tests with participant management and standardized interfaces.**
