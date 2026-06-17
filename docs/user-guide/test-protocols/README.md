# Test Protocols

Guide to standardized test protocols in labanalysis for biomechanical and physiological assessment.

## Overview

labanalysis provides standardized test protocols with automated analysis for:

- **Jump Tests** - Countermovement jump, squat jump, drop jump
- **Gait Analysis** - Walking and running kinematics
- **Balance Tests** - Static and dynamic balance assessment
- **Strength Tests** - Isokinetic and isometric testing
- **Agility Tests** - Change of direction performance
- **VO2max Tests** - Aerobic capacity estimation

## Quick Reference

| Protocol | Class | Guide |
|----------|-------|-------|
| **[Jump Tests](jump-tests.md)** | `JumpTest`, `JumpTestResults` | CMJ, SJ, DJ analysis |
| **[Gait Analysis](gait-analysis.md)** | `RunningTest`, `WalkingTest` | Walking/running kinematics |
| **[Balance Tests](balance-tests.md)** | `UprightBalanceTest`, `PlankBalanceTest` | Static/dynamic balance |
| **[Strength Tests](strength-tests.md)** | `Isokinetic1RMTest`, `IsometricTest` | Strength assessment |
| **[Agility Tests](agility-tests.md)** | `ShuttleTest` | COD performance |
| **[VO2max Tests](vo2max-tests.md)** | `SubmaximalVO2MaxTest` | Aerobic capacity |

## Quick Start

### Create a Participant

All test protocols require participant information:

```python
import labanalysis as laban

participant = laban.Participant(
    name="John",
    surname="Doe",
    gender="M",
    height=1.80,  # meters
    weight=75,    # kg
    age=25
)

# Computed properties
print(f"BMI: {participant.bmi:.1f} kg/m²")
print(f"Max HR (predicted): {participant.max_heart_rate:.0f} bpm")
```

### Run a Test Protocol

Example with jump test:

```python
from labanalysis.protocols import JumpTest

# Create and run test
test = JumpTest.from_tdf(
    "jump_data.tdf",
    participant=participant
)

# Access results
results = test.results
print(f"Jump height: {results['jump_height']:.2f} cm")
print(f"Peak power: {results['peak_power']:.0f} W")

# Generate report
report = test.report()
print(report)

# Visualize
fig = test.plot()
fig.show()
```

## Protocol Structure

All test protocols follow a consistent structure:

### 1. TestProtocol Interface

All protocols implement the `TestProtocol` interface:

```python
from typing import Protocol
from labanalysis.protocols import TestProtocol, TestResults

class MyCustomTest(TestProtocol):
    """Custom test protocol"""
    
    @property
    def results(self) -> TestResults:
        """Compute and return test results"""
        pass
    
    def plot(self):
        """Generate visualization"""
        pass
    
    def report(self) -> str:
        """Generate text report"""
        pass
```

### 2. Loading Data

Protocols use `.from_tdf()` or similar class methods:

```python
test = TestClass.from_tdf(
    "data.tdf",
    participant=participant,
    # ... test-specific parameters
)
```

### 3. Accessing Results

Results are accessed via the `.results` property:

```python
results = test.results

# Results is typically a dictionary
print(results.keys())
# Output: dict_keys(['metric1', 'metric2', ...])
```

### 4. Visualization

Generate interactive plots:

```python
fig = test.plot()
fig.show()  # Display in browser
fig.write_html("results.html")  # Save to file
```

### 5. Reporting

Generate formatted text reports:

```python
report = test.report()
print(report)
# Output: Formatted test report with metrics
```

## Available Protocols

### Jump Tests

Analyze jumping performance:

- Countermovement Jump (CMJ)
- Squat Jump (SJ)
- Drop Jump (DJ)
- Repeated Jumps

**Metrics:** Jump height, flight time, peak power, RSI, force-time characteristics

[→ Complete jump tests guide](jump-tests.md)

### Gait Analysis

Analyze walking and running:

- Walking kinematics
- Running kinematics
- Gait cycle extraction
- Temporal-spatial parameters

**Metrics:** Stride length, cadence, speed, contact time, flight time

[→ Complete gait analysis guide](gait-analysis.md)

### Balance Tests

Assess postural control:

- Upright standing balance
- Plank/prone balance
- Dynamic balance

**Metrics:** COP sway, path length, ellipse area, velocity

[→ Complete balance tests guide](balance-tests.md)

### Strength Tests

Measure muscular strength:

- Isokinetic testing (constant velocity)
- Isometric testing (constant length)
- 1RM estimation

**Metrics:** Peak torque, power, work, bilateral symmetry

[→ Complete strength tests guide](strength-tests.md)

### Agility Tests

Evaluate change of direction:

- Shuttle runs
- T-test
- Illinois agility

**Metrics:** Time, velocity, acceleration, deceleration

[→ Complete agility tests guide](agility-tests.md)

### VO2max Tests

Estimate aerobic capacity:

- Submaximal protocols
- Predictive equations
- Heart rate response

**Metrics:** Predicted VO2max, HR response, workload

[→ Complete VO2max tests guide](vo2max-tests.md)

## Common Workflows

### Batch Testing

Test multiple participants:

```python
participants = [
    laban.Participant(name="John", surname="Doe", height=1.80, weight=75, age=25),
    laban.Participant(name="Jane", surname="Smith", height=1.65, weight=60, age=28),
    # ... more participants
]

results_list = []
for participant in participants:
    test = JumpTest.from_tdf(f"{participant.name}_jump.tdf", participant=participant)
    results_list.append({
        'name': f"{participant.surname}, {participant.name}",
        **test.results
    })

# Create summary DataFrame
import pandas as pd
df = pd.DataFrame(results_list)
df.to_excel("batch_results.xlsx", index=False)
```

### Comparing Tests

Compare before/after or bilateral:

```python
# Test left and right legs
left_test = IsometricTest.from_file("left_leg.txt", participant=participant)
right_test = IsometricTest.from_file("right_leg.txt", participant=participant)

# Compare results
left_peak = left_test.results['peak_torque']
right_peak = right_test.results['peak_torque']

asymmetry = abs(left_peak - right_peak) / max(left_peak, right_peak) * 100
print(f"Bilateral asymmetry: {asymmetry:.1f}%")
```

## See Also

- **[API Reference: Protocols](../../api-reference/protocols/README.md)** - Complete API documentation
- **[Tutorials](../../tutorials/README.md)** - Complete workflow examples
- **[Biomechanics Guide](../biomechanics/README.md)** - Biomechanical analysis tools

---

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
