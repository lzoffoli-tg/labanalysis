# labanalysis.protocols.agility-tests

Agility test protocols for change-of-direction assessment.

**Source**: `src/labanalysis/protocols/agilitytests.py`

## Overview

Agility test protocol for shuttle run and change-of-direction assessment:

- **ShuttleTest**: Shuttle run test protocol with COD analysis
- **ShuttleTestResults**: Results with loading/propulsion phase analysis

The test analyzes multiple change-of-direction maneuvers, extracting contact times, phase durations, and velocity changes.

**Typical Use Cases:**
- Shuttle run performance assessment
- COD agility evaluation
- Deceleration-acceleration profiling
- Left-right asymmetry analysis

## Classes

### ShuttleTest

Shuttle run test protocol.

```python
class ShuttleTest(TestProtocol):
    """
    Protocol for shuttle run and change-of-direction assessment.
    
    Analyzes multiple COD maneuvers from shuttle run trials, extracting
    temporal metrics, force profiles, and velocity changes.
    
    Parameters
    ----------
    participant : Participant
        Participant information
    change_of_direction_exercises : list of ChangeOfDirectionExercise
        List of COD maneuver data
    normative_data : pd.DataFrame, optional
        Reference normative values
        Default: empty DataFrame
    
    Attributes
    ----------
    participant : Participant
        Participant information
    change_of_direction_exercises : list of ChangeOfDirectionExercise
        COD maneuver records
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline (6 Hz lowpass for markers, 10 Hz for forces)
    processed_data : ShuttleTest
        Copy with all signals processed
    
    Methods
    -------
    from_files(...)
        Create test from TDF files (class method)
    get_results()
        Process test and return results
    save(file_path)
        Save test protocol to file
    load(file_path)
        Load test protocol from file (class method)
    
    Examples
    --------
    >>> from labanalysis.protocols import ShuttleTest, Participant
    >>> 
    >>> # Create participant
    >>> participant = Participant(surname='Athlete', weight=75)
    >>> 
    >>> # Create test from multiple shuttle trials
    >>> test = ShuttleTest.from_files(
    ...     filenames=['shuttle1.tdf', 'shuttle2.tdf', 'shuttle3.tdf'],
    ...     participant=participant,
    ...     left_foot_ground_reaction_force='left_foot',
    ...     right_foot_ground_reaction_force='right_foot',
    ...     s2='s2'  # S2 marker for velocity calculation
    ... )
    >>> 
    >>> # Process
    >>> results = test.get_results()
    >>> 
    >>> # View summary
    >>> print(results.summary)
    """
```

**Signal Processing:**

```python
# Force platforms
- Lowpass filter: 10 Hz cutoff, 4th order Butterworth
- Contact threshold: MINIMUM_CONTACT_FORCE_N (20 N)
- Below threshold values set to NaN, then filled with zeros

# Markers (S2 for velocity)
- Lowpass filter: 6 Hz cutoff, 4th order Butterworth
- Phase-corrected (zero-lag)
```

**Key Metrics:**

1. **Temporal:**
   - Contact time (s): Total ground contact duration
   - Loading time (s): Deceleration phase duration
   - Propulsion time (s): Acceleration phase duration

2. **Percentages:**
   - Loading time (%): Loading as % of total contact
   - Propulsion time (%): Propulsion as % of total contact

3. **Kinematic:**
   - Max velocity (m/s): Entry velocity to COD

---

### from_files()

Create ShuttleTest from TDF files.

```python
@classmethod
def from_files(
    cls,
    filenames: list[str],
    participant: Participant,
    normative_data: pd.DataFrame = pd.DataFrame(),
    left_foot_ground_reaction_force: str | None = 'left_foot',
    right_foot_ground_reaction_force: str | None = 'right_foot',
    s2: str | None = 's2'
) -> 'ShuttleTest'
```

**Parameters:**
- `filenames`: List of TDF file paths (one per shuttle trial)
- `participant`: Participant object
- `normative_data`: Reference normative data
- `left_foot_ground_reaction_force`: Force platform key for left foot
- `right_foot_ground_reaction_force`: Force platform key for right foot
- `s2`: S2 marker key for velocity calculation

**Example:**

```python
from labanalysis.protocols import ShuttleTest, Participant

participant = Participant(surname='Sprinter', weight=80, gender='M')

# Multiple shuttle runs
test = ShuttleTest.from_files(
    filenames=[
        'shuttle_run_1.tdf',
        'shuttle_run_2.tdf',
        'shuttle_run_3.tdf',
        'shuttle_run_4.tdf'
    ],
    participant=participant,
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot',
    s2='s2'
)

results = test.get_results()
```

---

### ShuttleTestResults

Results container for shuttle tests.

```python
class ShuttleTestResults(TestResults):
    """
    Container for shuttle test results.
    
    Attributes
    ----------
    participant : Participant
        Participant information
    summary : pd.DataFrame
        Summary metrics per COD maneuver
    analytics : pd.DataFrame
        Time-series data (not implemented for shuttle tests)
    figures : dict
        Plotly figures for visualization
    
    Methods
    -------
    plot() -> go.Figure
        Generate comprehensive results visualization
    to_dataframe() -> pd.DataFrame
        Export all metrics to DataFrame
    save(file_path: str)
        Save results to pickle file
    load(file_path: str) -> ShuttleTestResults
        Load results from pickle file (class method)
    
    Examples
    --------
    >>> results = test.get_results()
    >>> 
    >>> # View summary
    >>> print(results.summary)
    >>> #   side  limb    type  n  Contact Time (s)  Loading Time (s)  ...
    >>> # 0  left  unilateral  cod  1     0.425          0.255       ...
    >>> # 1  right unilateral  cod  2     0.418          0.248       ...
    >>> 
    >>> # Plot
    >>> fig = results.plot()
    >>> fig.show()
    """
```

**Summary DataFrame Columns:**

- `side`: 'left' or 'right' (foot used for COD)
- `limb`: 'unilateral' (single-leg COD)
- `type`: 'cod' (change of direction)
- `n`: Trial number
- `Contact Time (s)`: Total contact duration
- `Loading Time (s)`: Deceleration phase duration
- `Loading Time (%)`: Loading as % of contact
- `Propulsion Time (s)`: Acceleration phase duration
- `Propulsion Time (%)`: Propulsion as % of contact
- `Max Velocity (m/s)`: Entry velocity

---

## Complete Example Workflow

```python
import labanalysis as laban
from labanalysis.protocols import ShuttleTest, Participant
from datetime import date
import pandas as pd

# 1. Create participant
participant = Participant(
    surname='AgilityStar',
    name='Pro',
    gender='F',
    height=170,
    weight=65,
    birthdate=date(1998, 3, 10)
)

# 2. Create test from multiple shuttle trials
test = ShuttleTest.from_files(
    filenames=[
        'shuttle_trial_1.tdf',
        'shuttle_trial_2.tdf',
        'shuttle_trial_3.tdf',
        'shuttle_trial_4.tdf',
        'shuttle_trial_5.tdf'
    ],
    participant=participant,
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot',
    s2='s2'
)

# 3. Process
results = test.get_results()

# 4. Analyze summary
summary = results.summary

print("=== Shuttle Test Summary ===")
print(f"Total COD maneuvers: {len(summary)}")

# Average metrics
print(f"\nAverage metrics:")
print(f"  Contact time: {summary['Contact Time (s)'].mean():.3f} s")
print(f"  Loading time: {summary['Loading Time (%)'].mean():.1f}%")
print(f"  Propulsion time: {summary['Propulsion Time (%)'].mean():.1f}%")
print(f"  Max velocity: {summary['Max Velocity (m/s)'].mean():.2f} m/s")

# 5. Analyze left-right asymmetry
left_cods = summary[summary['side'] == 'left']
right_cods = summary[summary['side'] == 'right']

contact_left = left_cods['Contact Time (s)'].mean()
contact_right = right_cods['Contact Time (s)'].mean()
contact_asymm = abs(contact_left - contact_right) / ((contact_left + contact_right) / 2) * 100

print(f"\nLeft-Right Asymmetry:")
print(f"  Contact time: {contact_asymm:.1f}%")

if contact_asymm < 5:
    print("  → Excellent symmetry")
elif contact_asymm < 10:
    print("  → Good symmetry")
else:
    print("  → Asymmetry detected (investigate weakness/injury)")

# 6. Phase analysis
loading_pct = summary['Loading Time (%)'].mean()
propulsion_pct = summary['Propulsion Time (%)'].mean()

print(f"\nPhase Distribution:")
print(f"  Loading: {loading_pct:.1f}%")
print(f"  Propulsion: {propulsion_pct:.1f}%")

# Interpretation
if loading_pct > 65:
    print("  → Slow deceleration (need eccentric strength)")
elif loading_pct < 50:
    print("  → Fast deceleration (good eccentric control)")
else:
    print("  → Balanced phase distribution")

# 7. Plot results
fig = results.plot()
fig.write_html("shuttle_test_results.html")
fig.show()

# 8. Export
results.save("shuttle_test_results.pkl")
summary.to_excel("shuttle_test_summary.xlsx", index=False)
```

---

## Advanced Features

### Fatigue Analysis Across Trials

```python
# Order trials chronologically
summary = results.summary.copy()
summary['trial_order'] = range(1, len(summary) + 1)

# Plot contact time progression
import plotly.express as px
fig = px.line(
    summary,
    x='trial_order',
    y='Contact Time (s)',
    color='side',
    markers=True,
    title='Contact Time Across Trials (Fatigue Analysis)'
)
fig.show()

# Check for fatigue (increasing contact time)
import numpy as np
slope, _ = np.polyfit(summary['trial_order'], summary['Contact Time (s)'], 1)

if slope > 0.01:
    print("Significant fatigue detected (contact time increasing)")
elif slope < -0.01:
    print("Performance improving (contact time decreasing - warm-up effect?)")
else:
    print("Consistent performance across trials")
```

### Velocity-Contact Time Relationship

```python
import plotly.express as px

# Analyze speed-contact relationship
fig = px.scatter(
    summary,
    x='Max Velocity (m/s)',
    y='Contact Time (s)',
    color='side',
    trendline='ols',
    title='Velocity vs Contact Time'
)
fig.show()

# Expected: Higher velocity → shorter contact time (better reactive strength)
```

---

## Troubleshooting

### Issue: No COD maneuvers detected

**Cause**: Missing S2 marker or force platform data

**Solution**: Verify required signals
```python
# Check TDF file contains required keys
import labanalysis as laban
data = laban.read_tdf('shuttle.tdf')
print(data.keys())  # Should include 's2', 'left_foot', 'right_foot'

test = ShuttleTest.from_files(
    filenames=['shuttle.tdf'],
    participant=participant,
    s2='s2',  # S2 marker required for inversion detection
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot'
)
```

### Issue: Unrealistic contact times

**Cause**: Incorrect force threshold or missing force data

**Solution**: Verify force platform calibration and check data
```python
# Check force platform data quality
data = laban.read_tdf('shuttle.tdf', force_keys=['left_foot', 'right_foot'])
fp = data['left_foot']
print(f"Peak force: {fp.force.module.max():.0f} N")  # Should be > 500 N for COD
print(f"Min force during contact: {fp.force.module.min():.0f} N")
```

---

## See Also

- [Protocols Base](protocols.md) - Base protocol classes
- [ChangeOfDirectionExercise](../records/agility.md#changeofdirectionexercise) - COD exercise record
- [Agility Tutorial](../../tutorials/05-agility-assessment.md) - Complete workflow guide

---

**Agility test protocol for shuttle run and change-of-direction assessment with phase analysis.**
