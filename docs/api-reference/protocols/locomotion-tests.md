# labanalysis.protocols.locomotion-tests

Gait analysis test protocols for running and walking assessment.

**Source**: `src/labanalysis/protocols/locomotiontests.py`

## Overview

Locomotion test protocols for comprehensive gait analysis:

- **RunningTest**: Running gait analysis with step detection
- **WalkingTest**: Walking gait analysis with stride detection

Both tests extend their respective exercise classes (RunningExercise, WalkingExercise) with TestProtocol capabilities, adding participant tracking, automated cycle detection, and structured results export.

**Test Configurations:**
- **Algorithm**: Kinematics (marker-based) or Kinetics (force platform-based)
- **Cycle Detection**: Automatic detection of steps (running) or strides (walking)
- **Full-Body Model**: Inherits WholeBody for joint angle calculations

## Classes

### RunningTest

Running gait analysis test protocol.

```python
class RunningTest(RunningExercise, TestProtocol):
    """
    Test protocol for running gait analysis and biomechanical assessment.
    
    Combines running gait analysis with clinical test protocol structure,
    enabling systematic assessment with participant tracking, step detection,
    and automated metrics extraction.
    
    Parameters
    ----------
    participant : Participant
        Participant information (demographics, anthropometrics)
    normative_data : pd.DataFrame, optional
        Reference data for performance comparison
        Default: empty DataFrame
    algorithm : {'kinematics', 'kinetics'}, optional
        Cycle detection algorithm
        'kinematics' = marker trajectories
        'kinetics' = force platform data
        Default: 'kinematics'
    ground_reaction_force_threshold : float or int, optional
        Minimum vertical GRF (N) for contact detection (kinetics algorithm)
        Default: MINIMUM_CONTACT_FORCE_N
    height_threshold : float or int, optional
        Maximum normalized height for contact detection (kinematics algorithm)
        Default: DEFAULT_MINIMUM_HEIGHT_PERCENTAGE
    left_foot_ground_reaction_force : ForcePlatform or None, optional
        Left foot force platform data
        Default: None
    right_foot_ground_reaction_force : ForcePlatform or None, optional
        Right foot force platform data
        Default: None
    left_heel : Point3D or None, optional
        Left heel marker trajectory
        Default: None
    right_heel : Point3D or None, optional
        Right heel marker trajectory
        Default: None
    left_toe : Point3D or None, optional
        Left toe marker trajectory
        Default: None
    right_toe : Point3D or None, optional
        Right toe marker trajectory
        Default: None
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional biomechanical signals (joint angles, EMG, etc.)
    
    Attributes
    ----------
    participant : Participant
        Participant demographics and anthropometrics
    normative_data : pd.DataFrame
        Reference data for normative comparisons
    cycles : list of RunningStep
        Detected running steps from continuous data
    get_results : dict
        Dictionary with 'summary' DataFrame and 'analytics' dict
    
    Methods
    -------
    from_tdf(...)
        Create test from TDF file (class method)
    get_results
        Property returning summary and analytics
    save(file_path)
        Save test protocol to file
    load(file_path)
        Load test protocol from file (class method)
    
    Examples
    --------
    >>> from labanalysis.protocols import RunningTest, Participant
    >>> from datetime import date
    >>> 
    >>> # Create participant
    >>> participant = Participant(
    ...     surname='Runner',
    ...     age=30,
    ...     weight=70,
    ...     height=175
    ... )
    >>> 
    >>> # Create test from TDF file (kinematics algorithm)
    >>> test = RunningTest.from_tdf(
    ...     file='running_trial.tdf',
    ...     participant=participant,
    ...     algorithm='kinematics',
    ...     left_heel='LHEE',
    ...     right_heel='RHEE',
    ...     left_toe='LTOE',
    ...     right_toe='RTOE'
    ... )
    >>> 
    >>> # Get results
    >>> results = test.get_results
    >>> 
    >>> # View summary (spatiotemporal parameters)
    >>> print(results['summary'])
    >>> 
    >>> # View time-series analytics
    >>> print(results['analytics']['ground_reaction_force'])
    """
```

**Running Gait Characteristics:**

- **Flight Phase**: Period when neither foot contacts ground (unique to running)
- **Contact Phase**: Footstrike to toe-off
- **Loading Response**: Footstrike to midstance (shock absorption)
- **Propulsion**: Midstance to toe-off (push-off)

**Extracted Metrics:**

1. **Temporal:**
   - Contact time (ms): Foot-ground contact duration
   - Flight time (ms): Aerial phase duration
   - Cycle time (ms): Total step duration

2. **Kinetic:**
   - Peak vertical force (N): Maximum GRF
   - Loading rate (N/s): Rate of force application

3. **Spatial:**
   - Lateral displacement (mm): Mediolateral COP excursion
   - Vertical displacement (mm): Vertical COP excursion

**Algorithm Requirements:**

- **Kinematics**: Requires `left_heel`, `right_heel`, `left_toe`, `right_toe` markers
- **Kinetics**: Requires `left_foot_ground_reaction_force`, `right_foot_ground_reaction_force` force platforms

---

### from_tdf() (RunningTest)

Create RunningTest from TDF file.

```python
@classmethod
def from_tdf(
    cls,
    file: str,
    participant: Participant,
    normative_data: pd.DataFrame = pd.DataFrame(),
    algorithm: Literal['kinematics', 'kinetics'] = 'kinematics',
    ground_reaction_force_threshold: float | int = MINIMUM_CONTACT_FORCE_N,
    height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
    left_foot_ground_reaction_force: str | None = None,
    right_foot_ground_reaction_force: str | None = None,
    left_heel: str | None = None,
    right_heel: str | None = None,
    left_toe: str | None = None,
    right_toe: str | None = None,
    # ... additional marker parameters for full-body model
    **kwargs
) -> 'RunningTest'
```

**Parameters:**
- `file`: Path to TDF file
- `participant`: Participant object
- `normative_data`: Reference data
- `algorithm`: 'kinematics' or 'kinetics'
- `ground_reaction_force_threshold`: Minimum GRF for contact (kinetics)
- `height_threshold`: Maximum height for contact (kinematics)
- `left_foot_ground_reaction_force`: Force platform key for left foot
- `right_foot_ground_reaction_force`: Force platform key for right foot
- `left_heel`, `right_heel`, `left_toe`, `right_toe`: Marker keys

**Example:**

```python
from labanalysis.protocols import RunningTest, Participant

participant = Participant(surname='Athlete', weight=75, height=180)

# Kinematics-based test
test = RunningTest.from_tdf(
    file='running_10mps.tdf',
    participant=participant,
    algorithm='kinematics',
    left_heel='LHEE',
    right_heel='RHEE',
    left_toe='LTOE',
    right_toe='RTOE'
)

# Get results
results = test.get_results

# Summary per cycle
summary = results['summary']
print(f"Number of steps detected: {len(summary)}")
print(f"Average contact time: {summary['contact_time_ms'].mean():.1f} ms")
print(f"Average flight time: {summary['flight_time_ms'].mean():.1f} ms")

# Time-series analytics
cop = results['analytics']['centre_of_pressure']
grf = results['analytics']['ground_reaction_force']
```

---

### get_results (RunningTest)

Get test results with summary and analytics.

```python
@property
def get_results(self) -> dict
```

**Returns:**

Dictionary with two keys:

1. **'summary'** (pd.DataFrame): Per-step metrics
   - Columns: `cycle`, `side`, `contact_time_ms`, `flight_time_ms`, `cycle_time_ms`, `peak_vertical_force_N`, etc.
   - One row per detected step

2. **'analytics'** (dict): Time-series data
   - `'centre_of_pressure'`: COP trajectory DataFrame (columns: Side, Cycle, Time, lateral_axis, anteroposterior_axis)
   - `'ground_reaction_force'`: GRF time-series DataFrame (columns: Side, Cycle, Time, vertical_axis)

**Example:**

```python
results = test.get_results

# Analyze left vs right asymmetry
summary = results['summary']
left_steps = summary[summary['side'] == 'left']
right_steps = summary[summary['side'] == 'right']

contact_left = left_steps['contact_time_ms'].mean()
contact_right = right_steps['contact_time_ms'].mean()
asymmetry = abs(contact_left - contact_right) / ((contact_left + contact_right) / 2) * 100

print(f"Contact time asymmetry: {asymmetry:.1f}%")

# Plot GRF for all cycles
import plotly.express as px
grf = results['analytics']['ground_reaction_force']
fig = px.line(grf, x='Time', y=grf.columns[-1], color='Cycle', facet_col='Side')
fig.show()
```

---

### WalkingTest

Walking gait analysis test protocol.

```python
class WalkingTest(WalkingExercise, TestProtocol):
    """
    Test protocol for walking gait analysis and biomechanical assessment.
    
    Combines walking gait analysis with clinical test protocol structure,
    enabling systematic assessment with participant tracking, stride detection,
    gait phase identification, and automated metrics extraction.
    
    Parameters
    ----------
    participant : Participant
        Participant information (demographics, anthropometrics)
    normative_data : pd.DataFrame, optional
        Reference data for performance comparison
        Default: empty DataFrame
    algorithm : {'kinematics', 'kinetics'}, optional
        Cycle detection algorithm
        'kinematics' = marker trajectories
        'kinetics' = force platform data
        Default: 'kinematics'
    ground_reaction_force_threshold : float or int, optional
        Minimum vertical GRF (N) for contact detection (kinetics algorithm)
        Default: MINIMUM_CONTACT_FORCE_N
    height_threshold : float or int, optional
        Maximum normalized height for contact detection (kinematics algorithm)
        Default: DEFAULT_MINIMUM_HEIGHT_PERCENTAGE
    left_foot_ground_reaction_force : ForcePlatform or None, optional
        Left foot force platform data
        Default: None
    right_foot_ground_reaction_force : ForcePlatform or None, optional
        Right foot force platform data
        Default: None
    left_heel : Point3D or None, optional
        Left heel marker trajectory
        Default: None
    right_heel : Point3D or None, optional
        Right heel marker trajectory
        Default: None
    left_toe : Point3D or None, optional
        Left toe marker trajectory
        Default: None
    right_toe : Point3D or None, optional
        Right toe marker trajectory
        Default: None
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional biomechanical signals (joint angles, EMG, etc.)
    
    Attributes
    ----------
    participant : Participant
        Participant demographics and anthropometrics
    normative_data : pd.DataFrame
        Reference data for normative comparisons
    cycles : list of WalkingStride
        Detected walking strides from continuous data
    get_results : dict
        Dictionary with 'summary' DataFrame and 'analytics' dict
    
    Methods
    -------
    from_tdf(...)
        Create test from TDF file (class method)
    get_results
        Property returning summary and analytics
    save(file_path)
        Save test protocol to file
    load(file_path)
        Load test protocol from file (class method)
    
    Examples
    --------
    >>> from labanalysis.protocols import WalkingTest, Participant
    >>> 
    >>> # Create participant
    >>> participant = Participant(
    ...     surname='Patient',
    ...     age=65,
    ...     weight=70,
    ...     height=170
    ... )
    >>> 
    >>> # Create test from TDF file
    >>> test = WalkingTest.from_tdf(
    ...     file='walking_trial.tdf',
    ...     participant=participant,
    ...     algorithm='kinematics',
    ...     left_heel='LHEE',
    ...     right_heel='RHEE',
    ...     left_toe='LTOE',
    ...     right_toe='RTOE'
    ... )
    >>> 
    >>> # Get results
    >>> results = test.get_results
    >>> 
    >>> # Analyze gait symmetry
    >>> summary = results['summary']
    >>> print(summary[['cycle', 'side', 'stride_time_ms', 'stance_time_ms', 'swing_time_ms']])
    """
```

**Walking Gait Phases:**

- **Stance Phase**: Foot in contact with ground (~60% of stride)
  - Initial contact (heel strike)
  - Loading response
  - Midstance
  - Terminal stance
  - Pre-swing
- **Swing Phase**: Foot in the air (~40% of stride)
  - Initial swing
  - Midswing
  - Terminal swing
- **Double Support**: Both feet on ground (two periods per stride, ~20% total)
- **Single Support**: Only one foot on ground (~40% per leg)

**Key Difference from Running:**

Walking lacks a flight phase - there is always at least one foot in contact with the ground. This fundamental difference affects cycle detection and phase segmentation.

**Extracted Metrics:**

1. **Temporal:**
   - Stride time (ms): Complete stride cycle duration
   - Stance time (ms): Stance phase duration
   - Swing time (ms): Swing phase duration
   - Double support time (ms): Both feet on ground
   - Single support time (ms): Single-leg stance

2. **Kinetic:**
   - Peak vertical force (N): Maximum GRF
   - Loading rate (N/s): Rate of force application
   - Push-off force (N): Propulsive force

3. **Spatial:**
   - Lateral displacement (mm): Mediolateral COP excursion
   - Vertical displacement (mm): Vertical COP excursion
   - Step length (m): Distance between successive foot contacts
   - Stride length (m): Distance for complete stride

---

### from_tdf() (WalkingTest)

Create WalkingTest from TDF file.

```python
@classmethod
def from_tdf(
    cls,
    file: str,
    participant: Participant,
    normative_data: pd.DataFrame = pd.DataFrame(),
    algorithm: Literal['kinematics', 'kinetics'] = 'kinematics',
    ground_reaction_force_threshold: float | int = MINIMUM_CONTACT_FORCE_N,
    height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
    left_foot_ground_reaction_force: str | None = None,
    right_foot_ground_reaction_force: str | None = None,
    left_heel: str | None = None,
    right_heel: str | None = None,
    left_toe: str | None = None,
    right_toe: str | None = None,
    # ... additional marker parameters
    **kwargs
) -> 'WalkingTest'
```

**Example:**

```python
from labanalysis.protocols import WalkingTest, Participant

participant = Participant(surname='Patient', weight=65, height=165, age=70)

# Walking test with full-body markers
test = WalkingTest.from_tdf(
    file='walking_comfortable_speed.tdf',
    participant=participant,
    algorithm='kinematics',
    left_heel='LHEE',
    right_heel='RHEE',
    left_toe='LTOE',
    right_toe='RTOE',
    left_ankle_lateral='LLAT',
    right_ankle_lateral='RLAT',
    left_knee_lateral='LKNEE',
    right_knee_lateral='RKNEE',
    left_asis='LASI',
    right_asis='RASI'
)

# Analyze results
results = test.get_results
summary = results['summary']

# Gait parameters
print(f"Average stride time: {summary['stride_time_ms'].mean():.0f} ms")
print(f"Average cadence: {60000 / summary['stride_time_ms'].mean():.0f} steps/min")
print(f"Stance/Swing ratio: {summary['stance_time_ms'].mean() / summary['swing_time_ms'].mean():.2f}")
```

---

## Complete Example Workflows

### Running Gait Analysis

```python
import labanalysis as laban
from labanalysis.protocols import RunningTest, Participant
from datetime import date
import pandas as pd

# 1. Create participant
participant = Participant(
    surname='Sprinter',
    name='Fast',
    gender='M',
    height=185,
    weight=80,
    birthdate=date(1995, 5, 15)
)

# 2. Create test from TDF file
test = RunningTest.from_tdf(
    file='running_maxspeed.tdf',
    participant=participant,
    algorithm='kinematics',
    left_heel='LHEE',
    right_heel='RHEE',
    left_toe='LTOE',
    right_toe='RTOE'
)

# 3. Get results
results = test.get_results
summary = results['summary']

# 4. Calculate key metrics
print("=== Running Gait Summary ===")
print(f"Steps detected: {len(summary)}")
print(f"\nAverage metrics:")
print(f"  Contact time: {summary['contact_time_ms'].mean():.1f} ± {summary['contact_time_ms'].std():.1f} ms")
print(f"  Flight time: {summary['flight_time_ms'].mean():.1f} ± {summary['flight_time_ms'].std():.1f} ms")
print(f"  Cycle time: {summary['cycle_time_ms'].mean():.1f} ± {summary['cycle_time_ms'].std():.1f} ms")

# 5. Analyze left-right symmetry
left = summary[summary['side'] == 'left']
right = summary[summary['side'] == 'right']

contact_left = left['contact_time_ms'].mean()
contact_right = right['contact_time_ms'].mean()
contact_asymm = abs(contact_left - contact_right) / ((contact_left + contact_right) / 2) * 100

print(f"\nLeft-Right Asymmetry:")
print(f"  Contact time: {contact_asymm:.1f}%")

# Interpretation
if contact_asymm < 5:
    print("  → Excellent symmetry")
elif contact_asymm < 10:
    print("  → Good symmetry")
else:
    print("  → Asymmetry detected (investigate injury/weakness)")

# 6. Plot GRF time-series
import plotly.express as px
grf = results['analytics']['ground_reaction_force']
fig = px.line(
    grf,
    x='Time',
    y=grf.columns[-1],
    color='Cycle',
    facet_col='Side',
    title='Ground Reaction Force - All Steps'
)
fig.show()

# 7. Export results
summary.to_excel("running_gait_summary.xlsx", index=False)
test.save("running_test.pkl")
```

### Walking Gait Analysis with Clinical Interpretation

```python
from labanalysis.protocols import WalkingTest, Participant

# Create elderly participant
participant = Participant(
    surname='Elder',
    age=75,
    weight=65,
    height=165,
    gender='F'
)

# Walking test
test = WalkingTest.from_tdf(
    file='walking_self_selected.tdf',
    participant=participant,
    algorithm='kinematics',
    left_heel='LHEE',
    right_heel='RHEE',
    left_toe='LTOE',
    right_toe='RTOE'
)

results = test.get_results
summary = results['summary']

# Calculate clinical gait parameters
stride_time_mean = summary['stride_time_ms'].mean()
stance_time_mean = summary['stance_time_ms'].mean()
swing_time_mean = summary['swing_time_ms'].mean()

cadence = 60000 / stride_time_mean  # steps/min
walking_speed = 1.0 / (stride_time_mean / 1000)  # Estimate from stride time

print("=== Clinical Gait Assessment ===")
print(f"Cadence: {cadence:.0f} steps/min")
print(f"Stride time: {stride_time_mean:.0f} ms")
print(f"Stance phase: {stance_time_mean:.0f} ms ({stance_time_mean/stride_time_mean*100:.1f}%)")
print(f"Swing phase: {swing_time_mean:.0f} ms ({swing_time_mean/stride_time_mean*100:.1f}%)")

# Clinical interpretation for elderly
print("\n=== Clinical Interpretation ===")

# Cadence norms for elderly: 100-120 steps/min
if cadence < 90:
    print("  Cadence: Very low (fall risk, mobility impairment)")
elif cadence < 100:
    print("  Cadence: Low (reduced mobility)")
elif cadence < 120:
    print("  Cadence: Normal for age")
else:
    print("  Cadence: High (good mobility)")

# Stance phase should be ~60% of stride
stance_pct = stance_time_mean / stride_time_mean * 100
if stance_pct > 65:
    print("  Stance phase: Prolonged (cautious gait, fear of falling)")
elif stance_pct > 55:
    print("  Stance phase: Normal")
else:
    print("  Stance phase: Reduced (unusual, check data)")

# Analyze variability (gait stability indicator)
stride_cv = (summary['stride_time_ms'].std() / stride_time_mean) * 100
print(f"\nStride time variability: {stride_cv:.1f}%")
if stride_cv > 5:
    print("  → High variability (reduced gait stability, fall risk)")
else:
    print("  → Normal variability (stable gait)")
```

### Comparing Running Speeds

```python
from labanalysis.protocols import RunningTest, Participant
import pandas as pd

participant = Participant(surname='Runner', weight=75)

# Test at different speeds
speeds = ['slow', 'medium', 'fast']
tests = {}

for speed in speeds:
    test = RunningTest.from_tdf(
        file=f'running_{speed}.tdf',
        participant=participant,
        algorithm='kinematics',
        left_heel='LHEE',
        right_heel='RHEE',
        left_toe='LTOE',
        right_toe='RTOE'
    )
    tests[speed] = test.get_results

# Compare metrics
comparison = []
for speed, results in tests.items():
    summary = results['summary']
    comparison.append({
        'speed': speed,
        'contact_time_ms': summary['contact_time_ms'].mean(),
        'flight_time_ms': summary['flight_time_ms'].mean(),
        'cycle_time_ms': summary['cycle_time_ms'].mean(),
        'peak_force_N': summary['peak_vertical_force_N'].mean()
    })

df = pd.DataFrame(comparison)
print(df)

# Expected trend: as speed increases
# - Contact time decreases
# - Flight time increases
# - Peak force increases
```

---

## Advanced Features

### Full-Body Joint Angle Analysis

```python
# Include full marker set for joint angles
test = RunningTest.from_tdf(
    file='running_full_body.tdf',
    participant=participant,
    algorithm='kinematics',
    left_heel='LHEE',
    right_heel='RHEE',
    left_toe='LTOE',
    right_toe='RTOE',
    left_ankle_lateral='LLAT',
    right_ankle_lateral='RLAT',
    left_knee_lateral='LKNEE',
    right_knee_lateral='RKNEE',
    left_asis='LASI',
    right_asis='RASI',
    left_psis='LPSI',
    right_psis='RPSI'
)

# Access joint angles (inherited from WholeBody)
for cycle in test.cycles:
    ankle_angle = cycle.left_ankle_flexionextension
    knee_angle = cycle.left_knee_flexionextension
    hip_angle = cycle.left_hip_flexionextension
    # ... analyze angles
```

### Kinetics-Based Detection (Force Platforms)

```python
# Use force platforms for cycle detection
test = RunningTest.from_tdf(
    file='running_forceplatform.tdf',
    participant=participant,
    algorithm='kinetics',  # Use force data
    left_foot_ground_reaction_force='FP1',
    right_foot_ground_reaction_force='FP2',
    ground_reaction_force_threshold=20  # 20 N threshold
)

results = test.get_results

# Force platform data provides accurate contact times
summary = results['summary']
print(f"Contact time (kinetics): {summary['contact_time_ms'].mean():.1f} ms")
```

---

## Troubleshooting

### Issue: No cycles detected

**Cause**: Insufficient markers or incorrect algorithm selection

**Solution**: Verify marker availability and algorithm requirements
```python
# Check required markers for kinematics
test = RunningTest.from_tdf(
    file='test.tdf',
    participant=participant,
    algorithm='kinematics',
    left_heel='LHEE',    # Required
    right_heel='RHEE',   # Required
    left_toe='LTOE',     # Required
    right_toe='RTOE'     # Required
)

# If markers missing, try kinetics
test = RunningTest.from_tdf(
    file='test.tdf',
    participant=participant,
    algorithm='kinetics',
    left_foot_ground_reaction_force='FP1',
    right_foot_ground_reaction_force='FP2'
)
```

### Issue: Incorrect cycle count

**Cause**: Threshold too sensitive/insensitive

**Solution**: Adjust detection thresholds
```python
# For kinematics: adjust height threshold
test = RunningTest.from_tdf(
    file='test.tdf',
    participant=participant,
    algorithm='kinematics',
    height_threshold=0.05,  # Default: 0.05 (5% of max height)
    left_heel='LHEE',
    right_heel='RHEE',
    left_toe='LTOE',
    right_toe='RTOE'
)

# For kinetics: adjust force threshold
test = RunningTest.from_tdf(
    file='test.tdf',
    participant=participant,
    algorithm='kinetics',
    ground_reaction_force_threshold=30,  # Default: 20 N
    left_foot_ground_reaction_force='FP1',
    right_foot_ground_reaction_force='FP2'
)
```

---

## See Also

- [Protocols Base](protocols.md) - Base protocol classes
- [RunningExercise](../records/locomotion.md#runningexercise) - Running gait record class
- [WalkingExercise](../records/locomotion.md#walkingexercise) - Walking gait record class
- [RunningStep](../records/locomotion.md#runningstep) - Individual running step
- [WalkingStride](../records/locomotion.md#walkingstride) - Individual walking stride
- [Gait Tutorial](../../tutorials/02-gait-analysis.md) - Complete workflow guide

---

**Gait analysis protocols for running and walking with automated cycle detection and spatiotemporal analysis.**
