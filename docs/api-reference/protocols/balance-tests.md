# labanalysis.protocols.balance-tests

Balance assessment protocols for upright and plank stability tests.

**Source**: `src/labanalysis/protocols/balancetests.py`

## Overview

Balance test protocols for center of pressure (COP) analysis and postural stability assessment:

- **UprightBalanceTest**: Bipedal stance balance protocol (eyes open/closed)
- **PlankBalanceTest**: Core stability protocol in prone position
- **UprightBalanceTestResults**: Results with COP sway analysis
- **PlankBalanceTestResults**: Results with force distribution analysis

**Test Configurations:**
- **Eyes condition**: Open vs Closed (affects visual feedback)
- **Stance**: Bilateral (both feet), Left (single-leg), Right (single-leg)
- **Plank**: 4 force platforms required (hands + feet)

## Classes

### UprightBalanceTest

Upright stance balance assessment protocol.

```python
class UprightBalanceTest(TestProtocol):
    """
    Protocol for upright postural stability assessment.
    
    Parameters
    ----------
    participant : Participant
        Participant information
    exercise : UprightPosture
        Upright balance exercise data
    eyes : {'open', 'closed'}
        Eyes condition during test
    normative_data : pd.DataFrame, optional
        Reference normative values
        Default: uprightbalance_normative_values
    emg_normalization_references : TimeseriesRecord or 'self', optional
        EMG normalization references
        Default: empty TimeseriesRecord
    emg_normalization_function : callable, optional
        Function to compute normalization value
        Default: np.mean
    emg_activation_references : TimeseriesRecord or 'self', optional
        References for muscle activation thresholds
        Default: empty TimeseriesRecord
    emg_activation_threshold : float, optional
        Threshold multiplier for activation detection
        Default: 3
    relevant_muscle_map : list of str or None, optional
        Muscles to include in analysis
        Default: None (all muscles)
    
    Attributes
    ----------
    participant : Participant
        Participant information
    exercise : UprightPosture
        Balance exercise data
    eyes : str
        Eyes condition ('open' or 'closed')
    side : str
        Stance type ('bilateral', 'left', 'right')
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline
    processed_data : UprightBalanceTest
        Copy with all signals processed
    
    Methods
    -------
    from_files(...)
        Create test from TDF file (class method)
    get_results(include_emg=True)
        Process test and return results
    save(file_path)
        Save test protocol to file
    load(file_path)
        Load test protocol from file (class method)
    
    Examples
    --------
    >>> from labanalysis.protocols import UprightBalanceTest, Participant
    >>> 
    >>> # Create participant
    >>> participant = Participant(surname='Smith', weight=75)
    >>> 
    >>> # Create test from file
    >>> test = UprightBalanceTest.from_files(
    ...     filename='bilateral_eyes_open.tdf',
    ...     participant=participant,
    ...     eyes='open',
    ...     left_foot_ground_reaction_force='left_foot',
    ...     right_foot_ground_reaction_force='right_foot'
    ... )
    >>> 
    >>> # Process and get results
    >>> results = test.get_results(include_emg=True)
    >>> 
    >>> # View summary
    >>> print(results.summary)
    >>> 
    >>> # Plot
    >>> fig = results.plot()
    >>> fig.show()
    """
```

**Signal Processing:**

```python
# Force platforms
- Lowpass filter: 30 Hz cutoff
- COP calculation: Automatic from force + moment
- Reference frame: Auto-aligned to bilateral force center

# EMG signals (if available)
- Bandpass filter: 20-450 Hz
- RMS envelope: 200ms window
- Normalization: User-defined reference
```

**Key Metrics:**

1. **COP Sway Area (mm²)**: Ellipse area encompassing 95% of COP trajectory
2. **Force Symmetry (%)**: Left-right force distribution balance
3. **Muscle Imbalance (%)**: EMG amplitude asymmetry between sides
4. **Performance Ranking**: Comparison to normative data (Excellent/Good/Fair/Poor)

---

### from_files()

Create UprightBalanceTest from TDF file.

```python
@classmethod
def from_files(
    cls,
    filename: str,
    participant: Participant,
    eyes: Literal['open', 'closed'],
    left_foot_ground_reaction_force: str | None = None,
    right_foot_ground_reaction_force: str | None = None,
    normative_data: pd.DataFrame = uprightbalance_normative_values,
    emg_normalization_references: TimeseriesRecord | str | Literal['self'] = TimeseriesRecord(),
    emg_normalization_function: Callable = np.mean,
    emg_activation_references: TimeseriesRecord | str | Literal['self'] = TimeseriesRecord(),
    emg_activation_threshold: float | int = 3,
    relevant_muscle_map: list[str] | None = None
) -> 'UprightBalanceTest'
```

**Parameters:**
- `filename`: Path to TDF file
- `participant`: Participant object
- `eyes`: 'open' or 'closed'
- `left_foot_ground_reaction_force`: Force platform key for left foot (None = single-leg right)
- `right_foot_ground_reaction_force`: Force platform key for right foot (None = single-leg left)
- `normative_data`: Normative reference data
- `emg_normalization_references`: EMG normalization reference
- `emg_normalization_function`: Normalization function (e.g., np.mean, np.max)
- `emg_activation_references`: Activation threshold references
- `emg_activation_threshold`: Activation threshold multiplier
- `relevant_muscle_map`: List of muscles to analyze

**Example:**

```python
from labanalysis.protocols import UprightBalanceTest, Participant

participant = Participant(surname='Athlete', weight=75, gender='M')

# Bilateral stance
test_bilateral = UprightBalanceTest.from_files(
    filename='balance_bilateral_open.tdf',
    participant=participant,
    eyes='open',
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot'
)

# Single-leg stance (right)
test_single = UprightBalanceTest.from_files(
    filename='balance_single_right.tdf',
    participant=participant,
    eyes='closed',
    left_foot_ground_reaction_force=None,  # No left foot
    right_foot_ground_reaction_force='right_foot'
)

results = test_bilateral.get_results(include_emg=True)
```

---

### UprightBalanceTestResults

Results container for upright balance tests.

```python
class UprightBalanceTestResults(TestResults):
    """
    Container for upright balance test results.
    
    Attributes
    ----------
    participant : Participant
        Participant information
    summary : pd.DataFrame
        Summary metrics (area, symmetry, ranking)
    analytics : pd.DataFrame
        Time-series data (COP coordinates, EMG)
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
    load(file_path: str) -> UprightBalanceTestResults
        Load results from pickle file (class method)
    
    Examples
    --------
    >>> results = test.get_results(include_emg=True)
    >>> 
    >>> # View summary metrics
    >>> print(results.summary)
    >>> #   type  eyes     side  bodymass_kg  area_of_stability_mm2  ...
    >>> # 0  UprightBalanceTest  open  bilateral  75.2  245.3  ...
    >>> 
    >>> # Plot sway + performance + muscle imbalance
    >>> fig = results.plot()
    >>> fig.show()
    >>> 
    >>> # Export analytics (time-series)
    >>> analytics = results.analytics
    >>> # Contains: cop_x_mm, cop_y_mm, EMG signals (if available)
    """
```

**Figures Generated:**

1. **Sway Plot**: COP trajectory with normative ellipses overlay
   - Color-coded zones: Excellent (green) → Good (yellow) → Fair (orange) → Poor (red)
   - Shows actual COP path (black line)
   
2. **Performance Overview** (if normative data available):
   - Bar chart showing % time in each performance zone
   
3. **Muscle Imbalance** (if EMG + bilateral stance):
   - Horizontal bar chart showing left-right asymmetry per muscle
   - Color gradient from green (balanced) to red (imbalanced)

**Summary DataFrame Columns:**

- `type`: Test type (UprightBalanceTest)
- `eyes`: 'open' or 'closed'
- `side`: 'bilateral', 'left', or 'right'
- `bodymass_kg`: Body mass (from vertical GRF)
- `area_of_stability_mm2`: COP sway area
- `force_slope`, `force_intercept`, `force_r2`, `force_symmetry_pct`: Force balance metrics
- `emg_slope`, `emg_intercept`, `emg_r2`, `emg_symmetry_pct`: EMG balance metrics (if available)

---

### PlankBalanceTest

Plank position core stability assessment protocol.

```python
class PlankBalanceTest(TestProtocol):
    """
    Protocol for plank core stability assessment.
    
    Requires 4 force platforms: left/right hands + left/right feet.
    
    Parameters
    ----------
    participant : Participant
        Participant information
    exercise : PronePosture
        Plank exercise data
    eyes : {'open', 'closed'}
        Eyes condition during test
    normative_data : pd.DataFrame, optional
        Reference normative values
        Default: plankbalance_normative_values
    emg_normalization_references : TimeseriesRecord or 'self', optional
        EMG normalization references
        Default: empty TimeseriesRecord
    emg_normalization_function : callable, optional
        Function to compute normalization value
        Default: np.mean
    emg_activation_references : TimeseriesRecord or 'self', optional
        References for muscle activation thresholds
        Default: empty TimeseriesRecord
    emg_activation_threshold : float, optional
        Threshold multiplier for activation detection
        Default: 3
    relevant_muscle_map : list of str or None, optional
        Muscles to include in analysis
        Default: None (all muscles)
    
    Attributes
    ----------
    participant : Participant
        Participant information
    exercise : PronePosture
        Plank exercise data
    eyes : str
        Eyes condition ('open' or 'closed')
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline
    processed_data : PlankBalanceTest
        Copy with all signals processed
    
    Methods
    -------
    from_files(...)
        Create test from TDF file (class method)
    get_results(include_emg=True)
        Process test and return results
    save(file_path)
        Save test protocol to file
    load(file_path)
        Load test protocol from file (class method)
    
    Examples
    --------
    >>> from labanalysis.protocols import PlankBalanceTest, Participant
    >>> 
    >>> # Create participant
    >>> participant = Participant(surname='Smith', weight=75)
    >>> 
    >>> # Create test from file
    >>> test = PlankBalanceTest.from_files(
    ...     filename='plank_eyes_open.tdf',
    ...     participant=participant,
    ...     eyes='open',
    ...     left_hand_ground_reaction_force='left_hand',
    ...     right_hand_ground_reaction_force='right_hand',
    ...     left_foot_ground_reaction_force='left_foot',
    ...     right_foot_ground_reaction_force='right_foot'
    ... )
    >>> 
    >>> # Process and get results
    >>> results = test.get_results(include_emg=True)
    >>> 
    >>> # View force distribution
    >>> summary = results.summary
    >>> print(summary[['type', 'eyes', 'area_of_stability_mm2']])
    """
```

**Force Distribution Targets (Optimal Plank):**

- **Hands**: 60-65% of body weight (combined)
- **Feet**: 35-40% of body weight (combined)
- **Left-Right Balance**: <10% asymmetry

---

### from_files() (PlankBalanceTest)

Create PlankBalanceTest from TDF file.

```python
@classmethod
def from_files(
    cls,
    filename: str,
    participant: Participant,
    eyes: Literal['open', 'closed'],
    left_foot_ground_reaction_force: str = 'left_foot',
    right_foot_ground_reaction_force: str = 'right_foot',
    left_hand_ground_reaction_force: str = 'left_hand',
    right_hand_ground_reaction_force: str = 'right_hand',
    normative_data: pd.DataFrame = plankbalance_normative_values,
    emg_normalization_references: TimeseriesRecord | str | Literal['self'] = TimeseriesRecord(),
    emg_normalization_function: Callable = np.mean,
    emg_activation_references: TimeseriesRecord | str | Literal['self'] = TimeseriesRecord(),
    emg_activation_threshold: float | int = 3,
    relevant_muscle_map: list[str] | None = None
) -> 'PlankBalanceTest'
```

**Example:**

```python
from labanalysis.protocols import PlankBalanceTest, Participant

participant = Participant(surname='Athlete', weight=75, gender='M')

test = PlankBalanceTest.from_files(
    filename='plank_eyes_closed.tdf',
    participant=participant,
    eyes='closed',
    left_hand_ground_reaction_force='left_hand',
    right_hand_ground_reaction_force='right_hand',
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot',
    relevant_muscle_map=[
        'left_rectus_abdominis',
        'right_rectus_abdominis',
        'left_erector_spinae',
        'right_erector_spinae'
    ]
)

results = test.get_results(include_emg=True)
```

---

### PlankBalanceTestResults

Results container for plank balance tests.

```python
class PlankBalanceTestResults(TestResults):
    """
    Container for plank balance test results.
    
    Attributes
    ----------
    participant : Participant
        Participant information
    summary : pd.DataFrame
        Summary metrics (area, symmetry, force distribution)
    analytics : pd.DataFrame
        Time-series data (COP coordinates, EMG)
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
    load(file_path: str) -> PlankBalanceTestResults
        Load results from pickle file (class method)
    
    Examples
    --------
    >>> results = test.get_results(include_emg=True)
    >>> 
    >>> # View force distribution
    >>> summary = results.summary
    >>> print(summary[['type', 'eyes', 'region']])
    >>> #   type              eyes     region
    >>> # 0  PlankBalanceTest  closed  upper/lower
    >>> # 1  PlankBalanceTest  closed  left/right
    >>> 
    >>> # Check force symmetry
    >>> upper_lower = summary[summary['region'] == 'upper/lower']
    >>> print(f"Upper-Lower balance: {upper_lower['force_symmetry_pct'].values[0]:.1f}%")
    """
```

**Summary DataFrame Columns:**

- `type`: Test type (PlankBalanceTest)
- `eyes`: 'open' or 'closed'
- `bodymass_kg`: Body mass (from vertical GRF)
- `area_of_stability_mm2`: COP sway area
- `region`: Force distribution region ('upper/lower' or 'left/right')
- `force_slope`, `force_intercept`, `force_r2`, `force_symmetry_pct`: Force balance metrics
- `emg_slope`, `emg_intercept`, `emg_r2`, `emg_symmetry_pct`: EMG balance metrics (if available)

---

## Complete Example Workflow

### Upright Balance Protocol

```python
import labanalysis as laban
from labanalysis.protocols import UprightBalanceTest, Participant
from datetime import date

# 1. Create participant
participant = Participant(
    surname='Runner',
    name='Pro',
    gender='M',
    height=180,
    weight=75,
    birthdate=date(1995, 1, 1)
)

# 2. Create tests for Romberg protocol
test_eyes_open = UprightBalanceTest.from_files(
    filename='balance_bilateral_open.tdf',
    participant=participant,
    eyes='open',
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot'
)

test_eyes_closed = UprightBalanceTest.from_files(
    filename='balance_bilateral_closed.tdf',
    participant=participant,
    eyes='closed',
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot'
)

# 3. Process tests
results_open = test_eyes_open.get_results(include_emg=True)
results_closed = test_eyes_closed.get_results(include_emg=True)

# 4. Calculate Romberg quotient
area_open = results_open.summary['area_of_stability_mm2'].values[0]
area_closed = results_closed.summary['area_of_stability_mm2'].values[0]
romberg_quotient = area_closed / area_open

print(f"Romberg Quotient: {romberg_quotient:.2f}")
# Interpretation:
# < 1.5: Good balance (visual input not dominant)
# 1.5-2.0: Normal (moderate visual dependence)
# > 2.0: High visual dependence (proprioception deficit)

# 5. Plot results
fig_open = results_open.plot()
fig_open.write_html("balance_eyes_open.html")

fig_closed = results_closed.plot()
fig_closed.write_html("balance_eyes_closed.html")

# 6. Export data
results_open.save("balance_open_results.pkl")
results_closed.save("balance_closed_results.pkl")
```

### Plank Balance Protocol

```python
from labanalysis.protocols import PlankBalanceTest, Participant

participant = Participant(surname='Athlete', weight=75, gender='M')

# Create plank test
test = PlankBalanceTest.from_files(
    filename='plank_eyes_open.tdf',
    participant=participant,
    eyes='open',
    left_hand_ground_reaction_force='left_hand',
    right_hand_ground_reaction_force='right_hand',
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot',
    relevant_muscle_map=[
        'left_rectus_abdominis',
        'right_rectus_abdominis',
        'left_external_oblique',
        'right_external_oblique',
        'left_erector_spinae',
        'right_erector_spinae'
    ]
)

# Process
results = test.get_results(include_emg=True)

# Analyze force distribution
summary = results.summary
upper_lower = summary[summary['region'] == 'upper/lower']
left_right = summary[summary['region'] == 'left/right']

print("Force Distribution:")
print(f"  Upper-Lower Symmetry: {upper_lower['force_symmetry_pct'].values[0]:.1f}%")
print(f"  Left-Right Symmetry: {left_right['force_symmetry_pct'].values[0]:.1f}%")

# Plot
fig = results.plot()
fig.show()
```

---

## Advanced Features

### EMG-Enhanced Balance Analysis

```python
# Use MVC normalization for EMG
test = UprightBalanceTest.from_files(
    filename='balance_bilateral.tdf',
    participant=participant,
    eyes='open',
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot',
    emg_normalization_references='mvc_trial.tdf',
    emg_normalization_function=np.max,
    relevant_muscle_map=[
        'left_gastrocnemius',
        'right_gastrocnemius',
        'left_tibialis_anterior',
        'right_tibialis_anterior'
    ]
)

results = test.get_results(include_emg=True)

# Muscle imbalance analysis
muscle_data = results.summary[[col for col in results.summary.columns if 'emg' in col]]
print(muscle_data)
```

### Single-Leg Balance Assessment

```python
# Right leg balance
test_right = UprightBalanceTest.from_files(
    filename='balance_single_right.tdf',
    participant=participant,
    eyes='open',
    left_foot_ground_reaction_force=None,  # No left foot
    right_foot_ground_reaction_force='right_foot'
)

# Left leg balance
test_left = UprightBalanceTest.from_files(
    filename='balance_single_left.tdf',
    participant=participant,
    eyes='open',
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force=None  # No right foot
)

results_right = test_right.get_results()
results_left = test_left.get_results()

# Compare sides
area_right = results_right.summary['area_of_stability_mm2'].values[0]
area_left = results_left.summary['area_of_stability_mm2'].values[0]
asymmetry = abs(area_right - area_left) / ((area_right + area_left) / 2) * 100

print(f"Single-leg balance asymmetry: {asymmetry:.1f}%")
# Target: < 10% asymmetry
```

### Custom Normative Data

```python
import pandas as pd

# Create custom normative data
custom_norms = pd.DataFrame({
    'side': ['bilateral', 'bilateral', 'left', 'right'],
    'eyes': ['open', 'closed', 'open', 'open'],
    'parameter': ['area_of_stability_mm2'] * 4,
    'mean': [200, 300, 350, 350],
    'std': [50, 75, 80, 80],
})

test = UprightBalanceTest.from_files(
    filename='balance_bilateral_open.tdf',
    participant=participant,
    eyes='open',
    left_foot_ground_reaction_force='left_foot',
    right_foot_ground_reaction_force='right_foot',
    normative_data=custom_norms
)

results = test.get_results()
```

---

## Troubleshooting

### Issue: "exercise must be an UprightPosture instance"

**Cause**: Wrong exercise type passed to constructor

**Solution**: Use `from_files()` class method or ensure exercise is UprightPosture
```python
# Correct
test = UprightBalanceTest.from_files(filename='...', participant=p, eyes='open')

# Also correct
from labanalysis.records import UprightPosture
exercise = UprightPosture.from_tdf('file.tdf')
test = UprightBalanceTest(participant=p, exercise=exercise, eyes='open')
```

### Issue: PlankBalanceTest requires 4 force platforms

**Cause**: Plank test needs all 4 contact points (both hands + both feet)

**Solution**: Ensure TDF file contains all 4 force platform channels
```python
test = PlankBalanceTest.from_files(
    filename='plank.tdf',
    participant=participant,
    eyes='open',
    left_hand_ground_reaction_force='left_hand',   # Required
    right_hand_ground_reaction_force='right_hand', # Required
    left_foot_ground_reaction_force='left_foot',   # Required
    right_foot_ground_reaction_force='right_foot'  # Required
)
```

### Issue: Normative ellipses not showing in plot

**Cause**: No normative data available for this eyes/side combination

**Solution**: Check normative data contains matching entry
```python
# Check what's in normative data
print(test.normative_data[['side', 'eyes', 'parameter']].drop_duplicates())

# If missing, results will still work but won't show performance zones
results = test.get_results()  # Works, but no normative comparison
```

---

## See Also

- [Protocols Base](protocols.md) - Base protocol classes
- [UprightPosture](../records/posture.md#uprightposture) - Upright posture record class
- [PronePosture](../records/posture.md#proneposture) - Plank posture record class
- [Balance Tutorial](../../tutorials/03-balance-assessment.md) - Complete workflow guide

---

**Balance assessment protocols with COP analysis, force symmetry, and normative comparison.**
