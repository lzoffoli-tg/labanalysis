# labanalysis.protocols.strength-tests

Strength assessment protocols for isokinetic and isometric testing.

**Source**: `src/labanalysis/protocols/strengthtests.py`

## Overview

Strength test protocols for Biostrength equipment (Technogym):

- **Isokinetic1RMTest**: Isokinetic test with 1RM estimation protocol
- **IsometricTest**: Isometric maximum voluntary contraction protocol
- **Isokinetic1RMTestResults**: Results with force profiles and 1RM estimates
- **IsometricTestResults**: Results with force profiles and RFD analysis

**Supported Equipment:**
- LEG PRESS / LEG PRESS REV
- LEG EXTENSION / LEG EXTENSION REV
- LEG CURL
- LOW ROW
- ADJUSTABLE PULLEY REV
- CHEST PRESS
- SHOULDER PRESS

**Test Configurations:**
- **Left**: Unilateral left side
- **Right**: Unilateral right side
- **Bilateral**: Both sides simultaneously

## Classes

### Isokinetic1RMTest

Isokinetic strength test with 1RM estimation.

```python
class Isokinetic1RMTest(TestProtocol):
    """
    Protocol for isokinetic strength assessment with 1RM estimation.
    
    Combines Biostrength force data with optional EMG to estimate
    one-repetition maximum (1RM) from isokinetic exercise.
    
    Parameters
    ----------
    participant : Participant
        Participant information
    rm1_coefs : dict
        1RM estimation coefficients {'beta0': float, 'beta1': float}
        Equation: 1RM = (F_peak / g) * beta1 + beta0
    left : IsokineticExercise or None
        Left side isokinetic exercise data
    right : IsokineticExercise or None
        Right side isokinetic exercise data
    bilateral : IsokineticExercise or None
        Bilateral isokinetic exercise data
    normative_data : pd.DataFrame, optional
        Reference normative values
        Default: isok_1rm_normative_values
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
    left : IsokineticExercise or None
        Left side test data
    right : IsokineticExercise or None
        Right side test data
    bilateral : IsokineticExercise or None
        Bilateral test data
    rm1_coefs : dict
        1RM estimation coefficients
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline
    processed_data : Isokinetic1RMTest
        Copy with all signals processed
    
    Methods
    -------
    from_files(...)
        Create test from Biostrength files (class method)
    get_results(include_emg=True, estimate_1rm=True, include_force_balance=True)
        Process test and return results
    save(file_path)
        Save test protocol to file
    load(file_path)
        Load test protocol from file (class method)
    
    Examples
    --------
    >>> from labanalysis.protocols import Isokinetic1RMTest, Participant
    >>> 
    >>> # Create participant
    >>> participant = Participant(surname='Smith', weight=80, gender='M')
    >>> 
    >>> # Create test from Biostrength files
    >>> test = Isokinetic1RMTest.from_files(
    ...     participant=participant,
    ...     product='LEG PRESS',
    ...     left_biostrength_filename='left_legpress.txt',
    ...     right_biostrength_filename='right_legpress.txt',
    ...     left_emg_filename='left_legpress_emg.tdf',
    ...     right_emg_filename='right_legpress_emg.tdf'
    ... )
    >>> 
    >>> # Process and get results
    >>> results = test.get_results(
    ...     include_emg=True,
    ...     estimate_1rm=True,
    ...     include_force_balance=True
    ... )
    >>> 
    >>> # View summary
    >>> print(results.summary)
    >>> 
    >>> # Plot force profiles
    >>> fig = results.plot()
    >>> fig.show()
    """
```

**Signal Processing:**

```python
# Force signal
- No filtering (raw from Biostrength device)

# EMG signals (if available)
- Bandpass filter: 20-450 Hz
- RMS envelope: 200ms window
- Normalization: User-defined reference
```

**Key Metrics:**

1. **Peak Force (N)**: Maximum force during concentric phase
2. **Estimated 1RM (kg)**: Predicted one-repetition maximum
3. **ROM (m)**: Range of motion
4. **Force Symmetry (%)**: Left-right force balance
5. **EMG Symmetry (%)**: Left-right muscle activation balance (if available)

---

### from_files() (Isokinetic1RMTest)

Create Isokinetic1RMTest from Biostrength TXT files.

```python
@classmethod
def from_files(
    cls,
    participant: Participant,
    product: Literal[
        'LEG PRESS', 'LEG PRESS REV', 'LEG EXTENSION', 'LEG EXTENSION REV',
        'LEG CURL', 'LOW ROW', 'ADJUSTABLE PULLEY REV', 'CHEST PRESS',
        'SHOULDER PRESS'
    ],
    left_biostrength_filename: str | None = None,
    right_biostrength_filename: str | None = None,
    bilateral_biostrength_filename: str | None = None,
    left_emg_filename: str | None = None,
    right_emg_filename: str | None = None,
    bilateral_emg_filename: str | None = None,
    normative_data: pd.DataFrame = isok_1rm_normative_values,
    emg_normalization_references: TimeseriesRecord | str | Literal['self'] = TimeseriesRecord(),
    emg_normalization_function: Callable = np.mean,
    emg_activation_references: TimeseriesRecord | str | Literal['self'] = TimeseriesRecord(),
    emg_activation_threshold: float = 3,
    relevant_muscle_map: list[str] | None = None
) -> 'Isokinetic1RMTest'
```

**Parameters:**
- `participant`: Participant object
- `product`: Biostrength equipment type (determines 1RM coefficients)
- `left_biostrength_filename`: Left side Biostrength TXT file (None = no left test)
- `right_biostrength_filename`: Right side Biostrength TXT file (None = no right test)
- `bilateral_biostrength_filename`: Bilateral Biostrength TXT file (None = no bilateral test)
- `left_emg_filename`: Left side EMG TDF file (optional)
- `right_emg_filename`: Right side EMG TDF file (optional)
- `bilateral_emg_filename`: Bilateral EMG TDF file (optional)
- `normative_data`: Normative reference data
- `emg_normalization_references`: EMG normalization reference
- `emg_normalization_function`: Normalization function
- `emg_activation_references`: Activation threshold references
- `emg_activation_threshold`: Activation threshold multiplier
- `relevant_muscle_map`: List of muscles to analyze

**Example:**

```python
from labanalysis.protocols import Isokinetic1RMTest, Participant

participant = Participant(surname='Athlete', weight=80, gender='M')

# Bilateral leg press test with EMG
test = Isokinetic1RMTest.from_files(
    participant=participant,
    product='LEG PRESS',
    bilateral_biostrength_filename='legpress_bilateral.txt',
    bilateral_emg_filename='legpress_bilateral_emg.tdf',
    relevant_muscle_map=[
        'left_vastus_lateralis',
        'right_vastus_lateralis',
        'left_biceps_femoris',
        'right_biceps_femoris',
        'left_gastrocnemius',
        'right_gastrocnemius'
    ]
)

results = test.get_results(
    include_emg=True,
    estimate_1rm=True,
    include_force_balance=True
)

# View estimated 1RM
summary = results.summary
rm1 = summary[summary['parameter'] == 'estimated 1RM (kg)']['bilateral'].values[0]
print(f"Estimated 1RM: {rm1:.1f} kg")
```

---

### get_results() (Isokinetic1RMTest)

Process isokinetic test and return results.

```python
def get_results(
    self,
    include_emg: bool = True,
    estimate_1rm: bool = True,
    include_force_balance: bool = True
) -> 'Isokinetic1RMTestResults'
```

**Parameters:**
- `include_emg` (bool): Include EMG analysis. Default: True
- `estimate_1rm` (bool): Calculate 1RM estimation. Default: True
- `include_force_balance` (bool): Calculate force symmetry. Default: True

**Returns:**
- `Isokinetic1RMTestResults`: Results object

**Example:**

```python
# Full analysis
results = test.get_results(
    include_emg=True,
    estimate_1rm=True,
    include_force_balance=True
)

# Force-only analysis (no EMG)
results = test.get_results(
    include_emg=False,
    estimate_1rm=True,
    include_force_balance=True
)
```

---

### Isokinetic1RMTestResults

Results container for isokinetic 1RM tests.

```python
class Isokinetic1RMTestResults(TestResults):
    """
    Container for isokinetic 1RM test results.
    
    Attributes
    ----------
    participant : Participant
        Participant information
    summary : pd.DataFrame
        Summary metrics (1RM, peak force, ROM, symmetry)
    analytics : pd.DataFrame
        Time-series data (force profiles, EMG)
    figures : dict
        Plotly figures for visualization
    estimate_1rm : bool
        Whether 1RM was estimated
    include_emg : bool
        Whether EMG was included
    include_force_balance : bool
        Whether force balance was calculated
    
    Methods
    -------
    plot() -> go.Figure
        Generate comprehensive results visualization
    to_dataframe() -> pd.DataFrame
        Export all metrics to DataFrame
    save(file_path: str)
        Save results to pickle file
    load(file_path: str) -> Isokinetic1RMTestResults
        Load results from pickle file (class method)
    
    Examples
    --------
    >>> results = test.get_results()
    >>> 
    >>> # View summary
    >>> print(results.summary)
    >>> #   parameter                  left   right  bilateral  symmetry (%)
    >>> # 0  rom (m)                  0.45    0.46      0.92          -2.2
    >>> # 1  estimated 1RM (kg)      120.5   118.3     245.8           1.8
    >>> # 2  peak force (N)         1250.2  1220.5    2480.6           2.4
    >>> 
    >>> # Plot force profiles + muscle balance
    >>> fig = results.plot()
    >>> fig.show()
    """
```

**Figures Generated:**

1. **Force Profiles** (one subplot per side):
   - Force vs. concentric phase (% of ROM)
   - Peak force annotation
   - Estimated 1RM annotation
   - RFD annotation (if available)
   - Time to peak force annotation

2. **Muscle Imbalance** (if EMG available):
   - Horizontal bar chart showing left-right asymmetry per muscle
   - Color gradient from green (balanced) to red (imbalanced)

**Summary DataFrame Columns:**

- `parameter`: Metric name
- `left`: Left side value (if available)
- `right`: Right side value (if available)
- `bilateral`: Bilateral value (if available)
- `symmetry (%)`: Left-right asymmetry percentage (if both sides available)

**Parameters include:**
- `rom (m)`: Range of motion
- `estimated 1RM (kg)`: Predicted 1RM (if `estimate_1rm=True`)
- `peak force (N)`: Maximum force
- `<muscle_name> (%)`: EMG amplitude (if EMG available and normalized)
- `<muscle_name> (uV)`: EMG amplitude (if EMG available but not normalized)

---

### IsometricTest

Isometric maximum voluntary contraction test.

```python
class IsometricTest(TestProtocol):
    """
    Protocol for isometric strength assessment.
    
    Assesses maximum voluntary contraction (MVC) force production
    with rate of force development (RFD) analysis.
    
    Parameters
    ----------
    participant : Participant
        Participant information
    left : IsometricExercise or None
        Left side isometric exercise data
    right : IsometricExercise or None
        Right side isometric exercise data
    bilateral : IsometricExercise or None
        Bilateral isometric exercise data
    normative_data : pd.DataFrame, optional
        Reference normative values
        Default: empty DataFrame
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
    left : IsometricExercise or None
        Left side test data
    right : IsometricExercise or None
        Right side test data
    bilateral : IsometricExercise or None
        Bilateral test data
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline (1 Hz lowpass)
    processed_data : IsometricTest
        Copy with all signals processed
    
    Methods
    -------
    from_files(...)
        Create test from Biostrength files (class method)
    get_results(include_emg=True)
        Process test and return results
    save(file_path)
        Save test protocol to file
    load(file_path)
        Load test protocol from file (class method)
    
    Examples
    --------
    >>> from labanalysis.protocols import IsometricTest, Participant
    >>> 
    >>> # Create participant
    >>> participant = Participant(surname='Smith', weight=80)
    >>> 
    >>> # Create test from Biostrength files
    >>> test = IsometricTest.from_files(
    ...     participant=participant,
    ...     product='LEG EXTENSION',
    ...     left_biostrength_filename='left_mvc.txt',
    ...     right_biostrength_filename='right_mvc.txt'
    ... )
    >>> 
    >>> # Process
    >>> results = test.get_results(include_emg=False)
    >>> 
    >>> # View RFD
    >>> summary = results.summary
    >>> print(summary[summary['parameter'] == 'rate of force development (kN/s)'])
    """
```

**Signal Processing:**

```python
# Force signal
- Lowpass filter: 1 Hz cutoff, 4th order Butterworth
- Phase-corrected (zero-lag)

# EMG signals (if available)
- Bandpass filter: 20-450 Hz
- RMS envelope: 200ms window
- Normalization: User-defined reference
```

**Key Metrics:**

1. **Peak Force (N)**: Maximum force during MVC
2. **Rate of Force Development (kN/s)**: RFD from start to peak
3. **Time to Peak Force (ms)**: Duration from start to peak force
4. **Force Symmetry (%)**: Left-right force balance
5. **EMG Symmetry (%)**: Left-right muscle activation balance (if available)

---

### from_files() (IsometricTest)

Create IsometricTest from Biostrength TXT files.

```python
@classmethod
def from_files(
    cls,
    participant: Participant,
    product: Literal[
        'LEG PRESS', 'LEG PRESS REV', 'LEG EXTENSION', 'LEG EXTENSION REV',
        'LEG CURL', 'LOW ROW', 'ADJUSTABLE PULLEY REV', 'CHEST PRESS',
        'SHOULDER PRESS'
    ],
    left_biostrength_filename: str | None = None,
    right_biostrength_filename: str | None = None,
    bilateral_biostrength_filename: str | None = None,
    left_emg_filename: str | None = None,
    right_emg_filename: str | None = None,
    bilateral_emg_filename: str | None = None,
    normative_data: pd.DataFrame = pd.DataFrame(),
    emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
    emg_normalization_function: Callable = np.mean,
    emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
    emg_activation_threshold: float = 3,
    relevant_muscle_map: list[str] | None = None
) -> 'IsometricTest'
```

**Example:**

```python
from labanalysis.protocols import IsometricTest, Participant

participant = Participant(surname='Athlete', weight=75, gender='F')

# Bilateral isometric leg extension MVC
test = IsometricTest.from_files(
    participant=participant,
    product='LEG EXTENSION',
    bilateral_biostrength_filename='legext_mvc.txt',
    bilateral_emg_filename='legext_mvc_emg.tdf',
    relevant_muscle_map=[
        'left_vastus_lateralis',
        'right_vastus_lateralis',
        'left_vastus_medialis',
        'right_vastus_medialis',
        'left_rectus_femoris',
        'right_rectus_femoris'
    ]
)

results = test.get_results(include_emg=True)

# Analyze RFD
summary = results.summary
rfd = summary[summary['parameter'] == 'rate of force development (kN/s)']['bilateral'].values[0]
print(f"RFD: {rfd:.2f} kN/s")
```

---

### IsometricTestResults

Results container for isometric tests.

```python
class IsometricTestResults(TestResults):
    """
    Container for isometric test results.
    
    Attributes
    ----------
    participant : Participant
        Participant information
    summary : pd.DataFrame
        Summary metrics (peak force, RFD, time to peak, symmetry)
    analytics : pd.DataFrame
        Time-series data (force profiles, EMG)
    figures : dict
        Plotly figures for visualization
    include_emg : bool
        Whether EMG was included
    
    Methods
    -------
    plot() -> go.Figure
        Generate comprehensive results visualization
    to_dataframe() -> pd.DataFrame
        Export all metrics to DataFrame
    save(file_path: str)
        Save results to pickle file
    load(file_path: str) -> IsometricTestResults
        Load results from pickle file (class method)
    
    Examples
    --------
    >>> results = test.get_results(include_emg=True)
    >>> 
    >>> # View summary
    >>> print(results.summary)
    >>> #   parameter                           left    right  bilateral  symmetry (%)
    >>> # 0  rate of force development (kN/s)   2.45     2.38      4.82          2.9
    >>> # 1  time to peak force (ms)          285.3    292.1     288.7         -2.4
    >>> # 2  peak force (N)                  1450.2   1410.3    2860.5          2.8
    >>> 
    >>> # Plot
    >>> fig = results.plot()
    >>> fig.show()
    """
```

**Summary DataFrame Columns:**

- `parameter`: Metric name
- `left`: Left side value (if available)
- `right`: Right side value (if available)
- `bilateral`: Bilateral value (if available)
- `symmetry (%)`: Left-right asymmetry percentage

**Parameters include:**
- `rate of force development (kN/s)`: RFD
- `time to peak force (ms)`: Time to reach peak force
- `peak force (N)`: Maximum force
- `<muscle_name> (%)`: EMG amplitude (if normalized)
- `<muscle_name> (uV)`: EMG amplitude (if not normalized)

---

## Complete Example Workflows

### Isokinetic 1RM Assessment

```python
import labanalysis as laban
from labanalysis.protocols import Isokinetic1RMTest, Participant
from datetime import date

# 1. Create participant
participant = Participant(
    surname='Powerlifter',
    name='Pro',
    gender='M',
    height=180,
    weight=90,
    birthdate=date(1990, 1, 1)
)

# 2. Create test from files (left vs right comparison)
test = Isokinetic1RMTest.from_files(
    participant=participant,
    product='LEG PRESS',
    left_biostrength_filename='left_legpress.txt',
    right_biostrength_filename='right_legpress.txt',
    left_emg_filename='left_legpress_emg.tdf',
    right_emg_filename='right_legpress_emg.tdf',
    relevant_muscle_map=[
        'left_vastus_lateralis',
        'right_vastus_lateralis',
        'left_gluteus_maximus',
        'right_gluteus_maximus'
    ]
)

# 3. Process
results = test.get_results(
    include_emg=True,
    estimate_1rm=True,
    include_force_balance=True
)

# 4. Analyze results
summary = results.summary

# Get 1RM estimates
rm1_left = summary[summary['parameter'] == 'estimated 1RM (kg)']['left'].values[0]
rm1_right = summary[summary['parameter'] == 'estimated 1RM (kg)']['right'].values[0]
rm1_symm = summary[summary['parameter'] == 'estimated 1RM (kg)']['symmetry (%)'].values[0]

print(f"Left 1RM: {rm1_left:.1f} kg")
print(f"Right 1RM: {rm1_right:.1f} kg")
print(f"Asymmetry: {abs(rm1_symm):.1f}%")

# Interpretation
if abs(rm1_symm) < 10:
    print("  → Balanced strength")
elif abs(rm1_symm) < 15:
    print("  → Mild asymmetry (monitor)")
else:
    print("  → Significant asymmetry (address imbalance)")

# 5. Plot
fig = results.plot()
fig.write_html("isokinetic_1rm_results.html")
fig.show()

# 6. Export
results.save("isokinetic_1rm_results.pkl")
summary.to_excel("isokinetic_1rm_summary.xlsx", index=False)
```

### Isometric MVC with RFD Analysis

```python
from labanalysis.protocols import IsometricTest, Participant

participant = Participant(surname='Sprinter', weight=75, gender='M')

# Bilateral MVC test
test = IsometricTest.from_files(
    participant=participant,
    product='LEG EXTENSION',
    bilateral_biostrength_filename='legext_mvc.txt',
    bilateral_emg_filename='legext_mvc_emg.tdf',
    relevant_muscle_map=[
        'left_vastus_lateralis',
        'right_vastus_lateralis',
        'left_rectus_femoris',
        'right_rectus_femoris'
    ]
)

results = test.get_results(include_emg=True)

# Analyze explosive strength
summary = results.summary
peak_force = summary[summary['parameter'] == 'peak force (N)']['bilateral'].values[0]
rfd = summary[summary['parameter'] == 'rate of force development (kN/s)']['bilateral'].values[0]
time_to_peak = summary[summary['parameter'] == 'time to peak force (ms)']['bilateral'].values[0]

print(f"Peak Force: {peak_force:.0f} N ({peak_force / (participant.weight * 9.81):.2f} BW)")
print(f"RFD: {rfd:.2f} kN/s")
print(f"Time to Peak: {time_to_peak:.0f} ms")

# RFD interpretation for sprinters
if rfd > 5.0:
    print("  → Excellent explosive strength")
elif rfd > 3.5:
    print("  → Good explosive strength")
elif rfd > 2.0:
    print("  → Moderate explosive strength")
else:
    print("  → Low explosive strength (focus on RFD training)")

# Plot
fig = results.plot()
fig.show()
```

### Unilateral vs Bilateral Comparison

```python
from labanalysis.protocols import Isokinetic1RMTest, Participant

participant = Participant(surname='Athlete', weight=80)

# Test all three conditions
test = Isokinetic1RMTest.from_files(
    participant=participant,
    product='LEG PRESS',
    left_biostrength_filename='left_legpress.txt',
    right_biostrength_filename='right_legpress.txt',
    bilateral_biostrength_filename='bilateral_legpress.txt'
)

results = test.get_results(estimate_1rm=True, include_emg=False)

# Calculate bilateral deficit
summary = results.summary
rm1_data = summary[summary['parameter'] == 'estimated 1RM (kg)']

rm1_left = rm1_data['left'].values[0]
rm1_right = rm1_data['right'].values[0]
rm1_bilateral = rm1_data['bilateral'].values[0]

unilateral_sum = rm1_left + rm1_right
bilateral_deficit_pct = (1 - rm1_bilateral / unilateral_sum) * 100

print(f"Left 1RM: {rm1_left:.1f} kg")
print(f"Right 1RM: {rm1_right:.1f} kg")
print(f"Bilateral 1RM: {rm1_bilateral:.1f} kg")
print(f"Expected (sum): {unilateral_sum:.1f} kg")
print(f"Bilateral Deficit: {bilateral_deficit_pct:.1f}%")

# Interpretation
if bilateral_deficit_pct < 0:
    print("  → Bilateral facilitation (unusual)")
elif bilateral_deficit_pct < 10:
    print("  → Normal bilateral deficit")
else:
    print("  → High bilateral deficit (coordination issue)")
```

---

## Advanced Features

### Custom 1RM Coefficients

```python
# Use custom regression coefficients instead of equipment defaults
custom_coefs = {
    'beta0': 25.5,  # Intercept
    'beta1': 0.85   # Slope
}

test = Isokinetic1RMTest(
    participant=participant,
    rm1_coefs=custom_coefs,
    bilateral=isokinetic_exercise,
    normative_data=pd.DataFrame()
)

results = test.get_results(estimate_1rm=True)
```

### EMG Normalization to MVC

```python
# Use MVC trial to normalize subsequent tests
mvc_test = IsometricTest.from_files(
    participant=participant,
    product='LEG EXTENSION',
    bilateral_biostrength_filename='mvc.txt',
    bilateral_emg_filename='mvc_emg.tdf'
)

# Use MVC as normalization reference
iso_test = Isokinetic1RMTest.from_files(
    participant=participant,
    product='LEG EXTENSION',
    bilateral_biostrength_filename='isokinetic.txt',
    bilateral_emg_filename='isokinetic_emg.tdf',
    emg_normalization_references='mvc_emg.tdf',
    emg_normalization_function=np.max  # Normalize to peak MVC
)

results = iso_test.get_results(include_emg=True)
```

---

## Troubleshooting

### Issue: "rm1_coefs must be a dict with keys 'beta0', 'beta1'"

**Cause**: Invalid 1RM coefficients format

**Solution**: Use correct dictionary format
```python
rm1_coefs = {'beta0': 25.5, 'beta1': 0.85}
test = Isokinetic1RMTest(
    participant=participant,
    rm1_coefs=rm1_coefs,
    bilateral=exercise
)
```

### Issue: No force data in Biostrength file

**Cause**: Invalid file format or wrong product type

**Solution**: Check file format and product name
```python
# Verify product name matches exactly (case-sensitive)
valid_products = [
    'LEG PRESS', 'LEG PRESS REV', 'LEG EXTENSION', 'LEG EXTENSION REV',
    'LEG CURL', 'LOW ROW', 'ADJUSTABLE PULLEY REV', 'CHEST PRESS',
    'SHOULDER PRESS'
]

# Use exact product name from list
test = Isokinetic1RMTest.from_files(
    participant=participant,
    product='LEG EXTENSION',  # Exact match required
    bilateral_biostrength_filename='test.txt'
)
```

### Issue: EMG channels mismatch between Biostrength and TDF files

**Cause**: EMG file recorded at different time or missing synchronization

**Solution**: Ensure EMG file corresponds to same trial, or skip EMG
```python
# Option 1: Skip EMG if not synchronized
test = Isokinetic1RMTest.from_files(
    participant=participant,
    product='LEG PRESS',
    bilateral_biostrength_filename='legpress.txt',
    bilateral_emg_filename=None  # Skip EMG
)

# Option 2: Use separate EMG normalization
test = Isokinetic1RMTest.from_files(
    participant=participant,
    product='LEG PRESS',
    bilateral_biostrength_filename='legpress.txt',
    bilateral_emg_filename='legpress_emg.tdf',
    emg_normalization_references='self'  # Self-normalize
)
```

---

## See Also

- [Protocols Base](protocols.md) - Base protocol classes
- [IsokineticExercise](../records/strength.md#isokineticexercise) - Isokinetic exercise record
- [IsometricExercise](../records/strength.md#isometricexercise) - Isometric exercise record
- [Strength Tutorial](../../tutorials/04-strength-assessment.md) - Complete workflow guide

---

**Strength assessment protocols for isokinetic 1RM estimation and isometric MVC testing.**
