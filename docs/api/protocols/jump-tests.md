# labanalysis.protocols.jump-tests

Jump test protocol for comprehensive vertical jump assessment.

**Source**: `src/labanalysis/protocols/jumptests.py`

## Overview

The jump test protocol manages multiple jump types with automated processing, EMG normalization, and normative comparisons:

- **JumpTest**: Protocol class for organizing and processing jump trials
- **JumpTestResults**: Results container with figures and normative rankings

**Supported Jump Types:**
1. **Squat Jump (SJ)**: Concentric-only from static position
2. **Counter-Movement Jump (CMJ)**: Jump with pre-stretch
3. **Drop Jump (DJ)**: Plyometric jump from elevated surface
4. **Repeated Jumps**: Continuous jumping for endurance/fatigue

## Classes

### JumpTest

Complete jump assessment protocol.

```python
class JumpTest(TestProtocol):
    """
    Protocol for comprehensive jump performance assessment.
    
    Parameters
    ----------
    participant : Participant
        Participant information (must have weight specified)
    normative_data : pd.DataFrame, optional
        Reference data for performance ranking
        Default: jumps_normative_values
    emg_normalization_references : TimeseriesRecord or 'self', optional
        EMG normalization references
        'self' = use test data for normalization
        Default: empty TimeseriesRecord
    emg_normalization_function : callable, optional
        Function to compute normalization value (e.g., np.mean, np.max)
        Default: np.mean
    emg_activation_references : TimeseriesRecord or 'self', optional
        References for muscle activation thresholds
        Default: empty TimeseriesRecord
    emg_activation_threshold : float, optional
        Threshold multiplier for activation detection
        Default: 3 (3x reference level)
    relevant_muscle_map : list of str or None, optional
        Muscles to include in analysis (None = all muscles)
        Default: None
    squat_jumps : list of SingleJump, optional
        SJ trials. Default: []
    counter_movement_jumps : list of SingleJump, optional
        CMJ trials. Default: []
    drop_jumps : list of DropJump, optional
        DJ trials. Default: []
    repeated_jumps : list of SingleJump, optional
        Individual jumps from sequences. Default: []
    
    Attributes
    ----------
    squat_jumps : list of SingleJump
        Squat jump trials
    counter_movement_jumps : list of SingleJump
        Counter-movement jump trials
    drop_jumps : list of DropJump
        Drop jump trials
    repeated_jumps : list of SingleJump
        Individual jumps from sequences
    jumps : list
        All jumps combined
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline (jump-specific configuration)
    processed_data : JumpTest
        Copy with all signals processed
    
    Methods
    -------
    add_squat_jumps(*jumps)
        Add SJ trials
    add_counter_movement_jumps(*jumps)
        Add CMJ trials
    add_drop_jumps(*jumps)
        Add DJ trials
    add_repeated_jumps(*jumps)
        Add individual jumps from sequences
    pop_squat_jumps(index)
        Remove SJ trial by index
    pop_counter_movement_jumps(index)
        Remove CMJ trial by index
    pop_drop_jumps(index)
        Remove DJ trial by index
    pop_repeated_jumps(index)
        Remove repeated jump by index
    from_files(...)
        Create test from TDF files (class method)
    get_results(include_emg=True)
        Process all jumps and return JumpTestResults object with
        summary tables, analytics, and visualizations
    save(file_path)
        Save test protocol to file
    load(file_path)
        Load test protocol from file (class method)
    
    Examples
    --------
    >>> from labanalysis.protocols import JumpTest, Participant
    >>> 
    >>> # Create participant
    >>> participant = Participant(
    ...     surname='Rossi',
    ...     weight=75,
    ...     gender='M'
    ... )
    >>> 
    >>> # Create test from files
    >>> test = JumpTest.from_files(
    ...     participant=participant,
    ...     squat_jump_files=['sj1.tdf', 'sj2.tdf'],
    ...     counter_movement_jump_files=['cmj1.tdf', 'cmj2.tdf', 'cmj3.tdf'],
    ...     drop_jump_files=['dj_30cm.tdf', 'dj_40cm.tdf'],
    ...     drop_jump_heights_cm=[30, 40]
    ... )
    >>> 
    >>> # Process and get results
    >>> results = test.process()
    >>> 
    >>> # Save test protocol
    >>> test.save("rossi_jump_test.pkl")
    >>> 
    >>> # Load later
    >>> loaded_test = JumpTest.load("rossi_jump_test.pkl")
    """
```

**Signal Processing Pipeline:**

JumpTest uses optimized processing:

```python
# Force platforms
- Lowpass filter: 30 Hz cutoff
- Moment calculation: Automatic
- Reference frame: Auto-aligned to bilateral force center

# EMG signals
- Bandpass filter: 20-450 Hz
- RMS envelope: 50ms window (vs. 200ms default)
- Normalization: User-defined reference
- Activation detection: 3x threshold (adjustable)

# Kinematic markers
- Standard WholeBody processing pipeline
```

**Performance Metrics Calculated:**

1. **Basic:**
   - Jump height (cm) - from flight time
   - Flight time (ms)
   - Contact time (ms)
   - Takeoff velocity (m/s)

2. **Advanced:**
   - RSI (cm/s) - Reactive Strength Index = height/contact_time
   - Force symmetry (%) - Left-right balance
   - Eccentric/concentric phase durations
   - Rate of force development (N/s)

3. **EMG (if available):**
   - Muscle activation timing (ms)
   - Pre-activation ratio (%)
   - Coordination indices

---

### from_files()

Create JumpTest from TDF files.

```python
@classmethod
def from_files(
    cls,
    participant: Participant,
    squat_jump_files: list[str] = [],
    counter_movement_jump_files: list[str] = [],
    drop_jump_files: list[str] = [],
    drop_jump_heights_cm: list[float] = [],
    repeated_jump_files: list[str] = [],
    emg_normalization_reference_file: str | None = None,
    **kwargs
) -> 'JumpTest'
```

**Parameters:**
- `participant`: Participant object
- `squat_jump_files`: List of SJ TDF file paths
- `counter_movement_jump_files`: List of CMJ TDF file paths
- `drop_jump_files`: List of DJ TDF file paths
- `drop_jump_heights_cm`: Drop heights for each DJ file (same length as drop_jump_files)
- `repeated_jump_files`: List of repeated jump TDF files
- `emg_normalization_reference_file`: Optional TDF file for EMG normalization
- `**kwargs`: Additional parameters passed to JumpTest constructor

**Example:**

```python
from labanalysis.protocols import JumpTest, Participant

participant = Participant(surname='Smith', weight=80, gender='M')

test = JumpTest.from_files(
    participant=participant,
    squat_jump_files=[
        'data/smith_sj_trial1.tdf',
        'data/smith_sj_trial2.tdf',
        'data/smith_sj_trial3.tdf'
    ],
    counter_movement_jump_files=[
        'data/smith_cmj_trial1.tdf',
        'data/smith_cmj_trial2.tdf',
        'data/smith_cmj_trial3.tdf'
    ],
    drop_jump_files=[
        'data/smith_dj_30cm.tdf',
        'data/smith_dj_40cm.tdf',
        'data/smith_dj_50cm.tdf'
    ],
    drop_jump_heights_cm=[30, 40, 50],
    emg_normalization_reference_file='data/smith_mvc.tdf'
)

results = test.process()
```

---

### process()

Process all jumps and generate results.

```python
def process(
    self,
    include_emg: bool = True,
    include_normative_comparison: bool = True
) -> 'JumpTestResults'
```

**Parameters:**
- `include_emg` (bool): Include EMG analysis. Default: True
- `include_normative_comparison` (bool): Compare to normative data. Default: True

**Returns:**
- `JumpTestResults`: Results object with figures and summaries

**Example:**

```python
# Process with all features
results = test.process(include_emg=True, include_normative_comparison=True)

# View summary
print(results.summary)

# Plot results
fig = results.plot()
fig.show()

# Export to Excel
results.to_excel("jump_results.xlsx")
```

---

### JumpTestResults

Results container with visualization and normative comparison.

```python
class JumpTestResults(TestResults):
    """
    Container for jump test results with visualization.
    
    Attributes
    ----------
    participant : Participant
        Participant information
    summary : pd.DataFrame
        Summary metrics for all jumps
    normative_rankings : pd.DataFrame
        Performance rankings vs. normative data
    figures : dict
        Plotly figures for visualization
    
    Methods
    -------
    plot() -> go.Figure
        Generate comprehensive results visualization
    to_dataframe() -> pd.DataFrame
        Export all metrics to DataFrame
    to_excel(file_path: str)
        Export results to Excel with multiple sheets
    save(file_path: str)
        Save results to pickle file
    load(file_path: str) -> JumpTestResults
        Load results from pickle file (class method)
    
    Examples
    --------
    >>> results = test.process()
    >>> 
    >>> # View summary
    >>> print(results.summary)
    >>> 
    >>> # Plot
    >>> fig = results.plot()
    >>> fig.show()
    >>> 
    >>> # Export
    >>> results.to_excel("athlete_jump_results.xlsx")
    >>> 
    >>> # Save/load
    >>> results.save("results.pkl")
    >>> loaded = JumpTestResults.load("results.pkl")
    """
```

**Figures Generated:**

1. **Jump Height Comparison**: Bar chart comparing all jump types
2. **Force-Time Curves**: Overlay of GRF for each jump type
3. **Normative Ranking**: Performance vs. reference population
4. **EMG Activation Patterns** (if EMG available): Timing and amplitude
5. **Symmetry Analysis**: Left-right force balance

**Summary DataFrame Columns:**

- `jump_type`: SJ, CMJ, DJ, Repeated
- `trial_number`: Trial number within type
- `jump_height_cm`: Jump height
- `flight_time_ms`: Flight duration
- `contact_time_ms`: Contact duration
- `takeoff_velocity_m_s`: Velocity at takeoff
- `rsi_cm_s`: Reactive strength index (DJ only)
- `peak_force_N`: Maximum vertical GRF
- `force_symmetry_pct`: Left-right balance
- Additional EMG metrics if available

---

## Complete Example Workflow

```python
import labanalysis as laban
from labanalysis.protocols import JumpTest, Participant

# 1. Create participant
participant = Participant(
    surname='Athlete',
    name='Pro',
    gender='M',
    height=180,
    weight=75,
    birthdate=date(1995, 5, 15)
)

# 2. Create test from files
test = JumpTest.from_files(
    participant=participant,
    squat_jump_files=['sj1.tdf', 'sj2.tdf', 'sj3.tdf'],
    counter_movement_jump_files=['cmj1.tdf', 'cmj2.tdf', 'cmj3.tdf'],
    drop_jump_files=['dj_20cm.tdf', 'dj_30cm.tdf', 'dj_40cm.tdf'],
    drop_jump_heights_cm=[20, 30, 40],
    emg_normalization_reference_file='mvc.tdf',
    relevant_muscle_map=[
        'left_gastrocnemius',
        'right_gastrocnemius',
        'left_vastus_lateralis',
        'right_vastus_lateralis'
    ]
)

# 3. Process test
results = test.process(include_emg=True, include_normative_comparison=True)

# 4. View summary
summary = results.summary
print("\nBest performances:")
print(f"  SJ: {summary[summary['jump_type']=='SJ']['jump_height_cm'].max():.1f} cm")
print(f"  CMJ: {summary[summary['jump_type']=='CMJ']['jump_height_cm'].max():.1f} cm")
print(f"  DJ RSI: {summary[summary['jump_type']=='DJ']['rsi_cm_s'].max():.1f} cm/s")

# 5. Plot results
fig = results.plot()
fig.write_html("jump_results.html")
fig.show()

# 6. Export to Excel
results.to_excel("athlete_jump_report.xlsx")

# 7. Save for later
test.save("athlete_jump_test.pkl")
results.save("athlete_jump_results.pkl")
```

---

## Advanced Features

### EMG Normalization

```python
# Method 1: Use MVC trial
test = JumpTest.from_files(
    participant=participant,
    counter_movement_jump_files=['cmj.tdf'],
    emg_normalization_reference_file='mvc.tdf',
    emg_normalization_function=np.max  # Normalize to peak MVC
)

# Method 2: Use test data (self-normalization)
test = JumpTest(
    participant=participant,
    emg_normalization_references='self',
    emg_normalization_function=np.max
)
# EMG normalized to max value found in jump trials

# Method 3: Custom reference
import labanalysis as laban
ref_data = laban.read_tdf('reference.tdf', emg_keys=['.*'])
test = JumpTest(
    participant=participant,
    emg_normalization_references=ref_data,
    emg_normalization_function=lambda x: np.percentile(x, 95)
)
```

### Muscle Selection

```python
# Include only specific muscles
test = JumpTest.from_files(
    participant=participant,
    counter_movement_jump_files=['cmj.tdf'],
    relevant_muscle_map=[
        'left_gastrocnemius',
        'right_gastrocnemius',
        'left_vastus_lateralis',
        'right_vastus_lateralis',
        'left_biceps_femoris',
        'right_biceps_femoris'
    ]
)
```

### Custom Normative Data

```python
import pandas as pd

# Create custom normative data
custom_norms = pd.DataFrame({
    'gender': ['M', 'M', 'F', 'F'],
    'age_group': ['18-25', '26-35', '18-25', '26-35'],
    'cmj_height_cm_mean': [45, 42, 35, 32],
    'cmj_height_cm_std': [5, 6, 4, 5],
    'sj_height_cm_mean': [40, 38, 30, 28],
    'sj_height_cm_std': [4, 5, 3, 4],
})

test = JumpTest(
    participant=participant,
    normative_data=custom_norms
)
```

---

## Troubleshooting

### Issue: "Participant weight not specified"

**Cause**: Weight required for jump height calculation

**Solution**:
```python
participant = Participant(surname='Athlete', weight=75)
```

### Issue: "Drop jump heights mismatch"

**Cause**: Number of heights doesn't match number of files

**Solution**:
```python
# Ensure same length
drop_files = ['dj1.tdf', 'dj2.tdf', 'dj3.tdf']
drop_heights = [30, 40, 50]  # Same length!

test = JumpTest.from_files(
    participant=participant,
    drop_jump_files=drop_files,
    drop_jump_heights_cm=drop_heights
)
```

### Issue: "EMG normalization reference not found"

**Cause**: Reference file path incorrect or EMG channels don't match

**Solution**: Use 'self' normalization
```python
test = JumpTest.from_files(
    participant=participant,
    counter_movement_jump_files=['cmj.tdf'],
    emg_normalization_references='self'  # Use jump data
)
```

---

## See Also

- [Protocols Base](protocols.md) - Base protocol classes
- [SingleJump](../records/jumping.md#singlejump) - Single jump record class
- [DropJump](../records/jumping.md#dropjump) - Drop jump record class
- [RepeatedJumps](../records/jumping.md#repeatedjumps) - Repeated jumps record class
- [Jump Tutorial](../../tutorials/01-jump-analysis.md) - Complete workflow guide

---

**Comprehensive jump assessment protocol with EMG normalization and normative comparison.**
