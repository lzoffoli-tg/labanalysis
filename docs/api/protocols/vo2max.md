# labanalysis.protocols.vo2max

VO2max estimation protocol from submaximal exercise testing.

**Source**: `src/labanalysis/protocols/vo2max.py`

## Overview

Submaximal VO2max test protocol for aerobic capacity assessment:

- **SubmaximalVO2MaxTest**: Submaximal exercise test with VO2max estimation
- **SubmaximalVO2MaxTestResults**: Results with VO2max, VT2, and FatMax analysis

The test estimates VO2max from submaximal metabolic data using heart rate extrapolation and respiratory quotient (RQ) analysis.

**Estimation Methods:**
1. **HR Extrapolation**: Linear extrapolation to age-predicted HRmax
2. **RQ Method**: VO2max estimation from RQ > 0.832 (Beck et al., 2018)

**Additional Metrics:**
- **VT2** (Ventilatory Threshold 2): Anaerobic threshold detection
- **FatMax**: Maximum fat oxidation rate and corresponding intensity

## Classes

### SubmaximalVO2MaxTest

Submaximal VO2max test protocol.

```python
class SubmaximalVO2MaxTest(TestProtocol):
    """
    Protocol for submaximal VO2max estimation with metabolic analysis.
    
    Estimates maximal oxygen uptake from submaximal exercise data using
    heart rate extrapolation and RQ-based methods. Also identifies
    ventilatory threshold 2 (VT2) and fat oxidation maximum (FatMax).
    
    Parameters
    ----------
    participant : Participant
        Participant information (age, weight, gender required)
    metabolic_record : MetabolicRecord
        Metabolic data (VO2, VCO2, HR, RQ)
    normative_data : pd.DataFrame, optional
        Reference normative values
        Default: vo2max_normative_values
    
    Attributes
    ----------
    participant : Participant
        Participant information
    metabolic_record : MetabolicRecord
        Metabolic exercise data
    normative_data : pd.DataFrame
        Reference normative values
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline
    processed_data : SubmaximalVO2MaxTest
        Copy with all signals processed
    
    Methods
    -------
    from_files(...)
        Create test from metabolic file (class method)
    get_results()
        Process test and return results
    save(file_path)
        Save test protocol to file
    load(file_path)
        Load test protocol from file (class method)
    
    Examples
    --------
    >>> from labanalysis.protocols import SubmaximalVO2MaxTest, Participant
    >>> from datetime import date
    >>> 
    >>> # Create participant (age, weight, gender required)
    >>> participant = Participant(
    ...     surname='Runner',
    ...     gender='Male',
    ...     weight=75,
    ...     birthdate=date(1990, 5, 15)
    ... )
    >>> 
    >>> # Create test from metabolic file
    >>> test = SubmaximalVO2MaxTest.from_files(
    ...     filename='submaximal_test.xlsx',
    ...     participant=participant,
    ...     breath_by_breath=False
    ... )
    >>> 
    >>> # Process
    >>> results = test.get_results()
    >>> 
    >>> # View VO2max estimation
    >>> print(results.summary)
    """
```

**Required Participant Information:**
- **Age** (or birthdate): For HRmax prediction (Gellish: 207 - 0.7 × age)
- **Weight**: For FatMax calculation and power/speed prediction
- **Gender**: For cycling power prediction at thresholds

**Metabolic Data Requirements:**
- **VO2** (ml/kg/min): Oxygen uptake
- **VCO2** (ml/kg/min): Carbon dioxide production
- **RQ**: Respiratory quotient (VCO2/VO2)
- **HR** (bpm): Heart rate (optional but recommended for HR-based estimation)
- **Fat oxidation** (g/min): Fat oxidation rate (for FatMax analysis)

---

### from_files()

Create SubmaximalVO2MaxTest from metabolic file.

```python
@classmethod
def from_files(
    cls,
    filename: str,
    participant: Participant,
    normative_data: pd.DataFrame = vo2max_normative_values,
    breath_by_breath: bool = False
) -> 'SubmaximalVO2MaxTest'
```

**Parameters:**
- `filename`: Path to metabolic data file (Excel, CSV, etc.)
- `participant`: Participant object (age, weight, gender required)
- `normative_data`: Reference normative values
- `breath_by_breath`: Whether data is breath-by-breath (True) or averaged (False)

**Example:**

```python
from labanalysis.protocols import SubmaximalVO2MaxTest, Participant
from datetime import date

participant = Participant(
    surname='Athlete',
    gender='Male',
    weight=75,
    height=180,
    birthdate=date(1990, 1, 1)
)

# From metabolic cart export (e.g., Cosmed, Cortex)
test = SubmaximalVO2MaxTest.from_files(
    filename='graded_exercise_test.xlsx',
    participant=participant,
    breath_by_breath=False  # Data already averaged
)

results = test.get_results()

# View estimated VO2max
vo2max = results.summary.loc[results.summary['Parameter'] == 'VO2max', 'Value'].values[0]
print(f"Estimated VO2max: {vo2max:.1f} ml/kg/min")
```

---

### SubmaximalVO2MaxTestResults

Results container for submaximal VO2max tests.

```python
class SubmaximalVO2MaxTestResults(TestResults):
    """
    Container for submaximal VO2max test results.
    
    Attributes
    ----------
    participant : Participant
        Participant information
    summary : pd.DataFrame
        Summary metrics (VO2max, VT2, FatMax)
    analytics : pd.DataFrame
        Time-series metabolic data
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
    load(file_path: str) -> SubmaximalVO2MaxTestResults
        Load results from pickle file (class method)
    
    Examples
    --------
    >>> results = test.get_results()
    >>> 
    >>> # View summary
    >>> print(results.summary)
    >>> #   Parameter             Value       Unit
    >>> # 0  VO2max               52.3     ml/kg/min
    >>> # 1  VT2 VO2              45.2     ml/kg/min
    >>> # 2  VT2 VO2              86.5     %VO2max
    >>> # 3  VT2 HR               165      bpm
    >>> # 4  VT2 Running Speed    14.2     km/h
    >>> # 5  FatMax               0.65     g/min
    >>> # 6  FatMax VO2           28.5     ml/kg/min
    >>> 
    >>> # Plot VO2-HR relationship with VT2
    >>> fig = results.plot()
    >>> fig.show()
    """
```

**Summary Metrics:**

1. **VO2max Estimation:**
   - `VO2max` (ml/kg/min): Estimated maximal oxygen uptake
   - Derived from minimum of HR extrapolation and RQ method

2. **VT2 (Ventilatory Threshold 2):**
   - `VT2 VO2` (ml/kg/min): VO2 at anaerobic threshold
   - `VT2 VO2` (%VO2max): Threshold as % of VO2max
   - `VT2 HR` (bpm): Heart rate at threshold
   - `VT2 HR` (%HRmax): HR as % of max
   - `VT2 Running Speed` (km/h): Predicted running speed at VT2
   - `VT2 Cycling Power` (W): Predicted cycling power at VT2

3. **FatMax:**
   - `FatMax` (g/min): Maximum fat oxidation rate
   - `FatMax VO2` (ml/kg/min): VO2 at maximum fat oxidation
   - `FatMax VO2` (%VO2max): Intensity as % of VO2max
   - `FatMax HR` (bpm): Heart rate at FatMax
   - `FatMax Running Speed` (km/h): Predicted running speed at FatMax
   - `FatMax Cycling Power` (W): Predicted cycling power at FatMax

---

## Complete Example Workflow

```python
import labanalysis as laban
from labanalysis.protocols import SubmaximalVO2MaxTest, Participant
from datetime import date
import pandas as pd

# 1. Create participant
participant = Participant(
    surname='Endurance',
    name='Athlete',
    gender='Male',
    height=178,
    weight=72,
    birthdate=date(1992, 6, 15)
)

print(f"Participant: {participant.fullname}")
print(f"Age: {participant.age} years")
print(f"Predicted HRmax: {participant.hrmax:.0f} bpm")

# 2. Create test from metabolic file
test = SubmaximalVO2MaxTest.from_files(
    filename='graded_exercise_test.xlsx',
    participant=participant,
    breath_by_breath=False
)

# 3. Process
results = test.get_results()

# 4. Analyze VO2max
summary = results.summary
vo2max = summary.loc[summary['Parameter'] == 'VO2max', 'Value'].values[0]

print(f"\n=== VO2max Assessment ===")
print(f"Estimated VO2max: {vo2max:.1f} ml/kg/min")

# Classify aerobic fitness (male, 30-39 years)
if vo2max < 35:
    classification = "Poor"
elif vo2max < 43:
    classification = "Fair"
elif vo2max < 51:
    classification = "Good"
elif vo2max < 57:
    classification = "Excellent"
else:
    classification = "Superior"

print(f"Classification: {classification}")

# 5. Analyze VT2 (Anaerobic Threshold)
vt2_vo2 = summary.loc[summary['Parameter'] == 'VT2', 'ml/kg/min'].values[0]
vt2_vo2_pct = summary.loc[summary['Parameter'] == 'VT2', '%VO2max'].values[0]
vt2_hr = summary.loc[summary['Parameter'] == 'VT2', 'bpm'].values[0]
vt2_speed = summary.loc[summary['Parameter'] == 'VT2', 'km/h'].values[0]

print(f"\n=== Ventilatory Threshold 2 (VT2) ===")
print(f"VT2 VO2: {vt2_vo2:.1f} ml/kg/min ({vt2_vo2_pct:.1f}% VO2max)")
print(f"VT2 HR: {vt2_hr:.0f} bpm")
print(f"VT2 Running Speed: {vt2_speed:.1f} km/h")

# Training zones
print(f"\nTraining Zones (Running):")
print(f"  Zone 1 (Recovery): < {vt2_speed * 0.75:.1f} km/h")
print(f"  Zone 2 (Endurance): {vt2_speed * 0.75:.1f} - {vt2_speed * 0.90:.1f} km/h")
print(f"  Zone 3 (Tempo): {vt2_speed * 0.90:.1f} - {vt2_speed:.1f} km/h")
print(f"  Zone 4 (Threshold): {vt2_speed:.1f} - {vt2_speed * 1.05:.1f} km/h")
print(f"  Zone 5 (VO2max): > {vt2_speed * 1.05:.1f} km/h")

# 6. Analyze FatMax
fatmax = summary.loc[summary['Parameter'] == 'FatMax', 'g/min'].values[0]
fatmax_vo2 = summary.loc[summary['Parameter'] == 'FatMax', 'ml/kg/min'].values[0]
fatmax_vo2_pct = summary.loc[summary['Parameter'] == 'FatMax', '%VO2max'].values[0]

print(f"\n=== Fat Oxidation ===")
print(f"Maximum Fat Oxidation: {fatmax:.2f} g/min")
print(f"FatMax Intensity: {fatmax_vo2:.1f} ml/kg/min ({fatmax_vo2_pct:.1f}% VO2max)")

# 7. Plot results
fig = results.plot()
fig.write_html("vo2max_test_results.html")
fig.show()

# 8. Export
results.save("vo2max_test_results.pkl")
summary.to_excel("vo2max_summary.xlsx", index=False)
```

---

## Advanced Features

### Comparing Pre/Post Training

```python
from labanalysis.protocols import SubmaximalVO2MaxTest, Participant

participant = Participant(surname='Trainee', weight=75, birthdate=date(1990, 1, 1))

# Pre-training test
test_pre = SubmaximalVO2MaxTest.from_files(
    filename='vo2_pre_training.xlsx',
    participant=participant
)
results_pre = test_pre.get_results()

# Post-training test (8 weeks later)
test_post = SubmaximalVO2MaxTest.from_files(
    filename='vo2_post_training.xlsx',
    participant=participant
)
results_post = test_post.get_results()

# Compare
vo2max_pre = results_pre.summary.loc[results_pre.summary['Parameter'] == 'VO2max', 'Value'].values[0]
vo2max_post = results_post.summary.loc[results_post.summary['Parameter'] == 'VO2max', 'Value'].values[0]
improvement = ((vo2max_post - vo2max_pre) / vo2max_pre) * 100

print("=== Training Adaptation ===")
print(f"Pre-training VO2max: {vo2max_pre:.1f} ml/kg/min")
print(f"Post-training VO2max: {vo2max_post:.1f} ml/kg/min")
print(f"Improvement: {improvement:.1f}%")

if improvement > 5:
    print("  → Excellent response to training")
elif improvement > 2:
    print("  → Good response to training")
elif improvement > 0:
    print("  → Modest improvement")
else:
    print("  → No improvement (check training load/recovery)")
```

### HR-Free VO2max Estimation (RQ Method Only)

```python
# When HR data unavailable (e.g., arrhythmia, no HR monitor)
participant = Participant(
    surname='NoHR',
    weight=70,
    birthdate=date(1985, 1, 1)
)

test = SubmaximalVO2MaxTest.from_files(
    filename='vo2_test_no_hr.xlsx',
    participant=participant
)

results = test.get_results()

# VO2max will be estimated from RQ method only (Beck et al., 2018)
# Based on RQ > 0.832: VO2_% = sqrt(2*RQ - 1.664) + 0.301
```

---

## VO2max Estimation Methods

### 1. RQ-Based Method (Beck et al., 2018)

```python
# For RQ > 0.832
# Percentage of VO2max = sqrt(2 * RQ - 1.664) + 0.301
# VO2max = VO2_measured / VO2_percentage

# Example: At VO2 = 40 ml/kg/min, RQ = 0.95
RQ = 0.95
VO2_measured = 40

vo2_percentage = (2 * RQ - 1.664)**0.5 + 0.301  # = 0.867 (86.7%)
vo2max_estimate = VO2_measured / vo2_percentage  # = 46.1 ml/kg/min
```

### 2. HR Extrapolation Method

```python
# Linear extrapolation from submaximal HR-VO2 relationship
# 1. Fit line through (HR, VO2) points where RQ > 0.95
# 2. Extrapolate to age-predicted HRmax (207 - 0.7 * age)
# 3. VO2max = predicted VO2 at HRmax

# Example: Age 35, HRmax = 207 - 0.7*35 = 182.5 bpm
# If linear fit is VO2 = 0.35*HR - 20
vo2max_hr = 0.35 * 182.5 - 20  # = 43.9 ml/kg/min
```

### 3. Final Estimation

```python
# Final VO2max = min(RQ_method, HR_method)
# Conservative estimate to avoid overestimation
```

---

## VT2 Detection

Ventilatory Threshold 2 (anaerobic threshold) is detected as the point where the VCO2/VO2 curve crosses the identity line (VCO2 = VO2), indicating RQ = 1.

```python
# Fit 3rd order polynomial to VCO2 vs VO2
# Find where polynomial crosses identity line (y = x)
# VT2_VO2 = solution where polynomial(VO2) = VO2
```

This represents the maximal sustainable aerobic intensity where lactate production equals clearance.

---

## Troubleshooting

### Issue: "participant's age or date of birth must be provided"

**Cause**: Age required for HRmax prediction

**Solution**: Provide age or birthdate
```python
participant = Participant(
    surname='Athlete',
    birthdate=date(1990, 5, 15)  # Age calculated automatically
)
# OR
participant = Participant(
    surname='Athlete',
    age=35
)
```

### Issue: "participant's weight must be provided"

**Cause**: Weight required for FatMax and power/speed predictions

**Solution**: Provide weight in kg
```python
participant = Participant(
    surname='Athlete',
    weight=75,
    birthdate=date(1990, 1, 1)
)
```

### Issue: "participant's gender must be provided as Male or Female"

**Cause**: Gender required for cycling power prediction equations

**Solution**: Provide gender exactly as 'Male' or 'Female'
```python
participant = Participant(
    surname='Athlete',
    gender='Male',  # Must be 'Male' or 'Female' (case-sensitive)
    weight=75,
    birthdate=date(1990, 1, 1)
)
```

### Issue: Unrealistic VO2max estimate

**Cause**: Insufficient high-intensity data (RQ < 0.832 throughout)

**Solution**: Ensure test reaches near-maximal intensities
```python
# Check RQ values in metabolic data
import labanalysis as laban
data = laban.MetabolicRecord.from_file('test.xlsx')
print(f"Max RQ reached: {data.rq.max():.3f}")
# Should reach > 0.90 for reliable estimation
# Ideally > 1.0 for best accuracy
```

---

## See Also

- [Protocols Base](protocols.md) - Base protocol classes
- [MetabolicRecord](../records/metabolic.md) - Metabolic data record
- [Cardio Equations](../equations/cardio.md) - VO2 prediction equations
- [VO2max Tutorial](../../tutorials/06-vo2max-assessment.md) - Complete workflow guide

---

**Submaximal VO2max test protocol with VT2 and FatMax analysis.**
