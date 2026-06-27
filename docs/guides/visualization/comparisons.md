# Comparison Plots

Comprehensive guide to method agreement analysis using `plot_comparisons()` for validation studies, model evaluation, and reliability assessment.

## Overview

`plot_comparisons()` creates a comprehensive validation figure with 5 subplots that analyze agreement between two measurement methods or datasets. This function is essential for method validation studies, machine learning model evaluation, and reliability analyses where you need to assess how well two measurement approaches agree.

**The 5-Subplot Structure:**
1. **Statistics Table** - RMSE, MAPE, R², t-tests, bias, and limits of agreement
2. **True vs Predicted** - Scatter plot with regression and identity lines
3. **Bland-Altman Plot** - Mean-difference plot with limits of agreement
4. **Error Distribution** - Histogram of prediction errors
5. **Link Plot** - Paired observations with color-coded differences

**Use Cases:**
- Validating new measurement devices against gold standards
- Evaluating machine learning model predictions
- Assessing inter-rater reliability
- Test-retest reliability studies
- Comparing measurement methods (e.g., force platform vs video analysis)

## Quick Reference

### From DataFrame

```python
import labanalysis as laban
import pandas as pd
import numpy as np

# Example: Jump height validation (force platform vs video analysis)
np.random.seed(42)
df = pd.DataFrame({
    'force_platform': [32.5, 35.2, 28.9, 40.1, 33.7, 29.8, 36.4, 31.2],
    'video_analysis': [31.8, 36.0, 28.5, 39.5, 34.2, 30.1, 35.8, 31.9],
    'athlete': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
})

# Create comprehensive comparison figure
fig = laban.plot_comparisons(
    data_frame=df,
    true_data='video_analysis',      # Reference method (column name)
    pred_data='force_platform',      # Method being validated
    color_data='athlete'             # Group by athlete (optional)
)

fig.show()

# Output: Interactive 5-subplot figure
# - Bias: -0.26 cm (force platform slightly underestimates)
# - R²: 0.98 (excellent correlation)
# - RMSE: 0.72 cm (typical error)
```

### From Arrays

```python
import numpy as np

# Generate synthetic data
np.random.seed(42)
true = np.array([10, 20, 30, 40, 50, 60, 70, 80])
pred = true + np.random.normal(0, 3, size=8)

# Create comparison figure
fig = laban.plot_comparisons(
    data_frame=None,
    true_data=true,
    pred_data=pred
)

fig.show()
```

## Understanding the 5 Subplots

### Subplot 1: Summary Statistics Table

The table presents key validation metrics:

**#** - Number of observations (sample size)

**RMSE (Root Mean Square Error)**
```
RMSE = sqrt(mean((predicted - true)²))
```
- Average magnitude of errors
- Same units as measured variable
- Lower is better
- Example: RMSE = 2.3 cm means typical error is ±2.3 cm

**MAPE (Mean Absolute Percentage Error)**
```
MAPE = mean(|predicted - true| / true) × 100%
```
- Error as percentage of true value
- Scale-independent metric
- Example: MAPE = 5.2% means typical 5.2% relative error

**R² (Coefficient of Determination)**
```
R² = correlation(true, predicted)²
```
- Proportion of variance explained (0-1)
- R² = 1.0 means perfect correlation
- R² > 0.9 indicates strong correlation
- **Warning:** High R² doesn't mean good agreement (see Interpreting Results)

**T_Paired (Paired t-test)**
- Tests if means differ between true and predicted
- df: degrees of freedom
- t: t-statistic
- p: p-value (p < 0.05 indicates significant difference)
- Used when same subjects measured twice

**T_Independent (Independent t-test)**
- Tests if distributions differ
- Less common for validation studies
- Useful when comparing different groups

**Bias (Systematic Error)**
- Mean difference: `mean(predicted - true)`
- Positive: method overestimates
- Negative: method underestimates
- Zero: no systematic error

**LOA_Lower and LOA_Upper (Limits of Agreement)**
- Range containing most differences
- Default: 95% of differences fall within these limits
- Parametric: Bias ± 1.96×SD
- Non-parametric: 2.5th and 97.5th percentiles

### Subplot 2: True vs Predicted Scatter

Scatter plot showing relationship between reference and test methods:

**Identity Line (black dashed)** - Perfect agreement (y = x)
- Points on line: perfect match
- Points above line: method overestimates
- Points below line: method underestimates

**Regression Line (implied by scatter pattern)**
- Slope = 1.0: no proportional bias
- Slope ≠ 1.0: errors scale with magnitude

**Deviation Patterns:**
- **Random scatter around identity** - Good agreement, random errors only
- **Systematic offset** - Fixed bias (all points above or below line)
- **Fan shape** - Proportional bias (errors increase with magnitude)
- **Curved pattern** - Non-linear relationship

### Subplot 3: Bland-Altman Plot

Mean-difference plot for assessing agreement:

**X-axis:** Mean of two methods `(true + predicted) / 2`
**Y-axis:** Difference `predicted - true`

**Bias Line (horizontal trend line):**
- Shows systematic error
- Flat line at y=0: no bias
- Sloped line: proportional bias (errors depend on magnitude)

**Limits of Agreement (LOA) Bands:**
- Upper and lower dashed lines
- Contains 95% of differences (by default)
- Narrower bands: better agreement

**Trend Line (dotted):**
- Linear fit to differences
- Flat (slope ≈ 0): fixed bias only
- Sloped: proportional bias present

**Interpretation:**
- Points within LOA: acceptable agreement
- Points outside LOA: outliers or poor agreement
- Funnel shape: heteroscedasticity (errors vary with magnitude)

### Subplot 4: Error Distribution

Histogram showing distribution of prediction errors:

**Normal distribution:**
- Symmetric bell curve
- Indicates random errors
- Parametric statistics appropriate

**Skewed distribution:**
- Asymmetric histogram
- Suggests systematic patterns
- Consider non-parametric analysis

**Outliers:**
- Isolated bars far from center
- May indicate measurement errors
- Investigate individual cases

### Subplot 5: Link Plot

Paired observations connected by lines:

**X-axis:** "TRUE" and "PREDICTED" categories
**Y-axis:** Measured values

**Line colors (gradient):**
- Colors indicate magnitude of difference
- Default color scale: 'temps' (blue→red)
- Blue: small difference (good agreement)
- Red: large difference (poor agreement)

**Patterns:**
- Horizontal lines: perfect agreement
- All lines sloped same direction: systematic bias
- Mixed slopes: random errors
- Long lines: large disagreements

## Function Parameters

### data_frame

**Type:** `pd.DataFrame` or `None`

Controls how data is provided:

```python
# Option 1: Use DataFrame (recommended for labeled data)
df = pd.DataFrame({'ref': [10, 20, 30], 'test': [9, 21, 29]})
fig = laban.plot_comparisons(
    data_frame=df,
    true_data='ref',    # Column name (string)
    pred_data='test'    # Column name (string)
)

# Option 2: Use arrays directly
true_arr = np.array([10, 20, 30])
pred_arr = np.array([9, 21, 29])
fig = laban.plot_comparisons(
    data_frame=None,
    true_data=true_arr,  # NumPy array
    pred_data=pred_arr   # NumPy array
)
```

**Best Practice:** Use DataFrame when data has meaningful labels or grouping variables.

### true_data and pred_data

**Type:** `str` (column name) or `np.ndarray` (array)

**true_data** - Reference method or gold standard
**pred_data** - Method being validated or predictions

```python
# Validation study: gold standard vs new device
fig = laban.plot_comparisons(
    data_frame=df,
    true_data='isokinetic_dynamometer',  # Gold standard
    pred_data='portable_device'          # Being validated
)

# ML model evaluation: actual vs predicted
fig = laban.plot_comparisons(
    data_frame=test_df,
    true_data='actual_jump_height',
    pred_data='model_predictions'
)
```

**Data Format Requirements:**
- Arrays must have same length
- Values must be numeric
- No NaN or infinite values

**Error Handling:**

```python
# Check for issues before plotting
assert len(true_data) == len(pred_data), "Arrays must have same length"
assert not np.any(np.isnan(true_data)), "Remove NaN values first"
assert not np.any(np.isnan(pred_data)), "Remove NaN values first"
```

### color_data

**Type:** `str` (column name), `np.ndarray`, or `None`

Groups data by category with color-coded points:

```python
# Example: Multi-athlete analysis
df = pd.DataFrame({
    'true': [30, 32, 35, 37, 28, 31],
    'pred': [29, 33, 34, 38, 27, 32],
    'athlete': ['A', 'A', 'B', 'B', 'C', 'C']
})

fig = laban.plot_comparisons(
    data_frame=df,
    true_data='true',
    pred_data='pred',
    color_data='athlete'  # Each athlete gets unique color
)

# Output: Separate statistics computed per athlete
# - Table shows columns for 'A', 'B', 'C'
# - Scatter points colored by athlete
```

**When to Use:**
- Comparing multiple athletes/patients
- Analyzing different test conditions
- Investigating sub-group differences
- Test-retest by session

**None (default):**
- All data treated as single group
- Single column in statistics table
- One color for all points

### confidence

**Type:** `float` (0 to 1)
**Default:** `0.95`

Confidence level for limits of agreement:

```python
# 95% confidence (default, most common)
fig = laban.plot_comparisons(..., confidence=0.95)
# LOA contains 95% of differences

# 99% confidence (stricter)
fig = laban.plot_comparisons(..., confidence=0.99)
# LOA contains 99% of differences (wider bands)

# 90% confidence (looser)
fig = laban.plot_comparisons(..., confidence=0.90)
# LOA contains 90% of differences (narrower bands)
```

**When to Adjust:**
- **0.95 (default):** Standard for most validation studies
- **0.99:** When stricter agreement required (clinical safety)
- **0.90:** Exploratory analyses

**Effect on Bland-Altman:**
```
Parametric LOA = Bias ± z×SD
- 95%: z = 1.96
- 99%: z = 2.576
- 90%: z = 1.645
```

### parametric

**Type:** `bool`
**Default:** `False`

Statistical approach for calculating limits of agreement:

```python
# Non-parametric (default, robust)
fig = laban.plot_comparisons(..., parametric=False)
# LOA based on percentiles (2.5th, 97.5th for 95% CI)
# Bias = median(differences)

# Parametric (assumes normal distribution)
fig = laban.plot_comparisons(..., parametric=True)
# LOA based on mean ± 1.96×SD
# Bias = mean(differences)
```

**When to Use Each:**

| Condition | Recommended |
|-----------|-------------|
| Normal error distribution | Parametric |
| Skewed error distribution | Non-parametric |
| Outliers present | Non-parametric |
| Small sample size (< 30) | Non-parametric |
| Large sample size (> 100) | Either (similar results) |
| Publication requirement | Check journal guidelines |

**Checking Normality:**

```python
from scipy.stats import shapiro

# Compute errors
errors = pred_data - true_data

# Shapiro-Wilk test for normality
stat, p_value = shapiro(errors)
if p_value > 0.05:
    print("Errors are normally distributed → use parametric=True")
else:
    print("Errors are not normal → use parametric=False")
```

### color_scale

**Type:** `str`
**Default:** `'temps'`

Plotly color scale for link plot gradient:

```python
# Default: temps (blue to red)
fig = laban.plot_comparisons(..., color_scale='temps')

# Alternative color scales
fig = laban.plot_comparisons(..., color_scale='Viridis')  # Colorblind-friendly
fig = laban.plot_comparisons(..., color_scale='RdYlGn')   # Red-Yellow-Green
fig = laban.plot_comparisons(..., color_scale='Blues')     # Single hue
```

**Popular Options:**
- `'Viridis'` - Perceptually uniform, colorblind-friendly
- `'Plasma'` - Purple to yellow
- `'RdBu'` - Red-blue diverging
- `'Greys'` - Grayscale (for publications)

**Full list:** See [Plotly color scales](https://plotly.com/python/builtin-colorscales/)

## Practical Applications

### Application 1: Method Validation

Validate a new portable device against laboratory gold standard:

```python
import labanalysis as laban
import pandas as pd
import numpy as np

# Data: countermovement jump height (cm)
np.random.seed(123)
n = 50
gold_standard = np.random.uniform(25, 45, n)
portable = gold_standard + np.random.normal(0, 1.5, n)  # Small systematic error

df = pd.DataFrame({
    'lab_system': gold_standard,
    'portable_device': portable
})

# Create validation figure
fig = laban.plot_comparisons(
    data_frame=df,
    true_data='lab_system',
    pred_data='portable_device',
    parametric=True,
    confidence=0.95
)

fig.update_layout(title="Portable Device Validation Study (n=50)")
fig.show()

# Interpretation:
# - RMSE: 1.48 cm (acceptable for field testing)
# - Bias: -0.12 cm (negligible systematic error)
# - R²: 0.99 (excellent correlation)
# - LOA: ±2.9 cm (95% of measurements within this range)

# Decision: Device validated for field use
```

**Acceptance Criteria Example:**
```python
# Define acceptable limits
RMSE_threshold = 2.0  # cm
Bias_threshold = 1.0  # cm
R2_threshold = 0.95

# Extract metrics (manual check or automate)
if RMSE < RMSE_threshold and abs(Bias) < Bias_threshold and R2 > R2_threshold:
    print("✓ Device validated")
else:
    print("✗ Device does not meet criteria")
```

### Application 2: Machine Learning Model Evaluation

Evaluate regression model performance:

```python
import labanalysis as laban
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Example: Predict VO2max from jump performance
data = pd.read_csv("athlete_data.csv")
X = data[['cmj_height', 'squat_jump', 'reactive_strength']]
y = data['vo2max']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation figure
test_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
})

fig = laban.plot_comparisons(
    data_frame=test_df,
    true_data='actual',
    pred_data='predicted',
    parametric=True
)

fig.update_layout(title="Random Forest Model: VO2max Prediction")
fig.show()

# Metrics guide model improvement:
# - High R² but large bias → model systematically off, check feature engineering
# - Low R² → poor model fit, try different algorithm or add features
# - Large RMSE with R² = 1 → proportional bias, check for outliers
```

### Application 3: Inter-Rater Reliability

Assess agreement between two raters:

```python
# Two physiotherapists measuring joint angle (degrees)
df = pd.DataFrame({
    'rater_1': [45, 52, 38, 61, 55, 48, 42, 50],
    'rater_2': [47, 51, 40, 59, 57, 46, 43, 51],
    'patient': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
})

fig = laban.plot_comparisons(
    data_frame=df,
    true_data='rater_1',  # Neither is "true", just first reference
    pred_data='rater_2',
    color_data='patient',
    parametric=False  # May have outliers
)

fig.update_layout(title="Inter-Rater Reliability: Knee Flexion Angle")
fig.show()

# Interpretation:
# - Bias close to 0: no systematic difference between raters
# - Narrow LOA: raters agree well
# - T_Paired p > 0.05: no significant difference
# - Conclusion: Raters are interchangeable
```

### Application 4: Test-Retest Reliability

Evaluate measurement consistency across sessions:

```python
# Athletes tested on two separate days
df = pd.DataFrame({
    'day_1': [34.2, 38.5, 29.8, 42.1, 36.7],
    'day_2': [33.8, 39.1, 30.2, 41.5, 37.2],
    'athlete': ['A', 'B', 'C', 'D', 'E']
})

fig = laban.plot_comparisons(
    data_frame=df,
    true_data='day_1',
    pred_data='day_2',
    color_data='athlete',
    parametric=True
)

fig.update_layout(title="Test-Retest Reliability: CMJ Height (7-day interval)")
fig.show()

# Metrics:
# - RMSE: typical day-to-day variation
# - LOA: minimum detectable change
# - R²: consistency across athletes

# Application: Changes > LOA are real improvements (not measurement noise)
```

## Interpreting Results

### RMSE and MAPE

**RMSE (Root Mean Square Error):**
- Typical magnitude of error
- Same units as measured variable
- Sensitive to outliers

**Acceptable Thresholds (context-dependent):**
| Measurement | Typical RMSE | Excellent |
|-------------|--------------|-----------|
| Jump height (cm) | < 2.0 | < 1.0 |
| Sprint time (s) | < 0.05 | < 0.02 |
| Heart rate (bpm) | < 5 | < 3 |
| Joint angle (°) | < 3 | < 2 |

**MAPE (Mean Absolute Percentage Error):**
- Relative error (scale-independent)
- Useful for comparing different measurements

**Guidelines:**
- MAPE < 5%: Excellent agreement
- MAPE 5-10%: Good agreement
- MAPE 10-20%: Moderate agreement
- MAPE > 20%: Poor agreement

**Limitation:** MAPE undefined when true value = 0

### R² Interpretation

**Value Ranges:**
- R² = 1.0: Perfect correlation
- R² > 0.9: Strong correlation
- R² = 0.7-0.9: Moderate correlation
- R² < 0.7: Weak correlation

**CRITICAL WARNING:**
High R² does NOT guarantee good agreement!

```python
# Example: Perfect correlation but poor agreement
true = np.array([10, 20, 30, 40, 50])
pred = true * 2  # Systematic doubling

fig = laban.plot_comparisons(data_frame=None, true_data=true, pred_data=pred)
# R² = 1.0 (perfect correlation)
# BUT bias = 30 (massive systematic error!)
```

**Why R² Can Be Misleading:**
- Measures correlation, not agreement
- Insensitive to systematic bias
- Can be high even with large errors

**Use R² with other metrics:**
- High R² + low bias + narrow LOA → good agreement
- High R² + high bias → systematic error (proportional or fixed)
- Low R² → poor correlation AND poor agreement

### Bland-Altman Analysis

**Bias (Systematic Error):**

```python
# Positive bias: method overestimates
# Example: Bias = +2.5 cm
# Interpretation: Test method measures 2.5 cm higher on average

# Negative bias: method underestimates
# Example: Bias = -1.8 bpm
# Interpretation: Device reads 1.8 bpm lower than reference

# Near-zero bias: no systematic error
# Example: Bias = -0.1 kg
# Interpretation: Negligible systematic difference
```

**Fixed vs Proportional Bias:**

**Fixed bias:** Constant offset at all magnitudes
- Bland-Altman: horizontal trend line
- Can be corrected with simple offset: `corrected = measured - bias`

**Proportional bias:** Error scales with magnitude
- Bland-Altman: sloped trend line
- Larger values have larger errors
- Correction requires scaling: `corrected = measured / slope`

**Limits of Agreement (LOA):**

LOA defines the range containing 95% of differences:

```
95% LOA = [Bias - 1.96×SD, Bias + 1.96×SD]  (parametric)
```

**Interpretation:**
- Narrow LOA: good agreement
- Wide LOA: poor agreement
- Values outside LOA: outliers or measurement errors

**Clinical Significance:**
LOA must be clinically acceptable:

```python
# Example: Blood pressure measurement
# LOA = ±10 mmHg → clinically acceptable
# LOA = ±30 mmHg → too wide, device unreliable
```

**Minimum Detectable Change (MDC):**
LOA defines MDC for test-retest reliability:

```python
# Test-retest LOA = ±3 cm
# MDC = 3 cm
# Changes < 3 cm likely measurement noise
# Changes > 3 cm likely real improvements
```

### T-Test Results

**Paired T-Test:**
- Tests if mean difference ≠ 0
- Appropriate for validation studies (same subjects)

**Interpretation:**
```python
# p < 0.05: Significant difference (means differ)
# → Systematic bias present
# → Investigate and possibly correct

# p ≥ 0.05: No significant difference
# → No evidence of systematic bias
# → Methods agree on average
```

**Independent T-Test:**
- Tests if distributions differ
- Less relevant for validation
- Useful for comparing groups

**Statistical vs Clinical Significance:**
- Large sample → small differences become "significant" (p < 0.05)
- Always check bias magnitude, not just p-value
- Example: p < 0.001 but bias = 0.1 cm → statistically significant but clinically negligible

## Best Practices

### Minimum Sample Size

**Recommendations:**
| Study Type | Minimum n | Recommended n |
|------------|-----------|---------------|
| Method validation | 30 | 50-100 |
| Inter-rater reliability | 20 | 30-50 |
| Test-retest reliability | 15 | 25-40 |
| ML model evaluation | 50 | 100-500 |

**Why larger samples:**
- Narrower confidence intervals
- Better detection of proportional bias
- More reliable LOA estimates
- Increased statistical power

### Outlier Handling

**Detection:**

```python
# Identify outliers using standardized residuals
errors = pred_data - true_data
z_scores = (errors - np.mean(errors)) / np.std(errors)
outliers = np.abs(z_scores) > 3

print(f"Outliers: {np.sum(outliers)} / {len(errors)}")
```

**Strategies:**

1. **Investigate first** - Outliers may indicate real measurement errors
2. **Do not automatically remove** - Report with and without outliers
3. **Use non-parametric analysis** - Robust to outliers

```python
# Robust analysis with outliers
fig = laban.plot_comparisons(..., parametric=False)

# Sensitivity analysis: compare with/without outliers
fig_all = laban.plot_comparisons(data_frame=df, ...)
fig_clean = laban.plot_comparisons(data_frame=df[~outliers], ...)
```

### Parametric vs Non-Parametric Selection

**Decision Flowchart:**

```
1. Plot error histogram (Subplot 4)
   ├─ Symmetric, bell-shaped → parametric=True
   └─ Skewed or multi-modal → parametric=False

2. Check sample size
   ├─ n < 30 → parametric=False (safer)
   └─ n > 100 → either (similar results)

3. Test normality (optional)
   └─ Shapiro-Wilk p > 0.05 → parametric=True
```

**When in doubt:** Use non-parametric (more conservative)

### Reporting Standards

Follow Bland & Altman (1986) guidelines:

**Minimum Report Contents:**
1. Sample size (n)
2. Bias (mean difference)
3. Limits of agreement (95% LOA)
4. Confidence intervals for bias and LOA (if n < 100)
5. Bland-Altman plot
6. Statement of clinical acceptability

**Example Report:**

```
"Validation of portable jump system (n=50 athletes). 
Mean bias = -0.12 cm (95% CI: -0.45 to +0.21 cm), 
indicating the portable system slightly underestimates 
jump height compared to the laboratory force platform. 
95% limits of agreement were -2.9 to +2.7 cm. 
No proportional bias detected (Bland-Altman slope p=0.23). 
Agreement is clinically acceptable for field testing."
```

**Citation:**
Bland JM, Altman DG. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet*, 1(8476), 307-310.

## Real-World Examples

### Example 1: Jump Height Validation

Validate smartphone app vs force platform:

```python
import labanalysis as laban
import pandas as pd

# Collect data from 40 athletes
df = pd.read_csv("jump_validation_data.csv")
# Columns: athlete_id, force_platform_height, smartphone_height

fig = laban.plot_comparisons(
    data_frame=df,
    true_data='force_platform_height',
    pred_data='smartphone_height',
    parametric=True,
    confidence=0.95
)

fig.update_layout(
    title="Smartphone App Validation: CMJ Height (n=40)",
    width=1400,
    height=700
)

# Save for publication
fig.write_image("validation_study_figure.png", width=1400, height=700, scale=2)
fig.write_html("validation_study_interactive.html")

# Results:
# RMSE: 1.82 cm
# Bias: +0.53 cm (app overestimates slightly)
# R²: 0.96
# LOA: -2.76 to +3.82 cm
# Conclusion: Acceptable for non-critical field testing
```

### Example 2: Heart Rate Monitor Comparison

Compare wearable vs chest strap across activity types:

```python
# Data includes rest, walking, running conditions
df = pd.DataFrame({
    'chest_strap': [65, 68, 72, 95, 102, 98, 155, 162, 158],
    'wearable': [64, 70, 71, 93, 105, 96, 148, 159, 155],
    'activity': ['rest', 'rest', 'rest', 'walk', 'walk', 'walk', 'run', 'run', 'run']
})

fig = laban.plot_comparisons(
    data_frame=df,
    true_data='chest_strap',
    pred_data='wearable',
    color_data='activity',  # Separate analysis per activity
    parametric=False  # May have outliers during transitions
)

fig.update_layout(title="Wearable HR Monitor Validation Across Activities")
fig.show()

# Insight: Check if bias/LOA differ by activity type
# - Rest: narrow LOA (low HR, stable)
# - Run: wider LOA (high HR, more motion artifact)
```

## Troubleshooting

**Problem:** Array length mismatch error

```python
# Error: ValueError: Arrays must have same length

# Solution: Check array lengths
print(f"True: {len(true_data)}, Pred: {len(pred_data)}")

# Ensure equal length before calling
assert len(true_data) == len(pred_data), "Fix data alignment"
```

**Problem:** Empty plots or figure doesn't display

```python
# Solution 1: Check for NaN values
print(f"NaN in true: {np.sum(np.isnan(true_data))}")
print(f"NaN in pred: {np.sum(np.isnan(pred_data))}")

# Remove NaN before plotting
mask = ~(np.isnan(true_data) | np.isnan(pred_data))
fig = laban.plot_comparisons(
    data_frame=None,
    true_data=true_data[mask],
    pred_data=pred_data[mask]
)
```

**Problem:** Outliers dominate plot axes

```python
# Solution: Set axis ranges manually after creation
fig = laban.plot_comparisons(...)

# Limit Bland-Altman y-axis to ±3 SD
errors = pred_data - true_data
limit = 3 * np.std(errors)
fig.update_yaxes(range=[-limit, limit], row=1, col=3)

fig.show()
```

**Problem:** Performance slow with large datasets (>10k points)

```python
# Solution 1: Downsample for visualization
stride = len(data) // 5000  # Keep ~5000 points
fig = laban.plot_comparisons(
    data_frame=df[::stride],  # Every nth row
    true_data='true',
    pred_data='pred'
)

# Solution 2: Compute metrics on full data, plot subset
# (Advanced: requires manual metric calculation)
```

## See Also

- [Plotly Basics](plotly-basics.md) - Introduction to Plotly integration
- [Custom Figures](custom-figures.md) - Customize plot appearance
- [Protocol Reports](protocol-reports.md) - Normative band charts
- [Regression Modeling](../modeling/regression.md) - Building prediction models
- [Bland & Altman 1986 Paper](https://www.ncbi.nlm.nih.gov/pubmed/2868172) - Original method description

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
