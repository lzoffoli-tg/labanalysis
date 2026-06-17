# Protocol Reports

Guide to creating automated athlete assessment reports with normative reference bands using `bars_with_normative_bands()`.

## Overview

`bars_with_normative_bands()` creates bar charts with semi-transparent normative reference bands overlaid in the background. This function automatically assigns color and rank to each bar based on where it falls within defined normative intervals, making it ideal for athlete assessment reports, patient evaluations, and performance tracking.

**Key Features:**
- **Automatic rank assignment** - Bars colored based on normative intervals
- **Transparent bands** - Visual reference ranges in background
- **Flexible orientation** - Vertical or horizontal bars
- **Pattern grouping** - Compare multiple conditions (pre/post, left/right)
- **Enriched output** - Returns figure plus DataFrame with assigned ranks/colors

**Common Use Cases:**
- Athlete test battery reports (CMJ, sprint, VO2max)
- Pre-post intervention comparisons
- Team performance dashboards
- Clinical patient assessments vs normative data
- Rehabilitation progress tracking

## Quick Reference

### Basic Example

```python
import labanalysis as laban
import pandas as pd
import numpy as np

# Athlete test results
tests = ["Jump Height", "Sprint 10m", "VO2max"]
scores = [35, 1.85, 58]

# Define normative intervals
intervals = pd.DataFrame({
    "Rank": ["Low", "Medium", "High"],
    "Lower": [0, 30, 45],
    "Upper": [29, 44, 100],
    "Color": ["#FF6B6B", "#FFD93D", "#6BCF7F"]  # Red, Yellow, Green
})

# Create figure
fig, dfr = laban.bars_with_normative_bands(
    data_frame=None,
    xarr=tests,
    yarr=scores,
    orientation="v",
    unit="cm / s / ml/kg/min",
    intervals=intervals
)

fig.show()

# Output:
# - Bar chart with 3 bars
# - Jump Height: green (High rank, 35 in [30, 44])
# - Sprint: yellow (Medium, 1.85 in [0, 29] - needs adjustment!)
# - VO2max: green (High, 58 in [45, 100])

# Check assigned ranks
print(dfr[['_Rank', '_Color']])
#    _Rank    _Color
# 0  Medium  #FFD93D
# 1  Low     #FF6B6B
# 2  High    #6BCF7F
```

### From DataFrame

```python
# Multi-athlete comparison
df = pd.DataFrame({
    'Athlete': ['Alice', 'Bob', 'Charlie'],
    'CMJ_Height': [38, 32, 41],
})

# Normative data for CMJ
cmj_norms = pd.DataFrame({
    "Rank": ["Below Average", "Average", "Above Average", "Elite"],
    "Lower": [0, 25, 35, 45],
    "Upper": [24, 34, 44, np.inf],  # np.inf for open-ended upper bound
    "Color": ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]
})

fig, dfr = laban.bars_with_normative_bands(
    data_frame=df,
    xarr='Athlete',
    yarr='CMJ_Height',
    orientation='v',
    unit='cm',
    intervals=cmj_norms
)

fig.update_layout(title="Team CMJ Assessment")
fig.show()
```

## Function Parameters

### data_frame

**Type:** `pd.DataFrame` or `None`
**Default:** `None`

Container for structured data:

```python
# Option 1: DataFrame (recommended for labeled data)
df = pd.DataFrame({
    'test_name': ['CMJ', 'SJ', 'Sprint'],
    'result': [35, 33, 1.85]
})

fig, dfr = laban.bars_with_normative_bands(
    data_frame=df,
    xarr='test_name',  # Column name
    yarr='result'       # Column name
)

# Option 2: Arrays directly
tests = ['CMJ', 'SJ', 'Sprint']
results = [35, 33, 1.85]

fig, dfr = laban.bars_with_normative_bands(
    data_frame=None,
    xarr=tests,   # Array
    yarr=results  # Array
)
```

### xarr and yarr

**Type:** `str` (column name) or `np.ndarray`/`list` (array)

Define bar positions and heights:

```python
# Categorical x-axis, numerical y-axis (most common)
xarr = ["Test 1", "Test 2", "Test 3"]
yarr = [45, 52, 38]

# From DataFrame
fig, dfr = laban.bars_with_normative_bands(
    data_frame=df,
    xarr='test_name',  # Categories
    yarr='score'        # Values
)
```

**Label Handling:**
- X-axis labels taken from `xarr` values
- Y-axis shows numerical scale
- Text on bars shows formatted `yarr` values + unit

### patterns

**Type:** `str` (column name), `np.ndarray`/`list`, or `None`
**Default:** `None`

Group bars by pattern for comparison:

```python
# Example: Pre-post training comparison
df = pd.DataFrame({
    'Test': ['CMJ', 'CMJ', 'SJ', 'SJ', 'Sprint', 'Sprint'],
    'Score': [32, 36, 30, 34, 1.95, 1.82],
    'Phase': ['Pre', 'Post', 'Pre', 'Post', 'Pre', 'Post']
})

fig, dfr = laban.bars_with_normative_bands(
    data_frame=df,
    xarr='Test',
    yarr='Score',
    patterns='Phase',  # Group by Pre/Post
    intervals=norms
)

# Output: Grouped bars (Pre and Post side-by-side for each test)
```

**Visual Effect:**
- `patterns=None`: Single bar per x-category
- `patterns` specified: Multiple bars per x-category with different fill patterns
- Each pattern gets unique visual style (solid, striped, dotted)

**Common Pattern Use Cases:**
| Comparison | Pattern Variable |
|------------|------------------|
| Pre-post intervention | 'Pre' vs 'Post' |
| Left-right asymmetry | 'Left' vs 'Right' |
| Multiple athletes | 'Athlete_A', 'Athlete_B' |
| Test-retest | 'Session_1', 'Session_2' |

### orientation

**Type:** `Literal["h", "v"]`
**Default:** `"v"` (vertical)

Bar direction:

```python
# Vertical bars (default, most common)
fig, dfr = laban.bars_with_normative_bands(
    xarr=['Test A', 'Test B'],
    yarr=[45, 52],
    orientation='v'
)
# X-axis: test names
# Y-axis: scores (with normative bands horizontal)

# Horizontal bars
fig, dfr = laban.bars_with_normative_bands(
    xarr=[45, 52],  # Now these are the values
    yarr=['Test A', 'Test B'],  # Now these are labels
    orientation='h'
)
# X-axis: scores (with normative bands vertical)
# Y-axis: test names
```

**When to Use Each:**

| Orientation | Best For |
|-------------|----------|
| Vertical (`'v'`) | Few tests (< 8), standard reports, easier to read labels |
| Horizontal (`'h'`) | Many tests (> 8), long test names, ranking visualization |

### unit

**Type:** `str` or `None`
**Default:** `None`

Measurement unit displayed on bars and axis:

```python
# Single unit
fig, dfr = laban.bars_with_normative_bands(
    xarr=['CMJ', 'SJ'],
    yarr=[35, 33],
    unit='cm'
)
# Bar labels: "35 cm", "33 cm"
# Y-axis title: "cm"

# Mixed units (not ideal, but possible)
fig, dfr = laban.bars_with_normative_bands(
    xarr=['Jump', 'Sprint', 'VO2max'],
    yarr=[35, 1.85, 58],
    unit='cm / s / ml/kg/min'  # Shows all units
)
# Bar labels: "35 cm / s / ml/kg/min" (misleading!)

# Better approach for mixed units: separate charts
```

**Best Practice:**
- Use `unit` when all bars have same measurement unit
- Omit `unit` for dimensionless or mixed-unit data
- For mixed units, create separate charts per test type

### intervals

**Type:** `pd.DataFrame`
**Default:** `pd.DataFrame()` (empty)

**Critical Parameter:** Defines normative reference bands and automatic color/rank assignment.

**Required Columns:**

```python
intervals = pd.DataFrame({
    "Rank": [str],      # Interpretation label (e.g., "Low", "Average", "High")
    "Lower": [float],   # Lower bound of interval
    "Upper": [float],   # Upper bound of interval
    "Color": [str]      # Color code (hex, RGB, or named color)
})
```

**Example:**

```python
import numpy as np

intervals = pd.DataFrame({
    "Rank": ["Poor", "Fair", "Good", "Excellent"],
    "Lower": [0, 20, 30, 40],
    "Upper": [19, 29, 39, np.inf],  # np.inf = no upper limit
    "Color": ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]
})

# Interpretation:
# Value in [0, 19]   → Poor (red)
# Value in [20, 29]  → Fair (orange)
# Value in [30, 39]  → Good (blue)
# Value ≥ 40         → Excellent (green)
```

**Color Specifications:**

```python
# Hex colors (most common)
"Color": ["#FF0000", "#00FF00", "#0000FF"]

# RGB format
"Color": ["rgb(255, 0, 0)", "rgb(0, 255, 0)", "rgb(0, 0, 255)"]

# Named colors
"Color": ["red", "green", "blue"]

# RGBA (with transparency, though bar opacity is controlled separately)
"Color": ["rgba(255, 0, 0, 0.5)", "rgba(0, 255, 0, 0.5)"]
```

**Open-Ended Intervals:**

```python
# No lower bound (use -np.inf)
intervals = pd.DataFrame({
    "Rank": ["Very Low", "Low", "Normal", "High"],
    "Lower": [-np.inf, 10, 20, 30],
    "Upper": [9, 19, 29, np.inf],
    "Color": ["#C0392B", "#E67E22", "#27AE60", "#2980B9"]
})
```

## Creating Normative Intervals

### Interval DataFrame Structure

**Minimal Valid Structure:**

```python
intervals = pd.DataFrame({
    "Rank": ["Low", "High"],
    "Lower": [0, 50],
    "Upper": [49, 100],
    "Color": ["red", "green"]
})
```

**Required:**
- At least 1 row (1 interval)
- Exactly 4 columns with exact names: `Rank`, `Lower`, `Upper`, `Color`
- `Lower` and `Upper` must be numeric (int or float)
- Intervals should cover all possible data values (gaps leave bars uncolored)

**Example with Gaps (Problematic):**

```python
# Problematic: value = 45 falls in gap [40, 50]
intervals = pd.DataFrame({
    "Rank": ["Low", "High"],
    "Lower": [0, 50],
    "Upper": [40, 100],  # Gap: (40, 50) undefined
    "Color": ["red", "green"]
})

fig, dfr = laban.bars_with_normative_bands(yarr=[45], ...)
print(dfr['_Rank'])  # None (no rank assigned!)
```

**Fix: Ensure Continuous Coverage:**

```python
intervals = pd.DataFrame({
    "Rank": ["Low", "Medium", "High"],
    "Lower": [0, 40, 50],
    "Upper": [40, 50, 100],  # Continuous, no gaps
    "Color": ["#E74C3C", "#F39C12", "#2ECC71"]
})
```

### Percentile-Based Norms

Create normative intervals from reference population:

```python
import numpy as np
import pandas as pd

# Reference population data (n=500 athletes)
reference_cmj = np.random.normal(35, 8, 500)  # Mean=35cm, SD=8cm

# Define percentile-based intervals
percentiles = [0, 25, 50, 75, 100]
bounds = np.percentile(reference_cmj, percentiles)

intervals = pd.DataFrame({
    "Rank": ["Bottom 25%", "Below Median", "Above Median", "Top 25%"],
    "Lower": bounds[:-1],
    "Upper": bounds[1:],
    "Color": ["#E74C3C", "#F39C12", "#F1C40F", "#2ECC71"]
})

print(intervals)
#           Rank      Lower      Upper     Color
# 0  Bottom 25%  16.789123  30.456789  #E74C3C
# 1 Below Median  30.456789  35.123456  #F39C12
# 2 Above Median  35.123456  39.876543  #F1C40F
# 3     Top 25%  39.876543  53.210987  #2ECC71
```

### Age and Gender-Specific Norms

```python
# Example: CMJ normative data by age group and gender
def get_cmj_norms(age, gender):
    """Return age/gender-specific CMJ norms."""
    norms_database = {
        ('18-25', 'M'): [25, 35, 45, 55],  # Percentile cutoffs
        ('18-25', 'F'): [18, 28, 38, 48],
        ('26-35', 'M'): [22, 32, 42, 52],
        ('26-35', 'F'): [15, 25, 35, 45],
    }
    
    cutoffs = norms_database.get((age, gender), [20, 30, 40, 50])
    
    return pd.DataFrame({
        "Rank": ["Below Average", "Average", "Above Average", "Elite"],
        "Lower": [0] + cutoffs[:-1],
        "Upper": cutoffs + [np.inf],
        "Color": ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]
    })

# Use for 23-year-old male athlete
athlete_age = '18-25'
athlete_gender = 'M'
intervals = get_cmj_norms(athlete_age, athlete_gender)

fig, dfr = laban.bars_with_normative_bands(
    xarr=['CMJ'],
    yarr=[42],
    intervals=intervals,
    unit='cm'
)
# Output: "Above Average" rank (42 in [35, 45])
```

### Sport-Specific Normative Data

```python
# Different norms for different sports
sport_norms = {
    'Basketball': pd.DataFrame({
        "Rank": ["Low", "Average", "High"],
        "Lower": [0, 40, 50],
        "Upper": [39, 49, np.inf],
        "Color": ["#E74C3C", "#F39C12", "#2ECC71"]
    }),
    'Soccer': pd.DataFrame({
        "Rank": ["Low", "Average", "High"],
        "Lower": [0, 35, 45],
        "Upper": [34, 44, np.inf],
        "Color": ["#E74C3C", "#F39C12", "#2ECC71"]
    }),
}

# Apply sport-specific norms
athlete_sport = 'Basketball'
fig, dfr = laban.bars_with_normative_bands(
    xarr=['CMJ'],
    yarr=[48],
    intervals=sport_norms[athlete_sport],
    unit='cm'
)
# Rank: "Average" for basketball (different from soccer!)
```

## Return Values

### Figure Object

**Type:** `plotly.graph_objects.FigureWidget`

Interactive Plotly figure with:
- Bar chart
- Normative bands (semi-transparent rectangles)
- Bar labels (value + unit)
- Legend for normative ranks

```python
fig, dfr = laban.bars_with_normative_bands(...)

# Display
fig.show()

# Customize after creation
fig.update_layout(title="Athlete Assessment", font_size=14)

# Export
fig.write_html("report.html")
fig.write_image("report.png", width=800, height=600)
```

### Enriched DataFrame

**Type:** `pd.DataFrame`

Original data plus computed columns:

```python
fig, dfr = laban.bars_with_normative_bands(
    xarr=['CMJ', 'SJ', 'Sprint'],
    yarr=[35, 33, 1.85],
    intervals=intervals
)

print(dfr)
#      X     Y    _Text    _Rank  _Color
# 0  CMJ  35.0  35.0 cm  Average  #F39C12
# 1   SJ  33.0  33.0 cm  Average  #F39C12
# 2  Sprint  1.85  1.85 s     High  #2ECC71
```

**Computed Columns:**

**`_Text`** - Formatted value with unit
```python
dfr['_Text']
# ['35.0 cm', '33.0 cm', '1.85 s']
```

**`_Rank`** - Assigned rank from intervals
```python
dfr['_Rank']
# ['Average', 'Average', 'High']
```

**`_Color`** - Assigned color from intervals
```python
dfr['_Color']
# ['#F39C12', '#F39C12', '#2ECC71']
```

**Using Enriched Data:**

```python
# Filter by rank
high_performers = dfr[dfr['_Rank'] == 'High']

# Export to CSV
dfr.to_csv("athlete_results_with_ranks.csv", index=False)

# Count ranks
rank_counts = dfr['_Rank'].value_counts()
print(rank_counts)
# High       2
# Average    1
```

## Practical Applications

### Application 1: Athlete Test Battery

Single athlete, multiple tests:

```python
import labanalysis as laban
import pandas as pd
import numpy as np

# Test results
df = pd.DataFrame({
    'Test': ['CMJ', 'Squat Jump', '10m Sprint', '30m Sprint', 'VO2max'],
    'Score': [38, 36, 1.75, 4.12, 62]
})

# Generic athletic norms (simplified)
norms = pd.DataFrame({
    "Rank": ["Below Average", "Average", "Above Average", "Elite"],
    "Lower": [0, 30, 40, 50],
    "Upper": [29, 39, 49, np.inf],
    "Color": ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]
})

fig, dfr = laban.bars_with_normative_bands(
    data_frame=df,
    xarr='Test',
    yarr='Score',
    orientation='v',
    intervals=norms
)

fig.update_layout(
    title="Athlete Performance Profile - John Doe",
    font_size=12,
    showlegend=True,
    height=500
)

fig.show()

# Export
fig.write_html("athlete_report.html")
```

### Application 2: Pre-Post Intervention

Training program effects visualization:

```python
# 8-week training program results
df = pd.DataFrame({
    'Test': ['CMJ', 'CMJ', 'SJ', 'SJ', '10m', '10m'],
    'Score': [32, 36, 30, 34, 1.95, 1.82],
    'Phase': ['Pre', 'Post', 'Pre', 'Post', 'Pre', 'Post']
})

# Norms
norms = pd.DataFrame({
    "Rank": ["Low", "Average", "High"],
    "Lower": [0, 30, 40],
    "Upper": [29, 39, np.inf],
    "Color": ["#E74C3C", "#F39C12", "#2ECC71"]
})

fig, dfr = laban.bars_with_normative_bands(
    data_frame=df,
    xarr='Test',
    yarr='Score',
    patterns='Phase',  # Group by Pre/Post
    intervals=norms
)

fig.update_layout(
    title="Training Program Effects (8 weeks)",
    barmode='group'  # Side-by-side comparison
)

fig.show()

# Analyze improvements
pre = dfr[dfr['Phase'] == 'Pre']
post = dfr[dfr['Phase'] == 'Post']
improvements = post['Score'].values - pre['Score'].values
print(f"CMJ: +{improvements[0]:.1f} cm")
print(f"SJ: +{improvements[1]:.1f} cm")
print(f"Sprint: {improvements[2]:.2f} s (faster)")
```

### Application 3: Team Report

Multiple athletes, same test:

```python
# Team CMJ assessment
df = pd.DataFrame({
    'Athlete': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'CMJ': [42, 35, 38, 31, 45]
})

norms = pd.DataFrame({
    "Rank": ["Below Average", "Average", "Above Average", "Elite"],
    "Lower": [0, 30, 40, 50],
    "Upper": [29, 39, 49, np.inf],
    "Color": ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]
})

fig, dfr = laban.bars_with_normative_bands(
    data_frame=df,
    xarr='Athlete',
    yarr='CMJ',
    unit='cm',
    intervals=norms
)

fig.update_layout(title="Team CMJ Performance")
fig.show()

# Identify athletes needing attention
needs_work = dfr[dfr['_Rank'] == 'Below Average']['Athlete'].tolist()
print(f"Focus training on: {needs_work}")
```

### Application 4: Clinical Rehabilitation

Patient progress vs normative data:

```python
# Knee flexion range of motion (degrees)
# Tracked across 6 weeks of rehab
df = pd.DataFrame({
    'Week': [0, 2, 4, 6],
    'ROM': [45, 65, 85, 110]
})

# Clinical norms for knee flexion
clinical_norms = pd.DataFrame({
    "Rank": ["Severe Limitation", "Moderate", "Mild", "Normal"],
    "Lower": [0, 60, 90, 120],
    "Upper": [59, 89, 119, 180],
    "Color": ["#C0392B", "#E67E22", "#F39C12", "#27AE60"]
})

fig, dfr = laban.bars_with_normative_bands(
    data_frame=df,
    xarr='Week',
    yarr='ROM',
    unit='degrees',
    intervals=clinical_norms,
    orientation='v'
)

fig.update_layout(
    title="Knee Flexion ROM - Rehabilitation Progress",
    xaxis_title="Weeks Post-Surgery"
)

fig.show()

# Progress tracking
print("Progress:")
for i, row in dfr.iterrows():
    print(f"Week {row['Week']}: {row['ROM']}° ({row['_Rank']})")
```

## Customizing the Output

### Title and Font

```python
fig, dfr = laban.bars_with_normative_bands(...)

# Update title
fig.update_layout(
    title="Custom Report Title",
    title_font_size=20,
    title_font_color="darkblue",
    title_x=0.5  # Center title
)

# Update font globally
fig.update_layout(font_family="Arial", font_size=14)
```

### Bar Appearance

```python
# Modify bar colors manually (overrides normative colors)
fig.update_traces(
    marker_color=['red', 'blue', 'green'],
    marker_line_color='black',
    marker_line_width=2
)

# Bar width and gap
fig.update_traces(width=0.6)  # Thinner bars
fig.update_layout(bargap=0.2)  # Gap between bars

# Corner radius (already set by default to 30%)
fig.update_traces(marker_cornerradius='10%')  # Less rounded
```

### Layout Adjustments

```python
# Figure size
fig.update_layout(width=1000, height=600)

# Margins
fig.update_layout(margin=dict(l=50, r=50, t=80, b=50))

# Grid and background
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_layout(plot_bgcolor='white')
```

### Export for Reports

```python
# HTML for interactive viewing
fig.write_html("athlete_report.html", include_plotlyjs='cdn')

# PNG for static reports (requires kaleido)
fig.write_image(
    "report.png",
    width=1200,
    height=700,
    scale=2  # 2x resolution for print quality
)

# PDF for official documents
fig.write_image("report.pdf", width=1200, height=700)
```

## Best Practices

### Selecting Appropriate Normative Data

**Match Population Characteristics:**
- Age group
- Gender
- Sport/activity level
- Geographic region (if relevant)
- Measurement protocol (must be identical)

**Example:**
```python
# Don't mix norms
# ✗ WRONG: Apply elite basketball norms to recreational soccer players
# ✓ CORRECT: Use age/gender-matched recreational norms
```

### Age, Gender, and Sport Matching

```python
def get_athlete_norms(age, gender, sport):
    """Retrieve appropriate normative data."""
    # Implement logic to select correct norms
    if sport == 'Basketball' and gender == 'M' and 18 <= age <= 25:
        return basketball_male_18_25_norms
    elif sport == 'Soccer' and gender == 'F' and 18 <= age <= 25:
        return soccer_female_18_25_norms
    else:
        return generic_norms  # Fallback
```

### Color Scheme Accessibility

**Use colorblind-friendly palettes:**

```python
# Red-green colorblind friendly
intervals = pd.DataFrame({
    "Rank": ["Low", "Medium", "High"],
    "Lower": [0, 30, 50],
    "Upper": [29, 49, np.inf],
    "Color": ["#D55E00", "#F0E442", "#0072B2"]  # Orange, yellow, blue
})
```

**Online tools:**
- [ColorBrewer](https://colorbrewer2.org/) - Colorblind-safe palettes
- [Viz Palette](https://projects.susielu.com/viz-palette) - Test palettes for colorblindness

### Bar Count Limitations

**Recommendation:** Keep bars under 12 for readability

```python
# ✓ GOOD: 6 bars, easy to compare
tests = ['CMJ', 'SJ', 'Sprint', 'VO2max', 'Plank', 'Grip']

# ✗ BAD: 20 bars, cluttered
# Solution: Split into multiple charts or use horizontal orientation
```

**For many items:**
- Use horizontal orientation (`orientation='h'`)
- Split into multiple charts by category
- Consider alternative visualizations (tables, radar charts)

## Troubleshooting

**Problem:** Bars not colored correctly

```python
# Check if intervals cover all data values
yarr = [15, 35, 55]
intervals = pd.DataFrame({
    "Rank": ["Low", "High"],
    "Lower": [0, 50],
    "Upper": [30, 100],  # Gap: (30, 50) undefined
    "Color": ["red", "green"]
})

fig, dfr = laban.bars_with_normative_bands(yarr=yarr, intervals=intervals, ...)
print(dfr['_Rank'])
# [Low, None, High]  # Middle bar has no rank!

# Solution: Ensure continuous coverage
intervals = pd.DataFrame({
    "Rank": ["Low", "Medium", "High"],
    "Lower": [0, 30, 50],
    "Upper": [30, 50, 100],  # Continuous
    "Color": ["red", "yellow", "green"]
})
```

**Problem:** Values outside all intervals

```python
# Value = 120, but max interval upper bound = 100
yarr = [120]
intervals = pd.DataFrame({
    "Rank": ["Low", "High"],
    "Lower": [0, 50],
    "Upper": [49, 100],  # Value 120 is outside
    "Color": ["red", "green"]
})

# Solution: Use np.inf for open upper bound
intervals.loc[intervals['Rank'] == 'High', 'Upper'] = np.inf
```

**Problem:** Pattern display issues

```python
# Patterns not showing distinct visual styles
fig, dfr = laban.bars_with_normative_bands(
    xarr=['A', 'A', 'B', 'B'],
    yarr=[10, 15, 20, 25],
    patterns=['P1', 'P2', 'P1', 'P2'],
    ...
)

# Solution: Check Plotly version supports pattern_shape parameter
# Update plotly: pip install --upgrade plotly
```

**Problem:** Legend cluttered or confusing

```python
# Too many normative bands in legend
fig, dfr = laban.bars_with_normative_bands(...)

# Solution: Simplify intervals or hide normative legend
fig.update_layout(showlegend=False)

# Or show only bar legend, hide normative bands
for trace in fig.data:
    if 'normative' in trace.name.lower():
        trace.showlegend = False
```

## See Also

- [Plotly Basics](plotly-basics.md) - Plotly integration fundamentals
- [Custom Figures](custom-figures.md) - Advanced bar chart customization
- [Comparison Plots](comparison-plots.md) - Method validation plots
- [Test Protocols](../test-protocols/overview.md) - Standardized testing procedures
- [Jump Tests](../test-protocols/jump-tests.md) - CMJ and SJ testing protocols

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
