# Report Generation

Generate professional analysis reports from labanalysis test results with automated metrics, visualizations, and normative comparisons.

## Overview

Labanalysis test protocols provide built-in report generation capabilities:

- **Automated metrics calculation** - All relevant KPIs computed automatically
- **Normative comparisons** - Compare against reference data
- **Visual summaries** - Charts and graphs included
- **Export formats** - PDF, HTML, Excel, or custom

## Quick Reference

```python
import labanalysis as laban

# Load jump test
data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])
jump = laban.SingleJump(**data)

# Get summary metrics
summary = jump.output_metrics
print(summary)

# Export to Excel
summary.to_excel("jump_report.xlsx")
```

---

## Test Protocol Reports

### Jump Test Reports

```python
import labanalysis as laban

# Load and analyze jump
data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])
jump = laban.SingleJump(**data)

# Get comprehensive metrics
metrics = jump.output_metrics

# Metrics include:
# - jump_height_m: Jump height
# - flight_time_s: Time in air
# - peak_force_N: Maximum force
# - rfd_N_s: Rate of force development
# - eccentric_duration_s: Eccentric phase time
# - concentric_duration_s: Concentric phase time
# - eccentric_peak_velocity_m_s: Downward velocity
# - concentric_peak_velocity_m_s: Upward velocity
```

### Gait Analysis Reports

```python
import labanalysis as laban
import pandas as pd

# Load running trial
data = laban.read_tdf("running.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])
running = laban.RunningExercise(algorithm='kinetics', **data)

# Get metrics for all cycles
all_metrics = []
for cycle in running.cycles:
    metrics = cycle.output_metrics
    all_metrics.append(metrics)

# Combine into report
report = pd.concat(all_metrics, ignore_index=True)

# Calculate summary statistics
summary = {
    'total_cycles': len(running.cycles),
    'mean_cycle_time': report['cycle_time_s'].mean(),
    'std_cycle_time': report['cycle_time_s'].std(),
    'mean_stance_pct': report['stance_%'].mean() if 'stance_%' in report.columns else None,
    'mean_flight_pct': report['flight_%'].mean() if 'flight_%' in report.columns else None,
}

print(pd.DataFrame([summary]))
```

### Balance Test Reports

```python
import labanalysis as laban

# Load balance test
data = laban.read_tdf("balance.tdf", forceplatform_keys=[".*"])
balance = laban.UprightPosture(**data)

# Get sway metrics
metrics = balance.output_metrics

# Metrics include:
# - cop_path_length_mm: Total COP displacement
# - cop_mean_velocity_mm_s: Average sway velocity
# - cop_area_mm2: 95% confidence ellipse area
# - cop_range_ml_mm: Medial-lateral sway range
# - cop_range_ap_mm: Anterior-posterior sway range
```

### Strength Test Reports

```python
import labanalysis as laban

# Load isokinetic test
data = laban.read_biostrength("isokinetic_knee.csv")
test = laban.Isokinetic1RMTest(**data)

# Get strength metrics
metrics = test.output_metrics

# Metrics include:
# - peak_torque_Nm: Maximum torque
# - angle_at_peak_deg: Joint angle at peak
# - work_J: Total work performed
# - power_W: Average power
# - predicted_1rm_kg: Estimated 1RM
```

---

## Normative Data Comparison

### Using Normative Intervals

```python
import labanalysis as laban
from labanalysis.plotting import bars_with_normative_bands

# Load jump test
data = laban.read_tdf("athlete_cmj.tdf", forceplatform_keys=[".*"])
jump = laban.SingleJump(**data)

# Get jump height
jump_height = jump.output_metrics['jump_height_m'].values[0]

# Define normative ranges (example for elite athletes)
normative = {
    'excellent': (0.50, float('inf')),  # > 50 cm
    'good': (0.40, 0.50),               # 40-50 cm
    'average': (0.30, 0.40),            # 30-40 cm
    'below_average': (0.20, 0.30),      # 20-30 cm
    'poor': (0.0, 0.20),                # < 20 cm
}

# Determine athlete's category
for category, (lower, upper) in normative.items():
    if lower <= jump_height < upper:
        print(f"Athlete performance: {category}")
        print(f"Jump height: {jump_height:.3f} m")
        break

# Visualize with normative bands
fig = bars_with_normative_bands(
    values=[jump_height],
    labels=['Athlete'],
    normative_intervals=normative,
    title="CMJ Performance vs. Normative Data",
    y_label="Jump Height (m)"
)
fig.show()
```

### Age/Gender-Specific Norms

```python
import labanalysis as laban
import pandas as pd

# Load normative data (example structure)
norms = pd.DataFrame({
    'age_group': ['18-25', '18-25', '26-35', '26-35'],
    'gender': ['M', 'F', 'M', 'F'],
    'jump_height_mean': [0.45, 0.35, 0.42, 0.32],
    'jump_height_std': [0.05, 0.04, 0.06, 0.05],
})

# Athlete data
athlete = {
    'age': 22,
    'gender': 'M',
    'jump_height': 0.48
}

# Find appropriate norm
norm = norms[
    (norms['age_group'] == '18-25') & 
    (norms['gender'] == athlete['gender'])
].iloc[0]

# Calculate z-score
z_score = (athlete['jump_height'] - norm['jump_height_mean']) / norm['jump_height_std']

print(f"Athlete jump height: {athlete['jump_height']:.3f} m")
print(f"Age/gender norm: {norm['jump_height_mean']:.3f} ± {norm['jump_height_std']:.3f} m")
print(f"Z-score: {z_score:.2f}")

if z_score > 1.5:
    print("Performance: Excellent (>1.5 SD above mean)")
elif z_score > 0.5:
    print("Performance: Above average")
elif z_score > -0.5:
    print("Performance: Average")
else:
    print("Performance: Below average")
```

---

## Multi-Test Reports

### Athlete Battery Report

```python
import labanalysis as laban
import pandas as pd

# Define athlete
athlete_id = "ATH001"

# Test 1: CMJ
cmj_data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])
cmj = laban.SingleJump(**cmj_data)
cmj_metrics = cmj.output_metrics
cmj_metrics['test'] = 'CMJ'

# Test 2: Sprint
sprint_data = laban.read_tdf("sprint.tdf", marker_keys=[".*"])
sprint = laban.RunningExercise(algorithm='kinematics', **sprint_data)
sprint_summary = pd.DataFrame([{
    'test': 'Sprint',
    'max_velocity_m_s': max([c.pelvis_center['Y'].to_numpy().max() for c in sprint.cycles]),
    'num_steps': len(sprint.cycles)
}])

# Test 3: Balance
balance_data = laban.read_tdf("balance.tdf", forceplatform_keys=[".*"])
balance = laban.UprightPosture(**balance_data)
balance_metrics = balance.output_metrics
balance_metrics['test'] = 'Balance'

# Combine all tests
report = pd.concat([
    cmj_metrics[['test', 'jump_height_m', 'peak_force_N']],
    sprint_summary,
    balance_metrics[['test', 'cop_path_length_mm', 'cop_mean_velocity_mm_s']]
], ignore_index=True)

# Export
report.to_excel(f"{athlete_id}_battery_report.xlsx", index=False)
print(f"✓ Multi-test report for {athlete_id} exported")
```

### Longitudinal Progress Report

```python
import labanalysis as laban
import pandas as pd
from datetime import datetime

# Define test dates
tests = [
    ('2024-01-15', 'baseline_cmj.tdf'),
    ('2024-02-15', 'week4_cmj.tdf'),
    ('2024-03-15', 'week8_cmj.tdf'),
    ('2024-04-15', 'week12_cmj.tdf'),
]

# Analyze each test
progress = []
for date_str, file in tests:
    data = laban.read_tdf(file, forceplatform_keys=[".*"])
    jump = laban.SingleJump(**data)
    metrics = jump.output_metrics
    metrics['date'] = datetime.strptime(date_str, '%Y-%m-%d')
    progress.append(metrics)

# Combine
progress_df = pd.concat(progress, ignore_index=True)

# Calculate improvements
baseline = progress_df.iloc[0]
latest = progress_df.iloc[-1]

improvement = {
    'metric': [],
    'baseline': [],
    'latest': [],
    'change': [],
    'change_pct': []
}

for metric in ['jump_height_m', 'peak_force_N', 'concentric_peak_velocity_m_s']:
    improvement['metric'].append(metric)
    improvement['baseline'].append(baseline[metric])
    improvement['latest'].append(latest[metric])
    improvement['change'].append(latest[metric] - baseline[metric])
    improvement['change_pct'].append((latest[metric] - baseline[metric]) / baseline[metric] * 100)

improvement_df = pd.DataFrame(improvement)

# Export
with pd.ExcelWriter("progress_report.xlsx") as writer:
    progress_df.to_excel(writer, sheet_name='All Tests', index=False)
    improvement_df.to_excel(writer, sheet_name='Improvements', index=False)

print("✓ Longitudinal progress report exported")
```

---

## Custom Report Templates

### Excel Report with Formatting

```python
import labanalysis as laban
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

# Analyze test
data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])
jump = laban.SingleJump(**data)
metrics = jump.output_metrics

# Export basic report
metrics.to_excel("report.xlsx", index=False)

# Apply formatting
wb = load_workbook("report.xlsx")
ws = wb.active

# Header formatting
header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF")

for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal="center")

# Number formatting
for row in ws.iter_rows(min_row=2):
    for cell in row:
        if isinstance(cell.value, float):
            cell.number_format = '0.000'

# Adjust column widths
for column in ws.columns:
    max_length = max(len(str(cell.value)) for cell in column)
    ws.column_dimensions[column[0].column_letter].width = max_length + 2

wb.save("report_formatted.xlsx")
print("✓ Formatted Excel report created")
```

### HTML Report

```python
import labanalysis as laban
import pandas as pd

# Analyze test
data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])
jump = laban.SingleJump(**data)
metrics = jump.output_metrics

# Create HTML
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Jump Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #366092; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #366092; color: white; }}
        .highlight {{ background-color: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>Countermovement Jump Test Report</h1>
    <h2>Summary Metrics</h2>
    {metrics.to_html(index=False)}
</body>
</html>
"""

# Save
with open("report.html", "w") as f:
    f.write(html)

print("✓ HTML report created")
```

---

## Automated Batch Reports

### Process Multiple Athletes

```python
from pathlib import Path
import labanalysis as laban
import pandas as pd

# Find all test files
test_files = list(Path("athlete_tests/").glob("*_cmj.tdf"))

# Process each athlete
all_reports = []
for file in test_files:
    # Extract athlete ID from filename
    athlete_id = file.stem.replace("_cmj", "")
    
    # Analyze
    data = laban.read_tdf(str(file), forceplatform_keys=[".*"])
    jump = laban.SingleJump(**data)
    metrics = jump.output_metrics
    
    # Add athlete info
    metrics['athlete_id'] = athlete_id
    metrics['test_date'] = file.stat().st_mtime  # File modification time
    
    all_reports.append(metrics)

# Combine all athletes
team_report = pd.concat(all_reports, ignore_index=True)

# Sort by performance
team_report = team_report.sort_values('jump_height_m', ascending=False)

# Export
team_report.to_excel("team_report.xlsx", index=False)
print(f"✓ Processed {len(test_files)} athletes")
```

---

## Visualization in Reports

### Include Plots in Excel

```python
import labanalysis as laban
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

# Analyze test
data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])
jump = laban.SingleJump(**data)

# Create plot
fp = data['left_foot_ground_reaction_force']
time = fp.force.index
force_z = fp.force['Z'].to_numpy()

plt.figure(figsize=(10, 6))
plt.plot(time, force_z)
plt.xlabel('Time (s)')
plt.ylabel('Vertical Force (N)')
plt.title('Ground Reaction Force')
plt.grid(True)
plt.savefig("force_plot.png", dpi=150, bbox_inches='tight')
plt.close()

# Create Excel report
metrics = jump.output_metrics
metrics.to_excel("report_with_plot.xlsx", index=False)

# Insert plot
wb = load_workbook("report_with_plot.xlsx")
ws = wb.active
img = Image("force_plot.png")
ws.add_image(img, 'A10')  # Place plot below data
wb.save("report_with_plot.xlsx")

print("✓ Excel report with plot created")
```

---

## See Also

- [](dataframes.md) - DataFrame export guide
- [](opensim-export.md) - OpenSim file export
- [](../visualization/protocol-reports.md) - Visualization with normative bands
- [](../test-protocols/jump-tests.md) - Jump test protocols

---

**Generate professional analysis reports with automated metrics and visualizations.**
