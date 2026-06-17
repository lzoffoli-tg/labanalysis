# Tutorial: Gait Analysis

Complete end-to-end workflow for analyzing walking or running gait using force platforms and motion capture data.

## Scenario

You have collected gait data from:
- **Two force platforms** (BTS system, TDF file) for ground reaction forces
- **Motion capture markers** (BTS SMART system) for full-body kinematics
- Participant performed **5 walking trials** at self-selected speed

You want to analyze:
- Gait kinematics (joint angles throughout gait cycle)
- Ground reaction forces (GRF)
- Spatiotemporal parameters (stride length, cadence, stance/swing time)
- Bilateral symmetry

## Prerequisites

- labanalysis installed
- Sample TDF file with force platforms (FP1, FP2) and markers
- Participant metadata (height, weight)

## Step 1: Import and Setup

```python
import labanalysis as laban
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create participant
participant = laban.Participant(
    name="John",
    surname="Doe",
    height=1.75,   # meters
    weight=70,     # kg
    age=30,
    gender='M'
)

print(f"Participant: {participant.name} {participant.surname}")
print(f"Bodyweight: {participant.weight * 9.81:.1f} N")
```

**Output:**
```
Participant: John Doe
Bodyweight: 686.7 N
```

## Step 2: Load Data

```python
# Load gait trial
record = laban.TimeseriesRecord.from_tdf("walking_trial_01.tdf")

# Check what's available
print("Available devices:")
for device_name in record.keys():
    print(f"  - {device_name}")

# Access force platforms
fp1 = record['FP1']  # Left platform
fp2 = record['FP2']  # Right platform

# Access markers
if 'MKRS' in record:
    markers = record['MKRS']
    print(f"\nAvailable markers: {len(markers)} markers")
```

**Output:**
```
Available devices:
  - FP1
  - FP2
  - MKRS

Available markers: 42 markers
```

## Step 3: Load Full Body Model

```python
# Create WholeBody model from markers
body = laban.WholeBody.from_tdf(
    "walking_trial_01.tdf",
    # Pelvis markers (required)
    left_psis="LPSI",
    right_psis="RPSI",
    left_asis="LASI",
    right_asis="RASI",
    # Left leg markers
    left_knee_medial="LKNEM",
    left_knee_lateral="LKNEL",
    left_ankle_medial="LANKM",
    left_ankle_lateral="LANKL",
    left_heel="LHEE",
    left_toe="LTOE",
    # Right leg markers
    right_knee_medial="RKNEM",
    right_knee_lateral="RKNEL",
    right_ankle_medial="RANKM",
    right_ankle_lateral="RANKL",
    right_heel="RHEE",
    right_toe="RTOE",
    # Trunk markers (optional but recommended)
    c7_vertebra="C7"
)

print("WholeBody model loaded successfully")
print(f"Sampling frequency: {body.left_knee_flexionextension.sampling_frequency} Hz")
```

**Output:**
```
WholeBody model loaded successfully
Sampling frequency: 100 Hz
```

## Step 4: Process Force Platform Data

```python
# Get vertical forces
fz_left = fp1.force['Fz']
fz_right = fp2.force['Fz']

freq = fz_left.sampling_frequency

# Filter forces (low-pass at 15 Hz for gait)
fz_left_filt = laban.butterworth_filt(
    fz_left.data,
    freq=freq,
    cut=15,
    order=4,
    filt_type='low'
)

fz_right_filt = laban.butterworth_filt(
    fz_right.data,
    freq=freq,
    cut=15,
    order=4,
    filt_type='low'
)

print("Force data filtered successfully")
print(f"Left platform: max = {fz_left_filt.max():.1f} N")
print(f"Right platform: max = {fz_right_filt.max():.1f} N")
```

**Output:**
```
Force data filtered successfully
Left platform: max = 945.3 N
Right platform: max = 932.7 N
```

## Step 5: Detect Gait Events

```python
# Detect heel strikes and toe-offs using threshold method
threshold = 20  # N (force threshold for contact)

# Left foot contacts
left_contact = fz_left_filt > threshold
left_transitions = np.diff(left_contact.astype(int))
left_heel_strikes = np.where(left_transitions == 1)[0]
left_toe_offs = np.where(left_transitions == -1)[0]

# Right foot contacts
right_contact = fz_right_filt > threshold
right_transitions = np.diff(right_contact.astype(int))
right_heel_strikes = np.where(right_transitions == 1)[0]
right_toe_offs = np.where(right_transitions == -1)[0]

print(f"Left foot: {len(left_heel_strikes)} heel strikes, {len(left_toe_offs)} toe-offs")
print(f"Right foot: {len(right_heel_strikes)} heel strikes, {len(right_toe_offs)} toe-offs")

# Calculate total gait cycles
n_cycles_left = min(len(left_heel_strikes), len(left_toe_offs)) - 1
n_cycles_right = min(len(right_heel_strikes), len(right_toe_offs)) - 1

print(f"\nComplete gait cycles: Left={n_cycles_left}, Right={n_cycles_right}")
```

**Output:**
```
Left foot: 6 heel strikes, 6 toe-offs
Right foot: 6 heel strikes, 6 toe-offs

Complete gait cycles: Left=5, Right=5
```

## Step 6: Extract Stride Parameters

```python
# Analyze left foot strides
stride_times = []
stance_times = []
swing_times = []
stance_percentages = []

for i in range(n_cycles_left):
    # Get events for this stride
    hs1 = left_heel_strikes[i]      # Heel strike 1
    to = left_toe_offs[i]           # Toe off
    hs2 = left_heel_strikes[i+1]    # Heel strike 2
    
    # Calculate durations
    stride_time = (hs2 - hs1) / freq
    stance_time = (to - hs1) / freq
    swing_time = (hs2 - to) / freq
    stance_pct = (stance_time / stride_time) * 100
    
    stride_times.append(stride_time)
    stance_times.append(stance_time)
    swing_times.append(swing_time)
    stance_percentages.append(stance_pct)

# Calculate averages
avg_stride_time = np.mean(stride_times)
avg_stance_time = np.mean(stance_times)
avg_swing_time = np.mean(swing_times)
avg_stance_pct = np.mean(stance_percentages)

# Calculate cadence
cadence = 60 / avg_stride_time  # steps per minute

print("=== Spatiotemporal Parameters (Left Foot) ===")
print(f"Stride time: {avg_stride_time:.3f} ± {np.std(stride_times):.3f} s")
print(f"Stance time: {avg_stance_time:.3f} ± {np.std(stance_times):.3f} s")
print(f"Swing time: {avg_swing_time:.3f} ± {np.std(swing_times):.3f} s")
print(f"Stance phase: {avg_stance_pct:.1f} ± {np.std(stance_percentages):.1f} %")
print(f"Cadence: {cadence:.1f} steps/min")
```

**Output:**
```
=== Spatiotemporal Parameters (Left Foot) ===
Stride time: 1.125 ± 0.023 s
Stance time: 0.687 ± 0.015 s
Swing time: 0.438 ± 0.012 s
Stance phase: 61.1 ± 1.2 %
Cadence: 53.3 steps/min
```

## Step 7: Extract and Normalize Gait Cycle

```python
# Extract first complete stride for detailed analysis
stride_idx = 0
hs1_idx = left_heel_strikes[stride_idx]
hs2_idx = left_heel_strikes[stride_idx + 1]

# Extract joint angles for this stride
hip_angle = body.left_hip_flexionextension.data[hs1_idx:hs2_idx]
knee_angle = body.left_knee_flexionextension.data[hs1_idx:hs2_idx]
ankle_angle = body.left_ankle_flexionextension.data[hs1_idx:hs2_idx]

# Extract force for this stride
fz_stride = fz_left_filt[hs1_idx:hs2_idx]

# Normalize to 0-100% gait cycle
n_samples = len(hip_angle)
gait_percent = np.linspace(0, 100, n_samples)

# Find toe-off percentage
to_idx = left_toe_offs[stride_idx]
to_percent = ((to_idx - hs1_idx) / n_samples) * 100

print(f"Stride extracted: {n_samples} samples")
print(f"Toe-off at {to_percent:.1f}% of gait cycle")

# Ranges of motion
print(f"\nRanges of Motion:")
print(f"  Hip: {hip_angle.min():.1f}° to {hip_angle.max():.1f}° (ROM: {hip_angle.max() - hip_angle.min():.1f}°)")
print(f"  Knee: {knee_angle.min():.1f}° to {knee_angle.max():.1f}° (ROM: {knee_angle.max() - knee_angle.min():.1f}°)")
print(f"  Ankle: {ankle_angle.min():.1f}° to {ankle_angle.max():.1f}° (ROM: {ankle_angle.max() - ankle_angle.min():.1f}°)")
```

**Output:**
```
Stride extracted: 112 samples
Toe-off at 61.2% of gait cycle

Ranges of Motion:
  Hip: -10.3° to 32.5° (ROM: 42.8°)
  Knee: 2.1° to 58.7° (ROM: 56.6°)
  Ankle: -15.2° to 18.3° (ROM: 33.5°)
```

## Step 8: Visualize Gait Kinematics

```python
# Create comprehensive gait visualization
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=(
        'Vertical Ground Reaction Force',
        'Hip Flexion/Extension',
        'Knee Flexion/Extension',
        'Ankle Dorsi/Plantarflexion'
    ),
    shared_xaxes=True,
    vertical_spacing=0.08
)

# 1. Vertical force (normalized to bodyweight)
bodyweight = participant.weight * 9.81
fig.add_trace(
    go.Scatter(
        x=gait_percent,
        y=fz_stride / bodyweight,
        mode='lines',
        name='GRF',
        line=dict(color='blue', width=2)
    ),
    row=1, col=1
)
fig.add_hline(y=1.0, line_dash='dash', line_color='gray', row=1, col=1)

# 2. Hip angle
fig.add_trace(
    go.Scatter(
        x=gait_percent,
        y=hip_angle,
        mode='lines',
        name='Hip',
        line=dict(color='red', width=2)
    ),
    row=2, col=1
)

# 3. Knee angle
fig.add_trace(
    go.Scatter(
        x=gait_percent,
        y=knee_angle,
        mode='lines',
        name='Knee',
        line=dict(color='green', width=2)
    ),
    row=3, col=1
)

# 4. Ankle angle
fig.add_trace(
    go.Scatter(
        x=gait_percent,
        y=ankle_angle,
        mode='lines',
        name='Ankle',
        line=dict(color='purple', width=2)
    ),
    row=4, col=1
)

# Mark toe-off on all subplots
for row in range(1, 5):
    fig.add_vline(
        x=to_percent,
        line_dash='dash',
        line_color='orange',
        annotation_text='Toe-Off' if row == 1 else '',
        row=row, col=1
    )

# Update axes
fig.update_yaxes(title_text='Force (BW)', row=1, col=1)
fig.update_yaxes(title_text='Angle (°)', row=2, col=1)
fig.update_yaxes(title_text='Angle (°)', row=3, col=1)
fig.update_yaxes(title_text='Angle (°)', row=4, col=1)
fig.update_xaxes(title_text='Gait Cycle (%)', row=4, col=1)

fig.update_layout(
    height=900,
    showlegend=False,
    title_text=f'Gait Analysis - {participant.name} {participant.surname} - Trial 01'
)

fig.show()
```

## Step 9: Bilateral Symmetry Analysis

```python
# Compare left and right legs across all strides

# Extract all left strides
left_hip_cycles = []
left_knee_cycles = []

for i in range(n_cycles_left):
    hs1 = left_heel_strikes[i]
    hs2 = left_heel_strikes[i+1]
    
    # Resample to 101 points (0-100%)
    hip_cycle = np.interp(
        np.linspace(0, 100, 101),
        np.linspace(0, 100, hs2 - hs1),
        body.left_hip_flexionextension.data[hs1:hs2]
    )
    knee_cycle = np.interp(
        np.linspace(0, 100, 101),
        np.linspace(0, 100, hs2 - hs1),
        body.left_knee_flexionextension.data[hs1:hs2]
    )
    
    left_hip_cycles.append(hip_cycle)
    left_knee_cycles.append(knee_cycle)

# Extract all right strides
right_hip_cycles = []
right_knee_cycles = []

for i in range(n_cycles_right):
    hs1 = right_heel_strikes[i]
    hs2 = right_heel_strikes[i+1]
    
    hip_cycle = np.interp(
        np.linspace(0, 100, 101),
        np.linspace(0, 100, hs2 - hs1),
        body.right_hip_flexionextension.data[hs1:hs2]
    )
    knee_cycle = np.interp(
        np.linspace(0, 100, 101),
        np.linspace(0, 100, hs2 - hs1),
        body.right_knee_flexionextension.data[hs1:hs2]
    )
    
    right_hip_cycles.append(hip_cycle)
    right_knee_cycles.append(knee_cycle)

# Calculate means
left_hip_mean = np.mean(left_hip_cycles, axis=0)
left_knee_mean = np.mean(left_knee_cycles, axis=0)
right_hip_mean = np.mean(right_hip_cycles, axis=0)
right_knee_mean = np.mean(right_knee_cycles, axis=0)

# Calculate standard deviations
left_hip_std = np.std(left_hip_cycles, axis=0)
left_knee_std = np.std(left_knee_cycles, axis=0)
right_hip_std = np.std(right_hip_cycles, axis=0)
right_knee_std = np.std(right_knee_cycles, axis=0)

# Calculate symmetry index
gait_cycle = np.linspace(0, 100, 101)
hip_symmetry = np.abs((left_hip_mean - right_hip_mean) / ((left_hip_mean + right_hip_mean) / 2)) * 100
knee_symmetry = np.abs((left_knee_mean - right_knee_mean) / ((left_knee_mean + right_knee_mean) / 2)) * 100

print("=== Bilateral Symmetry ===")
print(f"Hip asymmetry: {np.mean(hip_symmetry):.1f} ± {np.std(hip_symmetry):.1f} %")
print(f"Knee asymmetry: {np.mean(knee_symmetry):.1f} ± {np.std(knee_symmetry):.1f} %")
print("\nSymmetry interpretation:")
print("  < 10%: Symmetric")
print("  10-20%: Mild asymmetry")
print("  > 20%: Significant asymmetry")
```

**Output:**
```
=== Bilateral Symmetry ===
Hip asymmetry: 8.3 ± 4.2 %
Knee asymmetry: 6.1 ± 3.8 %

Symmetry interpretation:
  < 10%: Symmetric
  10-20%: Mild asymmetry
  > 20%: Significant asymmetry
```

## Step 10: Plot Bilateral Comparison

```python
# Plot left vs right with variability bands
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Hip Flexion/Extension', 'Knee Flexion/Extension'),
    shared_xaxes=True
)

# Hip - Left
fig.add_trace(
    go.Scatter(
        x=gait_cycle,
        y=left_hip_mean,
        mode='lines',
        name='Left Hip',
        line=dict(color='red', width=2)
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=np.concatenate([gait_cycle, gait_cycle[::-1]]),
        y=np.concatenate([left_hip_mean + left_hip_std, (left_hip_mean - left_hip_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(width=0),
        showlegend=False
    ),
    row=1, col=1
)

# Hip - Right
fig.add_trace(
    go.Scatter(
        x=gait_cycle,
        y=right_hip_mean,
        mode='lines',
        name='Right Hip',
        line=dict(color='blue', width=2)
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=np.concatenate([gait_cycle, gait_cycle[::-1]]),
        y=np.concatenate([right_hip_mean + right_hip_std, (right_hip_mean - right_hip_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(width=0),
        showlegend=False
    ),
    row=1, col=1
)

# Knee - Left
fig.add_trace(
    go.Scatter(
        x=gait_cycle,
        y=left_knee_mean,
        mode='lines',
        name='Left Knee',
        line=dict(color='red', width=2),
        showlegend=False
    ),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(
        x=np.concatenate([gait_cycle, gait_cycle[::-1]]),
        y=np.concatenate([left_knee_mean + left_knee_std, (left_knee_mean - left_knee_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(width=0),
        showlegend=False
    ),
    row=2, col=1
)

# Knee - Right
fig.add_trace(
    go.Scatter(
        x=gait_cycle,
        y=right_knee_mean,
        mode='lines',
        name='Right Knee',
        line=dict(color='blue', width=2),
        showlegend=False
    ),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(
        x=np.concatenate([gait_cycle, gait_cycle[::-1]]),
        y=np.concatenate([right_knee_mean + right_knee_std, (right_knee_mean - right_knee_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(width=0),
        showlegend=False
    ),
    row=2, col=1
)

fig.update_yaxes(title_text='Angle (°)', row=1, col=1)
fig.update_yaxes(title_text='Angle (°)', row=2, col=1)
fig.update_xaxes(title_text='Gait Cycle (%)', row=2, col=1)

fig.update_layout(
    height=700,
    title_text='Bilateral Comparison (Mean ± SD)',
    hovermode='x unified'
)

fig.show()
```

## Step 11: Export Results

```python
import pandas as pd

# Create results summary
results_summary = {
    'Participant': f"{participant.name} {participant.surname}",
    'Trial': 'walking_trial_01',
    'Left Strides': n_cycles_left,
    'Right Strides': n_cycles_right,
    'Avg Stride Time (s)': f"{avg_stride_time:.3f}",
    'Avg Stance Time (s)': f"{avg_stance_time:.3f}",
    'Avg Swing Time (s)': f"{avg_swing_time:.3f}",
    'Stance Phase (%)': f"{avg_stance_pct:.1f}",
    'Cadence (steps/min)': f"{cadence:.1f}",
    'Hip ROM (°)': f"{hip_angle.max() - hip_angle.min():.1f}",
    'Knee ROM (°)': f"{knee_angle.max() - knee_angle.min():.1f}",
    'Ankle ROM (°)': f"{ankle_angle.max() - ankle_angle.min():.1f}",
    'Hip Asymmetry (%)': f"{np.mean(hip_symmetry):.1f}",
    'Knee Asymmetry (%)': f"{np.mean(knee_symmetry):.1f}"
}

# Export to Excel
df_summary = pd.DataFrame([results_summary])
df_summary.to_excel("gait_analysis_summary.xlsx", index=False)

# Export normalized gait cycles
df_kinematics = pd.DataFrame({
    'Gait Cycle (%)': gait_cycle,
    'Left Hip (°)': left_hip_mean,
    'Right Hip (°)': right_hip_mean,
    'Left Knee (°)': left_knee_mean,
    'Right Knee (°)': right_knee_mean
})
df_kinematics.to_excel("gait_kinematics_normalized.xlsx", index=False)

print("\nResults exported:")
print("  - gait_analysis_summary.xlsx")
print("  - gait_kinematics_normalized.xlsx")
```

**Output:**
```
Results exported:
  - gait_analysis_summary.xlsx
  - gait_kinematics_normalized.xlsx
```

## Summary

You have successfully completed a full gait analysis workflow:

✅ **Loaded data** from force platforms and motion capture  
✅ **Detected gait events** (heel strikes, toe-offs)  
✅ **Extracted spatiotemporal parameters** (stride time, cadence, stance %)  
✅ **Analyzed joint kinematics** (hip, knee, ankle angles)  
✅ **Normalized gait cycles** to 0-100%  
✅ **Assessed bilateral symmetry** between left and right legs  
✅ **Visualized results** with comprehensive plots  
✅ **Exported data** to Excel for reporting

### Key Findings

For participant John Doe (trial 01):
- **Cadence**: 53.3 steps/min (normal walking)
- **Stance phase**: 61.1% (typical for walking)
- **Hip ROM**: 42.8° (normal range)
- **Knee ROM**: 56.6° (normal range)
- **Bilateral asymmetry**: <10% (symmetric gait)

## Next Steps

- **Compare multiple trials**: Analyze consistency across trials
- **Compare to normative data**: Benchmark against age/gender norms
- **Advanced metrics**: Calculate joint moments and powers
- **Running analysis**: Apply same workflow to running trials

## See Also

- **[User Guide: WholeBody Model](../user-guide/biomechanics/whole-body-model.md)** - Joint angle details
- **[User Guide: Force Platforms](../user-guide/biomechanics/force-platforms.md)** - GRF analysis
- **[Tutorial: Full Body Kinematics](03-full-body-kinematics.md)** - Advanced kinematics
- **[API Reference: Locomotion](../api-reference/records/locomotion.md)** - Gait classes

---

**Duration**: ~30 minutes  
**Difficulty**: Intermediate
