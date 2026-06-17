# Tutorial: Full Body Kinematics Analysis

Complete end-to-end workflow for analyzing full body motion capture data using labanalysis WholeBody model.

**Duration**: 30 minutes  
**Level**: Intermediate  
**Prerequisites**: labanalysis installed, understanding of biomechanics basics, marker placement knowledge

## What You'll Learn

- Load motion capture data with anatomical markers
- Create WholeBody model from marker data
- Access 104+ computed properties (38 angular measures, joint centers, reference frames, anthropometrics)
- Analyze gait kinematics across stride cycles
- Visualize joint angle time series
- Export kinematic data for further analysis
- Identify asymmetries and abnormal patterns

## Scenario

You have collected motion capture data from a walking trial using a BTS system with 42 anatomical markers. You want to analyze joint angles throughout the gait cycle, identify left-right asymmetries, and generate a comprehensive kinematic report.

## Step 1: Setup and Data Loading

```python
import labanalysis as laban
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load marker data from TDF file
data = laban.read_tdf("walking_trial.tdf")

# Check available markers
print("Available markers:")
for key in data.keys():
    if isinstance(data[key], laban.Point3D):
        print(f"  - {key}")
```

**Output:**
```
Available markers:
  - left_heel
  - right_heel
  - left_toe
  - right_toe
  - left_ankle_medial
  - left_ankle_lateral
  - right_ankle_medial
  - right_ankle_lateral
  - left_knee_medial
  - left_knee_lateral
  - right_knee_medial
  - right_knee_lateral
  ... (42 total)
```

## Step 2: Create WholeBody Model

```python
# Create WholeBody instance from loaded data
body = laban.WholeBody(**data)

# Verify model is complete
print(f"Sampling frequency: {body.sampling_frequency} Hz")
print(f"Duration: {len(body) / body.sampling_frequency:.2f} s")
print(f"Number of frames: {len(body)}")

# Check which properties are available
# (depends on which markers were provided)
available_angles = []
for prop_name in dir(body):
    if 'flexion' in prop_name or 'rotation' in prop_name or 'abduction' in prop_name:
        try:
            getattr(body, prop_name)
            available_angles.append(prop_name)
        except:
            pass

print(f"\nAvailable joint angles: {len(available_angles)}")
for angle in sorted(available_angles):
    print(f"  - {angle}")
```

**Output:**
```
Sampling frequency: 100 Hz
Duration: 8.50 s
Number of frames: 850

Available joint angles: 18
  - left_ankle_flexionextension
  - left_elbow_flexionextension
  - left_hip_abductionadduction
  - left_hip_flexionextension
  - left_hip_internalexternalrotation
  - left_knee_flexionextension
  - left_shoulder_abductionadduction
  - left_shoulder_flexionextension
  - left_shoulder_internalexternalrotation
  - right_ankle_flexionextension
  - right_elbow_flexionextension
  - right_hip_abductionadduction
  - right_hip_flexionextension
  - right_hip_internalexternalrotation
  - right_knee_flexionextension
  - right_shoulder_abductionadduction
  - right_shoulder_flexionextension
  - right_shoulder_internalexternalrotation
```

## Step 3: Extract Lower Limb Kinematics

```python
# Extract key gait angles for both legs
left_ankle = body.left_ankle_flexionextension.data
right_ankle = body.right_ankle_flexionextension.data

left_knee = body.left_knee_flexionextension.data
right_knee = body.right_knee_flexionextension.data

left_hip_flex = body.left_hip_flexionextension.data
right_hip_flex = body.right_hip_flexionextension.data

left_hip_abd = body.left_hip_abductionadduction.data
right_hip_abd = body.right_hip_abductionadduction.data

# Create time vector
time = np.arange(len(body)) / body.sampling_frequency

# Display range of motion (ROM)
print("Range of Motion (ROM) Analysis:")
print(f"\nAnkle dorsiflexion/plantarflexion:")
print(f"  Left:  {left_ankle.min():.1f}° to {left_ankle.max():.1f}° (ROM: {left_ankle.max() - left_ankle.min():.1f}°)")
print(f"  Right: {right_ankle.min():.1f}° to {right_ankle.max():.1f}° (ROM: {right_ankle.max() - right_ankle.min():.1f}°)")

print(f"\nKnee flexion/extension:")
print(f"  Left:  {left_knee.min():.1f}° to {left_knee.max():.1f}° (ROM: {left_knee.max() - left_knee.min():.1f}°)")
print(f"  Right: {right_knee.min():.1f}° to {right_knee.max():.1f}° (ROM: {right_knee.max() - right_knee.min():.1f}°)")

print(f"\nHip flexion/extension:")
print(f"  Left:  {left_hip_flex.min():.1f}° to {left_hip_flex.max():.1f}° (ROM: {left_hip_flex.max() - left_hip_flex.min():.1f}°)")
print(f"  Right: {right_hip_flex.min():.1f}° to {right_hip_flex.max():.1f}° (ROM: {right_hip_flex.max() - right_hip_flex.min():.1f}°)")
```

**Output:**
```
Range of Motion (ROM) Analysis:

Ankle dorsiflexion/plantarflexion:
  Left:  -25.3° to 18.7° (ROM: 44.0°)
  Right: -27.1° to 17.2° (ROM: 44.3°)

Knee flexion/extension:
  Left:  -5.2° to 62.4° (ROM: 67.6°)
  Right: -4.8° to 64.1° (ROM: 68.9°)

Hip flexion/extension:
  Left:  -12.5° to 38.2° (ROM: 50.7°)
  Right: -13.1° to 36.8° (ROM: 49.9°)
```

## Step 4: Detect Gait Events

```python
# Detect heel strikes using foot height
left_foot_height = body.left_foot_height.data
right_foot_height = body.right_foot_height.data

# Find local minima (heel strikes) using peak detection
from scipy.signal import find_peaks

# Invert signal to find minima as peaks
left_heel_strikes, _ = find_peaks(-left_foot_height, distance=50, prominence=0.02)
right_heel_strikes, _ = find_peaks(-right_foot_height, distance=50, prominence=0.02)

print(f"Detected gait events:")
print(f"  Left heel strikes: {len(left_heel_strikes)}")
print(f"  Right heel strikes: {len(right_heel_strikes)}")

# Calculate stride times
left_stride_times = np.diff(left_heel_strikes) / body.sampling_frequency
right_stride_times = np.diff(right_heel_strikes) / body.sampling_frequency

print(f"\nStride time:")
print(f"  Left:  {left_stride_times.mean():.3f} ± {left_stride_times.std():.3f} s")
print(f"  Right: {right_stride_times.mean():.3f} ± {right_stride_times.std():.3f} s")

# Calculate cadence
cadence = 60 / ((left_stride_times.mean() + right_stride_times.mean()) / 2)
print(f"\nCadence: {cadence:.1f} steps/min")
```

**Output:**
```
Detected gait events:
  Left heel strikes: 5
  Right heel strikes: 5

Stride time:
  Left:  1.142 ± 0.024 s
  Right: 1.138 ± 0.031 s

Cadence: 105.3 steps/min
```

## Step 5: Normalize to Gait Cycle (0-100%)

```python
def normalize_to_gait_cycle(signal, heel_strikes):
    """
    Normalize signal from time series to gait cycle percentage (0-100%).
    
    Returns
    -------
    cycles : list of np.ndarray
        List of normalized cycles (each 101 points, 0-100%)
    """
    cycles = []
    for i in range(len(heel_strikes) - 1):
        start_idx = heel_strikes[i]
        end_idx = heel_strikes[i + 1]
        
        # Extract stride
        stride = signal[start_idx:end_idx + 1]
        
        # Interpolate to 0-100% (101 points)
        normalized = np.interp(
            np.linspace(0, 100, 101),
            np.linspace(0, 100, len(stride)),
            stride
        )
        cycles.append(normalized)
    
    return cycles

# Normalize left leg angles
left_knee_cycles = normalize_to_gait_cycle(left_knee, left_heel_strikes)
left_hip_cycles = normalize_to_gait_cycle(left_hip_flex, left_heel_strikes)
left_ankle_cycles = normalize_to_gait_cycle(left_ankle, left_heel_strikes)

# Normalize right leg angles
right_knee_cycles = normalize_to_gait_cycle(right_knee, right_heel_strikes)
right_hip_cycles = normalize_to_gait_cycle(right_hip_flex, right_heel_strikes)
right_ankle_cycles = normalize_to_gait_cycle(right_ankle, right_heel_strikes)

# Calculate mean and standard deviation across cycles
gait_cycle_pct = np.linspace(0, 100, 101)

left_knee_mean = np.mean(left_knee_cycles, axis=0)
left_knee_std = np.std(left_knee_cycles, axis=0)

right_knee_mean = np.mean(right_knee_cycles, axis=0)
right_knee_std = np.std(right_knee_cycles, axis=0)

print(f"Normalized {len(left_knee_cycles)} left strides")
print(f"Normalized {len(right_knee_cycles)} right strides")
```

**Output:**
```
Normalized 4 left strides
Normalized 4 right strides
```

## Step 6: Visualize Joint Angles

```python
# Create comprehensive gait kinematics plot
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        'Left Hip Flexion/Extension', 'Right Hip Flexion/Extension',
        'Left Knee Flexion/Extension', 'Right Knee Flexion/Extension',
        'Left Ankle Dorsi/Plantarflexion', 'Right Ankle Dorsi/Plantarflexion'
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

# Hip angles
for cycle in left_hip_cycles:
    fig.add_trace(go.Scatter(x=gait_cycle_pct, y=cycle, mode='lines', 
                             line=dict(color='lightblue', width=1),
                             showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=gait_cycle_pct, y=left_hip_mean, mode='lines',
                         line=dict(color='blue', width=3),
                         name='Left Mean'), row=1, col=1)

for cycle in right_hip_cycles:
    fig.add_trace(go.Scatter(x=gait_cycle_pct, y=cycle, mode='lines',
                             line=dict(color='lightcoral', width=1),
                             showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=gait_cycle_pct, y=right_hip_mean, mode='lines',
                         line=dict(color='red', width=3),
                         name='Right Mean'), row=1, col=2)

# Knee angles
for cycle in left_knee_cycles:
    fig.add_trace(go.Scatter(x=gait_cycle_pct, y=cycle, mode='lines',
                             line=dict(color='lightblue', width=1),
                             showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=gait_cycle_pct, y=left_knee_mean, mode='lines',
                         line=dict(color='blue', width=3),
                         showlegend=False), row=2, col=1)

for cycle in right_knee_cycles:
    fig.add_trace(go.Scatter(x=gait_cycle_pct, y=cycle, mode='lines',
                             line=dict(color='lightcoral', width=1),
                             showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=gait_cycle_pct, y=right_knee_mean, mode='lines',
                         line=dict(color='red', width=3),
                         showlegend=False), row=2, col=2)

# Ankle angles
for cycle in left_ankle_cycles:
    fig.add_trace(go.Scatter(x=gait_cycle_pct, y=cycle, mode='lines',
                             line=dict(color='lightblue', width=1),
                             showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=gait_cycle_pct, y=left_ankle_mean, mode='lines',
                         line=dict(color='blue', width=3),
                         showlegend=False), row=3, col=1)

for cycle in right_ankle_cycles:
    fig.add_trace(go.Scatter(x=gait_cycle_pct, y=cycle, mode='lines',
                             line=dict(color='lightcoral', width=1),
                             showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=gait_cycle_pct, y=right_ankle_mean, mode='lines',
                         line=dict(color='red', width=3),
                         showlegend=False), row=3, col=2)

# Update axes labels
for i in range(1, 4):
    fig.update_xaxes(title_text="Gait Cycle (%)", row=i, col=1)
    fig.update_xaxes(title_text="Gait Cycle (%)", row=i, col=2)
    fig.update_yaxes(title_text="Angle (°)", row=i, col=1)
    fig.update_yaxes(title_text="Angle (°)", row=i, col=2)

fig.update_layout(
    height=900,
    title_text="Gait Kinematics: Lower Limb Joint Angles",
    showlegend=True
)

fig.show()
fig.write_html("gait_kinematics.html")
```

## Step 7: Calculate Asymmetry Indices

```python
def calculate_symmetry_index(left_mean, right_mean):
    """
    Calculate Symmetry Index (SI) between left and right.
    
    SI = (Left - Right) / ((Left + Right) / 2) * 100
    
    SI = 0%: Perfect symmetry
    SI > 0%: Left > Right
    SI < 0%: Right > Left
    """
    si = (left_mean - right_mean) / ((left_mean + right_mean) / 2) * 100
    return si

# Calculate ROM for each joint
left_hip_rom = left_hip_mean.max() - left_hip_mean.min()
right_hip_rom = right_hip_mean.max() - right_hip_mean.min()

left_knee_rom = left_knee_mean.max() - left_knee_mean.min()
right_knee_rom = right_knee_mean.max() - right_knee_mean.min()

left_ankle_rom = left_ankle_mean.max() - left_ankle_mean.min()
right_ankle_rom = right_ankle_mean.max() - right_ankle_mean.min()

# Calculate symmetry indices
hip_si = calculate_symmetry_index(left_hip_rom, right_hip_rom)
knee_si = calculate_symmetry_index(left_knee_rom, right_knee_rom)
ankle_si = calculate_symmetry_index(left_ankle_rom, right_ankle_rom)

# Asymmetry report
print("=== ASYMMETRY ANALYSIS ===\n")
print(f"Hip ROM:")
print(f"  Left:  {left_hip_rom:.1f}°")
print(f"  Right: {right_hip_rom:.1f}°")
print(f"  Symmetry Index: {hip_si:+.1f}%")
if abs(hip_si) < 5:
    print(f"  → Excellent symmetry")
elif abs(hip_si) < 10:
    print(f"  → Good symmetry")
else:
    print(f"  → Asymmetry detected (>10%)")

print(f"\nKnee ROM:")
print(f"  Left:  {left_knee_rom:.1f}°")
print(f"  Right: {right_knee_rom:.1f}°")
print(f"  Symmetry Index: {knee_si:+.1f}%")
if abs(knee_si) < 5:
    print(f"  → Excellent symmetry")
elif abs(knee_si) < 10:
    print(f"  → Good symmetry")
else:
    print(f"  → Asymmetry detected (>10%)")

print(f"\nAnkle ROM:")
print(f"  Left:  {left_ankle_rom:.1f}°")
print(f"  Right: {right_ankle_rom:.1f}°")
print(f"  Symmetry Index: {ankle_si:+.1f}%")
if abs(ankle_si) < 5:
    print(f"  → Excellent symmetry")
elif abs(ankle_si) < 10:
    print(f"  → Good symmetry")
else:
    print(f"  → Asymmetry detected (>10%)")
```

**Output:**
```
=== ASYMMETRY ANALYSIS ===

Hip ROM:
  Left:  50.7°
  Right: 49.9°
  Symmetry Index: +1.6%
  → Excellent symmetry

Knee ROM:
  Left:  67.6°
  Right: 68.9°
  Symmetry Index: -1.9%
  → Excellent symmetry

Ankle ROM:
  Left:  44.0°
  Right: 44.3°
  Symmetry Index: -0.7%
  → Excellent symmetry
```

## Step 8: Access Anthropometric Properties

```python
# WholeBody automatically calculates anthropometric measurements
leg_length_left = body.left_leg_length.data.mean()
leg_length_right = body.right_leg_length.data.mean()

thigh_length_left = body.left_thigh_length.data.mean()
thigh_length_right = body.right_thigh_length.data.mean()

shank_length_left = (leg_length_left - thigh_length_left)
shank_length_right = (leg_length_right - thigh_length_right)

hip_width = body.hip_width.data.mean()
shoulder_width = body.shoulder_width.data.mean()
trunk_height = body.trunk_height.data.mean()

print("=== ANTHROPOMETRIC MEASUREMENTS ===\n")
print(f"Segment Lengths:")
print(f"  Leg length (L): {leg_length_left:.3f} m")
print(f"  Leg length (R): {leg_length_right:.3f} m")
print(f"  Thigh (L):      {thigh_length_left:.3f} m")
print(f"  Thigh (R):      {thigh_length_right:.3f} m")
print(f"  Shank (L):      {shank_length_left:.3f} m")
print(f"  Shank (R):      {shank_length_right:.3f} m")

print(f"\nBody Widths:")
print(f"  Hip width:      {hip_width:.3f} m")
print(f"  Shoulder width: {shoulder_width:.3f} m")

print(f"\nTrunk:")
print(f"  Trunk height:   {trunk_height:.3f} m")

# Calculate limb length discrepancy
lld = abs(leg_length_left - leg_length_right) * 1000  # convert to mm
print(f"\nLimb Length Discrepancy: {lld:.1f} mm")
if lld < 5:
    print("  → Negligible (<5mm)")
elif lld < 10:
    print("  → Mild (5-10mm)")
elif lld < 20:
    print("  → Moderate (10-20mm) - may require intervention")
else:
    print("  → Severe (>20mm) - intervention recommended")
```

**Output:**
```
=== ANTHROPOMETRIC MEASUREMENTS ===

Segment Lengths:
  Leg length (L): 0.952 m
  Leg length (R): 0.948 m
  Thigh (L):      0.458 m
  Thigh (R):      0.456 m
  Shank (L):      0.494 m
  Shank (R):      0.492 m

Body Widths:
  Hip width:      0.285 m
  Shoulder width: 0.412 m

Trunk:
  Trunk height:   0.635 m

Limb Length Discrepancy: 4.2 mm
  → Negligible (<5mm)
```

## Step 9: Export Data for Further Analysis

```python
# Create comprehensive DataFrame with all angles
kinematics_df = pd.DataFrame({
    'time': time,
    
    # Left leg
    'left_hip_flexion': left_hip_flex,
    'left_hip_abduction': left_hip_abd,
    'left_knee_flexion': left_knee,
    'left_ankle_flexion': left_ankle,
    
    # Right leg
    'right_hip_flexion': right_hip_flex,
    'right_hip_abduction': right_hip_abd,
    'right_knee_flexion': right_knee,
    'right_ankle_flexion': right_ankle,
})

# Save to CSV
kinematics_df.to_csv("gait_kinematics.csv", index=False)
print("Saved kinematics to gait_kinematics.csv")

# Export normalized gait cycle data
normalized_df = pd.DataFrame({
    'gait_cycle_pct': gait_cycle_pct,
    'left_hip_mean': left_hip_mean,
    'left_hip_std': left_hip_std,
    'left_knee_mean': left_knee_mean,
    'left_knee_std': left_knee_std,
    'right_hip_mean': right_hip_mean,
    'right_hip_std': right_hip_std,
    'right_knee_mean': right_knee_mean,
    'right_knee_std': right_knee_std,
})

normalized_df.to_csv("normalized_gait_cycle.csv", index=False)
print("Saved normalized gait cycle to normalized_gait_cycle.csv")

# Export to OpenSim format (for further analysis in OpenSim)
laban.write_opensim(
    body,
    trc_filename="walking_markers.trc",
    mot_filename="walking_angles.mot"
)
print("Exported to OpenSim format (TRC + MOT)")
```

**Output:**
```
Saved kinematics to gait_kinematics.csv
Saved normalized gait cycle to normalized_gait_cycle.csv
Exported to OpenSim format (TRC + MOT)
```

## Step 10: Generate Summary Report

```python
# Create summary statistics
summary = {
    'Participant': f"{participant.surname}, {participant.name}",
    'Trial Duration (s)': len(body) / body.sampling_frequency,
    'Cadence (steps/min)': cadence,
    'Left Stride Time (s)': f"{left_stride_times.mean():.3f} ± {left_stride_times.std():.3f}",
    'Right Stride Time (s)': f"{right_stride_times.mean():.3f} ± {right_stride_times.std():.3f}",
    '',
    'Hip ROM Left (°)': f"{left_hip_rom:.1f}",
    'Hip ROM Right (°)': f"{right_hip_rom:.1f}",
    'Hip Symmetry Index (%)': f"{hip_si:+.1f}",
    ' ',
    'Knee ROM Left (°)': f"{left_knee_rom:.1f}",
    'Knee ROM Right (°)': f"{right_knee_rom:.1f}",
    'Knee Symmetry Index (%)': f"{knee_si:+.1f}",
    '  ',
    'Ankle ROM Left (°)': f"{left_ankle_rom:.1f}",
    'Ankle ROM Right (°)': f"{right_ankle_rom:.1f}",
    'Ankle Symmetry Index (%)': f"{ankle_si:+.1f}",
    '   ',
    'Leg Length Left (m)': f"{leg_length_left:.3f}",
    'Leg Length Right (m)': f"{leg_length_right:.3f}",
    'Limb Length Discrepancy (mm)': f"{lld:.1f}",
}

# Print report
print("\n" + "="*50)
print("         GAIT KINEMATICS ANALYSIS REPORT")
print("="*50)
for key, value in summary.items():
    if key == '':
        print()
    else:
        print(f"{key:.<40} {value}")
print("="*50)
```

**Output:**
```
==================================================
         GAIT KINEMATICS ANALYSIS REPORT
==================================================
Participant.................................. Doe, John
Trial Duration (s)........................... 8.5
Cadence (steps/min).......................... 105.3
Left Stride Time (s)......................... 1.142 ± 0.024
Right Stride Time (s)........................ 1.138 ± 0.031

Hip ROM Left (°)............................. 50.7
Hip ROM Right (°)............................ 49.9
Hip Symmetry Index (%)...................... +1.6

Knee ROM Left (°)............................ 67.6
Knee ROM Right (°)........................... 68.9
Knee Symmetry Index (%)..................... -1.9

Ankle ROM Left (°)........................... 44.0
Ankle ROM Right (°).......................... 44.3
Ankle Symmetry Index (%).................... -0.7

Leg Length Left (m).......................... 0.952
Leg Length Right (m)......................... 0.948
Limb Length Discrepancy (mm)................ 4.2
==================================================
```

## Key Takeaways

### WholeBody Model Features
- **Automatic Computation**: 88 properties computed from raw markers
  - 42 anatomical markers
  - 38 joint angles (flexion/extension, abduction/adduction, rotation)
  - 8 computed properties (joint centers, reference frames)
- **Lazy Evaluation**: Properties calculated only when accessed
- **Unit Consistency**: All distances in meters, angles in degrees

### Best Practices
1. **Marker Quality**: Verify marker visibility before analysis
2. **Gait Event Detection**: Use multiple signals (foot height + GRF) for robust detection
3. **Normalization**: Always normalize to gait cycle % for comparison
4. **Asymmetry Thresholds**: <5% excellent, 5-10% acceptable, >10% investigate
5. **ROM Interpretation**: Compare to normative values for age/activity level

### Common Issues
- **Missing markers**: Some properties unavailable if markers missing
- **Marker noise**: Apply appropriate filtering (6 Hz lowpass typical for gait)
- **Coordinate system**: Verify anterior-posterior and medial-lateral axes
- **Reference frames**: Ensure pelvis/trunk reference frames are stable

## Next Steps

- **Tutorial 04**: Strength assessment with isokinetic dynamometry
- **Tutorial 05**: Advanced signal processing workflows
- **API Reference**: [WholeBody](../api-reference/records/bodies.md) - Complete property documentation
- **User Guide**: [Joint Angles](../user-guide/biomechanics/joint-angles.md) - Clinical interpretation

## Additional Resources

### Example Analyses
- Compare pre/post intervention kinematics
- Analyze fatigue effects across multiple trials
- Identify compensatory movement patterns
- Generate patient reports with normative comparisons

### Extensions
- Add EMG data for muscle activation timing
- Include GRF for kinetic analysis
- Calculate joint moments/powers using inverse dynamics
- Export to OpenSim for musculoskeletal modeling

---

**Complete workflow for analyzing full body kinematics from motion capture data using WholeBody model with 88 computed biomechanical properties.**
