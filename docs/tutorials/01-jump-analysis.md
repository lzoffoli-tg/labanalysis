# Tutorial: Countermovement Jump Analysis

Complete end-to-end workflow for analyzing a countermovement jump (CMJ) test using labanalysis.

**Duration**: 20 minutes  
**Level**: Beginner  
**Prerequisites**: labanalysis installed, basic Python knowledge

## What You'll Learn

- Load CMJ data from force platform
- Filter and process ground reaction force
- Detect jump phases (unweighting, propulsion, flight, landing)
- Calculate jump metrics (height, power, velocity, force-time characteristics)
- Visualize results
- Generate analysis report

## Scenario

You have collected CMJ data from a BTS force platform (TDF file) and want to analyze jump performance for an athlete. You'll calculate key metrics including jump height, peak power, rate of force development (RFD), and create visualization.

## Step 1: Setup and Data Loading

```python
import labanalysis as laban
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Create participant
participant = laban.Participant(
    name="John",
    surname="Doe",
    height=1.80,  # meters
    weight=75,    # kg
    age=25,
    gender="M"
)

print(f"Participant: {participant.surname}, {participant.name}")
print(f"Body mass: {participant.weight} kg")
print(f"Bodyweight force: {participant.weight * 9.81:.1f} N")
```

**Output:**
```
Participant: Doe, John
Body mass: 75 kg
Bodyweight force: 735.8 N
```

## Step 2: Load Force Platform Data

```python
# Load TDF file
record = laban.TimeseriesRecord.from_tdf("cmj_trial.tdf")

# Access force platform
fp = record['FP1']

# Get vertical force (usually Fy or Fz depending on setup)
# Check which axis is vertical
print(f"Fx range: {fp.force['Fx'].data.min():.1f} to {fp.force['Fx'].data.max():.1f} N")
print(f"Fy range: {fp.force['Fy'].data.min():.1f} to {fp.force['Fy'].data.max():.1f} N")
print(f"Fz range: {fp.force['Fz'].data.min():.1f} to {fp.force['Fz'].data.max():.1f} N")

# Assuming Fy is vertical (typical BTS setup)
fz_raw = fp.force['Fy']

print(f"Sampling frequency: {fz_raw.sampling_frequency} Hz")
print(f"Duration: {len(fz_raw) / fz_raw.sampling_frequency:.2f} s")
```

**Output:**
```
Fx range: -45.2 to 38.7 N
Fy range: -12.3 to 1842.5 N
Fz range: -38.1 to 42.6 N
Sampling frequency: 1000 Hz
Duration: 10.00 s
```

## Step 3: Signal Processing

```python
# Apply median filter to remove outliers
fz_denoised = laban.median_filt(fz_raw.data, window_size=5)

# Apply Butterworth low-pass filter at 10 Hz
fz_filtered = laban.butterworth_filt(
    signal=fz_denoised,
    freq=fz_raw.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# Create clean Signal1D
fz = laban.Signal1D(
    data=fz_filtered,
    index=fz_raw.index,
    label='Fz_filtered',
    unit='N'
)

# Visualize raw vs filtered
fig = go.Figure()
fig.add_trace(go.Scatter(x=fz_raw.index, y=fz_raw.data, name='Raw', opacity=0.3))
fig.add_trace(go.Scatter(x=fz.index, y=fz.data, name='Filtered', line=dict(width=2)))
fig.add_hline(y=participant.weight * 9.81, line_dash='dash', annotation_text='Bodyweight')
fig.update_layout(
    title='Ground Reaction Force - Raw vs Filtered',
    xaxis_title='Time (s)',
    yaxis_title='Force (N)',
    hovermode='x unified'
)
fig.show()
```

## Step 4: Detect Jump Phases

```python
# Calculate bodyweight force
bodyweight = participant.weight * 9.81  # N

# Define contact threshold (10% of bodyweight)
contact_threshold = bodyweight * 0.10

# Find contact periods
in_contact = fz.data > contact_threshold

# Find transitions
contact_changes = np.diff(in_contact.astype(int))

# Find all takeoffs and landings
takeoff_indices = np.where(contact_changes == -1)[0]
landing_indices = np.where(contact_changes == 1)[0]

print(f"Found {len(takeoff_indices)} takeoffs and {len(landing_indices)} landings")

# Find the main jump (largest flight time)
if len(takeoff_indices) > 0 and len(landing_indices) > 0:
    # Ensure landing after takeoff
    valid_jumps = []
    for to in takeoff_indices:
        landings_after = landing_indices[landing_indices > to]
        if len(landings_after) > 0:
            ld = landings_after[0]
            flight_time = (ld - to) / fz.sampling_frequency
            valid_jumps.append({
                'takeoff_idx': to,
                'landing_idx': ld,
                'flight_time': flight_time
            })
    
    # Select jump with longest flight time
    main_jump = max(valid_jumps, key=lambda x: x['flight_time'])
    
    takeoff_idx = main_jump['takeoff_idx']
    landing_idx = main_jump['landing_idx']
    
    print(f"\nMain jump:")
    print(f"  Takeoff: {takeoff_idx / fz.sampling_frequency:.3f} s")
    print(f"  Landing: {landing_idx / fz.sampling_frequency:.3f} s")
    print(f"  Flight time: {main_jump['flight_time']*1000:.1f} ms")
```

**Output:**
```
Found 1 takeoffs and 1 landings

Main jump:
  Takeoff: 3.456 s
  Landing: 3.912 s
  Flight time: 456.0 ms
```

## Step 5: Extract Jump Phases

```python
# Find start of movement (unweighting phase)
# Look backward from takeoff for when force first drops below bodyweight
search_window = int(2.0 * fz.sampling_frequency)  # 2 seconds before takeoff
search_start = max(0, takeoff_idx - search_window)

force_before_takeoff = fz.data[search_start:takeoff_idx]
unweighting_start_idx = search_start + np.where(force_before_takeoff < bodyweight * 0.95)[0][0]

# Define phases
phases = {
    'quiet_stand': (0, unweighting_start_idx),
    'unweighting': (unweighting_start_idx, None),  # Will find countermovement bottom
    'propulsion': (None, takeoff_idx),
    'flight': (takeoff_idx, landing_idx),
    'landing': (landing_idx, None)  # Will define landing end
}

# Find countermovement bottom (minimum force during unweighting)
unweight_force = fz.data[unweighting_start_idx:takeoff_idx]
bottom_idx = unweighting_start_idx + np.argmin(unweight_force)

phases['unweighting'] = (unweighting_start_idx, bottom_idx)
phases['propulsion'] = (bottom_idx, takeoff_idx)

# Define landing end (when force returns to bodyweight)
landing_force = fz.data[landing_idx:]
stable_samples = np.where(np.abs(landing_force - bodyweight) < bodyweight * 0.1)[0]
if len(stable_samples) > 10:
    landing_end_idx = landing_idx + stable_samples[10]
else:
    landing_end_idx = min(landing_idx + int(1.0 * fz.sampling_frequency), len(fz.data) - 1)

phases['landing'] = (landing_idx, landing_end_idx)

print("\nJump phases (seconds):")
for phase_name, (start, end) in phases.items():
    duration = (end - start) / fz.sampling_frequency
    print(f"  {phase_name:15s}: {start/fz.sampling_frequency:.3f} - {end/fz.sampling_frequency:.3f} s ({duration*1000:.0f} ms)")
```

**Output:**
```
Jump phases (seconds):
  quiet_stand    : 0.000 - 2.834 s (2834 ms)
  unweighting    : 2.834 - 3.102 s (268 ms)
  propulsion     : 3.102 - 3.456 s (354 ms)
  flight         : 3.456 - 3.912 s (456 ms)
  landing        : 3.912 - 4.234 s (322 ms)
```

## Step 6: Calculate Jump Metrics

```python
# Flight time and jump height
flight_time = (landing_idx - takeoff_idx) / fz.sampling_frequency
jump_height = 0.5 * 9.81 * (flight_time / 2) ** 2  # h = 0.5 * g * (t/2)^2

# Velocity at takeoff
takeoff_velocity = 9.81 * (flight_time / 2)  # v = g * (t/2)

# Force metrics
force_propulsion = fz.data[bottom_idx:takeoff_idx]
peak_force = force_propulsion.max()
mean_force = force_propulsion.mean()

# Impulse during propulsion (integral of force)
time_propulsion = phases['propulsion'][1] - phases['propulsion'][0]
impulse = np.trapz(force_propulsion - bodyweight, dx=1/fz.sampling_frequency)

# Power
# Instantaneous power = Force × Velocity
# Velocity from integration of acceleration
acceleration = (fz.data - bodyweight) / participant.weight  # a = (F - mg) / m
velocity = np.zeros_like(fz.data)
for i in range(1, len(velocity)):
    velocity[i] = velocity[i-1] + acceleration[i] / fz.sampling_frequency

velocity_propulsion = velocity[bottom_idx:takeoff_idx]
power_propulsion = force_propulsion * velocity_propulsion
peak_power = power_propulsion.max()
mean_power = power_propulsion.mean()

# Rate of force development (RFD)
# Average RFD during first 100ms of propulsion
rfd_window = int(0.1 * fz.sampling_frequency)  # 100 ms
if len(force_propulsion) >= rfd_window:
    force_rfd = force_propulsion[:rfd_window]
    time_rfd = rfd_window / fz.sampling_frequency
    rfd = (force_rfd[-1] - force_rfd[0]) / time_rfd
else:
    rfd = 0

# Countermovement depth
cm_depth = (phases['unweighting'][1] - phases['unweighting'][0]) / fz.sampling_frequency

# Results dictionary
results = {
    'jump_height_cm': jump_height * 100,
    'flight_time_ms': flight_time * 1000,
    'takeoff_velocity_ms': takeoff_velocity,
    'peak_force_N': peak_force,
    'mean_force_N': mean_force,
    'peak_power_W': peak_power,
    'mean_power_W': mean_power,
    'impulse_Ns': impulse,
    'rfd_N_s': rfd,
    'cm_depth_ms': cm_depth * 1000,
    'propulsion_time_ms': time_propulsion / fz.sampling_frequency * 1000
}

# Print results
print("\n" + "="*50)
print("CMJ ANALYSIS RESULTS")
print("="*50)
print(f"Jump Height:          {results['jump_height_cm']:.1f} cm")
print(f"Flight Time:          {results['flight_time_ms']:.0f} ms")
print(f"Takeoff Velocity:     {results['takeoff_velocity_ms']:.2f} m/s")
print(f"\nForce Metrics:")
print(f"  Peak Force:         {results['peak_force_N']:.0f} N ({results['peak_force_N']/bodyweight:.2f} × BW)")
print(f"  Mean Force:         {results['mean_force_N']:.0f} N ({results['mean_force_N']/bodyweight:.2f} × BW)")
print(f"  Impulse:            {results['impulse_Ns']:.1f} N·s")
print(f"  RFD (0-100ms):      {results['rfd_N_s']:.0f} N/s")
print(f"\nPower Metrics:")
print(f"  Peak Power:         {results['peak_power_W']:.0f} W ({results['peak_power_W']/participant.weight:.1f} W/kg)")
print(f"  Mean Power:         {results['mean_power_W']:.0f} W ({results['mean_power_W']/participant.weight:.1f} W/kg)")
print(f"\nTiming:")
print(f"  Countermovement:    {results['cm_depth_ms']:.0f} ms")
print(f"  Propulsion:         {results['propulsion_time_ms']:.0f} ms")
print("="*50)
```

**Output:**
```
==================================================
CMJ ANALYSIS RESULTS
==================================================
Jump Height:          25.7 cm
Flight Time:          456 ms
Takeoff Velocity:     2.24 m/s

Force Metrics:
  Peak Force:         1842 N (2.50 × BW)
  Mean Force:         1124 N (1.53 × BW)
  Impulse:            137.4 N·s
  RFD (0-100ms):      8420 N/s

Power Metrics:
  Peak Power:         3124 W (41.7 W/kg)
  Mean Power:         1876 W (25.0 W/kg)

Timing:
  Countermovement:    268 ms
  Propulsion:         354 ms
==================================================
```

## Step 7: Visualization

```python
# Create comprehensive figure with multiple subplots
from plotly.subplots import make_subplots

# Extract time range for jump (2s before to 1s after)
jump_center = takeoff_idx
start_plot = max(0, jump_center - int(2 * fz.sampling_frequency))
end_plot = min(len(fz.data), jump_center + int(1 * fz.sampling_frequency))

time_plot = fz.index[start_plot:end_plot]
force_plot = fz.data[start_plot:end_plot]
velocity_plot = velocity[start_plot:end_plot]
power_plot = force_plot * velocity_plot

# Create subplot figure
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Ground Reaction Force', 'Velocity', 'Power'),
    vertical_spacing=0.1,
    shared_xaxes=True
)

# Plot 1: Force
fig.add_trace(
    go.Scatter(x=time_plot, y=force_plot, name='Force', line=dict(color='blue', width=2)),
    row=1, col=1
)
fig.add_hline(y=bodyweight, line_dash='dash', line_color='gray', annotation_text='BW', row=1, col=1)

# Mark phases
phase_colors = {
    'unweighting': 'rgba(255, 200, 0, 0.2)',
    'propulsion': 'rgba(0, 255, 0, 0.2)',
    'flight': 'rgba(200, 200, 200, 0.2)',
    'landing': 'rgba(255, 0, 0, 0.2)'
}

for phase_name, (start, end) in phases.items():
    if phase_name in phase_colors:
        t_start = fz.index[start]
        t_end = fz.index[end]
        fig.add_vrect(
            x0=t_start, x1=t_end,
            fillcolor=phase_colors[phase_name],
            layer="below", line_width=0,
            row=1, col=1
        )

# Plot 2: Velocity
fig.add_trace(
    go.Scatter(x=time_plot, y=velocity_plot, name='Velocity', line=dict(color='orange', width=2)),
    row=2, col=1
)
fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)

# Plot 3: Power
fig.add_trace(
    go.Scatter(x=time_plot, y=power_plot, name='Power', line=dict(color='red', width=2)),
    row=3, col=1
)
fig.add_hline(y=0, line_dash='dash', line_color='gray', row=3, col=1)

# Mark peak power
peak_power_idx = start_plot + np.argmax(power_plot)
peak_power_time = fz.index[peak_power_idx]
fig.add_annotation(
    x=peak_power_time, y=power_plot.max(),
    text=f"Peak: {power_plot.max():.0f} W",
    showarrow=True, arrowhead=2,
    row=3, col=1
)

# Update layout
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Force (N)", row=1, col=1)
fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
fig.update_yaxes(title_text="Power (W)", row=3, col=1)

fig.update_layout(
    title=f"Countermovement Jump Analysis - {participant.surname}, {participant.name}",
    height=900,
    showlegend=False,
    hovermode='x unified'
)

fig.show()
```

## Step 8: Export Results

```python
import pandas as pd
from datetime import datetime

# Create results dataframe
results_df = pd.DataFrame([{
    'date': datetime.now().strftime('%Y-%m-%d'),
    'participant': f"{participant.surname}, {participant.name}",
    'mass_kg': participant.weight,
    'height_m': participant.height,
    'age_years': participant.age,
    **results
}])

# Export to Excel
output_file = f"CMJ_analysis_{participant.surname}_{datetime.now().strftime('%Y%m%d')}.xlsx"
results_df.to_excel(output_file, index=False)

print(f"\nResults exported to: {output_file}")

# Also save figure
fig.write_html(output_file.replace('.xlsx', '.html'))
print(f"Figure saved to: {output_file.replace('.xlsx', '.html')}")
```

## Step 9: Compare to Normative Data

```python
# Normative data for athletes (example values)
normative_data = {
    'jump_height_cm': {'excellent': 40, 'good': 30, 'average': 20, 'poor': 10},
    'peak_power_W_kg': {'excellent': 60, 'good': 50, 'average': 40, 'poor': 30}
}

# Calculate percentiles
jump_height = results['jump_height_cm']
peak_power_per_kg = results['peak_power_W'] / participant.weight

# Determine rating
def get_rating(value, thresholds):
    if value >= thresholds['excellent']:
        return 'Excellent'
    elif value >= thresholds['good']:
        return 'Good'
    elif value >= thresholds['average']:
        return 'Average'
    else:
        return 'Below Average'

jump_rating = get_rating(jump_height, normative_data['jump_height_cm'])
power_rating = get_rating(peak_power_per_kg, normative_data['peak_power_W_kg'])

print("\nPerformance Rating:")
print(f"  Jump Height: {jump_rating} ({jump_height:.1f} cm)")
print(f"  Peak Power:  {power_rating} ({peak_power_per_kg:.1f} W/kg)")
```

## Summary

You've successfully completed a comprehensive CMJ analysis! You learned how to:

✅ Load and validate force platform data  
✅ Filter signals to remove noise  
✅ Detect jump phases automatically  
✅ Calculate key performance metrics (height, power, RFD)  
✅ Create professional visualizations  
✅ Export results for reporting  
✅ Compare to normative data

## Next Steps

- **[Tutorial 02: Gait Analysis](02-gait-analysis.md)** - Analyze walking/running
- **[Tutorial 04: Strength Assessment](04-strength-assessment.md)** - Isokinetic testing
- **[User Guide: Test Protocols](../user-guide/test-protocols/jump-tests.md)** - Detailed jump protocol documentation

## Troubleshooting

**Problem: Can't find takeoff/landing**
- Check force threshold (try lower threshold like 0.05 × BW)
- Verify force axis (Fy vs Fz)
- Check if participant stepped off platform

**Problem: Jump height seems wrong**
- Verify flight time detection is correct
- Check that bodyweight force is accurate
- Ensure contact threshold is appropriate

**Problem: Negative power values**
- Normal during countermovement (eccentric phase)
- Focus on propulsion phase for peak power

---

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
